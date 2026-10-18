"""Every model on one grid before it is scored.

Interpolating each model to the stations from its own grid scores unlike things: a finer grid
carries small-scale detail whose position errors count twice (the double penalty), and a model
published on a grid finer than its own resolution is smoother than its grid suggests. ``coarsen``
averages every field of a forecast onto one regular latitude-longitude grid, by area overlap
(first-order conservative remapping, so a 0.15 deg grid and a 0.25 deg grid both average onto
0.5 deg exactly), and the model's terrain, land fraction and roughness with it. Matching then
interpolates every model to the stations from cells of the same size.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
from scipy import sparse

from mverify.io.open import Field, Forecast, Grid

MIN_COVER = 0.5  # a target cell needs valid source values over at least this fraction of its area


def _edges(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) cell edges of the centres ``c``, halfway between neighbours, in c's order."""
    mid = 0.5 * (c[1:] + c[:-1])
    e = np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])
    return np.minimum(e[:-1], e[1:]), np.maximum(e[:-1], e[1:])


def _lat_weights(src: np.ndarray, dst: np.ndarray, res: float) -> sparse.csr_array:
    """(dst, src) overlap areas of latitude bands, in sin(latitude)."""
    s0, s1 = (np.clip(e, -90, 90) for e in _edges(src))
    d0, d1 = np.clip(dst - res / 2, -90, 90), np.clip(dst + res / 2, -90, 90)
    lo, hi = np.maximum(d0[:, None], s0[None]), np.minimum(d1[:, None], s1[None])
    w = np.where(hi > lo, np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo)), 0.0)
    return sparse.csr_array(w)


def _lon_weights(src: np.ndarray, dst: np.ndarray, res: float) -> sparse.csr_array:
    """(dst, src) overlap lengths of longitude intervals, in degrees, on the circle."""
    s0, s1 = _edges(src)
    shift = np.floor(s0 / 360.0) * 360.0
    s0, s1 = s0 - shift, s1 - shift
    d0, d1 = dst - res / 2, dst + res / 2
    w = np.zeros((dst.size, src.size))
    for k in (-360.0, 0.0, 360.0):
        w += np.clip(np.minimum(d1[:, None], s1[None] + k) - np.maximum(d0[:, None], s0[None] + k), 0, None)
    return sparse.csr_array(w)


def _centres(lo: float, hi: float, res: float) -> np.ndarray:
    """Target centres (multiples of ``res``) whose cells overlap [lo, hi]."""
    return res * np.arange(np.ceil(lo / res - 0.5), np.floor(hi / res + 0.5) + 1)


class Remap:
    """Area-weighted averaging from the regular grid ``src`` onto the regular ``res``-degree grid covering it."""

    def __init__(self, src: Grid, res: float, min_cover: float = MIN_COVER):
        if not src.regular:
            raise ValueError("coarsen needs a regular latitude-longitude grid")
        lat, lon = np.asarray(src.lat, float), np.asarray(src.lon, float)
        dlon = abs(lon[-1] - lon[0]) / (lon.size - 1)
        global_x = abs(lon.size * dlon - 360.0) < 1e-3 * dlon + 1e-6
        la0, la1 = _edges(lat)
        lo0, lo1 = _edges(lon)
        dlat = _centres(max(la0.min(), -90.0), min(la1.max(), 90.0), res)
        dlat = dlat[np.abs(dlat) <= 90.0 + 1e-9]
        dlon_c = res * np.arange(round(360.0 / res)) if global_x else _centres(lo0.min(), lo1.max(), res)
        self.wy = _lat_weights(lat, dlat, res)
        self.wx = _lon_weights(lon, dlon_c, res)
        top, bottom = np.clip(dlat + res / 2, -90, 90), np.clip(dlat - res / 2, -90, 90)
        band = np.sin(np.deg2rad(top)) - np.sin(np.deg2rad(bottom))
        self.area = band[:, None] * res  # a target cell's full area, in the units of the weights
        self.min_cover = min_cover
        self.grid = Grid(dlat, dlon_c)

    def _apply(self, a: np.ndarray) -> np.ndarray:
        return (self.wx @ (self.wy @ a).T).T

    def __call__(self, values: np.ndarray) -> np.ndarray:
        a = np.asarray(values, float)
        ok = np.isfinite(a)
        total = self._apply(np.where(ok, a, 0.0))
        weight = self._apply(ok.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = total / weight
        out[~(weight >= self.min_cover * self.area)] = np.nan
        return out


@dataclass
class CoarseField(Field):
    """A field read from its own grid and averaged onto the common one."""

    remap: Remap | None = field(default=None, repr=False)

    def values(self) -> np.ndarray:
        return self.remap(super().values())


def _add_static(g: Grid, remap: Remap, src: Grid) -> None:
    """The model's terrain, land fraction and roughness averaged onto ``g`` where it has none yet (roughness
    as the mean of its logarithm, the effective roughness of mixed terrain)."""
    if g.orog is None and src.orog is not None:
        g.orog = remap(src.orog)
    if g.lsm is None and src.lsm is not None:
        g.lsm = remap(src.lsm)
    if g.z0 is None and src.z0 is not None:
        g.z0 = np.exp(remap(np.log(np.maximum(src.z0, 1e-6))))


def coarsen(fc: Forecast, res: float) -> Forecast:
    """``fc`` with every field averaged onto the regular ``res``-degree grid (centres at multiples of res).

    Fields from several native grids that cover the same cells share one coarse grid, and with it one set of
    static fields, as matching expects of fields on one grid."""
    remaps: dict[tuple, Remap] = {}
    grids: dict[tuple, Grid] = {}
    out = []
    names = [f.name for f in fields(Field)]
    for f in fc.fields:
        if f.grid.key not in remaps:
            r = remaps[f.grid.key] = Remap(f.grid, res)
            g = grids.setdefault(r.grid.key, Grid(r.grid.lat, r.grid.lon))
            _add_static(g, r, f.grid)
        r = remaps[f.grid.key]
        out.append(CoarseField(**{**{n: getattr(f, n) for n in names}, "grid": grids[r.grid.key]}, remap=r))
    return Forecast(fc.name, out)
