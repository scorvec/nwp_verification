"""Grid-to-point interpolation, built once per grid and station set and reused for every field.

Regular lat/lon grids use bilinear weights (either longitude convention,
either latitude order, periodic in longitude when the grid is global).
Curvilinear grids (Lambert, polar stereographic, rotated, WRF) use inverse-
distance weights of the four nearest grid points on the sphere; stations more
than 1.5 grid spacings from the nearest point are outside the domain.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mverify.io.open import Grid


def _xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


class PointInterpolator:
    """Weights mapping a 2-D field on ``grid`` to the points (lat, lon)."""

    def __init__(self, grid: Grid, lat: np.ndarray, lon: np.ndarray):
        self.n = len(lat)
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        if grid.regular:
            self.idx, self.w, self.inside = self._bilinear(grid, lat, lon)
        else:
            self.idx, self.w, self.inside = self._idw(grid, lat, lon)

    @staticmethod
    def _bilinear(grid: Grid, lat, lon):
        glat, glon = grid.lat, grid.lon
        ny, nx = glat.size, glon.size
        flip = glat[0] > glat[-1]
        la = glat[::-1] if flip else glat
        lo0 = glon[0]
        dlon = (glon[-1] - glon[0]) / (nx - 1)
        periodic = abs(nx * dlon - 360.0) < 1e-3 * abs(dlon) + 1e-6
        x = ((lon - lo0) % 360.0) / dlon            # fractional column index, 0..nx (periodic)
        if not periodic:
            raw = (lon - lo0) / dlon
            alt = (lon - lo0 + (360.0 if lo0 >= 0 else -360.0)) / dlon
            x = np.where((raw >= 0) & (raw <= nx - 1), raw, alt)
        i0 = np.clip(np.searchsorted(la, lat) - 1, 0, ny - 2)
        fy = (lat - la[i0]) / (la[i0 + 1] - la[i0])
        j0 = np.floor(x).astype(int)
        fx = x - j0
        inside = (lat >= la[0]) & (lat <= la[-1])
        if periodic:
            j0 %= nx
            j1 = (j0 + 1) % nx
        else:
            inside &= (x >= 0) & (x <= nx - 1)
            j0 = np.clip(j0, 0, nx - 2)
            j1 = j0 + 1
            fx = np.clip(x - j0, 0, 1)
        if flip:
            r0, r1 = ny - 1 - i0, ny - 2 - i0
        else:
            r0, r1 = i0, i0 + 1
        idx = np.stack([r0 * nx + j0, r0 * nx + j1, r1 * nx + j0, r1 * nx + j1], axis=1)
        w = np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], axis=1)
        return idx, w, inside

    @staticmethod
    def _idw(grid: Grid, lat, lon):
        pts = _xyz(grid.lat.ravel(), grid.lon.ravel())
        tree = cKDTree(pts)
        d, idx = tree.query(_xyz(lat, lon), k=4)
        sample = pts[:: max(1, len(pts) // 2000)]
        spacing = np.median(tree.query(sample, k=2)[0][:, 1])
        inside = d[:, 0] <= 1.5 * spacing
        w = 1.0 / np.maximum(d, 1e-12) ** 2
        w /= w.sum(axis=1, keepdims=True)
        return idx, w, inside

    def __call__(self, field: np.ndarray, which: np.ndarray | None = None) -> np.ndarray:
        """Field values at all points, or at the point indices ``which``; NaN outside the grid."""
        flat = np.asarray(field, float).ravel()
        idx, w, inside = (self.idx, self.w, self.inside) if which is None else (
            self.idx[which], self.w[which], self.inside[which])
        out = (flat[idx] * w).sum(axis=1)
        out[~inside] = np.nan
        return out
