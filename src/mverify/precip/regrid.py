"""Forecast precipitation onto an analysis grid.

Precipitation is scored on the analysis grid, so every model is judged on the
same cells. Coarsening (a 0.25 deg model onto 0.5 deg GPCP) averages every
model cell that falls in an analysis cell, area-weighted, which conserves the
water a cell receives; interpolating instead would sample the field and keep
noise at scales the analysis does not resolve. Refining or matching (GFS onto
the ~4.8 km Stage IV grid) uses the bilinear / inverse-distance weights of
mverify.interp. The choice is made from the two grid spacings.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from mverify.interp import PointInterpolator
from mverify.io.open import Grid

EARTH_KM_PER_DEG = 111.2
COARSEN_BELOW = 0.75  # box-average when the model spacing is below this fraction of the analysis spacing


def analysis_grid(da: xr.DataArray) -> Grid:
    return Grid(np.asarray(da["latitude"].values, float), np.asarray(da["longitude"].values, float))


def spacing_km(grid: Grid) -> float:
    """Typical distance between neighbouring grid points."""
    if grid.regular:
        return float(abs(np.median(np.diff(grid.lat)))) * EARTH_KM_PER_DEG
    lat, lon = grid.lat, grid.lon
    dy = np.hypot(np.diff(lat, axis=0), np.diff(lon, axis=0) * np.cos(np.deg2rad(lat[1:])))
    return float(np.median(dy)) * EARTH_KM_PER_DEG


class Regridder:
    """Maps fields on ``src`` to the regular or projected analysis grid ``dst``."""

    def __init__(self, src: Grid, dst: Grid):
        self.dst_shape = dst.lat.shape if not dst.regular else (dst.lat.size, dst.lon.size)
        self.coarsen = dst.regular and spacing_km(src) < COARSEN_BELOW * spacing_km(dst)
        if self.coarsen:
            self._bins(src, dst)
        else:
            dlat, dlon = (np.meshgrid(dst.lat, dst.lon, indexing="ij") if dst.regular else (dst.lat, dst.lon))
            self.interp = PointInterpolator(src, dlat.ravel(), dlon.ravel())

    def _bins(self, src: Grid, dst: Grid) -> None:
        slat, slon = np.meshgrid(src.lat, src.lon, indexing="ij") if src.regular else (src.lat, src.lon)
        slat, slon = slat.ravel(), ((slon.ravel() + 180) % 360) - 180
        ny, nx = self.dst_shape
        dlat, dlon = dst.lat, ((dst.lon + 180) % 360) - 180
        # Cell edges halfway between centres; longitudes may run either way round from any start.
        dy, dx = dlat[1] - dlat[0], abs(np.median(np.diff(np.sort(dlon))))
        i = np.floor((slat - (dlat[0] - dy / 2)) / dy).astype(int)
        j = np.floor(((slon - (dlon[0] - dx / 2)) % 360) / dx).astype(int)
        global_x = abs(nx * dx - 360) < 1e-6
        ok = (i >= 0) & (i < ny) & (j >= 0) & ((j < nx) | global_x)
        j = j % nx if global_x else j
        self.cell = np.where(ok, i * nx + np.minimum(j, nx - 1), -1)
        self.w = np.cos(np.deg2rad(slat)) * ok

    def coverage(self, values: np.ndarray) -> np.ndarray:
        """When coarsening: the share of each analysis cell's source points that hold a value (0 where none fall)."""
        v = np.asarray(values, float).ravel()
        n = self.dst_shape[0] * self.dst_shape[1]
        inside = self.cell >= 0
        total = np.bincount(self.cell[inside], minlength=n)
        valid = np.bincount(self.cell[inside & np.isfinite(v)], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, valid / total, 0.0).reshape(self.dst_shape)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, float).ravel()
        if not self.coarsen:
            return self.interp(v).reshape(self.dst_shape)
        ok = (self.cell >= 0) & np.isfinite(v)
        n = self.dst_shape[0] * self.dst_shape[1]
        total = np.bincount(self.cell[ok], weights=(v * self.w)[ok], minlength=n)
        weight = np.bincount(self.cell[ok], weights=self.w[ok], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(weight > 0, total / weight, np.nan)
        return out.reshape(self.dst_shape)
