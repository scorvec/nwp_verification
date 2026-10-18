import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mverify.coarsen import CoarseField, Remap, coarsen
from mverify.io.open import Field, Forecast, Grid


def global_grid(res, lon0=0.0, descending=False):
    lat = np.arange(-90, 90 + res / 2, res)
    return Grid(lat[::-1] if descending else lat, np.arange(lon0, lon0 + 360 - res / 2, res))


def mesh(g):
    return np.meshgrid(g.lat, g.lon, indexing="ij")


def area_mean(values, lat):
    w = np.cos(np.deg2rad(lat))
    return float(np.nansum(values * w) / np.sum(w * np.isfinite(values)))


@pytest.mark.parametrize("res", [0.15, 0.25])
def test_a_constant_stays_constant_and_the_global_mean_is_kept(res):
    src = global_grid(res)
    la, lo = mesh(src)
    r = Remap(src, 0.5)
    assert r.grid.lat.size == 361 and r.grid.lon.size == 720
    assert np.allclose(r(np.full(la.shape, 7.0)), 7.0)
    f = 280 + 30 * np.cos(np.deg2rad(la)) + 5 * np.sin(np.deg2rad(3 * lo))
    out = r(f)
    tla, _ = mesh(r.grid)
    # cell-centre cos weights only approximate the polar cells, hence the tolerance
    assert area_mean(out, tla) == pytest.approx(area_mean(f, la), abs=0.02)


@pytest.mark.parametrize("col, want", [(10.0, [0.5, 0.0, 0.0]), (10.25, [0.25, 0.25, 0.0])])
def test_a_quarter_degree_column_counts_by_its_overlap_with_each_half_degree_cell(col, want):
    src = global_grid(0.25)
    a = np.zeros((src.lat.size, src.lon.size))
    a[:, np.flatnonzero(np.isclose(src.lon, col))[0]] = 1.0
    r = Remap(src, 0.5)
    k = [np.flatnonzero(np.isclose(r.grid.lon, x))[0] for x in (10.0, 10.5, 11.0)]
    assert r(a)[180, k] == pytest.approx(want)


def test_longitude_convention_and_latitude_order_do_not_matter():
    a, b = global_grid(0.25), global_grid(0.25, lon0=-180.0, descending=True)
    fa = np.add.outer(np.sin(np.deg2rad(a.lat)), np.cos(np.deg2rad(a.lon)))
    fb = np.add.outer(np.sin(np.deg2rad(b.lat)), np.cos(np.deg2rad(b.lon)))
    assert np.allclose(Remap(a, 0.5)(fa), Remap(b, 0.5)(fb))


def test_missing_values_are_left_out_and_cells_mostly_missing_are_nan():
    src = global_grid(0.25)
    a = np.ones((src.lat.size, src.lon.size))
    a[:, : src.lon.size // 2] = np.nan  # the western half of the 0..360 grid
    out = Remap(src, 0.5)(a)
    assert np.nanmax(np.abs(out - 1)) < 1e-12
    assert np.isnan(out[:, 100]).all() and np.isfinite(out[:, 500]).all()


def test_a_regional_grid_gets_a_regional_target():
    src = Grid(np.arange(20, 50.01, 0.15), np.arange(-130, -60.01, 0.15))
    r = Remap(src, 0.5)
    assert r.grid.lat.min() >= 19.5 and r.grid.lat.max() <= 50.5
    assert r.grid.lon.min() >= -130.5 and r.grid.lon.max() <= -59.5
    out = r(np.full((src.lat.size, src.lon.size), 3.0))
    assert np.allclose(out[np.isfinite(out)], 3.0) and np.isfinite(out[5:-5, 5:-5]).all()


def test_coarsen_keeps_each_field_and_averages_the_static_fields():
    src = global_grid(0.25)
    src.orog = np.full((src.lat.size, src.lon.size), 500.0)
    src.lsm = np.ones_like(src.orog)
    src.z0 = np.full_like(src.orog, 0.1)
    vals = np.full(src.orog.shape, 290.0)
    da = xr.DataArray(vals, dims=("latitude", "longitude"))
    f = Field("t2m", pd.Timestamp("2026-10-01"), pd.Timedelta("24h"), None, src, da, "K")
    out = coarsen(Forecast("M", [f]), 0.5)
    g = out.fields[0]
    assert isinstance(g, CoarseField) and g.var == "t2m" and g.lead == pd.Timedelta("24h")
    assert g.grid.lat.size == 361 and np.allclose(g.grid.orog, 500.0) and np.allclose(g.grid.z0, 0.1)
    assert g.values().shape == (361, 720)
