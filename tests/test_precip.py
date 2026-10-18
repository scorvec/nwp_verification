"""Precipitation: accumulations from files, regridding, scores, analyses in the store, end to end."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mverify.io.open import Field, Grid, open_forecast
from mverify.precip import scores as P
from mverify.precip.accum import Accumulations
from mverify.precip.analyses import PRODUCTS, _finish
from mverify.precip.regrid import Regridder
from mverify.verify import verify

H = pd.Timedelta("1h")
GRID = Grid(np.array([10.0, 0.0]), np.array([0.0, 10.0]))


def _field(start_h, end_h, value):
    da = xr.DataArray(np.full((2, 2), float(value)), dims=("y", "x"))
    return Field("tp", pd.Timestamp("2026-10-13"), end_h * H, None, GRID, da, "mm", accum=(start_h * H, end_h * H))


@pytest.mark.parametrize("fields, window, expected", [
    ([(0, 12, 3), (0, 36, 10)], (12, 36), 7),               # running totals: difference
    ([(12, 18, 1), (18, 24, 2), (24, 36, 4)], (12, 36), 7),  # buckets: chain
    ([(0, 30, 9), (24, 30, 2), (0, 12, 3), (12, 24, 4)], (12, 30), 6),  # GFS-style mix: 9 - 3
    ([(0, 12, 3), (0, 36, 2.9999)], (12, 36), 0),           # packing noise never goes negative
    ([(0, 12, 3)], (12, 36), None),                          # not derivable
])
def test_any_window_is_built_from_the_accumulations_a_file_has(fields, window, expected):
    acc = Accumulations([_field(*f) for f in fields])
    got = acc.total(window[0] * H, window[1] * H)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(np.full((2, 2), expected))


def test_grib_file_with_two_accumulations_of_the_same_field(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    pytest.importorskip("cfgrib")
    p = tmp_path / "gfs.t00z.pgrb2.f030"
    with open(p, "wb") as fh:
        for start, value in ((0, 0.009), (24, 0.002)):  # ECMWF-table tp is in metres
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set_key_vals(h, {"Ni": 36, "Nj": 19, "latitudeOfFirstGridPointInDegrees": 90,
                                           "longitudeOfFirstGridPointInDegrees": 0,
                                           "latitudeOfLastGridPointInDegrees": -90,
                                           "longitudeOfLastGridPointInDegrees": 350,
                                           "iDirectionIncrementInDegrees": 10, "jDirectionIncrementInDegrees": 10,
                                           "dataDate": 20261013, "dataTime": 0})
            eccodes.codes_set(h, "productDefinitionTemplateNumber", 8)
            eccodes.codes_set(h, "shortName", "tp")
            eccodes.codes_set_key_vals(h, {"stepType": "accum", "stepUnits": 1, "startStep": start, "endStep": 30})
            eccodes.codes_set_values(h, np.full(36 * 19, value))
            eccodes.codes_write(h, fh)
            eccodes.codes_release(h)
    fields = sorted(open_forecast(p).fields, key=lambda f: f.accum)
    assert [(f.accum[0] / H, f.accum[1] / H) for f in fields] == [(0, 30), (24, 30)]
    assert [f.values().mean() for f in fields] == pytest.approx([9.0, 2.0])


def test_coarsening_averages_the_cells_it_covers():
    src = Grid(np.arange(-89.875, 90, 0.25), np.arange(0.125, 360, 0.25))
    dst = Grid(np.arange(-89.75, 90, 0.5), np.arange(-179.75, 180, 0.5))
    rg = Regridder(src, dst)
    assert rg.coarsen
    lat2, lon2 = np.meshgrid(src.lat, src.lon, indexing="ij")
    out = rg(np.where(lon2 < 180, 4.0, 0.0))  # rain over the eastern hemisphere only
    east = dst.lon > 0
    assert out[:, east] == pytest.approx(4.0) and out[:, ~east] == pytest.approx(0.0)
    # Area weighting pulls a cell's mean latitude slightly equatorward, by ~0.06 deg next to the pole.
    smooth = rg(lat2 + 0.0)
    assert smooth == pytest.approx(np.repeat(dst.lat[:, None], dst.lon.size, axis=1), abs=0.07)
    mid = np.abs(dst.lat) < 60
    assert smooth[mid] == pytest.approx(np.repeat(dst.lat[mid, None], dst.lon.size, axis=1), abs=5e-3)


def test_refining_interpolates():
    src = Grid(np.arange(20.0, 50.1, 1.0), np.arange(-120.0, -79.9, 1.0))
    dst = Grid(np.array([[30.25, 30.5], [31.0, 31.75]]), np.array([[-100.5, -99.25], [-100.0, -98.75]]))
    rg = Regridder(src, dst)
    assert not rg.coarsen
    la, lo = np.meshgrid(src.lat, src.lon, indexing="ij")
    assert rg(2 * la + lo) == pytest.approx(2 * dst.lat + dst.lon)


def test_contingency_scores_and_fss_by_hand():
    o = np.zeros((20, 20))
    o[5:10, 5:10] = 20.0
    f = np.zeros((20, 20))
    f[5:10, 7:12] = 20.0  # displaced by two cells: 15 hits, 10 misses, 10 false alarms
    s = P.case_sums(f, o, np.ones_like(o, bool), np.ones_like(o), cell_km=10, periodic_x=False,
                    thresholds=(10.0,), fss_km=(10, 50))
    c = s[10.0]
    assert (c["hits"], c["misses"], c["false_alarms"]) == (15, 10, 10)
    rows = pd.DataFrame([{"m": "M", "threshold": 10.0, "n_cases": 1, **c}])
    t = P.finish(rows, ["m"], fss_km=(10, 50)).iloc[0]
    random_hits = 25 * 25 / 400
    assert t["ets"] == pytest.approx((15 - random_hits) / (35 - random_hits))
    assert t["fbias"] == 1 and t["pod"] == pytest.approx(0.6) and t["far"] == pytest.approx(0.4)
    assert t["fss_10km"] == pytest.approx(1 - 20 / 50)    # grid scale: 1 - (misses + false alarms) / (2 x 25)
    assert t["fss_50km"] > t["fss_10km"]                   # neighbourhoods forgive the displacement


def test_cells_without_analysis_are_left_out():
    o = np.array([[1.0, np.nan], [0.0, 30.0]])
    f = np.array([[1.0, 99.0], [0.0, 30.0]])
    s = P.case_sums(f, o, np.ones_like(o, bool), np.ones_like(o), 10, False, thresholds=(10.0,), fss_km=(10,))
    assert s["amount"]["w"] == 3 and s["amount"]["sum_err"] == 0
    assert s[10.0]["false_alarms"] == 0


def _analysis(valid, lat, lon, values):
    return _finish(values, lat, lon, "stageiv", valid, 24, ("y", "x"))


def test_analyses_round_trip_through_the_store_and_the_snapshot(filled_store, tmp_path):
    store = filled_store
    from mverify.db.store import ObsStore
    lat, lon = np.meshgrid([30.0, 31.0], [-100.0, -99.0], indexing="ij")
    valid = pd.Timestamp("2026-10-14 12:00")
    store.add_grid(_analysis(valid, lat, lon, np.array([[1.0, -9999.0], [3.0, 4.0]])))
    got = store.get_grid("stageiv", 24, valid)
    assert np.isnan(got.values[0, 1]) and got.values[1, 1] == 4.0
    assert store.grid_times("stageiv", 24, "2026-10-14", "2026-10-15") == [valid]
    store.export_parquet(tmp_path / "snap")
    with ObsStore.snapshot(str(tmp_path / "snap")) as snap:
        assert snap.get_grid("stageiv", 24, valid).values[1, 0] == 3.0
        assert snap.get_grid("stageiv", 24, valid + pd.Timedelta("1D")) is None


def test_product_windows():
    assert PRODUCTS["stageiv"].windows("2026-10-13", "2026-10-15") == [
        pd.Timestamp("2026-10-13 12:00"), pd.Timestamp("2026-10-14 12:00")]
    assert PRODUCTS["gpcp"].url(pd.Timestamp("2026-01-01")).endswith("GPCPDAY_L3_20251231_V3.3.nc4")


def test_a_perfect_precipitation_forecast(tmp_path, store):
    init = pd.Timestamp("2026-10-13 00:00")
    lat, lon = np.arange(50.0, 24.9, -0.5), np.arange(-125.0, -65.0, 0.5)
    _, lo = np.meshgrid(lat, lon, indexing="ij")
    day1 = np.where((lo > -100) & (lo < -90), 20.0, 0.5)  # a rain band
    leads = pd.to_timedelta([12, 36], unit="h")
    running = np.stack([np.zeros_like(day1), day1])[None]  # nothing in 0-12 h, the band in 12-36 h
    xr.Dataset({"tp": (("time", "step", "latitude", "longitude"), running, {"units": "mm"})},
               coords={"time": [init], "step": leads, "latitude": lat, "longitude": lon}).to_netcdf(tmp_path / "f.nc")
    store.add_grid(_finish(day1, lat, lon, "stageiv", init + pd.Timedelta("36h"), 24, ("latitude", "longitude")))
    t = verify({"M": [tmp_path / "f.nc"]}, store, variables=["tp"]).table.set_index("variable")
    assert t.loc["tp24", "lead_h"] == 36 and t.loc["tp24", "bias"] == pytest.approx(0.0, abs=1e-9)
    assert t.loc["tp24>=10mm", "ets"] == pytest.approx(1.0) and t.loc["tp24>=10mm", "fss_100km"] == pytest.approx(1.0)
    assert (t["network"] == "STAGEIV").all() and (t["region"] == "conus").all()


def test_imerg_daily_files_are_read_lat_by_lon(tmp_path):
    from mverify.precip.analyses import read_imerg
    lon, lat = np.arange(-179.95, 180, 0.1)[::360], np.arange(-89.95, 90, 0.1)[::180]
    a = np.zeros((1, lon.size, lat.size), "float32")  # IMERG stores (time, lon, lat)
    a[0, 0, -1] = 12.5
    a[0, 1, 0] = -9999.9
    ds = xr.Dataset({"precipitation": (("time", "lon", "lat"), a, {"_FillValue": np.float32(-9999.9)})},
                    coords={"time": [0], "lon": lon, "lat": lat})
    ds.to_netcdf(tmp_path / "3B-DAY-L.nc4")
    da = read_imerg(tmp_path / "3B-DAY-L.nc4", pd.Timestamp("2026-10-16"))
    assert da.dims == ("latitude", "longitude") and da.shape == (lat.size, lon.size)
    assert da.values[-1, 0] == 12.5 and np.isnan(da.values[0, 1])
    assert da.attrs["product"] == "imerg_late" and da.attrs["valid_time"] == "2026-10-16T00:00:00"
