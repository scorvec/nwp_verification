import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mverify.io.open import _embedded_level, _time_from_name, detect_format, open_forecast


def _by_var(fc):
    return {f.var: f for f in fc.fields}


def test_netcdf_init_and_lead(tmp_path, forecast_ds, init):
    p = tmp_path / "model.nc"
    forecast_ds.to_netcdf(p)
    assert detect_format(p) == "netcdf"
    fc = open_forecast(p, name="M")
    t2m = [f for f in fc.fields if f.var == "t2m"]
    assert [f.lead_h for f in t2m] == [0, 6, 12]
    assert all(f.init == init for f in t2m)
    assert t2m[0].values().max() < 60  # Kelvin converted
    assert t2m[0].grid.orog is not None


def test_zarr_init_and_valid(tmp_path, forecast_ds, init):
    ds = forecast_ds.rename(step="valid_time").assign_coords(valid_time=init + forecast_ds["step"].values)
    p = tmp_path / "model.zarr"
    ds.to_zarr(p)
    assert detect_format(p) == "zarr"
    fc = open_forecast(p)
    assert sorted({f.lead_h for f in fc.fields}) == [0, 6, 12]


def test_valid_time_only_takes_init_from_the_file_name(tmp_path, forecast_ds, init):
    ds = (forecast_ds.isel(init_time=0, drop=True).rename(step="time")
          .assign_coords(time=init + forecast_ds["step"].values + pd.Timedelta("12h")))
    p = tmp_path / "wrf_2026101212.nc"
    ds.to_netcdf(p)
    fc = open_forecast(p)
    assert {f.init for f in fc.fields} == {pd.Timestamp("2026-10-12 12:00")}
    assert sorted({f.lead_h for f in fc.fields}) == [24, 30, 36]


def test_pressure_levels_in_pa_and_embedded_level_names(tmp_path, init):
    lat, lon = np.linspace(50, 30, 5), np.linspace(-110, -90, 6)
    z = np.full((1, 2, 5, 6), 5600.0 * 9.80665)
    ds = xr.Dataset(
        {"z": (("time", "plev", "lat", "lon"), z, {"units": "m2 s-2"}),
         "TMP_850mb": (("time", "lat", "lon"), np.full((1, 5, 6), 271.0), {"units": "K"})},
        coords={"time": [init + pd.Timedelta("24h")], "plev": [50000.0, 85000.0], "lat": lat, "lon": lon},
        attrs={"init_time": str(init)})
    p = tmp_path / "x.nc"
    ds.to_netcdf(p)
    fc = _by_var(open_forecast(p))
    assert set(fc) == {"z500", "z850", "t850"}
    assert fc["z500"].values()[0, 0] == pytest.approx(5600.0)
    assert fc["t850"].values()[0, 0] == pytest.approx(-2.15)
    assert fc["z500"].lead_h == 24


def test_curvilinear_grid_and_ensemble_members(tmp_path, init):
    y, x = np.meshgrid(np.arange(10), np.arange(12), indexing="ij")
    lat2, lon2 = 30 + y * 1.0 + 0.1 * x, -110 + x * 1.2
    t = np.random.default_rng(0).normal(285, 1, (3, 1, 10, 12))
    ds = xr.Dataset({"T2": (("member", "time", "y", "x"), t, {"units": "K"})},
                    coords={"member": [0, 1, 2], "time": [init + pd.Timedelta("6h")],
                            "XLAT": (("y", "x"), lat2), "XLONG": (("y", "x"), lon2)},
                    attrs={"START_DATE": "2026-10-13_00:00:00"})
    p = tmp_path / "wrfout.nc"
    ds.to_netcdf(p)
    fc = open_forecast(p)
    assert sorted(f.member for f in fc.fields) == [0, 1, 2]
    assert not fc.fields[0].grid.regular
    assert fc.fields[0].init == init and fc.fields[0].lead_h == 6


def test_grib_round_trip(tmp_path, init):
    eccodes = pytest.importorskip("eccodes")
    pytest.importorskip("cfgrib")
    p = tmp_path / "gfs.t00z.pgrb2.f006"
    with open(p, "wb") as fh:
        for short, level, value in (("2t", 2, 290.0), ("prmsl", 0, 101300.0)):
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set_key_vals(h, {"Ni": 36, "Nj": 19, "latitudeOfFirstGridPointInDegrees": 90,
                                           "longitudeOfFirstGridPointInDegrees": 0,
                                           "latitudeOfLastGridPointInDegrees": -90,
                                           "longitudeOfLastGridPointInDegrees": 350,
                                           "iDirectionIncrementInDegrees": 10, "jDirectionIncrementInDegrees": 10,
                                           "dataDate": int(init.strftime("%Y%m%d")), "dataTime": 0,
                                           "stepUnits": 1, "endStep": 6})
            eccodes.codes_set(h, "shortName", short)
            if short == "2t":
                eccodes.codes_set_key_vals(h, {"typeOfFirstFixedSurface": 103, "scaledValueOfFirstFixedSurface": 2,
                                               "scaleFactorOfFirstFixedSurface": 0})
            eccodes.codes_set_values(h, np.full(36 * 19, value))
            eccodes.codes_write(h, fh)
            eccodes.codes_release(h)
    assert detect_format(p) == "grib"
    fc = _by_var(open_forecast(p))
    assert set(fc) == {"t2m", "msl"}
    assert fc["t2m"].values().mean() == pytest.approx(16.85)
    assert fc["msl"].values().mean() == pytest.approx(1013.0)
    assert fc["t2m"].lead_h == 6 and fc["t2m"].init == init


def test_unrecognised_bytes_are_refused(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="not GRIB"):
        detect_format(p)


@pytest.mark.parametrize("name, expected", [
    ("gfs.t12z.20261013.pgrb2.0p25.f024", pd.Timestamp("2026-10-13 12:00")),
    ("aifs_2026101300_f048.nc", pd.Timestamp("2026-10-13 00:00")),
    ("model-2026-10-13T06.zarr", pd.Timestamp("2026-10-13 06:00")),
])
def test_init_from_file_name(name, expected):
    assert _time_from_name(name) == expected


@pytest.mark.parametrize("name, expected", [
    ("TMP_850mb", ("t", 850.0)), ("z500", ("z", 500.0)), ("hgt_500", ("z", 500.0)), ("t2m", None),
])
def test_embedded_level(name, expected):
    assert _embedded_level(name) == expected
