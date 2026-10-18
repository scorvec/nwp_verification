"""Making forecasts and observations comparable: positions, sensor heights, terrain."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mverify import match as M
from mverify.io.open import open_forecast


def _grid_ds(init, **fields):
    lat = np.arange(60.0, 19.9, -0.5)
    lon = np.arange(220.0, 300.01, 0.5)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    data = {}
    for name, (fn, attrs, static) in fields.items():
        v = fn(la, lo)
        data[name] = (("latitude", "longitude"), v, attrs) if static else \
            (("time", "latitude", "longitude"), v[None], attrs)
    return xr.Dataset(data, coords={"time": [init], "latitude": lat, "longitude": lon})


def test_upper_air_is_matched_where_the_balloon_was(tmp_path, store, stations, init):
    store.upsert_stations(stations)
    z500 = lambda la, lo: 5500 + 20 * (la - 40) + 5 * (lo - 280)
    _grid_ds(init, z500=(z500, {"units": "m"}, False)).to_netcdf(tmp_path / "z.nc")
    drifted = (40.9, -79.1)  # 0.4 N and 1.1 E of the launch site
    store.upsert_obs(pd.DataFrame([{"station_id": "RAOB:72520", "valid_time": init, "variable": "z500",
                                    "value": z500(drifted[0], drifted[1] % 360), "lat": drifted[0],
                                    "lon": drifted[1], "dt_min": -15, "source": "test"}]))
    pairs = M.match(open_forecast(tmp_path / "z.nc", init=init), store)
    assert pairs["fcst"].iloc[0] == pytest.approx(pairs["obs"].iloc[0])
    assert tuple(pairs[["lat", "lon", "dt_min"]].iloc[0]) == (40.9, -79.1, -15)
    assert np.isnan(pairs["dz_m"].iloc[0])


def test_wind_from_a_high_anemometer_is_brought_to_10_m(tmp_path, store, stations, init):
    st = stations.copy()
    st["wind_height_m"] = [30.0, None, 10.0, None]
    store.upsert_stations(st)
    calm = lambda la, lo: np.full_like(la, 5.0)
    _grid_ds(init, u10=(calm, {"units": "m s-1"}, False),
             fsr=(lambda la, lo: np.full_like(la, 0.1), {"units": "m"}, True)).to_netcdf(tmp_path / "w.nc")
    store.upsert_obs(pd.DataFrame([{"station_id": s, "valid_time": init, "variable": "u10", "value": 6.0,
                                    "source": "test"} for s in ("GHCNH:A", "GHCNH:B", "GHCNH:C")]))
    p = M.match(open_forecast(tmp_path / "w.nc", init=init), store).set_index("station_id")
    assert p.loc["GHCNH:A", "obs"] == pytest.approx(6.0 * np.log(10 / 0.1) / np.log(30 / 0.1))
    assert p.loc["GHCNH:A", "sensor_h_m"] == 30.0
    assert p.loc["GHCNH:B", "obs"] == 6.0 and np.isnan(p.loc["GHCNH:B", "sensor_h_m"])  # unknown: assumed 10 m
    assert p.loc["GHCNH:C", "obs"] == pytest.approx(6.0)


def test_log_profile_default_roughness():
    assert M.log_wind_factor(np.array([10.0]), np.array([M.Z0_DEFAULT]))[0] == pytest.approx(1.0)
    assert M.log_wind_factor(np.array([2.0]), np.array([0.03]))[0] == pytest.approx(np.log(10 / 0.03) / np.log(2 / 0.03))


def test_thermometers_outside_the_siting_range_are_left_out(tmp_path, store, stations, init, forecast_ds):
    st = stations.copy()
    st["t_height_m"] = [1.5, 3.0, None, None]
    store.upsert_stations(st)
    store.upsert_obs(pd.DataFrame([{"station_id": s, "valid_time": init, "variable": "t2m", "value": 10.0,
                                    "source": "test"} for s in ("GHCNH:A", "GHCNH:B", "GHCNH:C")]))
    forecast_ds.to_netcdf(tmp_path / "t.nc")
    fc = open_forecast(tmp_path / "t.nc")
    kept = M.match(fc, store, max_dz=None)
    assert sorted(kept.loc[kept["lead_h"] == 0, "station_id"]) == ["GHCNH:A", "GHCNH:C"]
    assert sorted(kept["sensor_h_m"].dropna().unique()) == [1.5]
    every = M.match(fc, store, max_dz=None, t_height_range=None)
    assert sorted(every.loc[every["lead_h"] == 0, "station_id"]) == ["GHCNH:A", "GHCNH:B", "GHCNH:C"]


def test_coastal_stations_take_the_forecast_from_land_points(tmp_path, store, init):
    # Land west of 260E (-100), sea east of it; t2m is 20 over land and 10 over sea.
    store.upsert_stations(pd.DataFrame({
        "station_id": ["GHCNH:COAST", "GHCNH:SEA", "GHCNH:INLAND"], "network": "GHCNH",
        "lat": [40.2, 40.2, 40.2], "lon": [-100.3, -95.0, -110.0], "elev_m": 0.0}))
    land = lambda la, lo: (lo < 260).astype(float)
    _grid_ds(init, t2m=(lambda la, lo: 283.15 + 10 * land(la, lo), {"units": "K"}, False),
             lsm=(land, {"units": "1"}, True)).to_netcdf(tmp_path / "c.nc")
    store.upsert_obs(pd.DataFrame([{"station_id": s, "valid_time": init, "variable": "t2m", "value": 20.0,
                                    "source": "test"} for s in ("GHCNH:COAST", "GHCNH:SEA", "GHCNH:INLAND")]))
    p = M.match(open_forecast(tmp_path / "c.nc", init=init), store).set_index("station_id")
    assert "GHCNH:SEA" not in p.index  # no land point around it
    assert p.loc["GHCNH:COAST", "fcst"] == pytest.approx(20.0)  # bilinear would give 16
    assert p.loc["GHCNH:COAST", "land_frac"] == pytest.approx(0.6)
    assert p.loc["GHCNH:INLAND", "land_frac"] == 1.0
