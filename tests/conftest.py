import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mverify.db.store import ObsStore


@pytest.fixture
def store():
    s = ObsStore.memory()
    yield s
    s.close()


def analytic_t2m(lat, lon):
    """A smooth, exactly bilinear-friendly field (degC)."""
    return 15.0 - 0.5 * (np.asarray(lat) - 40.0) + 0.1 * (((np.asarray(lon) + 180) % 360) - 180 + 100.0)


@pytest.fixture
def stations():
    return pd.DataFrame({
        "station_id": ["GHCNH:A", "GHCNH:B", "GHCNH:C", "RAOB:72520"],
        "network": ["GHCNH", "GHCNH", "GHCNH", "RAOB"],
        "name": ["a", "b", "c", "pit"],
        "lat": [40.3, 41.7, 38.1, 40.53],
        "lon": [-100.2, -95.55, -104.9, -80.22],
        "elev_m": [500.0, 300.0, 1600.0, 360.0],
    })


@pytest.fixture
def init():
    return pd.Timestamp("2026-10-13 00:00")


@pytest.fixture
def forecast_ds(init):
    """GFS-like regular 0.5 deg grid over North America, 0..360 longitudes, latitudes north to south."""
    lat = np.arange(60.0, 19.9, -0.5)
    lon = np.arange(220.0, 300.01, 0.5)
    leads = pd.to_timedelta([0, 6, 12], unit="h")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    t = np.stack([analytic_t2m(la, lo) + 273.15 + 0.0 * k for k in range(len(leads))])[None]
    ds = xr.Dataset(
        {"t2m": (("init_time", "step", "latitude", "longitude"), t, {"units": "K"}),
         "orog": (("latitude", "longitude"), np.zeros_like(la), {"units": "m"})},
        coords={"init_time": [init], "step": leads, "latitude": lat, "longitude": lon})
    return ds


@pytest.fixture
def filled_store(store, stations, init, forecast_ds):
    """Observations equal to the analytic field moved to station height, so a perfect model scores 0."""
    store.upsert_stations(stations)
    rows = []
    for k in range(3):
        vt = init + pd.Timedelta(hours=6 * k)
        for _, s in stations[stations["network"] == "GHCNH"].iterrows():
            truth = analytic_t2m(s["lat"], s["lon"]) - 0.0065 * s["elev_m"]
            rows.append({"station_id": s["station_id"], "valid_time": vt, "variable": "t2m", "value": truth,
                         "source": "test"})
    store.upsert_obs(pd.DataFrame(rows))
    return store
