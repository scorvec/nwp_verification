"""Precipitation analyses: where they come from, how they are read, which windows they cover.

Each product is read into one ``xr.DataArray`` per accumulation window, named ``tp``
(mm), with ``latitude`` / ``longitude`` coordinates (1-D on a regular grid, 2-D on
a projected one), NaN where the analysis has no data, and the attributes
``product``, ``valid_time`` (end of the window, ISO) and ``accum_hours``. The
database stores them unchanged on their native grids (``ObsStore.add_grid``).

  stageiv   NCEP Stage IV, CONUS, ~4.8 km polar stereographic (HRAP), 24 h ending
            12Z: radar + gauges, quality-controlled by the River Forecast Centers.
            From the Iowa Environmental Mesonet archive (no login).
  gpcp      GPCP V3.3 daily, global 0.5 deg, 24 h ending 00Z: satellite + gauge,
            IMERG Final-based within 60S-60N, from NASA GES DISC (Earthdata login).
            The same family as the V3.2 climatology of the scorvec.com rain bias
            correction; V3.3 continues it past 2021, but months behind real time.
  imerg_late  GPM IMERG Late daily (V07), global 0.1 deg, 24 h ending 00Z, about
            14 h after real time: the satellite estimate GPCP V3 is built on,
            for recent forecasts GPCP does not reach yet. GES DISC, Earthdata login.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from mverify.ingest import http


@dataclass(frozen=True)
class Product:
    name: str
    label: str
    accum_hours: int
    end_hour: int                 # UTC hour at which every window ends
    regions: tuple[str, ...]      # regions scored by default (mverify.scores.REGIONS)
    url: Callable[[pd.Timestamp], str]
    read: Callable[[Path, pd.Timestamp], xr.DataArray]
    login: bool = False           # needs NASA Earthdata credentials

    def windows(self, start, end) -> list[pd.Timestamp]:
        """Window end times in [start, end]."""
        first = pd.Timestamp(start).floor("D") + pd.Timedelta(hours=self.end_hour)
        ends = pd.date_range(first, pd.Timestamp(end), freq=f"{self.accum_hours}h")
        return [t for t in ends if t >= pd.Timestamp(start)]


def _finish(a: np.ndarray, lat, lon, product: str, valid: pd.Timestamp, hours: int, dims) -> xr.DataArray:
    a = np.where((a >= 0) & (a < 2000), a, np.nan).astype("float32")  # fill values and gross errors
    if np.ndim(lat) == 1:  # regular grids are stored with ascending longitudes in -180..180
        order = np.argsort(lon, kind="stable")
        lon, a = np.asarray(lon)[order], a[:, order]
    coords = ({"latitude": (dims, np.asarray(lat, float)), "longitude": (dims, np.asarray(lon, float))}
              if np.ndim(lat) == 2 else {"latitude": np.asarray(lat, float), "longitude": np.asarray(lon, float)})
    return xr.DataArray(a, dims=dims, coords=coords, name="tp",
                        attrs={"units": "mm", "product": product, "valid_time": valid.isoformat(),
                               "accum_hours": hours})


def read_stageiv(path: Path, valid: pd.Timestamp) -> xr.DataArray:
    import cfgrib
    (ds,) = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    tp = ds["tp"]
    got = pd.Timestamp(ds["valid_time"].values)
    if got != valid:
        raise ValueError(f"{path.name}: valid {got}, expected {valid}")
    return _finish(tp.values, ds["latitude"].values, ((ds["longitude"].values + 180) % 360) - 180,
                   "stageiv", valid, 24, ("y", "x"))


def read_gpcp(path: Path, valid: pd.Timestamp) -> xr.DataArray:
    """GPCP V3.x daily: ``precip`` (time, lat, lon) in mm/day for the UTC day ending at ``valid``."""
    with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as ds:
        a = np.asarray(ds["precip"].values[0], float)
        lat, lon = ds["lat"].values, ds["lon"].values
    return _finish(a, lat, ((lon + 180) % 360) - 180, "gpcp", valid, 24, ("latitude", "longitude"))


def read_imerg(path: Path, valid: pd.Timestamp) -> xr.DataArray:
    """IMERG daily: ``precipitation`` (time, lon, lat) in mm/day for the UTC day ending at ``valid``."""
    with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
        da = ds["precipitation"].isel(time=0).transpose("lat", "lon")
        a, lat, lon = np.asarray(da.values, float), ds["lat"].values, ds["lon"].values
    return _finish(np.nan_to_num(a, nan=-1.0), lat, lon, "imerg_late", valid, 24, ("latitude", "longitude"))


GES_DISC = "https://measures.gesdisc.eosdis.nasa.gov/data/GPCP/GPCPDAY.3.3"
IMERG_LATE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07"
IEM = "https://mesonet.agron.iastate.edu/archive/data"


def imerg_url(valid: pd.Timestamp) -> str:
    """The IMERG Late daily file for the day ending at ``valid``; its minor version (V07B, V07C ...) is
    read from the directory listing, which needs no login."""
    day = valid - pd.Timedelta("1D")
    base = f"{IMERG_LATE}/{day:%Y/%m}"
    names = re.findall(rf'href="(3B-DAY-L\.MS\.MRG\.3IMERG\.{day:%Y%m%d}-S000000-E235959\.V07[A-Z]?\.nc4)"',
                       http.get(f"{base}/").decode(errors="replace"))
    if not names:
        raise FileNotFoundError(f"IMERG Late has no file for {day:%Y-%m-%d} yet")
    return f"{base}/{max(names)}"


PRODUCTS: dict[str, Product] = {
    "stageiv": Product("stageiv", "Stage IV (CONUS)", 24, 12, ("conus",),
                       lambda t: f"{IEM}/{t:%Y/%m/%d}/stage4/ST4.{t:%Y%m%d%H}.24h.grib", read_stageiv),
    # The file for day D holds the 24 h ending D+1 00Z.
    "gpcp": Product("gpcp", "GPCP V3.3 daily", 24, 0, ("global", "nh", "tropics", "sh"),
                    lambda t: f"{GES_DISC}/{t - pd.Timedelta('1D'):%Y}/GPCPDAY_L3_{t - pd.Timedelta('1D'):%Y%m%d}_V3.3.nc4",
                    read_gpcp, login=True),
    "imerg_late": Product("imerg_late", "IMERG Late daily", 24, 0, ("global", "nh", "tropics", "sh"),
                          imerg_url, read_imerg, login=True),
}


def fetch(product: Product, valid: pd.Timestamp, cache_dir: Path) -> xr.DataArray:
    """The analysis for the window ending ``valid``, downloaded once into ``cache_dir``."""
    opener = http.earthdata_opener() if product.login else None
    return product.read(http.cached(product.url(valid), cache_dir, opener=opener), valid)


def ingest(store, product: str, start, end, cache_dir: Path, progress=print) -> int:
    """Add every window of ``product`` ending in [start, end] that the database lacks. Returns windows added."""
    p = PRODUCTS[product]
    have = set(store.grid_times(product, p.accum_hours, start, end))
    todo = [t for t in p.windows(start, end) if t not in have]
    progress(f"{product}: {len(todo)} new windows")
    added = 0
    for t in todo:
        try:
            store.add_grid(fetch(p, t, cache_dir))
            added += 1
        except PermissionError:  # missing credentials: every window would fail the same way
            raise
        except Exception as e:  # noqa: BLE001 - one missing or bad day must not stop the rest
            progress(f"  {product} {t:%Y-%m-%d %H}Z skipped: {e}")
    return added
