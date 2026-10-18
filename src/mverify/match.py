"""Pair forecast fields with observations.

For every field (variable x init x lead x member) the observations valid at
the field's valid time are fetched once from the database, the field is
interpolated to their stations, and one pair row per station is emitted:

    model, init, lead_h, member, valid_time, station_id, network, lat, lon,
    variable, fcst, obs

2 m temperature is moved from the model's terrain height to the station's
with the standard-atmosphere lapse rate (6.5 K/km) when the model's
orography is known; the correction applied is kept in ``dz_m``. 10 m wind
speed is derived from u10/v10 when the model does not provide it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from mverify import variables as V
from mverify.db.store import ObsStore
from mverify.interp import PointInterpolator
from mverify.io.open import Field, Forecast

LAPSE = 0.0065  # K m-1
PAIR_COLUMNS = ["model", "init", "lead_h", "member", "valid_time", "station_id", "network", "lat", "lon",
                "variable", "fcst", "obs", "dz_m"]


def _grid_bbox(fc: Forecast) -> tuple[float, float, float, float] | None:
    """Station search box covering every grid, or None when one grid is global."""
    boxes = []
    for f in {f.grid.key: f for f in fc.fields}.values():
        lat, lon = f.grid.lat, ((f.grid.lon + 180) % 360) - 180
        if f.grid.regular and (lon.size * abs(lon[1] - lon[0]) >= 359):
            return None
        boxes.append((lon.min(), lat.min(), lon.max(), lat.max()))
    if not boxes:
        return None
    b = np.array(boxes)
    lo0, la0, lo1, la1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    return None if lo1 - lo0 > 300 else (lo0, la0, lo1, la1)


def match(fc: Forecast, store: ObsStore, variables: list[str] | None = None,
          bbox: tuple[float, float, float, float] | None = None, networks: list[str] | None = None,
          height_correction: bool = True) -> pd.DataFrame:
    have = set(fc.variables())
    derivable = {name for name, (parts, _) in V.DERIVED.items() if set(parts) <= have and name not in have}
    wanted = sorted((have | derivable) - {"orog"})
    if variables:
        wanted = [v for v in wanted if v in variables]
    if not wanted or not fc.fields:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    valid = [f.valid for f in fc.fields]
    obs = store.fetch_obs(wanted, min(valid), max(valid), bbox or _grid_bbox(fc), networks)
    if obs.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    stations = obs.drop_duplicates("station_id").reset_index(drop=True)
    sidx = pd.Series(np.arange(len(stations)), index=stations["station_id"])
    obs["sidx"] = sidx.loc[obs["station_id"]].to_numpy()
    groups = {k: g for k, g in obs.groupby(["variable", "valid_time"], sort=False)}

    interps: dict[tuple, PointInterpolator] = {}
    model_z: dict[tuple, np.ndarray] = {}

    def interp(f: Field) -> PointInterpolator:
        if f.grid.key not in interps:
            interps[f.grid.key] = PointInterpolator(f.grid, stations["lat"].to_numpy(), stations["lon"].to_numpy())
            if f.grid.orog is not None:
                model_z[f.grid.key] = interps[f.grid.key](f.grid.orog)
        return interps[f.grid.key]

    # Components for derived variables, keyed by (init, lead, member, grid).
    parts: dict[tuple, dict[str, Field]] = defaultdict(dict)
    for f in fc.fields:
        if any(f.var in p for p, _ in V.DERIVED.values()):
            parts[(f.init, f.lead, f.member, f.grid.key)][f.var] = f

    out = []

    def emit(f: Field, var: str, values_fn):
        g = groups.get((var, f.valid))
        if g is None:
            return
        which = g["sidx"].to_numpy()
        fcst = interp(f)(values_fn(), which)
        dz = np.zeros(len(which))
        if var == "t2m" and height_correction and f.grid.key in model_z:
            dz = model_z[f.grid.key][which] - g["elev_m"].to_numpy(float)
            dz = np.where(np.isfinite(dz), dz, 0.0)
            fcst = fcst + LAPSE * dz
        ok = np.isfinite(fcst)
        if not ok.any():
            return
        out.append(pd.DataFrame({
            "model": fc.name, "init": f.init, "lead_h": f.lead_h, "member": f.member,
            "valid_time": f.valid, "station_id": g["station_id"].to_numpy()[ok],
            "network": g["network"].to_numpy()[ok], "lat": g["lat"].to_numpy()[ok], "lon": g["lon"].to_numpy()[ok],
            "variable": var, "fcst": fcst[ok], "obs": g["value"].to_numpy()[ok], "dz_m": dz[ok]}))

    for f in fc.fields:
        if f.var in wanted:
            emit(f, f.var, f.values)
    for name in derivable & set(wanted):
        comps, fn = V.DERIVED[name]
        for fs in parts.values():
            if all(c in fs for c in comps):
                emit(fs[comps[0]], name, lambda fs=fs, fn=fn, comps=comps: fn(*(fs[c].values() for c in comps)))
    if not out:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(out, ignore_index=True)
