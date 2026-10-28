"""Pair forecast fields with observations.

For every field (variable x init x lead x member) the observations valid at
the field's valid time are fetched once from the database, the field is
interpolated to where each observation was made (the station, or the drifted
balloon position for a radiosonde level), and one pair row per observation is
emitted with the columns in PAIR_COLUMNS.

Making the two sides comparable:
  * terrain: ``dz_m`` is model terrain minus station elevation (surface
    variables, when the model's orography is known). 2 m temperature is moved
    to the station height with the standard-atmosphere lapse rate (6.5 K/km).
    Surface pairs with |dz_m| > ``max_dz`` (100 m) are dropped, because there
    the station samples a different climate than the model grid box (valley vs
    ridge, sheltered vs exposed wind) and no lapse rate fixes that. Pairs whose
    dz_m is unknown (no orography in the files) are kept. Surface pressure is
    compared unadjusted, so its cut is MAX_DZ_PRESSURE (5 m, about 0.6 hPa at
    sea level) and a pair with unknown dz_m is dropped. Mean sea-level pressure
    is reduced to sea level through the ground under the station, a reduction
    each model makes its own way and that grows with height, so its pairs also
    need model terrain of at most MAX_TERRAIN_MSL (500 m) at the station.
  * sensor height: an anemometer not at 10 m is brought to 10 m with the
    neutral log wind profile, u(10)/u(h) = ln(10/z0)/ln(h/z0), with z0 from the
    model's surface roughness when the file has it, else Z0_DEFAULT. A
    thermometer outside ``t_height_range`` (1.25-2.5 m, the WMO siting range)
    is dropped; within it the height is kept in ``sensor_h_m`` so scores can be
    stratified by it. Unknown heights are assumed standard.
  * coastlines: when the files carry a land-sea mask, surface variables at
    (land) stations are interpolated from land grid points only, so a coastal
    station is not handed the sea's temperature or wind; a station with no land
    point among its neighbours is dropped. The model land fraction at the
    station is kept in ``land_frac`` for stratifying coastal stations.
  * observation error: ``sigma_o`` from the variable catalogue travels with each
    pair for the error-aware scores in mverify.scores.

10 m wind speed is derived from u10/v10 when the model does not provide it. A gauge's 24 h total (``pr24``) is
paired with each run's total over the same 24 h, built from the file's precipitation accumulations
(mverify.precip.accum).
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from mverify import variables as V
from mverify.db.store import ObsStore
from mverify.interp import PointInterpolator
from mverify.io.open import Field, Forecast
from mverify.precip.accum import by_run

log = logging.getLogger(__name__)

LAPSE = 0.0065       # K m-1
MAX_DZ = 100.0       # m, largest model-station terrain difference for surface pairs
# m, the same for surface pressure, which falls ~0.12 hPa per metre near sea level and is not moved to the station
# height: 5 m keeps the mismatch (<= 0.6 hPa) within the observation error and well below a day-1 forecast error
MAX_DZ_PRESSURE = 5.0
MAX_TERRAIN_MSL = 500.0  # m, highest model terrain at a station whose mean sea-level pressure is scored
T_HEIGHT_RANGE = (1.25, 2.5)  # m, WMO siting range for thermometers
Z0_DEFAULT = 0.03    # m, roughness of open short grass, the WMO reference exposure
WIND_10M = ("u10", "v10", "ws10")
SCREEN = ("t2m", "td2m")
PAIR_COLUMNS = ["model", "init", "lead_h", "member", "valid_time", "station_id", "network", "lat", "lon",
                "variable", "fcst", "obs", "dz_m", "dt_min", "sensor_h_m", "land_frac", "sigma_o"]
LAND = 0.5           # grid points with a land fraction at least this are land


def log_wind_factor(h: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Factor taking a wind measured at height ``h`` to 10 m in a neutral log profile."""
    z0 = np.clip(z0, 1e-4, 2.0)
    h = np.maximum(h, 2 * z0)  # the profile is undefined at or below the roughness length
    return np.log(10.0 / z0) / np.log(h / z0)


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
          height_correction: bool = True, max_dz: float | None = MAX_DZ,
          t_height_range: tuple[float, float] | None = T_HEIGHT_RANGE) -> pd.DataFrame:
    """Pairs for every field of ``fc`` with an observation; see the module docstring for the adjustments.

    ``max_dz=None`` keeps every surface pair, ``t_height_range=None`` every thermometer height.
    """
    have = set(fc.variables())
    derivable = {name for name, (parts, _) in V.DERIVED.items() if set(parts) <= have and name not in have}
    gauges = {V.GAUGE.name} if "tp" in have else set()
    wanted = sorted((have | derivable | gauges) - set(V.STATIC) - set(V.GRIDDED))
    if variables:
        wanted = [v for v in wanted if v in variables]
    if not wanted or not fc.fields:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    valid = sorted({f.valid for f in fc.fields})
    # only the fields' valid times: a gauge total ends at one too, since accumulations are taken between steps
    obs = store.fetch_obs(wanted, valid[0], valid[-1], bbox or _grid_bbox(fc), networks, times=valid)
    if obs.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    if t_height_range is not None:
        h = obs["t_height_m"]
        off = obs["variable"].isin(SCREEN) & h.notna() & ~h.between(*t_height_range)
        if off.any():
            log.info("dropped %d t2m/td2m observations from thermometers outside %s m", off.sum(), t_height_range)
            obs = obs[~off]

    # Interpolation weights are computed once per grid for every distinct observation point.
    points = obs[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    obs = obs.merge(points.reset_index(names="pidx"), on=["lat", "lon"], how="left")
    groups = {k: g for k, g in obs.groupby(["variable", "valid_time"], sort=False)}

    interps: dict[tuple, PointInterpolator] = {}
    model_z: dict[tuple, np.ndarray] = {}
    model_z0: dict[tuple, np.ndarray] = {}
    land_frac: dict[tuple, np.ndarray] = {}
    land_interps: dict[tuple, PointInterpolator] = {}

    def interp(f: Field, surface: bool = False) -> PointInterpolator:
        if f.grid.key not in interps:
            ip = interps[f.grid.key] = PointInterpolator(f.grid, points["lat"].to_numpy(), points["lon"].to_numpy())
            if f.grid.lsm is not None:
                land_frac[f.grid.key] = ip(f.grid.lsm)
                land_interps[f.grid.key] = ip.over(f.grid.lsm >= LAND)
            # Terrain and roughness from the same (land) neighbours the surface forecast uses.
            surf = land_interps.get(f.grid.key, ip)
            if f.grid.orog is not None:
                model_z[f.grid.key] = surf(f.grid.orog)
            if f.grid.z0 is not None:
                model_z0[f.grid.key] = surf(f.grid.z0)
        if surface and f.grid.key in land_interps:
            return land_interps[f.grid.key]
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
        which = g["pidx"].to_numpy()
        surface = V.CATALOG[var].kind == "surface"
        fcst = interp(f, surface)(values_fn(), which)
        o = g["value"].to_numpy(float)
        lf = land_frac[f.grid.key][which] if surface and f.grid.key in land_frac else np.full(len(which), np.nan)
        zm = model_z[f.grid.key][which] if surface and f.grid.key in model_z else np.full(len(which), np.nan)
        dz = zm - g["elev_m"].to_numpy(float)
        if var == "t2m" and height_correction:
            fcst = fcst + LAPSE * np.nan_to_num(dz)
        sensor_h = np.full(len(which), np.nan)
        if var in SCREEN:
            sensor_h = g["t_height_m"].to_numpy(float)
        elif var in WIND_10M:
            sensor_h = g["wind_height_m"].to_numpy(float)
            z0 = model_z0[f.grid.key][which] if f.grid.key in model_z0 else np.full(len(which), Z0_DEFAULT)
            z0 = np.where(np.isfinite(z0), z0, Z0_DEFAULT)
            known = np.isfinite(sensor_h)
            o = np.where(known, o * log_wind_factor(np.where(known, sensor_h, 10.0), z0), o)
        ok = np.isfinite(fcst)
        if var == "sp" and max_dz is not None:
            ok &= np.abs(dz) <= min(max_dz, MAX_DZ_PRESSURE)  # unknown dz (NaN) is dropped: no height check
        elif surface and max_dz is not None:
            ok &= ~(np.abs(dz) > max_dz)  # unknown dz (NaN) is kept
            if var == "msl":
                ok &= zm <= MAX_TERRAIN_MSL  # unknown model terrain (NaN) is dropped
        if not ok.any():
            return
        out.append(pd.DataFrame({
            "model": fc.name, "init": f.init, "lead_h": f.lead_h, "member": f.member,
            "valid_time": f.valid, "station_id": g["station_id"].to_numpy()[ok],
            "network": g["network"].to_numpy()[ok], "lat": g["lat"].to_numpy()[ok], "lon": g["lon"].to_numpy()[ok],
            "variable": var, "fcst": fcst[ok], "obs": o[ok], "dz_m": dz[ok], "dt_min": g["dt_min"].to_numpy()[ok],
            "sensor_h_m": sensor_h[ok], "land_frac": lf[ok],
            # NaN, not None, where a variable has no observation error: one None makes the column object-typed
            # in every concatenation of pairs, and the scores' arithmetic on it fails
            "sigma_o": np.nan if V.CATALOG[var].sigma_o is None else V.CATALOG[var].sigma_o}))

    for f in fc.fields:
        if f.var in wanted:
            emit(f, f.var, f.values)
    for name in derivable & set(wanted):
        comps, fn = V.DERIVED[name]
        for fs in parts.values():
            if all(c in fs for c in comps):
                emit(fs[comps[0]], name, lambda fs=fs, fn=fn, comps=comps: fn(*(fs[c].values() for c in comps)))
    if V.GAUGE.name in wanted:
        hours = pd.Timedelta(hours=24)
        ends = sorted(v for var, v in groups if var == V.GAUGE.name)
        for acc in by_run([f for f in fc.fields if f.var == "tp"]).values():
            first = next(iter(acc.fields.values()))
            for end in ends:
                a, b = end - hours - first.init, end - first.init
                if a < pd.Timedelta(0) or (total := acc.total(a, b)) is None:
                    continue
                emit(dataclasses.replace(first, lead=b), V.GAUGE.name, lambda total=total: total)
    if not out:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(out, ignore_index=True)
