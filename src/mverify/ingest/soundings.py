"""Radiosonde mandatory levels from the scorvec.com sounding mirror.

The mirror (the ``skewt-data`` branch of scorvec/scorvec.github.io) already
merges WIS2 BUFR, IEM, ECCC, TAC and IGRA feeds for ~700 stations and keeps a
rolling window of recent ascents as one CSV per station and hour. Here each
ascent is reduced to the mandatory levels mverify verifies (1000-100 hPa, humidity to 300 hPa):
an exact level is used when reported, otherwise heights and temperatures are
interpolated linearly in log-pressure between levels no more than 60 hPa
apart (winds only from an exact level or levels 25 hPa apart). The vertical
hydrostatic check in mverify.db.qc then flags inconsistent heights.

Balloon drift. A sonde reaches 200 hPa some 40 minutes after launch and up to
100-200 km downwind in a jet, so every level is stored with where and when it
was measured (obs.lat, obs.lon, obs.dt_min relative to the nominal hour), and
matching interpolates the model to that point instead of the launch site:
  * time: the per-level timestamps when the ascent carries them (WIS2 BUFR and
    ECCC do, about 80% of the mirror); otherwise launch is assumed
    LAUNCH_LEAD_MIN before the nominal hour and the balloon rises at
    ASCENT_RATE, the WMO standard practice values;
  * position: the per-level positions when they vary within the ascent (GPS
    sondes in BUFR); otherwise the reported winds are integrated over time
    from the launch site.
The time offset is stored but not yet used: the model is taken at the nominal
hour, as in most operational verification.

IGRA2 (NCEI, mverify.ingest.igra) is the long archive for the same stations,
reduced by the same function.
"""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mverify import variables as V
from mverify.db import qc
from mverify.db.store import ObsStore, ValidationError, db_path
from mverify.ingest import http

log = logging.getLogger(__name__)

MIRROR = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-data"
SOURCE = "raob:scorvec-skewt"
COLS = {"pressure_hPa": "p", "geopotential height_m": "z", "temperature_C": "t",
        "dew point temperature_C": "td", "wind direction_degree": "wd", "wind speed_m/s": "ws", "time": "time",
        "latitude": "lat", "longitude": "lon"}
NUMERIC = ("p", "z", "t", "td", "wd", "ws", "lat", "lon")

ASCENT_RATE = 5.0         # m s-1, typical rise rate of a radiosonde balloon
LAUNCH_LEAD_MIN = 45.0    # launch this many minutes before the nominal hour (WMO practice)
MAX_DT_MIN = 180          # per-level times further than this from the nominal hour are not trusted
EARTH_RADIUS = 6.371e6    # m
# An ascent never changes once mirrored, so its ingest_log version is the version of the reduction
# below; bumping it makes the next update re-derive every ascent (2: balloon drift).
REDUCTION_VERSION = "3"  # 3: 400, 150 and 100 hPa, specific humidity


def _read(root: str, rel: str) -> bytes:
    if root.startswith("http"):
        return http.get(f"{root}/{rel}")
    return (Path(root) / rel).read_bytes()


def _at_level(p: np.ndarray, x: np.ndarray, lev: float, max_gap: float) -> float | None:
    """x at pressure ``lev``: the reported value, else log-p interpolation across a gap <= max_gap hPa."""
    ok = np.isfinite(x)
    p, x = p[ok], x[ok]
    exact = np.flatnonzero(p == lev)
    if exact.size:
        return float(x[exact[0]])
    below, above = np.flatnonzero(p > lev), np.flatnonzero(p < lev)
    if not below.size or not above.size:
        return None
    i0, i1 = below[-1], above[0]  # p is sorted high to low: nearest level on each side
    if p[i0] - p[i1] > max_gap:
        return None
    w = np.log(lev / p[i0]) / np.log(p[i1] / p[i0])
    return float(x[i0] + w * (x[i1] - x[i0]))


def _fill(x: np.ndarray, at: np.ndarray) -> np.ndarray | None:
    """``x`` with gaps filled by linear interpolation along ``at`` (held constant past the ends)."""
    ok = np.isfinite(x) & np.isfinite(at)
    if not ok.any():
        return None
    order = np.argsort(at[ok], kind="stable")  # np.interp needs ascending abscissae
    return np.interp(at, at[ok][order], x[ok][order])


def flight_time(prof: pd.DataFrame, hour: pd.Timestamp | None) -> np.ndarray | None:
    """Seconds from the nominal hour at which each row (sorted by falling pressure) was measured."""
    if hour is not None and prof["time"].notna().sum() >= 2:
        s = (pd.to_datetime(prof["time"], errors="coerce") - hour).dt.total_seconds().to_numpy(float)
        ok = np.isfinite(s)
        # Some feeds stamp every level with the launch or nominal time; such a column is not a
        # flight time. Real per-level times rise on most steps and span the ascent.
        rising = np.mean(np.diff(s[ok]) > 0) >= 0.5 if ok.sum() >= 2 else False
        if rising and np.ptp(s[ok]) >= 600 and np.all(np.abs(s[ok]) <= MAX_DT_MIN * 60):
            # Feeds occasionally repeat or reorder a timestamp; time cannot run backwards up the ascent.
            return np.fmax.accumulate(_fill(s, np.arange(len(s), dtype=float)))
    z = _fill(prof["z"].to_numpy(float), np.log(prof["p"].to_numpy(float)))
    if z is None:
        return None
    return -LAUNCH_LEAD_MIN * 60 + (z - z[0]) / ASCENT_RATE


def trajectory(prof: pd.DataFrame, t: np.ndarray, u: np.ndarray, v: np.ndarray,
               lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Latitude and longitude of each row: reported positions when they vary, else winds integrated over ``t``."""
    lat, lon = prof["lat"].to_numpy(float), prof["lon"].to_numpy(float)
    if np.isfinite(lat).sum() >= 2 and (np.nanmax(lat) - np.nanmin(lat) > 1e-4 or np.nanmax(lon) - np.nanmin(lon) > 1e-4):
        lat, lon = _fill(lat, t), _fill(lon, t)
        return lat, ((lon + 180) % 360) - 180
    u, v = _fill(u, t), _fill(v, t)
    if u is None or v is None:
        return None
    dt = np.diff(t, prepend=t[0])
    north = np.cumsum(0.5 * (v + np.roll(v, 1)) * dt * (dt > 0))   # trapezoid; first step is zero
    east = 0.5 * (u + np.roll(u, 1)) * dt * (dt > 0)
    lat = lat0 + np.rad2deg(north / EARTH_RADIUS)
    lon = lon0 + np.rad2deg(np.cumsum(east / (EARTH_RADIUS * np.cos(np.deg2rad(lat)))))
    return lat, ((lon + 180) % 360) - 180


def mandatory_levels(prof: pd.DataFrame, hour: pd.Timestamp | None = None,
                     lat0: float | None = None, lon0: float | None = None) -> pd.DataFrame:
    """One ascent at the mandatory levels: index ``variable`` (z500, t850, u250, q700 ...), columns
    ``value`` (canonical unit) and, where the flight could be reconstructed, ``dt_min``, ``lat``, ``lon``.

    ``prof`` has the mirror's column names or the short ones they map to (``COLS``)."""
    prof = prof.rename(columns=COLS).reindex(columns=list(COLS.values()))
    for c in NUMERIC:
        prof[c] = pd.to_numeric(prof[c], errors="coerce")
    prof = prof.dropna(subset=["p"]).sort_values("p", ascending=False, kind="stable").drop_duplicates("p")
    p = prof["p"].to_numpy(float)
    rad = np.deg2rad(prof["wd"].to_numpy(float))
    ws = prof["ws"].to_numpy(float)
    u, v = -ws * np.sin(rad), -ws * np.cos(rad)
    q = V.specific_humidity(prof["td"].to_numpy(float), p)
    series = {"z": (prof["z"].to_numpy(float), 60), "t": (prof["t"].to_numpy(float), 60), "u": (u, 25), "v": (v, 25),
              "q": (q, 60)}

    t = flight_time(prof, hour) if len(p) else None
    path = trajectory(prof, t, u, v, lat0, lon0) if t is not None and lat0 is not None else None
    rows = []
    for lev in V.LEVELS:
        where = {"dt_min": np.nan, "lat": np.nan, "lon": np.nan}
        if t is not None and len(p) and p[-1] <= lev <= p[0]:
            lp = np.log(p[::-1])  # ascending, as np.interp needs
            where["dt_min"] = np.interp(np.log(lev), lp, t[::-1]) / 60
            if path is not None:
                where["lat"] = np.interp(np.log(lev), lp, path[0][::-1])
                # Longitude through the sine/cosine, so a path across the dateline interpolates correctly.
                rl = np.deg2rad(path[1][::-1])
                where["lon"] = np.rad2deg(np.arctan2(np.interp(np.log(lev), lp, np.sin(rl)),
                                                     np.interp(np.log(lev), lp, np.cos(rl))))
        for var, (x, gap) in series.items():
            if var == "q" and lev not in V.HUMIDITY_LEVELS:
                continue
            val = _at_level(p, x, lev, gap)
            if val is not None:
                rows.append({"variable": f"{var}{lev}", "value": val, **where})
    out = pd.DataFrame(rows, columns=["variable", "value", "dt_min", "lat", "lon"]).set_index("variable")
    out["dt_min"] = out["dt_min"].round().clip(-MAX_DT_MIN, MAX_DT_MIN)
    return out


def _one(root: str, sid: str, hour: pd.Timestamp, lat: float, lon: float) -> tuple[pd.DataFrame, float | None] | None:
    rel = f"soundings/{sid}_{hour:%Y%m%d%H}.csv"
    try:
        prof = pd.read_csv(io.BytesIO(_read(root, rel)))
    except (OSError, ValueError, pd.errors.ParserError) as e:  # the mirror is best-effort; skip the ascent
        log.debug("skipping %s: %s", rel, e)
        return None
    z = pd.to_numeric(prof.get("geopotential height_m"), errors="coerce").dropna()
    return mandatory_levels(prof, hour, lat, lon), (float(z.iloc[0]) if len(z) else None)


MAX_POSITION_ERROR_DEG = 1.0


def reference_positions(cache_dir: Path) -> dict[str, tuple[float, float]]:
    """WMO index -> (lat, lon) from the GHCNh station list, an independent source for station positions."""
    from mverify.ingest import ghcnh
    sl = ghcnh.station_list(cache_dir).dropna(subset=["WMO_ID", "LATITUDE", "LONGITUDE"])
    sl = sl.drop_duplicates("WMO_ID")
    return {w.zfill(5): (la, lo) for w, la, lo in zip(sl["WMO_ID"], sl["LATITUDE"], sl["LONGITUDE"])}


def position_confirmed(sid: str, lat: float, lon: float, ref: dict[str, tuple[float, float]]) -> bool:
    """False when an independent list puts the station somewhere else.

    Neither source is authoritative (the mirror has had latitudes of 0.0, the GHCNh list maps
    some WMO indexes to the wrong station), so a disagreement is never resolved by picking
    one: the station is left out until the metadata is fixed. Stations absent from the
    reference are accepted.
    """
    if sid not in ref:
        return True
    rlat, rlon = ref[sid]
    dlon = abs(((lon - rlon) + 180) % 360 - 180)
    return abs(lat - rlat) <= MAX_POSITION_ERROR_DEG and dlon <= MAX_POSITION_ERROR_DEG


def ingest(store: ObsStore, root: str = MIRROR, since=None, bbox=None, workers: int = 16,
           batch_size: int = 500, reference: dict[str, tuple[float, float]] | None = None, progress=print) -> int:
    """Read every ascent in the mirror's manifest that is not in the database yet. Returns rows written.

    Station positions are checked against ``reference`` (default: the GHCNh station list), because a
    wrong position silently turns into a large "forecast error".
    """
    if reference is None:
        reference = reference_positions(db_path().parent / "cache")
    man = json.loads(_read(root, "manifest.json"))
    done = store.ingested_versions(SOURCE)
    jobs = []
    for sid, e in sorted(man["entries"].items()):
        if not position_confirmed(sid, e["la"], e["lo"], reference):
            log.warning("RAOB %s (%s): mirror position (%.2f, %.2f) disagrees with the WMO list %s; skipped",
                        sid, e.get("n"), e["la"], e["lo"], reference[sid])
            continue
        if bbox and not (bbox[0] <= e["lo"] <= bbox[2] and bbox[1] <= e["la"] <= bbox[3]):
            continue
        for h in e.get("hours", []):
            hour = pd.Timestamp(h)
            if since is not None and hour < pd.Timestamp(since):
                continue
            if done.get(f"{sid}/{hour:%Y%m%d%H}") != REDUCTION_VERSION:
                jobs.append((sid, hour, e))
    progress(f"soundings: {len(jobs)} new ascents")
    total = 0
    with ThreadPoolExecutor(workers) as ex:
        for k in range(0, len(jobs), batch_size):
            chunk = jobs[k:k + batch_size]
            results = list(ex.map(lambda j: _one(root, j[0], j[1], j[2]["la"], j[2]["lo"]), chunk))
            total += _write_batch(store, chunk, results)
    return total


def _write_batch(store: ObsStore, chunk, results) -> int:
    """One transaction for a batch of ascents; each ascent is its own ingest_log object."""
    stations, frames, objects = {}, [], []
    for (sid, hour, e), res in zip(chunk, results):
        if res is None:
            continue
        levels, z0 = res
        key = f"{sid}/{hour:%Y%m%d%H}"
        stations[sid] = {"station_id": f"RAOB:{sid}", "network": "RAOB", "name": e.get("n"),
                         "lat": e["la"], "lon": e["lo"], "elev_m": z0, "wmo_id": sid}
        obs = levels.reset_index().assign(station_id=f"RAOB:{sid}", valid_time=hour, qc=0, source=SOURCE,
                                          _object=key)
        frames.append(qc.run_sounding_levels(obs[obs["variable"].isin(list(V.CATALOG))]))
        objects.append((key, REDUCTION_VERSION, len(levels)))
    if not objects:
        return 0
    try:
        return store.ingest_batch(SOURCE, objects, pd.DataFrame(list(stations.values())), pd.concat(frames))
    except ValidationError as e:  # nothing logged, so the next update retries the batch
        log.warning("sounding batch starting %s rejected: %s", objects[0][0], e)
        return 0
