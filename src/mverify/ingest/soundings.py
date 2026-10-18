"""Radiosonde mandatory levels from the scorvec.com sounding mirror.

The mirror (the ``skewt-data`` branch of scorvec/scorvec.github.io) already
merges WIS2 BUFR, IEM, ECCC, TAC and IGRA feeds for ~700 stations and keeps a
rolling window of recent ascents as one CSV per station and hour. Here each
ascent is reduced to the mandatory levels mverify verifies (1000-200 hPa):
an exact level is used when reported, otherwise heights and temperatures are
interpolated linearly in log-pressure between levels no more than 60 hPa
apart (winds only from an exact level or levels 25 hPa apart). The vertical
hydrostatic check in mverify.db.qc then flags inconsistent heights.

IGRA2 (NCEI) is the long archive for the same stations; its parser plugs in
the same way once NCEI is reachable from the build machine.
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
LEVELS = (1000, 925, 850, 700, 500, 300, 250, 200)
COLS = {"pressure_hPa": "p", "geopotential height_m": "z", "temperature_C": "t",
        "wind direction_degree": "wd", "wind speed_m/s": "ws"}


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


def mandatory_levels(prof: pd.DataFrame) -> dict[str, float]:
    """Canonical {z500: m, t850: degC, u250: m/s, ...} for one ascent."""
    prof = prof.rename(columns=COLS).reindex(columns=list(COLS.values())).apply(pd.to_numeric, errors="coerce")
    prof = prof.dropna(subset=["p"]).sort_values("p", ascending=False).drop_duplicates("p")
    p = prof["p"].to_numpy(float)
    rad = np.deg2rad(prof["wd"].to_numpy(float))
    ws = prof["ws"].to_numpy(float)
    series = {"z": (prof["z"].to_numpy(float), 60), "t": (prof["t"].to_numpy(float), 60),
              "u": (-ws * np.sin(rad), 25), "v": (-ws * np.cos(rad), 25)}
    out: dict[str, float] = {}
    for lev in LEVELS:
        for var, (x, gap) in series.items():
            val = _at_level(p, x, lev, gap)
            if val is not None:
                out[f"{var}{lev}"] = val
    return out


def _one(root: str, sid: str, hour: pd.Timestamp) -> tuple[dict[str, float], float | None] | None:
    rel = f"soundings/{sid}_{hour:%Y%m%d%H}.csv"
    try:
        prof = pd.read_csv(io.BytesIO(_read(root, rel)))
    except (OSError, ValueError, pd.errors.ParserError) as e:  # the mirror is best-effort; skip the ascent
        log.debug("skipping %s: %s", rel, e)
        return None
    z = pd.to_numeric(prof.get("geopotential height_m"), errors="coerce").dropna()
    return mandatory_levels(prof), (float(z.iloc[0]) if len(z) else None)


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
            if store.ingested_version(SOURCE, f"{sid}/{hour:%Y%m%d%H}") is None:
                jobs.append((sid, hour, e))
    progress(f"soundings: {len(jobs)} new ascents")
    total = 0
    with ThreadPoolExecutor(workers) as ex:
        for k in range(0, len(jobs), batch_size):
            chunk = jobs[k:k + batch_size]
            results = list(ex.map(lambda j: _one(root, j[0], j[1]), chunk))
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
        obs = pd.DataFrame({"station_id": f"RAOB:{sid}", "valid_time": hour, "variable": list(levels),
                            "value": list(levels.values()), "qc": 0, "dt_min": 0, "source": SOURCE, "_object": key})
        frames.append(qc.run_sounding_levels(obs[obs["variable"].isin(list(V.CATALOG))]))
        objects.append((key, "1", len(levels)))  # an ascent never changes once mirrored: its key is its version
    if not objects:
        return 0
    try:
        return store.ingest_batch(SOURCE, objects, pd.DataFrame(list(stations.values())), pd.concat(frames))
    except ValidationError as e:  # nothing logged, so the next update retries the batch
        log.warning("sounding batch starting %s rejected: %s", objects[0][0], e)
        return 0
