"""Radiosondes from IGRA2, NCEI's Integrated Global Radiosonde Archive.

IGRA2 is the long archive: some 2,800 stations, many back to the 1940s, about 800
reporting now, updated daily. Each station is one zipped text file, either the
current year (``data-y2d``, ``{id}-data-beg{year}.txt.zip``) or the whole record
(``data-por``, ``{id}-data.txt.zip``), the archive to use for hindcasts and
reforecasts. Ascents are reduced to the mandatory levels exactly as the scorvec.com
mirror's are (mverify.ingest.soundings.mandatory_levels), so both sources write the
same rows and either fills the other's gaps.

Format (IGRA2 data format description, NCEI): a header line per ascent, then one
fixed-width line per level, with -8888 (removed by NCEI's QC) and -9999 (missing)
for absent values. Pressure is in Pa, temperature and dew-point depression in tenths
of a degree, wind speed in tenths of m/s, elapsed time since launch as MMMSS.
IGRA2 gives no per-level positions, so drift comes from integrating the winds.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mverify import variables as V
from mverify.db import qc
from mverify.db.store import ObsStore, ValidationError, db_path
from mverify.ingest import http
from mverify.ingest.soundings import mandatory_levels

log = logging.getLogger(__name__)

BASE = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive"
STATION_LIST = f"{BASE}/doc/igra2-station-list.txt"
ARCHIVES = {"y2d": r"([A-Z0-9]{11})-data-beg\d{4}\.txt\.zip", "por": r"([A-Z0-9]{11})-data\.txt\.zip"}
SOURCE = "raob:igra2"
REDUCTION_VERSION = "1"

# 0-based [start, end) columns of a level line
FIELDS = {"etime": (3, 8), "p": (9, 15), "z": (16, 21), "t": (22, 27), "dpdp": (34, 39), "wd": (40, 45),
          "ws": (46, 51)}
WIDTH = 51


def station_id(igra_id: str) -> str:
    """Database id: the WMO index for WMO stations (network code M), as the mirror uses; else the IGRA id."""
    return f"RAOB:{igra_id[-5:]}" if igra_id[2] == "M" else f"RAOB:{igra_id}"


def station_list(cache_dir: Path) -> pd.DataFrame:
    text = http.cached(STATION_LIST, cache_dir).read_text(errors="replace")
    sl = pd.DataFrame([(ln[0:11], ln[12:20], ln[21:30], ln[31:37], ln[41:71].strip(), ln[77:81])
                       for ln in text.splitlines() if len(ln) >= 81],
                      columns=["id", "lat", "lon", "elev_m", "name", "last_year"])
    for c in ("lat", "lon", "elev_m", "last_year"):
        sl[c] = pd.to_numeric(sl[c], errors="coerce")
    sl.loc[sl["elev_m"] < -998, "elev_m"] = np.nan  # -998.8: unknown, -999.9: missing
    return sl.dropna(subset=["lat", "lon"])


def listing(archive: str) -> dict[str, tuple[str, str]]:
    """IGRA id -> (file name, version) of every station file in ``archive``; the version is the listed
    modification time and size, so an unchanged file is not downloaded again."""
    html = http.get(f"{BASE}/access/data-{archive}/").decode(errors="replace")
    out = {}
    for m in re.finditer(rf'href="({ARCHIVES[archive]})">.*?</a></td>\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*>([^<]*)<',
                         html):
        out[m.group(2)] = (m.group(1), f"{m.group(3).strip()}/{m.group(4).strip()}")
    return out


def _levels(lines: list[str], launch: pd.Timestamp | None) -> pd.DataFrame:
    """One ascent's level lines as a profile with mandatory_levels' column names."""
    buf = np.frombuffer("".join(ln[:WIDTH].ljust(WIDTH) for ln in lines).encode("ascii", "replace"), "S1")
    arr = buf.reshape(len(lines), WIDTH)
    col = {}
    for k, (a, b) in FIELDS.items():
        raw = np.char.strip(arr[:, a:b].copy().view(f"S{b - a}").ravel())
        x = np.where(raw == b"", b"-9999", raw).astype(float)
        x[x <= -8888] = np.nan
        col[k] = x
    prof = pd.DataFrame({"p": col["p"] / 100.0, "z": col["z"], "t": col["t"] / 10.0,
                         "td": col["t"] / 10.0 - col["dpdp"] / 10.0, "wd": col["wd"], "ws": col["ws"] / 10.0})
    if launch is not None:
        sec = (col["etime"] // 100) * 60 + col["etime"] % 100
        prof["time"] = launch + pd.to_timedelta(sec, unit="s")
    return prof


def parse(text: str, since=None, until=None) -> list[tuple[pd.Timestamp, float, float, pd.DataFrame]]:
    """(nominal hour, launch lat, launch lon, profile) for every ascent of a station file in [since, until]."""
    since = pd.Timestamp(since) if since is not None else None
    until = pd.Timestamp(until) if until is not None else None
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        h = lines[i]
        n = int(h[32:36]) if h.startswith("#") else 0
        body = lines[i + 1:i + 1 + n]
        i += 1 + n
        hh = int(h[24:26]) if n else 99
        if hh == 99:  # no nominal hour: cannot be matched to a forecast valid time
            continue
        hour = pd.Timestamp(int(h[13:17]), int(h[18:20]), int(h[21:23]), hh)
        if (since is not None and hour < since) or (until is not None and hour > until):
            continue
        rel = h[27:31]
        launch = None
        if rel.isdigit() and rel[:2] != "99":
            # RELTIME is the launch in HHMM UTC (MM = 99 when only the hour is known); it can fall on the previous day
            launch = hour.normalize() + pd.Timedelta(hours=int(rel[:2]), minutes=0 if rel[2:] == "99" else int(rel[2:]))
            if launch - hour > pd.Timedelta(hours=12):
                launch -= pd.Timedelta(days=1)
            elif hour - launch > pd.Timedelta(hours=12):
                launch += pd.Timedelta(days=1)
        out.append((hour, int(h[55:62]) / 1e4, int(h[63:71]) / 1e4, _levels(body, launch)))
    return out


def _read(archive: str, name: str) -> str:
    with zipfile.ZipFile(io.BytesIO(http.get(f"{BASE}/access/data-{archive}/{name}"))) as z:
        return z.read(z.namelist()[0]).decode("ascii", "replace")


def ingest(store: ObsStore, since=None, until=None, archive: str = "y2d", bbox=None,
           stations: list[str] | None = None, workers: int = 8, progress=print) -> int:
    """Read the ascents in [since, until] of every station file that changed since the last update.

    ``archive`` is ``"y2d"`` (this year) or ``"por"`` (each station's whole record); ``stations``
    limits the update to these IGRA ids. Returns rows written.
    """
    sl = station_list(db_path().parent / "cache").set_index("id")
    files = listing(archive)
    done = store.ingested_versions(SOURCE)
    jobs = []
    for sid, (name, version) in sorted(files.items()):
        if (stations and sid not in stations) or sid not in sl.index:
            continue
        lat, lon = sl.at[sid, "lat"], sl.at[sid, "lon"]
        if bbox and not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
            continue
        if since is not None and sl.at[sid, "last_year"] < pd.Timestamp(since).year:
            continue
        if done.get(f"file:{archive}/{name}") != version:
            jobs.append((sid, name, version))
    progress(f"igra2 {archive}: {len(jobs)} station files to read")

    def one(job):
        name = job[1]
        try:
            return parse(_read(archive, name), since, until)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            log.warning("skipping %s: %s", name, e)
            return None

    total = 0
    with ThreadPoolExecutor(workers) as ex:
        for job, ascents in zip(jobs, ex.map(one, jobs)):
            if ascents is not None:
                total += _write_station(store, archive, job, sl.loc[job[0]], ascents, done)
    return total


def _write_station(store: ObsStore, archive: str, job, meta: pd.Series, ascents, done: dict) -> int:
    """One transaction per station file: its new ascents and the file's version."""
    igra_id, name, version = job
    sid = station_id(igra_id)
    frames, objects = [], [(f"file:{archive}/{name}", version, len(ascents))]
    for hour, lat, lon, prof in ascents:
        key = f"{sid[5:]}/{hour:%Y%m%d%H}"
        if done.get(key) == REDUCTION_VERSION:
            continue
        levels = mandatory_levels(prof, hour, lat, lon)
        obs = levels.reset_index().assign(station_id=sid, valid_time=hour, qc=0, source=SOURCE, _object=key)
        frames.append(qc.run_sounding_levels(obs[obs["variable"].isin(list(V.CATALOG))]))
        objects.append((key, REDUCTION_VERSION, len(levels)))
    station = pd.DataFrame([{"station_id": sid, "network": "RAOB", "name": meta["name"], "lat": meta["lat"],
                             "lon": meta["lon"], "elev_m": meta["elev_m"],
                             "wmo_id": sid[5:] if igra_id[2] == "M" else None}])
    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["station_id", "valid_time", "variable", "value", "qc", "source", "_object"])
    try:
        return store.ingest_batch(SOURCE, objects, station, obs)
    except ValidationError as e:  # nothing logged, so the next update retries the file
        log.warning("IGRA2 %s rejected: %s", name, e)
        return 0
