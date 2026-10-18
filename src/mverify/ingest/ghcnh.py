"""Surface observations from NOAA GHCNh (Global Historical Climatology Network - hourly).

GHCNh is NCEI's successor to ISD: ~15,000 active stations, METAR + SYNOP +
other feeds merged per station, with NCEI's own QC flags, updated daily on the
NOAA Open Data bucket (``s3://noaa-ghcnh-pds``). Each station-year is one
Parquet file, so an update re-fetches only the files whose ETag changed.

Cleaning applied here:
  * upstream QC: the Copernicus-style letter flags (a-z) and legacy
    suspect/erroneous codes (2, 3, 6, 7) set ``qc.UPSTREAM``
  * de-duplication to one value per station x variable x whole hour: the report
    closest to the hour within +-30 min wins, ties broken by report type
    (routine METAR FM-15 > SYNOP FM-12 > SPECI FM-16 > other)
  * wind is stored as u10/v10 (calm = 0, 0) and as ws10
  * mverify.db.qc surface checks (range, spike, stuck sensor, dew point > temperature)
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mverify.db import qc
from mverify.db.store import ObsStore, ValidationError, db_path
from mverify.ingest import http

BUCKET = "https://noaa-ghcnh-pds.s3.amazonaws.com"
STATION_LIST = f"{BUCKET}/hourly/doc/ghcnh-station-list.csv"
SOURCE = "ghcnh"

FIELDS = {  # GHCNh column -> canonical variable
    "temperature": "t2m",
    "dew_point_temperature": "td2m",
    "sea_level_pressure": "msl",
    "station_level_pressure": "sp",
    "precipitation_24_hour": "pr24",
}
REPORT_PRIORITY = {"FM15": 0, "FM-15": 0, "FM12": 1, "FM-12": 1, "FM16": 2, "FM-16": 2}
BAD_LEGACY = {"2", "3", "6", "7"}


def station_url(ghcn_id: str, year: int) -> str:
    return f"{BUCKET}/hourly/access/by-year/{year}/parquet/GHCNh_{ghcn_id}_{year}.parquet"


def station_list(cache_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(http.cached(STATION_LIST, cache_dir), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in ("LATITUDE", "LONGITUDE", "ELEVATION"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _flag_bad(code: pd.Series) -> np.ndarray:
    c = code.astype("string").str.strip()
    return (c.str.fullmatch(r"[a-z]+").fillna(False) | c.isin(BAD_LEGACY).fillna(False)).to_numpy(bool)


def parse(raw: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """One GHCNh station-year frame -> long canonical rows (before mverify QC)."""
    t = pd.to_datetime(raw["DATE"], utc=False)
    hour = t.dt.round("h")
    dt_min = ((t - hour).dt.total_seconds() / 60).round().astype(int)
    keep = dt_min.abs() <= 30
    parts = []

    def emit(var, value, code, rtype):
        prio = rtype.map(REPORT_PRIORITY).fillna(3).to_numpy()
        parts.append(pd.DataFrame({
            "station_id": station_id, "valid_time": hour, "variable": var,
            "value": value.to_numpy(float), "qc": np.where(_flag_bad(code), qc.UPSTREAM, 0).astype("uint16"),
            "dt_min": dt_min, "source": "ghcnh:" + rtype.fillna("?").astype(str), "_prio": prio,
        })[keep.to_numpy() & value.notna().to_numpy()])

    for col, var in FIELDS.items():
        if col in raw:
            emit(var, pd.to_numeric(raw[col], errors="coerce"), raw[f"{col}_Quality_Code"],
                 raw[f"{col}_Report_Type"])
    if "wind_speed" in raw:
        ws = pd.to_numeric(raw["wind_speed"], errors="coerce")
        wd = pd.to_numeric(raw["wind_direction"], errors="coerce")
        code = raw["wind_speed_Quality_Code"].astype("string").fillna(
            raw["wind_direction_Quality_Code"].astype("string"))
        rtype = raw["wind_speed_Report_Type"]
        emit("ws10", ws, raw["wind_speed_Quality_Code"], rtype)
        calm = ws == 0
        rad = np.deg2rad(wd)
        u = (-ws * np.sin(rad)).where(~calm, 0.0).where(calm | (wd > 0), np.nan)
        v = (-ws * np.cos(rad)).where(~calm, 0.0).where(calm | (wd > 0), np.nan)
        emit("u10", u, code, rtype)
        emit("v10", v, code, rtype)
    if not parts:
        return pd.DataFrame(columns=["station_id", "valid_time", "variable", "value", "qc", "dt_min", "source"])
    df = pd.concat(parts, ignore_index=True)
    df["_abs"] = df["dt_min"].abs()
    df = (df.sort_values(["variable", "valid_time", "qc", "_abs", "_prio"])
            .drop_duplicates(["variable", "valid_time"], keep="first"))
    return df.drop(columns=["_abs", "_prio"]).reset_index(drop=True)


MISSING = -999.9


def _clean_num(x) -> float | None:
    x = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(x) or abs(x - MISSING) < 1e-6 else float(x)


def _station_row(ghcn_id: str, meta: pd.Series, raw: pd.DataFrame | None) -> dict:
    """Station metadata; the file's own latest position beats the station list, which lags moves."""
    lat, lon, elev = (_clean_num(meta["LATITUDE"]), _clean_num(meta["LONGITUDE"]), _clean_num(meta["ELEVATION"]))
    if raw is not None and len(raw):
        last = raw.iloc[-1]
        lat = _clean_num(last.get("LATITUDE")) or lat
        lon = _clean_num(last.get("LONGITUDE")) or lon
        elev = _clean_num(last.get("ELEVATION")) if _clean_num(last.get("ELEVATION")) is not None else elev
    return {"station_id": f"GHCNH:{ghcn_id}", "network": "GHCNH", "name": str(meta["NAME"]).strip(),
            "lat": lat, "lon": lon, "elev_m": elev, "wmo_id": meta.get("WMO_ID"), "icao": meta.get("ICAO"),
            "country": meta.get("ISO_CODE")}


COLUMNS = ["DATE", "LATITUDE", "LONGITUDE", "ELEVATION"] + [
    f"{c}{suffix}" for c in (*FIELDS, "wind_speed", "wind_direction")
    for suffix in ("", "_Quality_Code", "_Report_Type")]


def _fetch(ghcn_id: str, year: int) -> pd.DataFrame:
    """One station-year, only the columns parse() reads (the files carry ~330)."""
    buf = io.BytesIO(http.get(station_url(ghcn_id, year)))
    names = set(pq.ParquetFile(buf).schema_arrow.names)
    return pd.read_parquet(buf, columns=[c for c in COLUMNS if c in names])


def select(sl: pd.DataFrame, n: int, cell_deg: float = 2.0) -> pd.DataFrame:
    """``n`` stations spread over the map: one per ``cell_deg`` box first, then a second per box, and so on.
    Within a box, stations that also report to WMO and ICAO (the real-time network models are judged on)
    come first. Taking the first ``n`` of the list instead would fill the sample alphabetically, i.e. with
    the countries whose codes come first."""
    rank = sl["ICAO"].isna().astype(int) + sl["WMO_ID"].isna().astype(int)
    cell = (np.floor(sl["LATITUDE"] / cell_deg).astype("Int64").astype(str) + ","
            + np.floor(sl["LONGITUDE"] / cell_deg).astype("Int64").astype(str))
    sl = sl.assign(_rank=rank, _cell=cell).sort_values(["_rank", "GHCN_ID"], kind="stable")
    sl = sl.assign(_pass=sl.groupby("_cell", sort=False).cumcount())
    return sl.sort_values(["_pass", "_rank", "GHCN_ID"], kind="stable").head(n).drop(columns=["_rank", "_cell", "_pass"])


def ingest(store: ObsStore, years: list[int], bbox=None, ids: list[str] | None = None,
           max_stations: int | None = None, cache_dir: Path | None = None, workers: int = 8, progress=print) -> int:
    """Fetch, clean and write GHCNh station-years; unchanged upstream files are skipped. Returns rows written."""
    cache_dir = cache_dir or db_path().parent / "cache"
    sl = station_list(cache_dir)
    if ids:
        sl = sl[sl["GHCN_ID"].isin(ids)]
    if bbox:
        lo0, la0, lo1, la1 = bbox
        sl = sl[sl["LONGITUDE"].between(lo0, lo1) & sl["LATITUDE"].between(la0, la1)]
    if max_stations:
        sl = select(sl, max_stations)
    meta = sl.set_index("GHCN_ID")
    total = 0
    for year in years:
        ids_ = list(meta.index)
        with ThreadPoolExecutor(workers) as ex:
            versions = dict(zip(ids_, ex.map(lambda g, y=year: http.etag(station_url(g, y)), ids_)))
        todo = [g for g in ids_ if versions[g] and store.ingested_version(SOURCE, f"{g}/{year}") != versions[g]]
        progress(f"ghcnh {year}: {len(todo)} of {len(ids_)} station files new or changed")
        with ThreadPoolExecutor(workers) as ex:
            # Downloads run ahead of the writer by at most one chunk, which bounds memory.
            chunks = [todo[i:i + workers] for i in range(0, len(todo), workers)]
            pending = ex.map(lambda g, y=year: _fetch(g, y), chunks[0]) if chunks else iter(())
            for k, chunk in enumerate(chunks):
                raws = list(pending)
                if k + 1 < len(chunks):
                    pending = ex.map(lambda g, y=year: _fetch(g, y), chunks[k + 1])
                total += _write_chunk(store, chunk, raws, meta, year, versions, progress)
    return total


def _write_chunk(store: ObsStore, ids: list[str], raws: list[pd.DataFrame], meta: pd.DataFrame, year: int,
                 versions: dict[str, str], progress) -> int:
    total = 0
    for gid, raw in zip(ids, raws):
        rows = qc.run_surface(parse(raw, f"GHCNH:{gid}"))
        station = pd.DataFrame([_station_row(gid, meta.loc[gid], raw)])
        try:
            window = ([f"GHCNH:{gid}"], pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31, 23))
            total += store.ingest_batch(SOURCE, [(f"{gid}/{year}", versions[gid], len(raw))], station,
                                        rows[rows["valid_time"].dt.year == year], window)
        except ValidationError as e:  # not logged, so the next update retries it
            progress(f"ghcnh {gid}/{year}: rejected, {e}")
    return total
