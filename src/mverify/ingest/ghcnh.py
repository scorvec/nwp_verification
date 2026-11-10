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
  * gauge totals (``pr24``): the 24 h totals stations report, joined by totals built from hourly amounts
    where a station gives them (``gauge_totals``); a reported total that disagrees with the built one for
    the same hours is flagged ``qc.INCONSISTENT``, and so is a large total through hours the station's own
    weather reports show as dry (``dry_weather``)
  * mverify.db.qc surface checks (range, spike, stuck sensor, dew point > temperature, and the gauge
    totals of a station with a large total that failed)
"""

from __future__ import annotations

import contextlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
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
ROWS = ["station_id", "valid_time", "variable", "value", "qc", "dt_min", "source"]

TOTAL = FIELDS["precipitation_24_hour"]
METAR = ("FM15", "FM-15")  # routine aviation reports: at a station with a gauge they carry the hour's amount
HOURLY_FEED = ("EnvCan",)  # feeds that give every hour's amount, zero included
GAUGE_HOURS = (0, 6, 12, 18)  # UTC ends of the totals built from hourly amounts
PROOF_DAYS = 30  # an hour without an amount is dry only if the station reported one this recently
AGREE_MM, AGREE_FRAC = 1.0, 0.10  # reported and built totals agree within the larger of the two
REMARKS = r"\sRMK\s"  # the part of a METAR that holds the hour's amount; some copies come without it
GAUGE_OUT = r"\bPNO\b"  # METAR remark: precipitation amount not available
FROM_REMARKS = r"(?s)\sRMK\b.*"  # cut off to leave the body of a METAR
# rain, drizzle, snow or ice falling at the station in the body of a METAR (not in the vicinity, not blowing)
FALLING = r"\d{6}Z\s(?:\S+\s)*?[-+]?(?:TS|SH|FZ)*(?:DZ|RA|SN|SG|PL|GR|GS|UP)+(?:\s|$)"
# any sign of precipitation in the body of a METAR: falling, in the vicinity, recent, a shower, a thunderstorm
WET = r"\d{6}Z\s(?:\S+\s)*?(?:[-+]|VC|RE)?(?:TS|SH|FZ)*(?:DZ|RA|SN|SG|PL|GR|GS|UP|TS|SH)+(?:\s|$)"
REPORT_TIME = r"\d{6}Z\s"  # the day and time group every METAR body carries
NO_SENSOR = r"\d{6}Z\s(?:\S+\s)*?//(?:\s|$)"  # automatic station that cannot observe the weather
# amounts over shorter periods than the 24 h total
AMOUNTS = ("precipitation", *(f"precipitation_{n}_hour" for n in range(3, 24, 3)),
           "precipitation_5_minute", "precipitation_15_minute")
SEEN_HOURS = 18  # hours of a total's 24 with a weather report before the reports can call it dry
WET_HOURS = 27  # a total is checked against the reports of this many hours to its end
FAILED = pa.schema([("station_id", pa.string()), ("valid_time", pa.timestamp("us")), ("qc", pa.uint16())])


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
        return gauge_totals(raw, station_id)
    df = pd.concat(parts, ignore_index=True)
    df["_abs"] = df["dt_min"].abs()
    df = (df.sort_values(["variable", "valid_time", "qc", "_abs", "_prio"])
            .drop_duplicates(["variable", "valid_time"], keep="first"))
    return _flag_dry(_merge_totals(df.drop(columns=["_abs", "_prio"]), gauge_totals(raw, station_id)), raw)


def gauge_totals(raw: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """24 h totals built from a station's hourly amounts: canonical ``pr24`` rows ending at ``GAUGE_HOURS``.

    Two kinds of report give an hour's amount. A routine METAR at a station with a gauge carries the amount
    since the previous routine report and leaves the group out after a dry hour, so an hour without one
    counts as dry only where the station is known to be measuring: it reported an amount (a trace counts)
    within the ``PROOF_DAYS`` before, the report came with its remarks, they do not say the gauge is out,
    and it does not have rain or snow falling. The routine report is the one at the station's usual minute;
    the specials between two routine reports repeat a running total and are not summed. An hourly feed
    gives every hour's amount. A total needs all 24 of its hours, so a station that reports no hourly
    amounts gets none, and neither does the first day of a station-year file.
    """
    if "precipitation" not in raw:
        return pd.DataFrame(columns=ROWS)
    ptype = raw["precipitation_Report_Type"].astype("string")
    amount = pd.to_numeric(raw["precipitation"], errors="coerce")
    if not (amount.notna() & ptype.isin((*METAR, *HOURLY_FEED))).any():  # most of the world: nothing to build from
        return pd.DataFrame(columns=ROWS)
    t = pd.to_datetime(raw["DATE"], utc=False)
    hour = t.dt.round("h")
    dt_min = ((t - hour).dt.total_seconds() / 60).round().astype(int)
    near = dt_min.abs() <= 15
    bad = pd.Series(_flag_bad(raw["precipitation_Quality_Code"]), raw.index)
    metar = raw.filter(regex="_Report_Type$").astype("string").isin(METAR).any(axis=1) & near
    usual = t[metar].dt.minute.mode()
    routine = metar & (t.dt.minute == (usual.iloc[0] if len(usual) else -1))
    own = routine & ptype.isin(METAR)
    remark = raw.get("REM", pd.Series(pd.NA, raw.index)).astype("string")
    body = remark.str.replace(FROM_REMARKS, " ", regex=True)
    rows = pd.DataFrame({
        "hour": hour, "dt_min": dt_min, "routine": routine,
        "own": amount.where(own & ~bad), "feed": amount.where(ptype.isin(HOURLY_FEED) & near & ~bad),
        "full": routine & remark.str.contains(REMARKS, na=False),
        "out": routine & ((own & bad) | remark.str.contains(GAUGE_OUT, na=False)),
        "falling": routine & body.str.contains(FALLING, na=False),
    })
    rows = rows[rows["routine"] | rows["feed"].notna()]
    if rows.empty:
        return pd.DataFrame(columns=ROWS)
    hours = rows.groupby("hour")
    index = pd.date_range(rows["hour"].min(), rows["hour"].max(), freq="h")
    own, feed = hours["own"].max().reindex(index), hours["feed"].max().reindex(index)
    full, out, falling = (hours[c].any().reindex(index, fill_value=False) for c in ("full", "out", "falling"))
    proven = own.notna().rolling(24 * PROOF_DAYS, min_periods=1).max().astype(bool)
    dry = full & proven & ~out & ~falling & own.isna()
    hourly = feed.fillna(own.where(~out).mask(dry, 0.0))
    total = hourly.rolling(24, min_periods=24).sum().round(1)
    total = total[total.notna() & total.index.hour.isin(GAUGE_HOURS)]
    return pd.DataFrame({
        "station_id": station_id, "valid_time": total.index, "variable": TOTAL,
        "value": total.to_numpy(float), "qc": np.zeros(len(total), "uint16"),
        "dt_min": hours["dt_min"].first().reindex(total.index).to_numpy(int), "source": "ghcnh:hourly",
    })


def _merge_totals(df: pd.DataFrame, built: pd.DataFrame) -> pd.DataFrame:
    """Reported and built 24 h totals as one series. A reported total stands, flagged where the total built
    for the same hours differs from it by more than ``AGREE_MM`` and ``AGREE_FRAC`` of the larger; one the
    provider flagged gives way to the built total; an hour without a reported total takes the built one."""
    if built.empty:
        return df.reset_index(drop=True)
    reported = (df["variable"] == TOTAL).to_numpy()
    value, flags = df["value"].to_numpy(float), df["qc"].to_numpy()
    other = built.set_index("valid_time")["value"].reindex(df["valid_time"]).to_numpy(float)
    replaced = reported & (flags != 0) & ~np.isnan(other)
    differ = reported & (np.abs(value - other) > np.maximum(AGREE_MM, AGREE_FRAC * np.maximum(value, other)))
    df = df.assign(qc=np.where(differ, flags | qc.INCONSISTENT, flags).astype("uint16"))[~replaced]
    held = df.loc[df["variable"] == TOTAL, "valid_time"]
    return pd.concat([df, built[~built["valid_time"].isin(held)]], ignore_index=True)


def dry_weather(raw: pd.DataFrame, ends: pd.DatetimeIndex) -> np.ndarray:
    """Which of the 24 h totals ending at ``ends`` the station's own reports show as dry.

    A total passed in dry weather when at least ``SEEN_HOURS`` of its hours have a METAR that can show the
    weather (not the ``//`` of a station without a sensor), no report of the ``WET_HOURS`` to its end shows
    precipitation or carries an amount above zero for a shorter period, and the station did report
    precipitation in the ``PROOF_DAYS`` before, so its reports are known to show it.
    """
    hour = pd.to_datetime(raw["DATE"], utc=False).dt.round("h")
    index = pd.date_range(hour.min(), hour.max(), freq="h")
    remark = raw["REM"].astype("string")

    def before(times, hours):  # the reports of the ``hours`` to any of ``times``
        mark = pd.Series(0.0, index)
        mark.loc[times] = 1.0
        return hour.map(mark[::-1].rolling(hours, min_periods=1).max()[::-1]).to_numpy(bool)

    def count(flag, rows, hours):  # at each of ``ends``: how many of the last ``hours`` have a flagged report
        hit = flag.groupby(hour[rows]).any().reindex(index, fill_value=False).astype(float)
        return hit.rolling(hours, min_periods=1).sum().reindex(ends).to_numpy()

    rows = before(ends, WET_HOURS)
    body = remark[rows].str.replace(FROM_REMARKS, " ", regex=True)
    sees = body.str.contains(REPORT_TIME, na=False) & ~body.str.contains(NO_SENSOR, na=False)
    wet = body.str.contains(WET, na=False)
    for col in AMOUNTS:
        if col in raw:
            wet |= pd.to_numeric(raw.loc[rows, col], errors="coerce") > 0
    dry = (count(sees, rows, 24) >= SEEN_HOURS) & (count(wet, rows, WET_HOURS) == 0)
    if dry.any():  # only now the month before: has the station ever shown precipitation?
        rows = before(ends[dry], 24 * PROOF_DAYS)
        proof = remark[rows].str.replace(FROM_REMARKS, " ", regex=True).str.contains(WET, na=False)
        dry &= count(proof, rows, 24 * PROOF_DAYS) > 0
    return dry


def _flag_dry(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Flag the large 24 h totals that ``dry_weather`` contradicts."""
    large = ((df["variable"] == TOTAL) & (df["value"] >= qc.LARGE_TOTAL)).to_numpy()
    if "REM" not in raw or not large.any():
        return df
    flags = df["qc"].to_numpy().copy()
    flags[large] |= np.where(dry_weather(raw, pd.DatetimeIndex(df.loc[large, "valid_time"])),
                             qc.INCONSISTENT, 0).astype("uint16")
    return df.assign(qc=flags)


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


COLUMNS = ["DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "REM", *AMOUNTS[1:]] + [
    f"{c}{suffix}" for c in (*FIELDS, "wind_speed", "wind_direction", "precipitation")
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
           max_stations: int | None = None, cache_dir: Path | None = None, workers: int = 8, progress=print,
           since=None, until=None, failed_totals: Path | None = None) -> int:
    """Fetch, clean and write GHCNh station-years; unchanged upstream files are skipped. Returns rows written.

    ``since`` / ``until`` keep only the observations in that window. Each station-year is still read and
    quality-controlled whole, so the spike and stuck-sensor checks see the hours either side of it, but
    only the window is stored: every station for a few weeks takes a small part of the disk of a year.

    ``failed_totals`` (a directory) takes ``failed_totals_<year>.parquet``: the 24 h gauge totals that the
    checks failed in the station-years read, over the whole year whatever the window (station_id,
    valid_time, qc). Pairs kept from an earlier window may hold totals that a later failure of their
    station makes suspect; the file names them.
    """
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
    lo = pd.Timestamp.min if since is None else pd.Timestamp(since)
    hi = pd.Timestamp.max if until is None else pd.Timestamp(until)
    # a file read for one window is not ingested for another: the window is part of the version logged
    tag = "" if since is None and until is None else f"@{since}..{until}"
    total = 0
    for year in years:
        start, end = max(pd.Timestamp(year, 1, 1), lo), min(pd.Timestamp(year, 12, 31, 23), hi)
        if start > end:
            continue
        ids_ = list(meta.index)
        with ThreadPoolExecutor(workers) as ex:
            etags = ex.map(lambda g, y=year: http.etag(station_url(g, y)), ids_)
            versions = {g: v and v + tag for g, v in zip(ids_, etags)}
        todo = [g for g in ids_ if versions[g] and store.ingested_version(SOURCE, f"{g}/{year}") != versions[g]]
        progress(f"ghcnh {year}: {len(todo)} of {len(ids_)} station files new or changed")
        with _failed_writer(failed_totals, year) as failed, ThreadPoolExecutor(workers) as ex:
            # Downloads run ahead of the writer by at most one chunk, which bounds memory.
            chunks = [todo[i:i + workers] for i in range(0, len(todo), workers)]
            pending = ex.map(lambda g, y=year: _fetch(g, y), chunks[0]) if chunks else iter(())
            for k, chunk in enumerate(chunks):
                raws = list(pending)
                if k + 1 < len(chunks):
                    pending = ex.map(lambda g, y=year: _fetch(g, y), chunks[k + 1])
                total += _write_chunk(store, chunk, raws, meta, (start, end), versions, progress, failed)
    return total


def _failed_writer(root: Path | None, year: int):
    """The year's file of failed gauge totals in ``root``, open for writing; None without a ``root``."""
    if root is None:
        return contextlib.nullcontext()
    Path(root).mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(Path(root) / f"failed_totals_{year}.parquet", FAILED, compression="zstd")


def _write_chunk(store: ObsStore, ids: list[str], raws: list[pd.DataFrame], meta: pd.DataFrame,
                 span: tuple[pd.Timestamp, pd.Timestamp], versions: dict[str, str], progress,
                 failed: pq.ParquetWriter | None = None) -> int:
    """Station-years read whole, written for the hours of ``span`` (inside one year); their failed gauge
    totals of any hour go to ``failed``."""
    start, end = span
    total = 0
    for gid, raw in zip(ids, raws):
        rows = qc.run_surface(parse(raw, f"GHCNH:{gid}"))
        bad = rows[(rows["variable"] == TOTAL) & (rows["qc"] != 0)]
        if failed is not None and len(bad):
            failed.write_table(pa.Table.from_pandas(bad[FAILED.names], schema=FAILED, preserve_index=False))
        station = pd.DataFrame([_station_row(gid, meta.loc[gid], raw)])
        try:
            window = ([f"GHCNH:{gid}"], start, end)
            total += store.ingest_batch(SOURCE, [(f"{gid}/{start.year}", versions[gid], len(raw))], station,
                                        rows[rows["valid_time"].between(start, end)], window)
        except ValidationError as e:  # not logged, so the next update retries it
            progress(f"ghcnh {gid}/{start.year}: rejected, {e}")
    return total
