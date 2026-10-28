"""The observation database: creation, validated writes and the queries scoring needs.

``ObsStore`` wraps one DuckDB connection. Where the database file lives is
resolved in this order: the ``path`` argument, ``$MVERIFY_DB``,
``~/.cache/mverify/obs.duckdb``. ``ObsStore.snapshot(root)`` opens the published
Parquet snapshot read-only instead, so a user can verify without building
anything; every query below runs unchanged against either.

Writes are validated before they reach SQL (``validate_obs`` / ``validate_stations``)
so a bad batch fails with a message naming the column and the offending rows,
and each batch is written in one transaction together with its ingest_log row:
an object is either fully ingested and logged, or not at all.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

from mverify import variables as V
from mverify.db import migrations, qc

SCHEMA_VERSION = 2
DEFAULT_PATH = Path.home() / ".cache" / "mverify" / "obs.duckdb"

OBS_COLUMNS = ("station_id", "valid_time", "variable", "value", "qc", "dt_min", "lat", "lon", "source")
STATION_COLUMNS = ("station_id", "network", "name", "lat", "lon", "elev_m", "t_height_m", "wind_height_m",
                   "wmo_id", "icao", "country")
MAX_DT_MIN = 180  # observation time may differ from valid_time by this much (balloon flight time)


class SchemaError(RuntimeError):
    """The database was written by an incompatible schema version."""


class ValidationError(ValueError):
    """A batch handed to the store breaks the schema's rules."""


def _lit(s: str | os.PathLike) -> str:
    """A SQL string literal, for the statements (DDL, COPY) DuckDB cannot parameterise."""
    return "'" + str(s).replace("'", "''") + "'"


def db_path(path: str | os.PathLike | None = None) -> Path:
    return Path(path or os.environ.get("MVERIFY_DB") or DEFAULT_PATH).expanduser()


def _grid_root(db: Path) -> str:
    """Gridded analyses of ``obs.duckdb`` live in ``obs_grids/`` next to it."""
    return str(db.parent / f"{db.stem}_grids")


# --------------------------------------------------------------------------- validation

def _fail(what: str, df: pd.DataFrame, mask) -> None:
    mask = np.asarray(mask, bool)
    if mask.any():
        sample = df.loc[mask].head(3).to_dict("records")
        raise ValidationError(f"{mask.sum()} row(s) {what}; first: {sample}")


def validate_stations(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"station_id", "network", "lat", "lon"} - set(df.columns)
    if missing:
        raise ValidationError(f"stations batch lacks columns {sorted(missing)}")
    out = df.reindex(columns=list(STATION_COLUMNS)).copy()
    for c in ("lat", "lon", "elev_m", "t_height_m", "wind_height_m"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("station_id", "network", "name", "wmo_id", "icao", "country"):
        out[c] = out[c].astype("string").str.strip().replace("", pd.NA)
    out["lon"] = ((out["lon"] + 180.0) % 360.0) - 180.0
    _fail("have no station_id/network", out, out["station_id"].isna() | out["network"].isna())
    _fail("have a station_id not prefixed by its network", out,
          [not (isinstance(s, str) and isinstance(n, str) and s.startswith(n + ":"))
           for s, n in zip(out["station_id"], out["network"])])
    _fail("have latitude outside -90..90 or missing", out, ~out["lat"].between(-90, 90))
    _fail("have elevation outside -500..9000 m", out, out["elev_m"].notna() & ~out["elev_m"].between(-500, 9000))
    _fail("have a thermometer height outside 0.5..30 m", out,
          out["t_height_m"].notna() & ~out["t_height_m"].between(0.5, 30))
    _fail("have an anemometer height outside 1..100 m", out,
          out["wind_height_m"].notna() & ~out["wind_height_m"].between(1, 100))
    _fail("repeat a station_id within the batch", out, out["station_id"].duplicated(keep=False))
    return out


def validate_obs(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"station_id", "valid_time", "variable", "value", "source"} - set(df.columns)
    if missing:
        raise ValidationError(f"obs batch lacks columns {sorted(missing)}")
    out = df.reindex(columns=list(OBS_COLUMNS)).copy()
    out["valid_time"] = pd.to_datetime(out["valid_time"])
    if getattr(out["valid_time"].dt, "tz", None) is not None:
        out["valid_time"] = out["valid_time"].dt.tz_convert("UTC").dt.tz_localize(None)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["qc"] = out["qc"].fillna(0).astype("uint16")
    out["dt_min"] = out["dt_min"].fillna(0).astype("int16")
    for c in ("lat", "lon"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["lon"] = ((out["lon"] + 180.0) % 360.0) - 180.0
    _fail("name a variable outside the catalogue", out, ~out["variable"].isin(list(V.CATALOG)))
    _fail("have a non-finite value", out, ~np.isfinite(out["value"].to_numpy(float)))
    _fail("have a valid_time that is not a whole hour", out, out["valid_time"] != out["valid_time"].dt.floor("h"))
    _fail(f"have dt_min outside -{MAX_DT_MIN}..{MAX_DT_MIN}", out, ~out["dt_min"].between(-MAX_DT_MIN, MAX_DT_MIN))
    _fail("have only one of lat/lon", out, out["lat"].isna() != out["lon"].isna())
    _fail("have latitude outside -90..90", out, out["lat"].notna() & ~out["lat"].between(-90, 90))
    _fail("have no source", out, out["source"].isna())
    known = sum(qc.FLAGS)
    _fail("carry qc bits not defined in mverify.db.qc", out, (out["qc"].to_numpy() & ~np.uint16(known)) != 0)
    _fail("repeat a (station_id, valid_time, variable) key within the batch", out,
          out.duplicated(["station_id", "valid_time", "variable"], keep=False))
    return out


# --------------------------------------------------------------------------- the store

class ObsStore:
    def __init__(self, con: duckdb.DuckDBPyConnection, read_only: bool = False, grid_root: str | None = None):
        self.con = con
        self.read_only = read_only
        # Gridded analyses live as Zarr stores under grid_root; the grids table holds paths relative to it.
        self.grid_root = grid_root
        self._temporary_grids = False

    # ---- opening

    @classmethod
    def open(cls, path: str | os.PathLike | None = None, read_only: bool = False) -> ObsStore:
        p = db_path(path)
        if read_only:
            if not p.exists():
                raise FileNotFoundError(f"no observation database at {p}; run `mverify db init` first")
            store = cls(duckdb.connect(str(p), read_only=True), read_only=True, grid_root=_grid_root(p))
            store._check_version()
            return store
        p.parent.mkdir(parents=True, exist_ok=True)
        store = cls(duckdb.connect(str(p)), grid_root=_grid_root(p))
        store._create()
        return store

    @classmethod
    def memory(cls) -> ObsStore:
        store = cls(duckdb.connect(":memory:"), grid_root=tempfile.mkdtemp(prefix="mverify-grids-"))
        store._temporary_grids = True
        store._create()
        return store

    @classmethod
    def snapshot(cls, root: str) -> ObsStore:
        """Read-only views over a published snapshot (local directory or https:// / s3:// URL).

        Layout: ``{root}/stations.parquet``, ``{root}/variables.parquet``,
        ``{root}/obs/variable=*/year=*/month=*/*.parquet``, and for gridded analyses
        ``{root}/grids.parquet`` plus the Zarr stores under ``{root}/grids/``. Hive
        partitioning means a query reads only the variables and months it touches.
        """
        con = duckdb.connect(":memory:")
        if root.startswith(("http://", "https://", "s3://")):
            con.execute("INSTALL httpfs; LOAD httpfs;")
        for t in ("stations", "variables"):
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet({_lit(f'{root}/{t}.parquet')})")
        con.execute("CREATE VIEW obs AS SELECT * EXCLUDE (year, month) FROM "
                    f"read_parquet({_lit(f'{root}/obs/*/*/*/*.parquet')}, hive_partitioning = true)")
        try:
            con.execute(f"CREATE VIEW grids AS SELECT * FROM read_parquet({_lit(f'{root}/grids.parquet')})")
        except duckdb.Error:  # a snapshot without gridded analyses
            con.execute("CREATE VIEW grids AS SELECT NULL::VARCHAR AS product, NULL::VARCHAR AS variable, "
                        "NULL::SMALLINT AS accum_hours, NULL::TIMESTAMP AS valid_time, NULL::VARCHAR AS uri "
                        "WHERE false")
        return cls(con, read_only=True, grid_root=f"{root}/grids")

    def close(self) -> None:
        self.con.close()
        if self._temporary_grids:
            shutil.rmtree(self.grid_root, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _create(self) -> None:
        version = self._version()
        if version is not None and version > SCHEMA_VERSION:
            raise SchemaError(f"database has schema v{version}, newer than this mverify (v{SCHEMA_VERSION}); "
                              "upgrade mverify")
        schema_sql = resources.files("mverify.db").joinpath("schema.sql").read_text()
        with self.transaction():
            if version is not None and version < SCHEMA_VERSION:
                migrations.upgrade(self.con, schema_sql, version, SCHEMA_VERSION)
            self.con.execute(schema_sql)
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", [str(SCHEMA_VERSION)])
            self.con.executemany(
                "INSERT INTO variables (name, label, unit, kind, level_hpa, lo, hi, sigma_o) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (name) DO UPDATE SET label = excluded.label, "
                "unit = excluded.unit, lo = excluded.lo, hi = excluded.hi, sigma_o = excluded.sigma_o",
                [(v.name, v.label, v.unit, v.kind, v.level, v.lo, v.hi, v.sigma_o) for v in V.CATALOG.values()])
            self.con.executemany(
                "INSERT INTO qc_flags VALUES (?, ?, ?) ON CONFLICT (bit) DO UPDATE SET "
                "name = excluded.name, description = excluded.description",
                [(bit, name, desc) for bit, (name, desc) in qc.FLAGS.items()])

    def _version(self) -> int | None:
        try:
            row = self.con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        except duckdb.CatalogException:
            return None
        return int(row[0]) if row else None

    def _check_version(self) -> None:
        v = self._version()
        if v != SCHEMA_VERSION:
            raise SchemaError(f"database has schema v{v}, this mverify reads v{SCHEMA_VERSION}; "
                              "run `mverify db init` to upgrade it")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.con.execute("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("COMMIT")

    # ---- writes

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("store is read-only")

    def upsert_stations(self, df: pd.DataFrame) -> int:
        self._require_writable()
        df = validate_stations(df)
        if df.empty:
            return 0
        df["updated_at"] = pd.Timestamp.now("UTC").tz_localize(None)
        cols = list(df.columns)
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "station_id")
        self.con.register("_stations", df)
        try:
            self.con.execute(f"INSERT INTO stations ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _stations "
                             f"ON CONFLICT (station_id) DO UPDATE SET {updates}")
        finally:
            self.con.unregister("_stations")
        return len(df)

    def set_sensor_heights(self, df: pd.DataFrame) -> tuple[int, list[str]]:
        """Set thermometer / anemometer heights from rows naming a station by station_id, wmo or icao.

        A row may carry either height; a missing one leaves the stored value alone. Returns
        the number of stations updated and the identifiers of rows that matched no station.
        """
        self._require_writable()
        keys = [c for c in ("station_id", "wmo", "icao") if c in df.columns]
        heights = [c for c in ("t_height_m", "wind_height_m") if c in df.columns]
        if not keys or not heights:
            raise ValidationError("sensor file needs a station_id, wmo or icao column and t_height_m and/or "
                                  "wind_height_m")
        rows = df.reindex(columns=["station_id", "wmo", "icao", "t_height_m", "wind_height_m"]).copy()
        for c in ("station_id", "wmo", "icao"):
            rows[c] = rows[c].astype("string").str.strip().replace("", pd.NA)
        rows["wmo"] = rows["wmo"].str.zfill(5)
        for c in ("t_height_m", "wind_height_m"):
            rows[c] = pd.to_numeric(rows[c], errors="coerce")
        _fail("have a thermometer height outside 0.5..30 m", rows,
              rows["t_height_m"].notna() & ~rows["t_height_m"].between(0.5, 30))
        _fail("have an anemometer height outside 1..100 m", rows,
              rows["wind_height_m"].notna() & ~rows["wind_height_m"].between(1, 100))
        rows["_row"] = np.arange(len(rows))
        self.con.register("_sensors", rows)
        try:
            # Every station a row names: by its own id, or all stations sharing the WMO index or ICAO code.
            match = ("FROM _sensors x JOIN stations s ON s.station_id = x.station_id "
                     "OR (x.station_id IS NULL AND (s.wmo_id = x.wmo OR s.icao = x.icao))")
            unmatched = self.con.execute(f"SELECT coalesce(station_id, wmo, icao) FROM _sensors WHERE _row NOT IN "
                                         f"(SELECT x._row {match})").fetchall()
            with self.transaction():
                n = self.con.execute(
                    "UPDATE stations SET t_height_m = coalesce(m.t_height_m, stations.t_height_m), "
                    "wind_height_m = coalesce(m.wind_height_m, stations.wind_height_m), updated_at = now() "
                    f"FROM (SELECT s.station_id, any_value(x.t_height_m) AS t_height_m, "
                    f"any_value(x.wind_height_m) AS wind_height_m {match} GROUP BY s.station_id) m "
                    "WHERE stations.station_id = m.station_id").fetchone()[0]
        finally:
            self.con.unregister("_sensors")
        return int(n), [str(r[0]) for r in unmatched]

    def _check_references(self, obs: pd.DataFrame) -> None:
        ids = obs["station_id"].drop_duplicates()
        known = self.con.execute("SELECT station_id FROM stations WHERE station_id IN (SELECT unnest(?::VARCHAR[]))",
                                 [ids.tolist()]).df()["station_id"]
        unknown = sorted(set(ids) - set(known))
        if unknown:
            raise ValidationError(f"obs name {len(unknown)} station(s) not in stations, e.g. {unknown[:3]}")

    def _insert_obs(self, obs: pd.DataFrame, delete_sql: str, delete_args: list) -> int:
        self.con.register("_obs", obs)
        try:
            self.con.execute(delete_sql, delete_args)
            self.con.execute(f"INSERT INTO obs ({', '.join(OBS_COLUMNS)}) SELECT {', '.join(OBS_COLUMNS)} FROM _obs")
        finally:
            self.con.unregister("_obs")
        return len(obs)

    def upsert_obs(self, df: pd.DataFrame) -> int:
        """Insert observations, replacing any row with the same (station_id, valid_time, variable)."""
        self._require_writable()
        obs = validate_obs(df)
        if obs.empty:
            return 0
        self._check_references(obs)
        return self._insert_obs(obs, "DELETE FROM obs o USING _obs b WHERE o.station_id = b.station_id "
                                     "AND o.valid_time = b.valid_time AND o.variable = b.variable", [])

    def replace_obs(self, df: pd.DataFrame, station_ids: Sequence[str], start, end) -> int:
        """Make ``df`` the complete set of observations of ``station_ids`` valid in [start, end].

        This is how an upstream object (a station-year file, one ascent) is written:
        rows it no longer contains, because the provider withdrew or re-timed them,
        disappear too. Every row of ``df`` must fall inside the window.
        """
        self._require_writable()
        obs = validate_obs(df)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        _fail("fall outside the replace window", obs,
              ~obs["station_id"].isin(list(station_ids)) | ~obs["valid_time"].between(start, end))
        if len(obs):
            self._check_references(obs)
        return self._insert_obs(
            obs, "DELETE FROM obs WHERE station_id IN (SELECT unnest(?::VARCHAR[])) AND valid_time BETWEEN ? AND ?",
            [list(station_ids), start.to_pydatetime(), end.to_pydatetime()])

    def ingest_batch(self, source: str, objects: Sequence[tuple[str, str, int]], stations: pd.DataFrame | None,
                     obs: pd.DataFrame, window: tuple | None = None) -> int:
        """Write upstream objects atomically: their stations, observations and ingest_log rows.

        ``objects`` lists ``(object_key, version, rows_in)`` for every object the
        batch carries. With ``window = (station_ids, start, end)`` the batch
        replaces exactly the rows it owns (see ``replace_obs``), the right mode for
        one authoritative file such as a station-year; without it rows are upserted
        by key, the right mode for many small immutable objects such as ascents.
        """
        with self.transaction():
            if stations is not None and len(stations):
                self.upsert_stations(stations)
            n = self.replace_obs(obs, *window) if window is not None else self.upsert_obs(obs)
            kept = obs.groupby("_object").size() if "_object" in obs else None
            self.con.executemany(
                "INSERT INTO ingest_log (source, object_key, version, rows_in, rows_kept) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (source, object_key) DO UPDATE SET version = excluded.version, "
                "rows_in = excluded.rows_in, rows_kept = excluded.rows_kept, ingested_at = now()",
                [(source, key, version, rows_in,
                  int(kept.get(key, 0)) if kept is not None else (n if len(objects) == 1 else 0))
                 for key, version, rows_in in objects])
        return n

    # ---- gridded analyses

    def add_grid(self, da: xr.DataArray) -> str:
        """Store one analysis window (see mverify.precip.analyses) and catalogue it; returns its uri."""
        self._require_writable()
        product, hours = str(da.attrs["product"]), int(da.attrs["accum_hours"])
        valid = pd.Timestamp(da.attrs["valid_time"])
        if valid != valid.floor("h"):
            raise ValidationError(f"{product}: window end {valid} is not a whole hour")
        uri = f"{product}/{hours}h/{valid:%Y/%Y%m%d%H}.zarr"
        dest = Path(self.grid_root) / uri
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        da.astype("float32").to_dataset(name="tp").to_zarr(tmp, mode="w", consolidated=False)
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
        self.con.execute("INSERT OR REPLACE INTO grids VALUES (?, 'tp', ?, ?, ?)",
                         [product, hours, valid.to_pydatetime(), uri])
        return uri

    def grid_times(self, product: str, accum_hours: int, start, end) -> list[pd.Timestamp]:
        rows = self.con.execute(
            "SELECT valid_time FROM grids WHERE product = ? AND variable = 'tp' AND accum_hours = ? "
            "AND valid_time BETWEEN ? AND ? ORDER BY 1",
            [product, accum_hours, pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()]).fetchall()
        return [pd.Timestamp(r[0]) for r in rows]

    def get_grid(self, product: str, accum_hours: int, valid_time) -> xr.DataArray | None:
        row = self.con.execute(
            "SELECT uri FROM grids WHERE product = ? AND variable = 'tp' AND accum_hours = ? AND valid_time = ?",
            [product, accum_hours, pd.Timestamp(valid_time).to_pydatetime()]).fetchone()
        if row is None:
            return None
        with xr.open_zarr(f"{self.grid_root}/{row[0]}", consolidated=False) as ds:
            return ds["tp"].load()

    def check_integrity(self) -> dict[str, int]:
        """Counts of rows breaking each rule the schema does not enforce itself; all zero when healthy."""
        q = {
            "duplicate_keys": "SELECT count(*) FROM (SELECT 1 FROM obs GROUP BY station_id, valid_time, variable "
                              "HAVING count(*) > 1)",
            "orphan_station": "SELECT count(*) FROM obs o ANTI JOIN stations s USING (station_id)",
            "unknown_variable": "SELECT count(*) FROM obs o ANTI JOIN variables v ON o.variable = v.name",
            "undefined_qc_bits": f"SELECT count(*) FROM obs WHERE qc & ~{sum(qc.FLAGS)}::USMALLINT <> 0",
            "range_not_flagged": "SELECT count(*) FROM obs o JOIN variables v ON o.variable = v.name "
                                 f"WHERE (o.value < v.lo OR o.value > v.hi) AND o.qc & {qc.RANGE} = 0",
            "stations_without_obs": "SELECT count(*) FROM stations s ANTI JOIN obs o USING (station_id)",
        }
        return {k: int(self.con.execute(sql).fetchone()[0]) for k, sql in q.items()}

    # ---- reads

    def ingested_version(self, source: str, object_key: str) -> str | None:
        row = self.con.execute("SELECT version FROM ingest_log WHERE source = ? AND object_key = ?",
                               [source, object_key]).fetchone()
        return row[0] if row else None

    def ingested_versions(self, source: str) -> dict[str, str]:
        """object_key -> version of everything ingested from ``source``, for update planning in bulk."""
        return dict(self.con.execute("SELECT object_key, version FROM ingest_log WHERE source = ?",
                                     [source]).fetchall())

    def fetch_obs(self, variables: Sequence[str], start, end,
                  bbox: tuple[float, float, float, float] | None = None,
                  networks: Sequence[str] | None = None, clean_only: bool = True,
                  times: Sequence | None = None) -> pd.DataFrame:
        """Observations with station metadata for ``variables`` valid in [start, end].

        ``bbox`` selects by station position; the returned lat/lon are the observation's own.
        ``times`` keeps only those valid times (a forecast's steps, say, of hourly reports).

        ``bbox`` is (lon_min, lat_min, lon_max, lat_max) in degrees, lon in -180..180;
        a box with lon_min > lon_max crosses the dateline.
        """
        where = ["o.variable IN (SELECT unnest(?::VARCHAR[]))", "o.valid_time BETWEEN ? AND ?"]
        args: list = [list(variables), pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()]
        if clean_only:
            where.append("o.qc = 0")
        if times is not None:
            where.append("o.valid_time IN (SELECT unnest(?::TIMESTAMP[]))")
            args.append([pd.Timestamp(t).to_pydatetime() for t in times])
        if networks:
            where.append("s.network IN (SELECT unnest(?::VARCHAR[]))")
            args.append(list(networks))
        if bbox:
            lo0, la0, lo1, la1 = bbox
            where.append("s.lat BETWEEN ? AND ?")
            args += [la0, la1]
            if lo0 <= lo1:
                where.append("s.lon BETWEEN ? AND ?")
            else:
                where.append("(s.lon >= ? OR s.lon <= ?)")
            args += [lo0, lo1]
        # lat/lon are where the value was observed: the drifted balloon position when known, else the station.
        return self.con.execute(
            "SELECT o.station_id, o.valid_time, o.variable, o.value, o.dt_min, coalesce(o.lat, s.lat) AS lat, "
            "coalesce(o.lon, s.lon) AS lon, s.elev_m, s.t_height_m, s.wind_height_m, s.network "
            "FROM obs o JOIN stations s USING (station_id) WHERE " + " AND ".join(where)
            + " ORDER BY o.variable, o.valid_time, o.station_id", args).df()

    def summary(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT s.network, o.variable, count(*) AS n, count(*) FILTER (WHERE o.qc = 0) AS clean, "
            "count(DISTINCT o.station_id) AS stations, min(o.valid_time) AS first, max(o.valid_time) AS last "
            "FROM obs o JOIN stations s USING (station_id) GROUP BY ALL ORDER BY 1, 2").df()

    def qc_report(self) -> pd.DataFrame:
        """Share of rows failing each QC check, by variable."""
        return self.con.execute(
            "SELECT o.variable, f.name AS flag, count(*) FILTER (WHERE o.qc & f.bit <> 0) AS flagged, "
            "count(*) AS n, round(100.0 * flagged / n, 3) AS pct "
            "FROM obs o CROSS JOIN qc_flags f GROUP BY ALL HAVING flagged > 0 ORDER BY 1, 2").df()

    def export_parquet(self, root: str | os.PathLike) -> None:
        """Write the snapshot layout read by ``ObsStore.snapshot``."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for t in ("stations", "variables", "grids"):
            self.con.execute(f"COPY {t} TO {_lit(root / (t + '.parquet'))} (FORMAT parquet, COMPRESSION zstd)")
        if self.grid_root and Path(self.grid_root).is_dir():
            shutil.copytree(self.grid_root, root / "grids", dirs_exist_ok=True)
        self.con.execute(
            "COPY (SELECT *, year(valid_time) AS year, month(valid_time) AS month FROM obs "
            "ORDER BY variable, valid_time, station_id) "
            f"TO {_lit(root / 'obs')} (FORMAT parquet, COMPRESSION zstd, PARTITION_BY (variable, year, month), "
            "OVERWRITE_OR_IGNORE true)")
