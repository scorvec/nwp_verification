"""Upgrades of an existing database to the current schema, applied by ``ObsStore`` on open.

DuckDB cannot add or change a CHECK constraint on an existing table, so a
migration that touches constraints rebuilds the table: the old one is renamed,
``schema.sql`` creates the current definition, the rows are copied across by
column name and the old table is dropped. A migrated database is therefore
identical to a freshly created one, which the tests assert. Each step runs in
the caller's transaction, so a failed upgrade leaves the database untouched.
"""

from __future__ import annotations

from collections.abc import Callable

import duckdb


def _rebuild(con: duckdb.DuckDBPyConnection, schema_sql: str, tables: tuple[str, ...]) -> None:
    for t in tables:
        con.execute(f"ALTER TABLE {t} RENAME TO {t}_old")
    con.execute(schema_sql)
    for t in tables:
        con.execute(f"INSERT INTO {t} BY NAME SELECT * FROM {t}_old")
        con.execute(f"DROP TABLE {t}_old")


def _v1_to_v2(con: duckdb.DuckDBPyConnection, schema_sql: str) -> None:
    # v2: variables.sigma_o, stations.t_height_m / wind_height_m, obs.lat / lon for
    # drifting radiosondes, and obs.dt_min widened from +-59 to +-180 minutes.
    _rebuild(con, schema_sql, ("variables", "stations", "obs"))


# MIGRATIONS[v] upgrades a database from schema v to v + 1.
MIGRATIONS: dict[int, Callable[[duckdb.DuckDBPyConnection, str], None]] = {1: _v1_to_v2}


def upgrade(con: duckdb.DuckDBPyConnection, schema_sql: str, version: int, target: int) -> None:
    for v in range(version, target):
        if v not in MIGRATIONS:
            raise RuntimeError(f"no migration from schema v{v} to v{v + 1}")
        MIGRATIONS[v](con, schema_sql)
