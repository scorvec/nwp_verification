-- mverify observation database, schema version 2 (DuckDB). Upgrades from older versions are in
-- mverify/db/migrations/; ObsStore applies them in order when it opens a database.
--
-- Design rules
--   * One long table, obs, holds every cleaned observation: surface reports and
--     radiosonde mandatory levels alike, one row per station x valid hour x
--     variable. Long beats wide because matching is generic over variables and a
--     new variable is a row in `variables`, not a migration; columnar storage
--     dictionary-encodes the repeated keys, so the cost is small.
--   * The database describes itself: variables (units, physical limits) and QC
--     flags are reference tables.
--   * obs carries CHECK constraints but no PRIMARY KEY or FOREIGN KEYs, on
--     purpose. In DuckDB both are backed by ART indexes, which on this table made
--     the file 4-6x larger and bulk loads 2-3x slower (measured on GHCNh
--     station-years). Uniqueness of (station_id, valid_time, variable) and the
--     references to stations and variables are instead guaranteed by the only
--     write path, mverify.db.store.ObsStore (batch validation, and each upstream
--     object replacing exactly the rows it owns), and audited by
--     ObsStore.check_integrity(), which the build runs before every publish.
--   * Nothing is deleted by QC. A failed check sets a bit in obs.qc; scoring
--     reads qc = 0 by default, and flag rates stay auditable.
--   * Every row is traceable: obs.source names the upstream product and report
--     type, ingest_log records which upstream object (and which version of it)
--     each batch came from.
--   * All times are UTC TIMESTAMP (no time zone). Longitudes are -180..180.

CREATE TABLE IF NOT EXISTS meta (
    key          VARCHAR PRIMARY KEY,
    value        VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS variables (
    name         VARCHAR PRIMARY KEY,         -- canonical name: t2m, msl, z500, t850 ...
    label        VARCHAR NOT NULL,
    unit         VARCHAR NOT NULL,            -- canonical unit every value is stored in
    kind         VARCHAR NOT NULL CHECK (kind IN ('surface', 'upper')),
    level_hpa    DOUBLE,                      -- pressure level for kind = 'upper'
    lo           DOUBLE NOT NULL,             -- physical limits used by the range check
    hi           DOUBLE NOT NULL,
    sigma_o      DOUBLE CHECK (sigma_o > 0),  -- observation error SD (instrument + representativeness), unit
    CHECK (lo < hi),
    CHECK ((kind = 'upper') = (level_hpa IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS qc_flags (
    bit          USMALLINT PRIMARY KEY,       -- value of the bit in obs.qc (1, 2, 4, ...)
    name         VARCHAR NOT NULL UNIQUE,
    description  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS stations (
    station_id   VARCHAR PRIMARY KEY,         -- network-qualified: 'GHCNH:USW00094846', 'RAOB:72520'
    network      VARCHAR NOT NULL,            -- 'GHCNH', 'RAOB', ...
    name         VARCHAR,
    lat          DOUBLE NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lon          DOUBLE NOT NULL CHECK (lon >= -180 AND lon <= 180),
    elev_m       DOUBLE CHECK (elev_m BETWEEN -500 AND 9000),
    t_height_m   DOUBLE CHECK (t_height_m BETWEEN 0.5 AND 30),     -- thermometer above ground; NULL = unknown
    wind_height_m DOUBLE CHECK (wind_height_m BETWEEN 1 AND 100),  -- anemometer above ground; NULL = unknown
    wmo_id       VARCHAR,
    icao         VARCHAR,
    country      VARCHAR,
    updated_at   TIMESTAMP NOT NULL DEFAULT current_timestamp,
    CHECK (starts_with(station_id, network || ':'))
);

CREATE TABLE IF NOT EXISTS obs (
    station_id   VARCHAR NOT NULL,            -- stations.station_id
    valid_time   TIMESTAMP NOT NULL,          -- the whole hour the report represents
    variable     VARCHAR NOT NULL,            -- variables.name
    value        DOUBLE NOT NULL CHECK (isfinite(value)),
    qc           USMALLINT NOT NULL DEFAULT 0,  -- bitwise OR of qc_flags.bit; 0 = clean
    dt_min       SMALLINT NOT NULL DEFAULT 0 CHECK (dt_min BETWEEN -180 AND 180),  -- observation time - valid_time
    lat          DOUBLE CHECK (lat BETWEEN -90 AND 90),             -- where it was observed, when that is not the
    lon          DOUBLE CHECK (lon >= -180 AND lon <= 180),         -- station (a drifting balloon); else NULL
    source       VARCHAR NOT NULL,            -- upstream product and report type, e.g. 'ghcnh:FM15'
    CHECK ((lat IS NULL) = (lon IS NULL)),
    -- logical key: (station_id, valid_time, variable), see the note at the top
    CHECK (date_part('minute', valid_time) = 0 AND date_part('second', valid_time) = 0)
);

-- One row per upstream object (a GHCNh station-year file, one radiosonde ascent),
-- with the version that was ingested: an update re-fetches only objects whose
-- version changed, and any obs row can be traced to the object it came from.
CREATE TABLE IF NOT EXISTS ingest_log (
    source       VARCHAR NOT NULL,            -- 'ghcnh', 'raob:scorvec-skewt', ...
    object_key   VARCHAR NOT NULL,            -- e.g. 'USW00094846/2026'
    version      VARCHAR NOT NULL,            -- upstream ETag / Last-Modified / content hash
    rows_in      BIGINT NOT NULL CHECK (rows_in >= 0),
    rows_kept    BIGINT NOT NULL CHECK (rows_kept >= 0),
    ingested_at  TIMESTAMP NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (source, object_key)
);

-- Gridded truth (precipitation analyses: Stage IV, MRMS, CCPA, IMERG) is not
-- stored row by row; the arrays live in Zarr and this table catalogues them.
CREATE TABLE IF NOT EXISTS grids (
    product      VARCHAR NOT NULL,            -- 'stageiv', 'mrms_qpe', 'imerg_final', ...
    variable     VARCHAR NOT NULL,            -- 'tp' (accumulated precipitation, mm)
    accum_hours  SMALLINT NOT NULL CHECK (accum_hours > 0),
    valid_time   TIMESTAMP NOT NULL,          -- end of the accumulation window
    uri          VARCHAR NOT NULL,            -- zarr store and group
    PRIMARY KEY (product, variable, accum_hours, valid_time)
);
