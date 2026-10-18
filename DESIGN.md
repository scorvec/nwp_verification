# mverify design

Working name: `mverify` (to be settled with the repository name).

## Goal

Point the tool at model output in any common format and get verification scores back, with no
observation downloading or preprocessing on the user's side:

```
mverify score GFS=gfs/*.grib2 AIFS=aifs.zarr --region conus --out scores.json
```

Three properties make that possible, and each is a separate part of the package:

1. **A ready-made observation database.** Cleaned, quality-controlled surface and upper-air
   observations, built and kept current centrally, published as a snapshot users query in
   place. This is the core of the project and gets the strictest engineering.
2. **Format-agnostic ingestion.** GRIB 1/2, netCDF 3/4 and Zarr are recognised by content, and
   every field is mapped to one canonical variable, unit, time convention and grid description.
3. **Automatic matching and scoring.** Fields are interpolated to the stations that observed at
   their valid times, paired, and scored; results come out as a table and as scorecard JSON a
   web page can draw directly.

```
           upstream feeds                         central build (GitHub Actions, daily)
 GHCNh hourly (NOAA AWS) ─┐
 radiosondes (WIS2/IEM/   ├─► ingest ─► QC ─► DuckDB build db ─► integrity audit ─► Parquet snapshot
   ECCC/IGRA mirror)      │    (per upstream object, atomic)                          (object store / CDN)
 precip analyses (Zarr) ──┘                                                                 │
                                                                                            ▼
 user's GRIB / netCDF / Zarr ─► open_forecast ─► match (interp + time/level) ─► scores ─► table / JSON
                                                                                            │
                                                         scorvec.com verification page ◄────┘
```

## 1. Observation database

### Storage engine

DuckDB. It is SQL, needs no server, and the same queries run against a local `.duckdb` file
during the build and against a published set of Parquet files at use time (`ObsStore.snapshot`),
including over HTTPS, where hive partitioning plus row-group statistics mean a query for one
month of 2 m temperature downloads only that month's 2 m temperature. That is what lets users
"already have" the database without downloading all of it. PostgreSQL/PostGIS stays a possible
mirror (the schema is plain SQL apart from a few DuckDB types), but a server is not required for
anyone.

### Schema (`src/mverify/db/schema.sql`, version 1)

| table | one row per | notes |
|---|---|---|
| `obs` | station x valid hour x variable | `value` in canonical units, `qc` bitmask, `dt_min` report offset, `source` |
| `stations` | station | network-qualified id (`GHCNH:USW00094846`, `RAOB:72520`), lat/lon/elevation, WMO/ICAO ids |
| `variables` | canonical variable | unit, kind, level, physical limits; generated from `mverify.variables` |
| `qc_flags` | QC check | bit, name, description; the database documents its own flags |
| `ingest_log` | upstream object | source, key, upstream version (ETag), rows in/kept, time |
| `grids` | gridded truth field | catalogue of precipitation analyses stored as Zarr |
| `meta` | setting | `schema_version` |

Decisions and why:

* **Long table.** Matching is generic over variables, a new variable is a row in `variables`
  rather than a migration, and radiosonde levels fit the same table as surface reports.
  Columnar storage dictionary-encodes the repeated keys, so the cost is small.
* **Nothing is deleted by QC.** A failed check sets a bit; scoring reads `qc = 0` by default.
  Flag rates stay auditable (`mverify db status` prints them) and a check can be retuned
  without re-downloading anything.
* **Constraints where they are cheap, guarantees where they are not.** `obs` has CHECK
  constraints (finite values, whole-hour times, offsets, ranges in `stations`) but no primary
  or foreign keys. In DuckDB those are ART indexes, which on GHCNh station-years made the file
  4-6x larger and loads 2-3x slower. Uniqueness and references are instead guaranteed by the
  single write path (`ObsStore`) and verified by `ObsStore.check_integrity()`, which the build
  runs before every publish and the tests exercise by writing bad rows behind the store's back.
* **Every write is validated and atomic.** `validate_obs` / `validate_stations` reject a batch
  with a message naming the rule and the first offending rows. `ingest_batch` writes an upstream
  object's stations, observations and `ingest_log` row in one transaction: an object is fully
  ingested and logged, or not at all, so a crashed build resumes cleanly.
* **Upstream objects own their rows.** A GHCNh station-year file *replaces* that station's year
  (`replace_obs`), so values NCEI withdraws or re-times disappear too; immutable objects such as
  radiosonde ascents are upserted by key. `ingest_log.version` holds the upstream ETag, so a
  daily update re-fetches only files that changed.
* **Times are UTC, longitudes -180..180, units canonical**, and the store enforces all three.

### Sources

| network | source | coverage | status |
|---|---|---|---|
| Surface (`GHCNH`) | NOAA GHCNh on AWS (`noaa-ghcnh-pds`), NCEI's successor to ISD | ~15k active stations, METAR + SYNOP + others merged, updated daily (≈3-day lag) | implemented |
| Upper air (`RAOB`) | the scorvec.com sounding mirror (`skewt-data` branch: WIS2 BUFR, IEM, ECCC, TAC, IGRA) | ~700 stations, rolling recent window, 00/12Z plus intermediate | implemented |
| Upper air archive | IGRA2 (NCEI) | 1900s to present | planned; NCEI was not reachable from this build machine, the parser plugs into the same ingest |
| Precipitation | Stage IV, MRMS QPE (AWS), CCPA, IMERG | gridded | planned (section 4) |
| Surface, real time | METAR/SYNOP (IEM, WIS2) for the 3-day GHCNh lag | | planned |
| Daily max/min/precip | GHCN-Daily (AWS) | ~100k stations | planned |

### Cleaning and QC (`src/mverify/db/qc.py`)

| bit | check | applies to |
|---|---|---|
| 1 `upstream` | provider's own QC (GHCNh letter codes a-z, legacy 2/3/6/7) | surface |
| 2 `range` | physical limits from `variables` | all |
| 4 `spike` | departs from both hourly neighbours in the same direction beyond a per-variable limit | surface |
| 8 `stuck` | identical value for longer than plausible | surface |
| 16 `inconsistent` | dew point above temperature | surface |
| 32 `vertical` | layer thickness disagrees with the hypsometric thickness; isolates the bad level | radiosonde |

Station positions are part of QC: radiosonde stations are checked against an independent list
(the GHCNh station list's WMO indexes) and left out when the two disagree by more than 1°.
On the first real build this caught five stations with broken metadata in the scorvec.com
mirror, for example Sovetskaya Gavan (31770) at latitude 0.0 and Kavali (43243) at longitude
0.01; before the check, Sovetskaya Gavan alone raised the global 500 hPa height RMSE of the GFS
analysis from 15 m to 20 m.

GHCNh reports are reduced to one value per station x variable x whole hour (nearest report
within ±30 min, routine METAR preferred over SYNOP over SPECI on ties). Radiosondes are reduced
to the mandatory levels 1000-200 hPa, interpolated in log-pressure only across small gaps.

Planned, in priority order: a background check (observation minus a short-range analysis,
rejecting only departures far beyond normal model error; the remaining gross radiosonde
errors, such as 500 hPa heights 125 m too high at Tozeur, are this kind), buddy check against neighbouring stations, station-history table for moves, a
per-station climatological outlier check once a few years are loaded, and wind/pressure
consistency checks.

### Size and hosting

Measured on the first build: 148 US stations for 2026 to mid-October (5.7M rows) loaded in
60 s, and 4,700 recent radiosonde ascents from ~700 stations (136k rows); the integrity audit
came back clean. Scaling to ~15k
stations and 8 variables gives about 1 billion rows a year; at Parquet zstd densities that is
on the order of a few GB per year of surface data, and well under 100 MB per year of radiosonde
levels. The snapshot belongs on an object store with a CDN (the S3-compatible store behind
`frames.scorvec.com` already exists) or as GitHub Release assets per year; the repository itself
holds code only.

### Keeping it current

A scheduled GitHub Actions workflow in the package repository:

1. restores the build database from the previous snapshot,
2. runs `mverify db ghcnh --year <this year>` and `mverify db soundings` (both skip unchanged
   upstream objects),
3. runs `check_integrity()` and the test suite; any non-zero count fails the job and nothing
   is published,
4. exports the snapshot and uploads only changed partitions (current month for most days).

## 2. Ingesting model output (`src/mverify/io/open.py`)

* **Format by content**: GRIB magic, CDF/HDF5 signatures, Zarr metadata files. Extensions are
  ignored, so `gfs.t00z.pgrb2.0p25.f024` needs no renaming.
* **Variables**: GRIB shortName + level type; CF `standard_name`; aliases from WRF, CMIP, ERA5,
  ECMWF open data and wgrib2-converted names; names with embedded levels (`TMP_850mb`, `z500`).
* **Units**: from `units`, falling back to magnitude (Kelvin, Pa, geopotential in m² s⁻²).
* **Times**: cfgrib time/step; netCDF/Zarr init+lead, init+valid, or valid alone with the init
  from attributes (WRF `START_DATE` etc.) or the file name; `--init` overrides.
* **Grids**: regular lat/lon in either longitude convention and latitude order, or curvilinear
  2-D lat/lon (Lambert, polar stereographic, rotated, WRF). Orography found in any file is
  attached to every field on the same grid.
* **Ensembles**: member dimensions (`number`, `member`, `realization` ...).
* **Laziness**: fields load one at a time, so a season of 0.25° output never sits in memory.

Verified on real data: one GFS 0.25° run read as GRIB2, as netCDF with CF names (`tas`, `psl`,
`uas`) and as Zarr with -180..180 longitudes and south-to-north latitudes gives identical scores
to two decimals at every lead.

## 3. Matching and scoring

* **Interpolation** (`interp.py`): bilinear on regular grids (periodic when global), 4-point
  inverse distance on curvilinear grids; weights are computed once per grid and station set.
  Stations outside a regional domain drop out.
* **Pairing** (`match.py`): exact valid hour against `obs`; 2 m temperature moved from model
  terrain to station height at 6.5 K/km when orography is known (`dz_m` is kept per pair);
  10 m wind speed derived from u/v when needed.
* **Scores** (`scores.py`): n, bias, MAE, RMSE, error SD, correlation; vector wind RMSE; for
  ensembles fair CRPS, ensemble-mean RMSE, spread and spread/skill; bootstrap CIs resampling
  whole valid times (stations at one time are correlated, so resampling stations would
  overstate confidence). Regions: global, NH, tropics, SH, CONUS, North America, Europe.
* **Planned**: anomaly correlation against station climatologies (the `skewt-climo` medians
  your AIFS page already uses), categorical scores for precipitation, significance tests for
  model-vs-model differences (paired, with the same block bootstrap), diurnal-cycle and
  per-station bias maps.

## 4. Precipitation analyses (planned)

Precipitation is verified grid-to-grid: analyses (Stage IV 6/24 h, MRMS QPE 1-24 h, CCPA, IMERG)
are stored as Zarr on their native grids and catalogued in `grids`; model accumulations are
derived from the file (running totals differenced, rates integrated) and regridded
conservatively to the analysis grid. Scores: FSS over neighbourhood sizes, ETS / frequency bias /
POD / FAR at fixed thresholds, and spatial bias. Station precipitation from GHCNh is used only
where its accumulation period is unambiguous.

## 5. Feeding scorvec.com

`mverify score ... --out scores.json` writes a scorecard shaped for direct drawing: one row per
truth x variable x region, each with a score array per model aligned to a shared `leads` list,
in the same spirit as `assets/verify/aifs_scorecard.json`.

Recommended wiring, matching how the site's other products are built:

* the site gets a workflow (like `aifs-verify`) that installs the pinned package release, runs
  `mverify score` for the models it already downloads (GFS, AIFS, IFS, HRRR, RRFS ...) against
  the published snapshot, and commits the JSON under `assets/verify/` with the usual
  snapshot-and-resync step;
* a verification page fetches that JSON and draws lead-time curves and a scorecard grid.

This keeps the package independent of the site (anyone can run it) while the site stays a
consumer, like it is of `quantmet`. Nothing in `scorvec.github.io` has been changed.

## 6. Code quality

* Typed, documented modules with one responsibility each; no hidden global state (the
  database path resolves from an argument, `$MVERIFY_DB`, then a default).
* Tests (`pytest`, 63 so far) cover schema creation and versioning, every validation rule,
  atomicity, replace semantics, the integrity audit, snapshot round trip, each QC check, GHCNh
  and radiosonde parsing, every format path including real GRIB written with ecCodes,
  interpolation exactness, scores with known answers, and the CLI end to end. A perfect model
  must score exactly zero; a model warmed by 1.5 K must score a bias of exactly 1.5 K from both
  netCDF and Zarr.
* `ruff` clean; CI should run lint, tests, and a small real-data smoke test.

## Repository layout

```
src/mverify/
  variables.py        canonical variable catalogue and unit conversion
  db/schema.sql       the database schema
  db/store.py         ObsStore: validated, atomic writes; queries; integrity audit; snapshots
  db/qc.py            quality-control checks and flag definitions
  ingest/ghcnh.py     GHCNh hourly surface observations
  ingest/soundings.py radiosonde mandatory levels
  io/open.py          format detection and normalisation of model output
  interp.py           grid-to-point interpolation
  match.py            forecast/observation pairing
  scores.py           verification scores
  report.py           scorecard JSON and tables
  verify.py           one-call API
  cli.py              command line
tests/                pytest suite
examples/             helper scripts (GFS subset download)
```
