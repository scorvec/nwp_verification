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

### Schema (`src/mverify/db/schema.sql`, version 2)

| table | one row per | notes |
|---|---|---|
| `obs` | station x valid hour x variable | `value` in canonical units, `qc` bitmask, `dt_min` observation time minus valid hour, `lat`/`lon` where it was observed when that is not the station (a drifting balloon), `source` |
| `stations` | station | network-qualified id (`GHCNH:USW00094846`, `RAOB:72520`), lat/lon/elevation, thermometer and anemometer heights, WMO/ICAO ids |
| `variables` | canonical variable | unit, kind, level, physical limits, observation error `sigma_o`; generated from `mverify.variables` |
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
* **Migrations rebuild, so an upgraded database equals a new one.** `ObsStore` upgrades an older
  file when it opens it writable (`mverify db init`), in one transaction. DuckDB cannot add a
  CHECK to an existing table, so a step that changes constraints renames the old table, creates
  the current one from `schema.sql` and copies rows by column name (`db/migrations.py`). The tests
  build a v1 file and assert that the upgraded columns and constraints match a fresh database
  exactly. Upgrading the 5.9M-row test database took 3 s.

### Sources

| network | source | coverage | status |
|---|---|---|---|
| Surface (`GHCNH`) | NOAA GHCNh on AWS (`noaa-ghcnh-pds`), NCEI's successor to ISD | ~15k active stations, METAR + SYNOP + others merged, updated daily (≈3-day lag) | implemented |
| Upper air (`RAOB`) | the scorvec.com sounding mirror (`skewt-data` branch: WIS2 BUFR, IEM, ECCC, TAC, IGRA) | ~700 stations, rolling recent window, 00/12Z plus intermediate | implemented |
| Upper air archive | IGRA2 (NCEI): this year's files (`data-y2d`, updated daily) or each station's whole record (`data-por`) | ~2,800 stations, 1900s to present, ~800 reporting | implemented (`ingest/igra.py`); writes the same rows as the mirror, so either fills the other's gaps |
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
to the standard levels 1000-100 hPa, interpolated in log-pressure only across small gaps;
specific humidity comes from the dew point (Bolton 1980) and only up to 300 hPa, above which
sonde humidity is too slow and dry-biased to verify a model with.

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
   upstream objects; an ascent is re-derived when `soundings.REDUCTION_VERSION` changes),
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
  2-D lat/lon (Lambert, polar stereographic, rotated, WRF). Static fields found in any file
  (orography, land-sea mask, surface roughness) are attached to every field on the same grid,
  so they can come from a separate file.
* **Ensembles**: member dimensions (`number`, `member`, `realization` ...). The members of one
  time step are read together, because ensemble files store them in one chunk; reading them one
  by one made a 15-day, 31-member run 3.6x slower.
* **Laziness**: one time step loads at a time, so a season of 0.25° output never sits in memory.

Verified on real data: one GFS 0.25° run read as GRIB2, as netCDF with CF names (`tas`, `psl`,
`uas`) and as Zarr with -180..180 longitudes and south-to-north latitudes gives identical scores
to two decimals at every lead.

## 3. Matching and scoring

* **Interpolation** (`interp.py`): bilinear on regular grids (periodic when global), 4-point
  inverse distance on curvilinear grids; weights are computed once per grid and station set.
  Stations outside a regional domain drop out.
* **Pairing** (`match.py`): exact valid hour against `obs`; 10 m wind speed derived from u/v
  when needed. The adjustments that make a point observation and a grid box comparable are
  in the next section.
* **Scores** (`scores.py`): n, bias, MAE, RMSE, error SD, correlation; vector wind RMSE; for
  ensembles fair CRPS (computed from sorted members, O(m log m) per case), ensemble-mean RMSE,
  spread and spread/skill; the observation-error-aware scores below; bootstrap CIs resampling
  whole valid times (stations at one time are correlated, so resampling stations would
  overstate confidence). Regions: global, NH, tropics, SH, CONUS, North America, Europe.
* **One sample for all models** (`verify.common_sample`): with several models, each is scored
  only on the observations every model was matched to at the same init and lead. Models lose
  different pairs (the terrain cut follows each model's orography, the land-only
  interpolation its land-sea mask), so without this their scores are not comparable.
  `--all-pairs` turns it off.
* **Significance** (`significance.py`): for every ordered pair of models, a paired t-test on
  the per-run mean squared errors, with the variance inflated for lag-1 autocorrelation of
  the run-to-run differences (Geer 2016), since runs a day apart share their weather. The
  p values go into the scorecard JSON as `p`, so a page can colour only differences that
  hold up.
* **Planned**: anomaly correlation against station climatologies (the `skewt-climo` medians
  your AIFS page already uses), diurnal-cycle and per-station bias maps.

### Making observations and the model comparable

Each pair keeps what was done to it (`dz_m`, `dt_min`, `sensor_h_m`, `land_frac`, `sigma_o`),
so scores can be stratified by any of them.

* **Balloon drift** (`ingest/soundings.py`). A sonde reaches 200 hPa about 40 minutes after
  launch and drifts tens of km, over 100 km in a jet. Each level is stored with where and when
  it was measured: per-level times when the ascent carries them (WIS2 BUFR and ECCC, about 80%
  of the mirror), otherwise a launch 45 minutes before the nominal hour and a 5 m/s ascent;
  per-level positions when they vary (GPS sondes), otherwise the reported winds integrated over
  time from the launch site. Matching interpolates the model to the drifted point. Over the
  4,708 mirrored ascents the median drift is 1.4 km at 850 hPa, 7.7 km at 500 hPa and 28 km at
  200 hPa (95th percentile 71 km). The time offset is stored but the model is taken at the
  nominal hour, as in most operational verification; interpolating between output times is a
  later step. On one GFS run, using the drifted position lowered the 250 hPa vector wind RMSE at
  42 h from 5.68 to 5.26 m/s; heights barely change because they vary slowly in space.
* **Terrain** (`match.py`). `dz_m` is model terrain minus station elevation. 2 m temperature is
  moved to the station height at 6.5 K/km, and every surface pair with |dz_m| > 100 m is
  dropped (`--max-dz`): there the station samples a valley or a ridge the model does not
  resolve, and no lapse rate fixes that. Surface pressure is compared with the station-level
  pressure as it is, with no reduction (sea-level pressure is a reduction below ground that every
  model and station makes its own way); it falls about 0.12 hPa per metre near sea level, so its
  pairs need |dz_m| <= 5 m (`MAX_DZ_PRESSURE`, at most 0.6 hPa, within the observation error),
  and a pair with unknown terrain is dropped.
* **Resolution** (`coarsen.py`, `--grid DEG`). Interpolated from its own grid, a finer model
  carries small-scale detail whose position errors count twice (the double penalty), and a model
  published on a grid finer than its own resolution looks sharper than it is. With a common grid,
  every field is averaged onto one regular DEG-degree grid by area overlap (first-order
  conservative, so 0.15 and 0.25 deg grids both average onto 0.5 deg exactly), and the model
  terrain, land fraction and roughness with it (roughness as the mean of its logarithm); a cell
  needs valid values over half its area. Matching then interpolates every model to the stations
  from cells of the same size. Precipitation is regridded again onto each analysis grid.
* **Coastlines**. With a land-sea mask in the files (GFS `LAND`, ECMWF `lsm`, CMIP `sftlf`),
  surface variables at stations are interpolated from land grid points only, so a coastal
  station is not handed the sea's temperature or wind, and a station with no land point among
  its neighbours is dropped. The model land fraction at the station is kept in `land_frac`.
* **Sensor heights**. Thermometers outside the WMO 1.25-2.5 m siting range are excluded
  (`--any-thermometer-height` keeps them); within it the height is kept for stratification.
  Winds from an anemometer not at 10 m are brought to 10 m with the neutral log profile
  u(10)/u(h) = ln(10/z0)/ln(h/z0), with z0 from the model's roughness field when the files carry
  it, else 0.03 m (WMO open-terrain reference). No automatic source of sensor heights is
  reachable (OSCAR/Surface is blocked here), so heights are loaded from a CSV with
  `mverify db sensors FILE` (columns: `station_id`, `wmo` or `icao`, plus `t_height_m` and/or
  `wind_height_m`); unknown heights are assumed standard.
* **Observation error**. `sigma_o` per variable (instrument plus representativeness error;
  t2m 1.4 K, td2m 1.6 K, surface pressure 0.5 hPa, MSLP 1.0 hPa, 10 m wind components 1.8 m/s; radiosonde temperature
  0.8-1.2 K, height 5 m at 1000 hPa to 14 m at 200 hPa, wind components 1.8-2.6 m/s) is an
  approximate starting point of the size global data assimilation assigns to these observation
  types, not a measured value. It should be replaced by estimates from the database itself
  (Desroziers et al. 2005 diagnostics against a reanalysis). The raw scores stay the headline
  numbers; next to them `rmse_oc` = sqrt(max(MSE - sigma_o^2, 0)) (vector wind removes the
  variance of both components), and for ensembles `crps_oe`, the CRPS of members perturbed with
  seeded N(0, sigma_o) noise (Saetra et al. 2004), `spread_oe` = sqrt(spread^2 + sigma_o^2) and
  `ssr_oe` = spread_oe / RMSE of the mean, which is 1 for a calibrated ensemble (a test draws one
  and checks this).

### Lead times and scale

Deterministic runs to day 10 and ensembles to day 15 run through the same code. Measured here:
a GFS 0.25° run 0-240 h every 12 h against the CONUS test database scored in 14 s using 0.9 GB.
A synthetic 31-member, 1° ensemble 0-360 h every 6 h (7,564 fields, 508k pairs) matched in 23 s
and scored in 4 s using 2.3 GB. GFS, IFS and AIFS, ten 0.25° runs each to day 10 against 2,200
stations worldwide and Stage IV: 3.1M pairs in 13 min at a 3.8 GB peak. `verify` holds the pairs in
memory as one long table. A sample too large for that, such as every pressure level of many runs or a
51-member ensemble on a global station set (about 10^8 pairs), is matched run by run into Parquet files
and scored with `score_pair_files`, which reads one variable group at a time (one pressure level, or
the 10 m wind components together): no score or significance test of one variable needs another's
pairs.

## 4. Precipitation analyses (`src/mverify/precip/`)

Precipitation is verified grid-to-grid, on the analysis grid, so every model is judged on the
same cells.

| product | coverage | window | source |
|---|---|---|---|
| `stageiv` | CONUS, ~4.8 km polar stereographic | 24 h ending 12Z | NCEP Stage IV (radar + gauges, RFC quality control), Iowa Environmental Mesonet archive, no login |
| `gpcp` | global, 0.5° | 24 h ending 00Z | GPCP V3.3 daily, NASA GES DISC, Earthdata login; the successor of the V3.2 climatology behind the scorvec.com rain bias correction; months behind real time |
| `imerg_late` | global, 0.1° | 24 h ending 00Z | GPM IMERG Late daily V07, GES DISC, Earthdata login; about 14 h behind real time, the satellite estimate GPCP V3 is built on |

* **Storage.** `mverify db precip --start ... --product stageiv gpcp` stores each window unchanged
  as a small Zarr store in `obs_grids/` next to the database and catalogues it in `grids`. The
  snapshot carries `grids.parquet` and the stores, so users still download nothing.
* **Model totals** (`accum.py`). Files hold running totals (ECMWF, ERA5, the GFS 0-N h field),
  buckets (GFS 6 h, HRRR 1 h) or both. The window is built directly, as the difference of two
  totals with the same start, or by chaining contiguous buckets; differences are clipped at 0.
  GRIB files with two accumulations of one field (GFS: 0-30 h and 24-30 h) are opened one
  `startStep` at a time, because cfgrib otherwise keeps one and silently drops the other. In
  netCDF/Zarr an accumulation is taken as running from the init, the common convention.
* **Regridding** (`regrid.py`). Coarsening (0.25° model onto 0.5° GPCP) averages every model
  cell inside an analysis cell, area-weighted, which conserves water; refining (GFS onto Stage IV)
  interpolates bilinearly. The choice follows the two grid spacings.
* **Scores** (`scores.py`). ETS, frequency bias, POD and FAR at 1, 5, 10, 25 and 50 mm/24 h;
  FSS at 50, 100, 200 and 400 km; mean forecast and analysed amounts, bias, MAE, RMSE. All of
  them are built from sums that add across windows and inits, so a season is scored exactly
  instead of as an average of daily scores. Cells are area-weighted, and cells the analysis
  does not cover are left out, including from the neighbourhood fractions. On the lat/lon GPCP
  grid a neighbourhood is n x n cells with n from the latitude spacing, so it narrows in
  longitude towards the poles. An ensemble is scored through its member mean for now.
* **Results** join the station table: `network` is the product, `variable` is `tp24` for
  amounts and `tp24>=10mm` for a threshold, and `lead_h` is the lead at the end of the window.
  On the GFS run of 2026-10-13 00Z against Stage IV for the 24 h ending 14 Oct 12Z (36 h lead):
  ETS 0.41 / 0.38 / 0.29 / 0.02 at 1 / 10 / 25 / 50 mm, frequency bias 1.33 at 1 mm and 0.13 at
  50 mm (too much light rain, too little heavy rain), FSS at 10 mm from 0.69 (50 km) to 0.91
  (400 km).
* **Maps** (`maps.py`, `report.precip_maps`). On a forecast already on a coarse common grid
  (`coarsen`), each analysis window is averaged onto that grid where it covers at least half a
  cell, and every run's total over the window is compared with it cell by cell. Windows are pooled
  into lead groups (days 1-2, 3-5, 6-9) for each cell's bias, RMSE and mean forecast and analysed
  amounts. A cell's bias is tested over the valid days (the day's error averaged over the runs that
  verify it), with the paired test and day-to-day allowance of `significance.py`, once it has at
  least 8 days; the p values of all cells of a map go through Benjamini-Hochberg at a false-discovery
  rate of 10 % (`significance.fdr`), so a map shades only the cells whose bias is unlikely to be
  noise.
* **Latency, and why there are two global products.** Stage IV is final within about a day.
  GPCP V3.3 daily currently ends at 2025-12-31, so it scores the archive; IMERG Late scores the
  recent months. They are kept as separate truths (`GPCP`, `IMERG_LATE` rows) rather than spliced
  into one series, because a satellite-only estimate and a gauge-adjusted one differ in bias and
  a splice would show up as a jump in the scores.
* **Checking the providers.** The `analyses` workflow fetches one window of each product weekly
  and on changes to `precip/`, and fails when a file no longer reads as a plausible field. GPCP
  and IMERG need `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` as repository secrets; without them
  those two are skipped. Their readers follow the documented file layouts and are unit-tested on
  synthetic files, but have not yet read a real file (this build machine has no Earthdata login).

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
consumer, like it is of `quantmet`. The page is `verification.html` on the site.

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
  db/migrations.py    upgrades of older databases to the current schema
  db/store.py         ObsStore: validated, atomic writes; queries; integrity audit; snapshots
  db/qc.py            quality-control checks and flag definitions
  ingest/ghcnh.py     GHCNh hourly surface observations
  ingest/soundings.py radiosonde mandatory levels
  ingest/igra.py      IGRA2 radiosonde archive, reduced the same way
  io/open.py          format detection and normalisation of model output
  interp.py           grid-to-point interpolation
  match.py            forecast/observation pairing
  scores.py           verification scores
  report.py           scorecard JSON and tables
  precip/             precipitation analyses, model totals, regridding and grid scores
  verify.py           one-call API
  cli.py              command line
tests/                pytest suite (tests/data: the v1 schema, for the migration test)
examples/             helper scripts (GFS subset download)
```
