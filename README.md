# mverify

Verify weather model output against quality-controlled observations. Point it at forecast files (GRIB, netCDF
or Zarr): it recognises the variables, units, times, grid and ensemble members, interpolates each field to the
stations that observed at its valid time, and scores it.

**Status: prototype.** Architecture and plan: [DESIGN.md](DESIGN.md).

## Install

```bash
pip install "mverify[all] @ git+https://github.com/scorvec/nwp_verification"
```

## Use

```bash
# 1. Build the observation database once (default $MVERIFY_DB, else ~/.cache/mverify/obs.duckdb)
mverify db ghcnh --year 2026 --bbox conus      # surface stations (GHCNh)
mverify db soundings                           # recent radiosondes
mverify db precip --start 2026-10-01           # Stage IV and GPCP analyses (GPCP needs EARTHDATA_USERNAME/PASSWORD)

# 2. Check what was recognised in the files, then score
mverify inspect "gfs/*.grib2"
mverify score GFS="gfs/*.grib2" AIFS=aifs.zarr --region conus --out scores.json
```

`web/scorecard.html` draws a `scores.json` placed next to it. From Python:

```python
from mverify.verify import verify

result = verify({"GFS": ["gfs/*.grib2"]}, regions=["conus"])
result.table          # one row per model x network x variable x region x lead
result.scorecard()    # the JSON that --out writes
```

## What it scores

| Truth | Variables | Scores |
|---|---|---|
| Surface stations (GHCNh) | 2 m temperature and dew point, station and sea-level pressure, 10 m wind, 24 h gauge precipitation | bias, MAE, RMSE, error SD, correlation; wind-speed distribution |
| Radiosondes | height, temperature and wind at 1000–100 hPa, specific humidity at 1000–300 hPa | the same, plus vector wind RMSE |
| Ensembles | any of the above | fair CRPS, ensemble-mean RMSE, spread, spread/skill, paired tests between ensembles |
| Precipitation analyses (Stage IV, GPCP) | 24 h precipitation | ETS, frequency bias, POD, FAR, FSS at 50–400 km, bias and RMSE maps |

Models named together are compared on the observations they all have, and the scorecard marks which
differences are significant. `--grid DEG` first averages every model onto one regular grid. Each score also has
a version that allows for observation error (`rmse_oc`, `crps_oe`, `ssr_oe`). How stations and model grid points
are made comparable: [DESIGN.md](DESIGN.md#making-observations-and-the-model-comparable).

## Commands

```
mverify inspect FILES...                 variables, inits, leads, grid and members recognised
mverify score [NAME=]FILES... [options]  verify; --out scorecard JSON, --pairs matched pairs (Parquet),
                                         --snapshot DIR_OR_URL to read a published snapshot instead of --db
mverify db init | status                 create or upgrade a database; rows, stations, time span, QC rates
mverify db ghcnh --year Y [--bbox B]     surface observations (--since/--until limit the window)
mverify db soundings                     recent radiosonde levels from the scorvec.com mirror
mverify db igra --since DATE             radiosondes from IGRA2 (--archive por for the whole record)
mverify db precip --start DATE           precipitation analyses
mverify db sensors FILE.csv              thermometer and anemometer heights
mverify db export DIR                    write the Parquet snapshot
```

`mverify <command> --help` lists every option.

## Develop

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```
