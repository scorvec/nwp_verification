# mverify

Verify weather model output against real observations, without downloading or cleaning any
observations yourself.

```
pip install "mverify[all]"
mverify score GFS="gfs/*.grib2" AIFS=aifs.zarr --region conus --out scores.json
```

mverify reads GRIB, netCDF and Zarr by content, recognises the variables, units, times, grid and
ensemble members on its own, interpolates every field to the stations that observed at its valid
time, and scores it. The observations come from a database of quality-controlled surface
stations (GHCNh) and radiosondes that is built and kept current centrally.

**Status: prototype.** See [DESIGN.md](DESIGN.md) for the architecture and the plan.

## What it scores today

| truth | variables | scores |
|---|---|---|
| surface stations (GHCNh) | 2 m temperature and dew point, sea-level pressure, 10 m wind (u, v, speed, vector) | bias, MAE, RMSE, error SD, correlation |
| radiosondes | height, temperature and wind at 1000-200 hPa | same, plus vector wind RMSE |
| ensembles | any of the above | fair CRPS, ensemble-mean RMSE, spread, spread/skill |

## Commands

```
mverify inspect FILES...                  what was recognised: variables, inits, leads, grid, members
mverify score [NAME=]FILES... [options]   verify; --out scorecard JSON, --pairs matched pairs (Parquet)
mverify db status                         rows, stations and time span per variable, QC flag rates
mverify db ghcnh --year 2026 --bbox conus build or update surface observations
mverify db soundings                      build or update radiosonde levels
mverify db export DIR                     write the Parquet snapshot users query
```

From Python:

```python
from mverify.verify import verify
result = verify({"GFS": ["gfs/*.grib2"]}, regions=["conus"])
result.table          # one row per model x network x variable x region x lead
result.scorecard()    # the JSON written by --out
```

## Development

```
pip install -e ".[dev]"
pytest
ruff check src tests
```
