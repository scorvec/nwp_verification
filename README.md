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

Source: [scorvec/nwp_verification](https://github.com/scorvec/nwp_verification). The Python package and command are named `mverify`.

**Status: prototype.** See [DESIGN.md](DESIGN.md) for the architecture and the plan.

## What it scores today

| truth | variables | scores |
|---|---|---|
| surface stations (GHCNh) | 2 m temperature and dew point, sea-level pressure, 10 m wind (u, v, speed, vector) | bias, MAE, RMSE, error SD, correlation |
| radiosondes | height, temperature and wind at 1000-100 hPa, specific humidity at 1000-300 hPa | same, plus vector wind RMSE |
| ensembles | any of the above | fair CRPS, ensemble-mean RMSE, spread, spread/skill |
| precipitation analyses (Stage IV over CONUS, GPCP globally) | 24 h precipitation | ETS, frequency bias, POD, FAR, FSS at 50-400 km, bias, RMSE |

Every score also has an observation-error-aware version (`rmse_oc`, `crps_oe`, `ssr_oe`). Radiosonde
levels are matched where the balloon drifted to. Surface stations more than 100 m from the model terrain
are left out, coastal stations are matched to land grid points when the files have a land-sea mask, and
winds from anemometers that are not at 10 m are corrected to 10 m. Several models are compared on
the observations they all have, and the scorecard says which differences are significant; with `--grid DEG`
every model is first averaged onto one regular grid, so models of different resolution are compared from
cells of the same size. See [DESIGN.md](DESIGN.md#making-observations-and-the-model-comparable).

## Commands

```
mverify inspect FILES...                  what was recognised: variables, inits, leads, grid, members
mverify score [NAME=]FILES... [options]   verify; --out scorecard JSON, --pairs matched pairs (Parquet)
mverify db status                         rows, stations and time span per variable, QC flag rates
mverify db ghcnh --year 2026 --bbox conus build or update surface observations
mverify db soundings                      build or update radiosonde levels (recent, scorvec.com mirror)
mverify db igra --since 2026-01-01        radiosondes from IGRA2 (NCEI); --archive por for the whole record
mverify db sensors FILE.csv               load thermometer / anemometer heights
mverify db precip --start 2026-10-01      add Stage IV and GPCP analyses (GPCP needs EARTHDATA_USERNAME/PASSWORD)
mverify db init                           create a database, or upgrade an older one
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
