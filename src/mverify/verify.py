"""One-call verification: forecast files in, score table (and scorecard) out."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from mverify import scores, significance
from mverify import variables as V
from mverify.coarsen import coarsen
from mverify.db.store import ObsStore
from mverify.io.open import Forecast, open_forecast
from mverify.match import MAX_DZ, T_HEIGHT_RANGE, match
from mverify.precip.verify import verify_precip
from mverify.report import scorecard


@dataclass
class Result:
    pairs: pd.DataFrame
    table: pd.DataFrame
    inits: dict[str, list[str]] = field(default_factory=dict)
    tests: pd.DataFrame = field(default_factory=pd.DataFrame)  # model-pair p values (mverify.significance)
    stations: pd.DataFrame = field(default_factory=pd.DataFrame)  # scores per station (scores.by_station)

    def scorecard(self) -> dict:
        return scorecard(self.table, self.inits, self.tests)

    @classmethod
    def concat(cls, results: Sequence[Result]) -> Result:
        """One result from several scored on different variables (surface and upper air, say)."""
        inits: dict[str, set[str]] = {}
        for r in results:
            for m, i in r.inits.items():
                inits.setdefault(m, set()).update(i)
        return cls(_cat([r.pairs for r in results]), _cat([r.table for r in results]),
                   {m: sorted(i) for m, i in inits.items()}, _cat([r.tests for r in results]),
                   _cat([r.stations for r in results]))


def _cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def verify(forecasts: dict[str, list] | list | str, store: ObsStore | None = None,
           regions: list[str] | None = None, variables: list[str] | None = None,
           networks: list[str] | None = None, height_correction: bool = True, init=None,
           max_dz: float | None = MAX_DZ, t_height_range: tuple[float, float] | None = T_HEIGHT_RANGE,
           common: bool = True, grid: float | None = None) -> Result:
    """Score one or more models.

    ``forecasts`` is a path / glob / list of paths (one model, named from the
    file names), or ``{"GFS": [...], "IFS": [...]}`` for several. ``regions``
    defaults to every region in mverify.scores.REGIONS that holds pairs. ``max_dz`` and
    ``t_height_range`` are passed to mverify.match.match (None keeps every pair). With several
    models, ``common`` scores each only on the pairs every model has (see ``common_sample``).
    ``grid`` (degrees) first averages every model onto that one regular grid (mverify.coarsen), so
    models of different resolution are interpolated to the stations from cells of the same size.
    """
    own = store is None
    store = store or ObsStore.open(read_only=True)
    try:
        if isinstance(forecasts, dict):
            fcs = [open_forecast(p, name=n, init=init) for n, p in forecasts.items()]
        else:
            fcs = [open_forecast(forecasts, init=init)]
        if grid is not None:
            fcs = [coarsen(fc, grid) for fc in fcs]
        pairs = pd.concat([match(fc, store, variables, networks=networks, height_correction=height_correction,
                                 max_dz=max_dz, t_height_range=t_height_range)
                           for fc in fcs], ignore_index=True)
        precip = [verify_precip(fc, store, regions=regions) for fc in fcs
                  if not variables or any(v.startswith("tp") for v in variables)]
    finally:
        if own:
            store.close()
    if common and len(fcs) > 1:
        pairs = common_sample(pairs)
    result = score_pairs(pairs, regions, fcs)
    precip = [t for t in precip if not t.empty]
    if precip:
        result.table = pd.concat([result.table, *precip], ignore_index=True)
    return result


SAMPLE_KEY = ["network", "station_id", "variable", "init", "lead_h", "valid_time"]


def common_sample(pairs: pd.DataFrame) -> pd.DataFrame:
    """Only the observations every model was matched to, at the same init and lead.

    Models lose different pairs: the terrain cut depends on each model's orography, the
    land-only interpolation on its land-sea mask, and a model may lack a lead or a run.
    Comparing their scores is fair only on one sample, as in the WMO lead centre's practice.
    """
    if pairs.empty or pairs["model"].nunique() < 2:
        return pairs
    n = pairs.groupby(SAMPLE_KEY, dropna=False, sort=False)["model"].transform("nunique")
    return pairs[n == pairs["model"].nunique()].reset_index(drop=True)


def variable_group(name: str) -> str:
    """Variables that are scored together: one pressure level (z500, t500, u500, v500, q500), the
    10 m wind components, or a surface variable on its own."""
    if name in ("u10", "v10", "ws10"):
        return "wind10"
    var = V.CATALOG.get(name)
    return re.sub(r"^[a-z]+", "", name) if var is not None and var.kind == "upper" else name


def score_pair_files(paths: Sequence[str | Path], regions: list[str] | None = None, common: bool = True) -> Result:
    """Score pairs saved as Parquet (``match`` output, e.g. one file per run), one variable group at a time.

    This is how a sample too large to hold at once is scored, such as every pressure level of
    many runs: scores and significance tests of one variable never need another's pairs, except
    the wind components, which ``variable_group`` keeps together. The returned ``pairs`` is empty.
    """
    data = ds.dataset([str(p) for p in paths], format="parquet")
    names = data.to_table(columns=["variable"]).column("variable").unique().to_pylist()
    groups: dict[str, list[str]] = {}
    for n in sorted(names):
        groups.setdefault(variable_group(n), []).append(n)
    results = []
    for members in groups.values():
        pairs = data.to_table(filter=ds.field("variable").isin(members)).to_pandas()
        if common:
            pairs = common_sample(pairs)
        r = score_pairs(pairs, regions)
        r.inits = {m: sorted(g.dt.strftime("%Y-%m-%dT%H").unique()) for m, g in pairs.groupby("model")["init"]}
        r.pairs = pairs.iloc[:0]
        results.append(r)
    return Result.concat(results)


def score_pairs(pairs: pd.DataFrame, regions: list[str] | None = None, fcs: list[Forecast] = ()) -> Result:
    inits = {fc.name: sorted({f.init.strftime("%Y-%m-%dT%H") for f in fc.fields}) for fc in fcs}
    if pairs.empty:
        return Result(pairs, pd.DataFrame(), inits)
    by = ["model", "network", "variable", "region", "lead_h"]
    tables, tests = [], []
    # One region at a time: a pair belongs to several (global, NH, CONUS, North America), and
    # holding every copy at once multiplies the memory of a long or multi-model run.
    for region in regions or list(scores.REGIONS):
        rp = scores.assign_regions(pairs, [region])
        if rp.empty:
            continue
        tables.append(_score_region(rp, by))
        tests.append(significance.tests(rp))
    if not tables:
        return Result(pairs, pd.DataFrame(), inits)
    table = pd.concat(tables, ignore_index=True)
    return Result(pairs, table.sort_values(by).reset_index(drop=True), inits, pd.concat(tests, ignore_index=True),
                  scores.by_station(pairs))


def _score_region(rp: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    det_pairs = rp
    if rp["member"].notna().any():  # deterministic scores of an ensemble are those of its mean
        keys = ["model", "network", "variable", "region", "lead_h", "init", "valid_time", "station_id"]
        det_pairs = rp.groupby(keys, dropna=False).agg(fcst=("fcst", "mean"), obs=("obs", "first"),
                                                     sigma_o=("sigma_o", "first")).reset_index()
    tables = [scores.deterministic(det_pairs, by), scores.vector_wind(rp.assign(member=rp["member"].fillna(-1)), by)]
    if rp["member"].notna().any():
        tables.append(scores.ensemble(rp, by))
    table = tables[0]
    for t in tables[1:]:
        if not t.empty:
            table = table.merge(t, on=by, how="outer", suffixes=("", "_dup"))
            for c in [c for c in table.columns if c.endswith("_dup")]:
                base = c[:-4]
                table[base] = table[base].fillna(table[c])
                table = table.drop(columns=c)
    return table
