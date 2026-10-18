"""One-call verification: forecast files in, score table (and scorecard) out."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mverify import scores
from mverify.db.store import ObsStore
from mverify.io.open import Forecast, open_forecast
from mverify.match import MAX_DZ, T_HEIGHT_RANGE, match
from mverify.report import scorecard


@dataclass
class Result:
    pairs: pd.DataFrame
    table: pd.DataFrame
    inits: dict[str, list[str]] = field(default_factory=dict)

    def scorecard(self) -> dict:
        return scorecard(self.table, self.inits)


def verify(forecasts: dict[str, list] | list | str, store: ObsStore | None = None,
           regions: list[str] | None = None, variables: list[str] | None = None,
           networks: list[str] | None = None, height_correction: bool = True, init=None,
           max_dz: float | None = MAX_DZ, t_height_range: tuple[float, float] | None = T_HEIGHT_RANGE) -> Result:
    """Score one or more models.

    ``forecasts`` is a path / glob / list of paths (one model, named from the
    file names), or ``{"GFS": [...], "IFS": [...]}`` for several. ``regions``
    defaults to every region in mverify.scores.REGIONS that holds pairs. ``max_dz`` and
    ``t_height_range`` are passed to mverify.match.match (None keeps every pair).
    """
    own = store is None
    store = store or ObsStore.open(read_only=True)
    try:
        if isinstance(forecasts, dict):
            fcs = [open_forecast(p, name=n, init=init) for n, p in forecasts.items()]
        else:
            fcs = [open_forecast(forecasts, init=init)]
        pairs = pd.concat([match(fc, store, variables, networks=networks, height_correction=height_correction,
                                 max_dz=max_dz, t_height_range=t_height_range)
                           for fc in fcs], ignore_index=True)
    finally:
        if own:
            store.close()
    return score_pairs(pairs, regions, fcs)


def score_pairs(pairs: pd.DataFrame, regions: list[str] | None = None, fcs: list[Forecast] = ()) -> Result:
    inits = {fc.name: sorted({f.init.strftime("%Y-%m-%dT%H") for f in fc.fields}) for fc in fcs}
    if pairs.empty:
        return Result(pairs, pd.DataFrame(), inits)
    rp = scores.assign_regions(pairs, regions or list(scores.REGIONS))
    by = ["model", "network", "variable", "region", "lead_h"]
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
    return Result(pairs, table.sort_values(by).reset_index(drop=True), inits)
