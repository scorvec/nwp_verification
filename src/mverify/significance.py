"""Is one model's error really smaller than another's? Paired tests over forecast runs.

Each run (init) gives one mean squared error per model, variable, region and lead, on
the same observations for every model (``verify.common_sample``). The test is a
paired t-test on the per-run differences of those errors. Runs a day or two apart
see the same weather, so their differences are correlated, more so at long leads;
following Geer (2016, Tellus A 68, 30229) the variance is inflated for lag-1
autocorrelation r of the differences, which shrinks the sample to
n (1 - r) / (1 + r) effective runs. Without that the test reports far more
significant differences than there are.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats

BY = ["network", "variable", "region", "lead_h"]
KEY = ["init", "lead_h", "valid_time", "station_id"]


def squared_errors(pairs: pd.DataFrame) -> pd.DataFrame:
    """One squared error per model and observation; an ensemble is scored through its mean.
    u/v pairs also give the vector error of ``wind10``, ``wind250`` ... as in ``scores.vector_wind``."""
    keys = ["model", "network", "variable", "region", *KEY]
    p = pairs.groupby(keys, dropna=False, sort=False).agg(fcst=("fcst", "mean"), obs=("obs", "first")).reset_index()
    out = [p.assign(se=(p["fcst"] - p["obs"]) ** 2)]
    for var_u in sorted(v for v in p["variable"].unique() if v.startswith("u")):
        u, v = p[p["variable"] == var_u], p[p["variable"] == "v" + var_u[1:]]
        j = u.merge(v, on=[k for k in keys if k != "variable"], suffixes=("_u", "_v"))
        if not j.empty:
            se = (j["fcst_u"] - j["obs_u"]) ** 2 + (j["fcst_v"] - j["obs_v"]) ** 2
            out.append(j[[k for k in keys if k != "variable"]].assign(variable="wind" + var_u[1:], se=se))
    return pd.concat(out, ignore_index=True)[[*keys, "se"]]


def effective_runs(d: np.ndarray) -> float:
    """Runs left after allowing for lag-1 autocorrelation of the differences (never more than len(d))."""
    n = len(d)
    if n < 4:
        return float(n)
    a = d - d.mean()
    den = (a * a).sum()
    r = float((a[1:] * a[:-1]).sum() / den) if den > 0 else 0.0
    r = min(max(r, 0.0), 0.9)
    return n * (1 - r) / (1 + r)


def paired_test(d: np.ndarray) -> float:
    """Two-sided p value for a mean difference of zero; NaN with fewer than three runs."""
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() != 0 else 1.0
    n_eff = effective_runs(d)
    t = d.mean() / (sd / np.sqrt(n_eff))
    return float(2 * stats.t.sf(abs(t), df=max(n_eff - 1, 1.0)))


def fdr(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: which of the p values (NaN = not tested) are discoveries at false-discovery rate q."""
    p = np.asarray(p, float)
    out = np.zeros(p.shape, bool)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    passed = p[ok][order] <= q * np.arange(1, m + 1) / m
    k = int(np.flatnonzero(passed).max()) + 1 if passed.any() else 0
    hit = np.zeros(m, bool)
    hit[order[:k]] = True
    out[ok] = hit
    return out


def tests(pairs: pd.DataFrame) -> pd.DataFrame:
    """p values for every ordered pair of models: columns ``BY`` + ``model``, ``reference``, ``runs`` (paired
    runs), ``runs_eff`` (the effective runs the test counts, ``effective_runs``) and ``p``.

    ``pairs`` carries a ``region`` column (``scores.assign_regions``) and should be the
    common sample, so that both models of a pair are judged on the same observations.
    """
    cols = [*BY, "model", "reference", "runs", "runs_eff", "p"]
    if pairs.empty or pairs["model"].nunique() < 2:
        return pd.DataFrame(columns=cols)
    mse = (squared_errors(pairs).groupby(["model", *BY, "init"], dropna=False)["se"].mean()
           .unstack("model").sort_index(level="init"))
    rows = []
    for a, b in permutations(mse.columns, 2):
        d = (mse[a] - mse[b]).dropna()
        for key, g in d.groupby(level=BY, sort=True):
            d_run = g.to_numpy()
            rows.append((*key, a, b, len(g), effective_runs(d_run), paired_test(d_run)))
    return pd.DataFrame(rows, columns=cols)
