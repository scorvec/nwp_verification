"""Grid-to-grid precipitation scores, accumulated over cases and finished once.

Every score is built from sums that add across windows and inits, so a season is
scored exactly rather than as an average of daily scores:

  * contingency at each threshold (forecast and analysis >= threshold): hits,
    misses, false alarms, correct negatives -> ETS (Gilbert skill score),
    frequency bias, POD, FAR;
  * FSS (Roberts and Lean 2008) at neighbourhood widths in km: the numerator
    sum (Pf - Po)^2 and denominator sum Pf^2 + Po^2 of neighbourhood fractions;
  * amounts: mean forecast and analysis, bias, MAE, RMSE over valid cells.

Cells are area-weighted (cos latitude on a lat/lon grid; equal on a projected
grid such as Stage IV's, whose cells vary by a few percent). Cells the analysis
does not cover are left out of everything, including the neighbourhood
fractions, which are computed over covered cells only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

THRESHOLDS_MM = (1.0, 5.0, 10.0, 25.0, 50.0)  # per 24 h
FSS_KM = (50, 100, 200, 400)
SUMS = ("w", "sum_f", "sum_o", "sum_err", "sum_abs", "sum_sq")
COUNTS = ("hits", "misses", "false_alarms")


def _fractions(binary: np.ndarray, valid: np.ndarray, n: int, periodic_x: bool) -> np.ndarray:
    if n <= 1:
        return np.where(valid, binary, 0.0)
    mode = ("constant", "wrap") if periodic_x else "constant"
    num = uniform_filter(np.where(valid, binary, 0.0), size=n, mode=mode)
    den = uniform_filter(valid.astype(float), size=n, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def case_sums(fcst: np.ndarray, obs: np.ndarray, mask: np.ndarray, weights: np.ndarray, cell_km: float,
              periodic_x: bool, thresholds=THRESHOLDS_MM, fss_km=FSS_KM) -> dict:
    """Additive sums for one forecast window against one analysis, over the cells in ``mask``."""
    valid = np.isfinite(obs) & np.isfinite(fcst)
    m = mask & valid
    w = np.where(m, weights, 0.0)
    f, o = np.where(m, fcst, 0.0), np.where(m, obs, 0.0)
    e = f - o
    out: dict = {"amount": {"w": w.sum(), "sum_f": (w * f).sum(), "sum_o": (w * o).sum(),
                            "sum_err": (w * e).sum(), "sum_abs": (w * np.abs(e)).sum(),
                            "sum_sq": (w * e * e).sum()}}
    for t in thresholds:
        bf, bo = (f >= t) & m, (o >= t) & m
        c = {"hits": (w * (bf & bo)).sum(), "misses": (w * (~bf & bo)).sum(),
             "false_alarms": (w * (bf & ~bo)).sum(), "w": w.sum()}
        for km in fss_km:
            n = max(1, round(km / cell_km))
            pf = _fractions(bf.astype(float), valid, n, periodic_x)
            po = _fractions(bo.astype(float), valid, n, periodic_x)
            c[f"fss_num_{km}"] = (w * (pf - po) ** 2).sum()
            c[f"fss_den_{km}"] = (w * (pf ** 2 + po ** 2)).sum()
        out[t] = c
    return out


def finish(rows: pd.DataFrame, by: list[str], fss_km=FSS_KM) -> pd.DataFrame:
    """Scores from summed case sums. ``rows`` has ``by`` columns, ``threshold`` (NaN for amounts) and sums."""
    rows = rows.reindex(columns=list(dict.fromkeys([*rows.columns, *SUMS, *COUNTS,
                                                    *(f"fss_{p}_{km}" for km in fss_km for p in ("num", "den"))])))
    g = rows.groupby([*by, "threshold"], dropna=False).sum(numeric_only=True, min_count=0).reset_index()
    out = g[[*by, "threshold"]].copy()
    amount = g["threshold"].isna()
    w = g["w"]
    out["n"] = g["n_cases"]
    out.loc[amount, "fcst_mean"] = g["sum_f"] / w
    out.loc[amount, "obs_mean"] = g["sum_o"] / w
    out.loc[amount, "bias"] = g["sum_err"] / w
    out.loc[amount, "mae"] = g["sum_abs"] / w
    out.loc[amount, "rmse"] = np.sqrt(g["sum_sq"] / w)
    h, mi, fa = g["hits"], g["misses"], g["false_alarms"]
    with np.errstate(invalid="ignore", divide="ignore"):
        random_hits = (h + mi) * (h + fa) / w
        out.loc[~amount, "ets"] = (h - random_hits) / (h + mi + fa - random_hits)
        out.loc[~amount, "fbias"] = (h + fa) / (h + mi)
        out.loc[~amount, "pod"] = h / (h + mi)
        out.loc[~amount, "far"] = fa / (h + fa)
        out.loc[~amount, "base_rate"] = (h + mi) / w
        for km in fss_km:
            out.loc[~amount, f"fss_{km}km"] = 1 - g[f"fss_num_{km}"] / g[f"fss_den_{km}"]
    return out
