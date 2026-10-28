"""Ensemble pairs summarised to one row per case, for a database that keeps every case of every run.

A case is one observation against one run of an ensemble: model, init, lead, valid time, station (or cell) and
variable. ``summarise`` turns the members' pairs (mverify.match, one row per member) into one row per case with what
the ensemble scores need, so a window of runs is scored from these rows without the members:

  * n_members, mean, spread (the members' sample standard deviation), min and max
  * crps, the fair CRPS (``mverify.scores.crps_ensemble``), and crps_oe, the same with each member perturbed by
    N(0, sigma_o) noise (``mverify.scores.ensemble``; NaN where sigma_o is unknown)
  * below and ties: members below the observation and equal to it, the observation's rank for rank histograms
    (a tie, as with no rain, is split at random when the histogram is drawn)
  * p_ge_<t>: the share of members at or above each threshold of the variable (``THRESHOLDS``), for Brier scores and
    reliability against the observed exceedance

Every other column of the pairs that is the same for all members of a case (the station's position and height, the
terrain difference, sigma_o ...) is kept as it is. A case missing some members keeps the members it has
(n_members says how many); the scores of mverify.scores.ensemble over the cases with every member are these rows'
crps and spread exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASE = ["model", "init", "lead_h", "valid_time", "station_id", "variable"]
CELL = ["model", "init", "lead_h", "valid_time", "lat", "lon", "variable"]  # a case on an analysis grid
# Event thresholds in the variable's canonical unit: 24 h precipitation (mm) at gauges and on the analysis grid, 10 m
# wind speed (m s-1, about 20, 30 and 40 kt).
THRESHOLDS = {"pr24": (1, 5, 10, 25), "tp": (1, 5, 10, 25), "ws10": (10, 15, 20)}


def _crps(x: np.ndarray, m: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS per row of members ``x`` (NaN-padded on the right after sorting), ``m`` of them in each row."""
    term1 = np.nansum(np.abs(x - obs[:, None]), axis=1) / m
    i = np.arange(x.shape[1])[None, :]
    w = np.where(i < m[:, None], 2 * i - m[:, None] + 1, 0)
    pair_sum = 2 * np.nansum(x * w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m > 1, term1 - pair_sum / (2 * m * (m - 1)), term1)


def summarise(
    pairs: pd.DataFrame, keys: list[str] = CASE, seed: int = 0, thresholds: dict[str, tuple] = THRESHOLDS
) -> pd.DataFrame:
    """One row per case of the members' ``pairs`` (columns ``keys`` - CASE for stations, CELL for grid cells -,
    member, fcst, obs, sigma_o where known, and any others); ``seed`` fixes the observation-error noise of crps_oe."""
    ens = pairs.dropna(subset=["member"])
    if ens.empty:
        return pd.DataFrame()
    ens = ens.sort_values([*keys, "member"], kind="stable")
    gid = ens.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
    first = np.r_[0, np.flatnonzero(np.diff(gid)) + 1]
    m = np.diff(np.r_[first, len(gid)])
    pos = np.arange(len(gid)) - np.repeat(first, m)
    x = np.full((len(first), int(m.max())), np.nan)
    x[gid, pos] = ens["fcst"].to_numpy(float)

    meta = [c for c in ens.columns if c not in ("member", "fcst")]
    out = ens.iloc[first][meta].reset_index(drop=True)
    obs = out["obs"].to_numpy(float)
    sigma = out["sigma_o"].to_numpy(float) if "sigma_o" in out else np.full(len(out), np.nan)
    known = np.isfinite(sigma)
    noisy = x + np.random.default_rng(seed).standard_normal(x.shape) * np.where(known, sigma, 0.0)[:, None]
    srt = np.sort(x, axis=1)  # NaN last
    with np.errstate(invalid="ignore"):
        out = out.assign(
            n_members=m.astype("int16"),
            mean=np.nanmean(x, axis=1),
            spread=np.where(m > 1, np.nanstd(x, axis=1, ddof=1), np.nan),
            min=np.nanmin(x, axis=1),
            max=np.nanmax(x, axis=1),
            crps=_crps(srt, m, obs),
            crps_oe=np.where(known, _crps(np.sort(noisy, axis=1), m, obs), np.nan),
            below=(x < obs[:, None]).sum(axis=1).astype("int16"),
            ties=(x == obs[:, None]).sum(axis=1).astype("int16"),
        )
    for t in sorted({t for ts in thresholds.values() for t in ts}):
        applies = out["variable"].isin([v for v, ts in thresholds.items() if t in ts]).to_numpy()
        out[f"p_ge_{t:g}"] = np.where(applies, (x >= t).sum(axis=1) / m, np.nan)
    return out
