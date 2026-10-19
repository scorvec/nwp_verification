"""How a model's distribution of a skewed variable such as wind speed compares with the observed one.

Root-mean-square error and bias reward a forecast that hedges towards the middle: a wind forecast can score well on
both with too few calms and too few gales. For each model, region and lead group, on the pairs every model has
(``verify.common_sample``), so model and observed distributions come from the same places and times:

  * quantiles of the forecasts and of the observations at the same probabilities (a QQ plot), with a bootstrap
    interval for each forecast-minus-observed difference;
  * the hybrid Weibull of Takle and Brown (1978, J. Appl. Meteor. 17, 556-559): the calm fraction, and a Weibull
    fitted by maximum likelihood (shape k, scale c) to the values of at least ``CALM``;
  * how often each threshold is exceeded, observed and forecast, with an interval for the forecast/observed ratio.

The bootstrap resamples whole runs (inits) and keeps model and observation paired, as errors within a run are
correlated across stations and leads. Each run's values are first binned at 0.05 (m/s), so a resample is a sum of
histograms and every statistic is read off it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mverify.scores import REGIONS, assign_regions

PROBS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
CALM = 0.5               # values below are calm (METAR wind speeds come in whole knots)
THRESHOLDS = (5.0, 10.0, 15.0)
LEAD_GROUPS = (("days 1-2", 0, 48), ("days 3-5", 48, 120), ("days 6-10", 120, 240))  # (label, after, up to) hours
N_BOOT = 400
LEVEL = 0.95
MIN_RUNS = 5
EDGES = np.arange(0.0, 60.0 + 1e-9, 0.05)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])


def _json(v) -> list | float | None:
    """Rounded, with None for NaN (no observed exceedance to divide by, a fit with nothing above calm)."""
    a = np.round(np.asarray(v, float), 4)
    return None if a.ndim == 0 and not np.isfinite(a) else (float(a) if a.ndim == 0 else
                                                             [_json(x) for x in a])


def _hist(x: np.ndarray) -> np.ndarray:
    return np.histogram(np.clip(x, 0.0, EDGES[-1] - 1e-6), EDGES)[0]


def quantiles(h: np.ndarray, probs=PROBS) -> np.ndarray:
    """Quantiles of the binned values ``h`` (..., bins), interpolated within bins."""
    cdf = np.cumsum(h, axis=-1) / np.maximum(h.sum(axis=-1, keepdims=True), 1)
    flat = cdf.reshape(-1, cdf.shape[-1])
    out = np.array([np.interp(probs, np.concatenate([[0.0], c]), EDGES) for c in flat])
    return out.reshape(*cdf.shape[:-1], len(probs))


def weibull(h: np.ndarray, calm: float = CALM, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood Weibull shape k and scale c of the binned values above ``calm``, for each row of ``h``.
    Newton's method on the profile likelihood's equation for k, started from the log-moment estimate."""
    keep = CENTRES >= calm
    x, w = CENTRES[keep], np.atleast_2d(h)[:, keep].astype(float)
    lx = np.log(x)
    n = np.where(w.sum(1) > 0, w.sum(1), np.nan)
    mean_l = (w * lx).sum(1) / n
    sd_l = np.sqrt(np.maximum((w * lx**2).sum(1) / n - mean_l**2, 1e-12))
    k = np.pi / (np.sqrt(6.0) * sd_l)
    for _ in range(iters):
        xk = x[None, :] ** k[:, None]
        s0, s1, s2 = (w * xk).sum(1), (w * xk * lx).sum(1), (w * xk * lx**2).sum(1)
        f = s1 / s0 - 1.0 / k - mean_l
        df = (s2 * s0 - s1**2) / s0**2 + 1.0 / k**2
        k = np.clip(k - f / df, 0.2, 20.0)
    c = ((w * x[None, :] ** k[:, None]).sum(1) / n) ** (1.0 / k)
    return k, c


def _stats(h: np.ndarray, probs, thresholds) -> dict[str, np.ndarray]:
    total = np.maximum(h.sum(-1), 1)
    k, c = weibull(h.reshape(-1, h.shape[-1]))
    return {"q": quantiles(h, probs), "k": k.reshape(h.shape[:-1]), "c": c.reshape(h.shape[:-1]),
            "calm": h[..., CENTRES < CALM].sum(-1) / total,
            "exceed": np.stack([h[..., CENTRES >= t].sum(-1) / total for t in thresholds], -1)}


def compare(pairs: pd.DataFrame, variable: str = "ws10", regions: list[str] | None = None, groups=LEAD_GROUPS,
            probs=PROBS, thresholds=THRESHOLDS, n_boot: int = N_BOOT, level: float = LEVEL, seed: int = 0) -> dict:
    """The comparison as a JSON-ready dict: ``rows`` per region and lead group, each with the observed statistics
    (``obs``: q, k, c, calm, exceed) and per model the same plus bootstrap intervals for q - q_obs (``q_lo``,
    ``q_hi``), k - k_obs, c - c_obs, calm - calm_obs and exceed / exceed_obs (``ratio_lo``, ``ratio_hi``)."""
    p = pairs[(pairs["variable"] == variable) & pairs["fcst"].notna() & pairs["obs"].notna()]
    models = sorted(p["model"].unique())
    out = {"variable": variable, "probs": list(probs), "thresholds": list(thresholds), "calm": CALM, "level": level,
           "n_boot": n_boot, "groups": [g[0] for g in groups], "models": models, "rows": []}
    if p.empty:
        return out
    rng = np.random.default_rng(seed)
    a = (1 - level) / 2
    for region in regions or list(REGIONS):
        rp = assign_regions(p, [region])
        for label, after, upto in groups:
            g = rp[(rp["lead_h"] > after) & (rp["lead_h"] <= upto)]
            if g.empty:
                continue
            runs = sorted(g["init"].unique())
            if len(runs) < MIN_RUNS:
                continue
            first = g[g["model"] == models[0]]
            H_obs = np.stack([_hist(first.loc[first["init"] == r, "obs"].to_numpy(float)) for r in runs])
            draws = rng.integers(0, len(runs), size=(n_boot, len(runs)))
            obs_all, obs_b = _stats(H_obs.sum(0), probs, thresholds), _stats(H_obs[draws].sum(1), probs, thresholds)
            row = {"region": region, "group": label, "runs": len(runs), "n": int(H_obs.sum()),
                   "obs": {k: _json(v) for k, v in obs_all.items()}, "models": {}}
            for m in models:
                gm = g[g["model"] == m]
                H = np.stack([_hist(gm.loc[gm["init"] == r, "fcst"].to_numpy(float)) for r in runs])
                st, b = _stats(H.sum(0), probs, thresholds), _stats(H[draws].sum(1), probs, thresholds)
                ci = {}
                for key in ("q", "k", "c", "calm"):
                    d = b[key] - obs_b[key]
                    ci[f"{key}_lo"], ci[f"{key}_hi"] = np.quantile(d, a, axis=0), np.quantile(d, 1 - a, axis=0)
                # forecast/observed exceedance; no ratio in a resample that observed none
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(obs_b["exceed"] > 0, b["exceed"] / obs_b["exceed"], np.nan)
                ok = np.isfinite(r).any(0)
                ci["ratio_lo"], ci["ratio_hi"] = np.full(r.shape[1], np.nan), np.full(r.shape[1], np.nan)
                ci["ratio_lo"][ok] = np.nanquantile(r[:, ok], a, axis=0)
                ci["ratio_hi"][ok] = np.nanquantile(r[:, ok], 1 - a, axis=0)
                row["models"][m] = {k: _json(v) for k, v in {**st, **ci}.items()}
            out["rows"].append(row)
    return out
