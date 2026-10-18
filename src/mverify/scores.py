"""Verification scores computed from matched pairs.

Deterministic (per group): n, bias (forecast minus observed), MAE, RMSE,
error standard deviation and correlation. Vector wind RMSE combines the u and
v pairs at each level. Ensembles (pairs carrying a ``member``) add the fair
CRPS, ensemble-mean RMSE, spread and the spread/skill ratio. Confidence
intervals come from a bootstrap over valid times, which keeps the strong
spatial correlation between stations at one time inside each resample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (lon_min, lat_min, lon_max, lat_max)
REGIONS = {
    "global": (-180, -90, 180, 90),
    "nh": (-180, 20, 180, 90),
    "tropics": (-180, -20, 180, 20),
    "sh": (-180, -90, 180, -20),
    "conus": (-125, 24, -66, 50),
    "north_america": (-170, 15, -50, 75),
    "europe": (-12, 35, 40, 72),
}
REGION_LABELS = {"global": "Global", "nh": "NH 20-90N", "tropics": "Tropics 20S-20N", "sh": "SH 20-90S",
                 "conus": "CONUS", "north_america": "North America", "europe": "Europe"}


def assign_regions(pairs: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """One copy of each pair per region that contains its station."""
    parts = []
    for r in regions:
        lo0, la0, lo1, la1 = REGIONS[r]
        m = pairs["lat"].between(la0, la1) & pairs["lon"].between(lo0, lo1)
        if m.any():
            parts.append(pairs[m].assign(region=r))
    return pd.concat(parts, ignore_index=True) if parts else pairs.assign(region=pd.Series(dtype=str))


def _det(g: pd.DataFrame) -> pd.Series:
    e = g["fcst"].to_numpy() - g["obs"].to_numpy()
    n = len(e)
    corr = np.corrcoef(g["fcst"], g["obs"])[0, 1] if n > 2 and g["obs"].std() > 0 else np.nan
    return pd.Series({"n": n, "bias": e.mean(), "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()),
                      "sde": e.std(ddof=1) if n > 1 else np.nan, "corr": corr})


def deterministic(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=[*by, "n", "bias", "mae", "rmse", "sde", "corr"])
    return pairs.groupby(by, sort=True, dropna=False)[["fcst", "obs"]].apply(_det).reset_index()


def vector_wind(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Vector RMSE and speed bias from u/v pairs: rows named ``wind10``, ``wind850`` ..."""
    key = ["model", "init", "lead_h", "member", "valid_time", "station_id"]
    rows = []
    for v_u in sorted(pairs["variable"].unique()):
        if not v_u.startswith("u"):
            continue
        v_v = "v" + v_u[1:]
        u, v = pairs[pairs["variable"] == v_u], pairs[pairs["variable"] == v_v]
        if v.empty:
            continue
        j = u.merge(v[key + ["fcst", "obs"]], on=key, suffixes=("_u", "_v"))
        if j.empty:
            continue
        j["variable"] = "wind" + v_u[1:]
        j["se"] = (j["fcst_u"] - j["obs_u"]) ** 2 + (j["fcst_v"] - j["obs_v"]) ** 2
        j["dspd"] = np.hypot(j["fcst_u"], j["fcst_v"]) - np.hypot(j["obs_u"], j["obs_v"])
        g = j.groupby(by, dropna=False)
        rows.append(pd.DataFrame({"n": g.size(), "rmse": np.sqrt(g["se"].mean()), "bias": g["dspd"].mean()})
                    .reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*by, "n", "rmse", "bias"])


def crps_ensemble(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS of an ensemble (members on axis 1) against observations, per case."""
    m = members.shape[1]
    term1 = np.abs(members - obs[:, None]).mean(axis=1)
    if m < 2:
        return term1
    diffs = np.abs(members[:, :, None] - members[:, None, :]).sum(axis=(1, 2))
    return term1 - diffs / (2 * m * (m - 1))


def ensemble(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    case = ["model", "init", "lead_h", "valid_time", "station_id", "variable", *[b for b in by if b == "region"]]
    ens = pairs.dropna(subset=["member"])
    if ens.empty:
        return pd.DataFrame(columns=[*by, "n", "crps", "rmse_mean", "spread", "ssr"])
    wide = ens.pivot_table(index=case, columns="member", values="fcst").dropna()
    obs = ens.groupby(case)["obs"].first().reindex(wide.index)
    mem = wide.to_numpy()
    c = pd.DataFrame({"crps": crps_ensemble(mem, obs.to_numpy()),
                      "se_mean": (mem.mean(axis=1) - obs.to_numpy()) ** 2,
                      "var": mem.var(axis=1, ddof=1)}, index=wide.index).reset_index()
    g = c.groupby(by, dropna=False)
    out = pd.DataFrame({"n": g.size(), "crps": g["crps"].mean(), "rmse_mean": np.sqrt(g["se_mean"].mean()),
                        "spread": np.sqrt(g["var"].mean())}).reset_index()
    out["ssr"] = out["spread"] / out["rmse_mean"]
    return out


def bootstrap(pairs: pd.DataFrame, by: list[str], stat: str = "rmse", n_boot: int = 500, level: float = 0.95,
              seed: int = 0) -> pd.DataFrame:
    """Percentile CI of ``stat`` ('rmse' or 'bias'), resampling whole valid times."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in pairs.groupby(by, dropna=False):
        e = (g["fcst"] - g["obs"]).to_numpy()
        t = pd.factorize(g["valid_time"])[0]
        k = t.max() + 1
        sums = np.bincount(t, e ** 2 if stat == "rmse" else e, minlength=k)
        cnt = np.bincount(t, minlength=k)
        draws = rng.integers(0, k, size=(n_boot, k))
        s, c = sums[draws].sum(axis=1), cnt[draws].sum(axis=1)
        vals = np.sqrt(s / c) if stat == "rmse" else s / c
        lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), f"{stat}_lo": lo, f"{stat}_hi": hi})
    return pd.DataFrame(rows)
