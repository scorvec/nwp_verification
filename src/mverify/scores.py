"""Verification scores computed from matched pairs.

Deterministic (per group): n, bias (forecast minus observed), MAE, RMSE,
error standard deviation and correlation. Vector wind RMSE combines the u and
v pairs at each level. Ensembles (pairs carrying a ``member``) add the fair
CRPS, ensemble-mean RMSE, spread and the spread/skill ratio. Confidence
intervals come from a bootstrap over valid times, which keeps the strong
spatial correlation between stations at one time inside each resample.

Observation error. An observation differs from the grid-box truth by an error
of standard deviation sigma_o (instrument plus representativeness, carried on
each pair from the variable catalogue), so even a perfect model scores
RMSE = sigma_o and a perfectly calibrated ensemble looks under-dispersive.
Alongside the raw scores, which stay the headline numbers and are comparable
with other centres, three error-aware ones are reported:
  * rmse_oc = sqrt(max(MSE - sigma_o^2, 0)), the RMSE with the observation
    error variance removed (errors are assumed independent of the forecast);
  * crps_oe, the CRPS of the members perturbed with N(0, sigma_o) noise
    (Saetra et al. 2004), which compares the ensemble with observations on an
    equal footing;
  * spread_oe = sqrt(spread^2 + sigma_o^2) and ssr_oe = spread_oe / rmse_mean,
    the spread-skill ratio that is 1 for a calibrated ensemble.
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


def _obs_var(g: pd.DataFrame) -> float:
    """Mean observation-error variance of a group; NaN when sigma_o is unknown."""
    if "sigma_o" not in g:
        return np.nan
    return float(np.mean(g["sigma_o"].to_numpy(float) ** 2))


def _det(g: pd.DataFrame) -> pd.Series:
    e = g["fcst"].to_numpy() - g["obs"].to_numpy()
    n = len(e)
    corr = np.corrcoef(g["fcst"], g["obs"])[0, 1] if n > 2 and g["obs"].std() > 0 else np.nan
    mse = (e ** 2).mean()
    return pd.Series({"n": n, "bias": e.mean(), "mae": np.abs(e).mean(), "rmse": np.sqrt(mse),
                      "rmse_oc": np.sqrt(max(mse - _obs_var(g), 0.0)),
                      "sde": e.std(ddof=1) if n > 1 else np.nan, "corr": corr})


DET_STATS = ["n", "bias", "mae", "rmse", "rmse_oc", "sde", "corr"]


def deterministic(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=[*by, *DET_STATS])
    cols = ["fcst", "obs", *(["sigma_o"] if "sigma_o" in pairs else [])]
    return pairs.groupby(by, sort=True, dropna=False)[cols].apply(_det).reset_index()


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
        u, v = u.assign(sigma_o=u.get("sigma_o", np.nan)), v.assign(sigma_o=v.get("sigma_o", np.nan))
        j = u.merge(v[key + ["fcst", "obs", "sigma_o"]], on=key, suffixes=("_u", "_v"))
        if j.empty:
            continue
        j["variable"] = "wind" + v_u[1:]
        j["se"] = (j["fcst_u"] - j["obs_u"]) ** 2 + (j["fcst_v"] - j["obs_v"]) ** 2
        j["ov"] = j["sigma_o_u"] ** 2 + j["sigma_o_v"] ** 2  # error variance of the observed vector
        j["dspd"] = np.hypot(j["fcst_u"], j["fcst_v"]) - np.hypot(j["obs_u"], j["obs_v"])
        g = j.groupby(by, dropna=False)
        mse = g["se"].mean()
        rows.append(pd.DataFrame({"n": g.size(), "rmse": np.sqrt(mse),
                                  "rmse_oc": np.sqrt((mse - g["ov"].mean(skipna=False)).clip(lower=0)),
                                  "bias": g["dspd"].mean()}).reset_index())
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=[*by, "n", "rmse", "rmse_oc", "bias"]))


STATION_KEY = ["model", "network", "variable", "station_id", "lead_h"]


def by_station(pairs: pd.DataFrame) -> pd.DataFrame:
    """RMSE and bias of each model at each station and lead, for maps: columns ``STATION_KEY``,
    ``lat``, ``lon``, ``n``, ``rmse``, ``bias``.

    An ensemble is scored through its mean. u/v pairs also give the vector wind (``wind10``,
    ``wind500`` ...) with the speed bias as its bias, as in ``vector_wind``. The position is
    where the observations were: the median over the sample, for a drifting balloon.
    """
    cols = [*STATION_KEY, "lat", "lon", "n", "rmse", "bias"]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    one = ["model", "network", "variable", "init", "lead_h", "valid_time", "station_id"]
    p = pairs
    if p["member"].notna().any():
        p = p.groupby(one, dropna=False, sort=False).agg(fcst=("fcst", "mean"), obs=("obs", "first"),
                                                         lat=("lat", "first"), lon=("lon", "first")).reset_index()
    p = p[[*one, "lat", "lon", "fcst", "obs"]].assign(err=lambda d: d["fcst"] - d["obs"])
    parts = [p.assign(se=p["err"] ** 2)]
    for var_u in sorted(v for v in p["variable"].unique() if v.startswith("u")):
        u, v = p[p["variable"] == var_u], p[p["variable"] == "v" + var_u[1:]]
        j = u.merge(v[[k for k in one if k != "variable"] + ["fcst", "obs"]],
                    on=[k for k in one if k != "variable"], suffixes=("_u", "_v"))
        if not j.empty:
            parts.append(j[[k for k in one if k != "variable"] + ["lat", "lon"]].assign(
                variable="wind" + var_u[1:], se=(j["fcst_u"] - j["obs_u"]) ** 2 + (j["fcst_v"] - j["obs_v"]) ** 2,
                err=np.hypot(j["fcst_u"], j["fcst_v"]) - np.hypot(j["obs_u"], j["obs_v"])))
    g = pd.concat(parts, ignore_index=True).groupby(STATION_KEY, sort=True)
    out = pd.DataFrame({"lat": g["lat"].median(), "lon": g["lon"].median(), "n": g.size(),
                        "rmse": np.sqrt(g["se"].mean()), "bias": g["err"].mean()})
    return out.reset_index()[cols]


def crps_ensemble(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS of an ensemble (members on axis 1) against observations, per case.

    The sum of |x_i - x_j| over member pairs is taken from the sorted members,
    sum_i (2i - m + 1) x_(i), which is O(m log m) per case instead of O(m^2):
    a 51-member ensemble over 15 days of stations stays a few arrays of members.
    """
    m = members.shape[1]
    term1 = np.abs(members - obs[:, None]).mean(axis=1)
    if m < 2:
        return term1
    weights = 2 * np.arange(m) - m + 1
    pair_sum = 2 * (np.sort(members, axis=1) * weights).sum(axis=1)  # over ordered pairs (i, j), i != j
    return term1 - pair_sum / (2 * m * (m - 1))


ENS_STATS = ["n", "crps", "crps_oe", "rmse_mean", "spread", "spread_oe", "ssr", "ssr_oe"]


def ensemble(pairs: pd.DataFrame, by: list[str], seed: int = 0) -> pd.DataFrame:
    """Ensemble scores per group; ``seed`` fixes the observation-error perturbations of crps_oe."""
    case = list(dict.fromkeys(["model", "init", "lead_h", "valid_time", "station_id", "variable", *by]))
    ens = pairs.dropna(subset=["member"])
    if ens.empty:
        return pd.DataFrame(columns=[*by, *ENS_STATS])
    wide = ens.pivot_table(index=case, columns="member", values="fcst").dropna()
    first = ens.assign(sigma_o=ens.get("sigma_o", np.nan)).groupby(case)[["obs", "sigma_o"]].first()
    first = first.reindex(wide.index)
    obs, sigma = first["obs"].to_numpy(float), first["sigma_o"].to_numpy(float)
    mem = wide.to_numpy()
    noise = np.random.default_rng(seed).standard_normal(mem.shape) * sigma[:, None]
    c = pd.DataFrame({"crps": crps_ensemble(mem, obs),
                      "crps_oe": crps_ensemble(mem + noise, obs),  # NaN where sigma_o is unknown
                      "se_mean": (mem.mean(axis=1) - obs) ** 2,
                      "var": mem.var(axis=1, ddof=1), "ov": sigma ** 2}, index=wide.index).reset_index()
    g = c.groupby(by, dropna=False)
    out = pd.DataFrame({"n": g.size(), "crps": g["crps"].mean(), "crps_oe": g["crps_oe"].mean(skipna=False),
                        "rmse_mean": np.sqrt(g["se_mean"].mean()), "spread": np.sqrt(g["var"].mean()),
                        "spread_oe": np.sqrt(g["var"].mean() + g["ov"].mean(skipna=False))}).reset_index()
    out["ssr"] = out["spread"] / out["rmse_mean"]
    out["ssr_oe"] = out["spread_oe"] / out["rmse_mean"]
    return out[[*by, *ENS_STATS]]


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
