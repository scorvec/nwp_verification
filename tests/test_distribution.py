"""Distribution checks for wind speed: binned quantiles, the Weibull fit, and the model-against-observed comparison."""

import json

import numpy as np
import pandas as pd
import pytest

from mverify import distribution as D


def test_the_weibull_fit_is_the_maximum_likelihood_fit_to_the_speeds_above_calm():
    stats = pytest.importorskip("scipy.stats")
    x = np.random.default_rng(0).weibull(2.0, 200_000) * 6.0  # shape 2, scale 6 m/s
    k, c = D.weibull(D._hist(x))
    k_ref, _, c_ref = stats.weibull_min.fit(x[x >= D.CALM], floc=0)
    assert k[0] == pytest.approx(k_ref, rel=0.003) and c[0] == pytest.approx(c_ref, rel=0.003)
    assert k[0] == pytest.approx(2.0, rel=0.03)  # leaving the calms out moves k a little


def test_binned_quantiles_match_the_exact_ones():
    x = np.random.default_rng(1).weibull(1.8, 50_000) * 5.0
    assert D.quantiles(D._hist(x), (0.1, 0.5, 0.9, 0.99)) == pytest.approx(
        np.quantile(x, [0.1, 0.5, 0.9, 0.99]), abs=0.05)


def _pairs(rng, runs=10, stations=400):
    rows = []
    for r in range(runs):
        obs = rng.weibull(2.0, stations) * 6.0
        # WEAK: a fifth too slow; FAIR: an independent draw from the observed distribution
        for model, f in (("WEAK", 0.8 * obs + rng.normal(0, 0.3, stations)), ("FAIR", rng.weibull(2.0, stations) * 6.0)):
            rows.append(pd.DataFrame({"model": model, "init": pd.Timestamp("2026-10-01") + pd.Timedelta(days=r),
                                      "lead_h": 24.0, "variable": "ws10", "lat": 40.0, "lon": -100.0,
                                      "fcst": np.maximum(f, 0), "obs": obs}))
    return pd.concat(rows, ignore_index=True)


def test_a_model_too_weak_by_a_fifth_is_flagged_and_a_fair_one_is_not():
    out = D.compare(_pairs(np.random.default_rng(2)), regions=["conus"])
    row = next(r for r in out["rows"] if r["group"] == "days 1-2")
    assert row["runs"] == 10 and row["n"] == 4000 and row["obs"]["k"] == pytest.approx(2.0, rel=0.08)
    weak, fair = row["models"]["WEAK"], row["models"]["FAIR"]
    i90 = out["probs"].index(0.9)
    assert weak["q_hi"][i90] < 0 and weak["c_hi"] < 0  # its 90th percentile and scale are significantly low
    assert weak["ratio_hi"][out["thresholds"].index(10.0)] < 1  # and it reaches 10 m/s too rarely
    assert fair["q_lo"][out["probs"].index(0.5)] <= 0 <= fair["q_hi"][out["probs"].index(0.5)]
    assert fair["k_lo"] <= 0 <= fair["k_hi"]
    json.dumps(out, allow_nan=False)


def test_too_few_runs_give_no_row():
    p = _pairs(np.random.default_rng(3), runs=3)
    assert D.compare(p, regions=["conus"])["rows"] == []
