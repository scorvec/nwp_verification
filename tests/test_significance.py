"""Paired tests between models over forecast runs."""

import numpy as np
import pandas as pd
import pytest

from mverify import report, significance
from mverify.verify import common_sample


def _pairs(errors: dict[str, np.ndarray], n_stations=300, seed=0):
    """Pairs for models whose error standard deviation in run k is errors[model][k]."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(15, 5, (len(next(iter(errors.values()))), n_stations))
    rows = []
    for model, sd in errors.items():
        fc = obs + rng.normal(0, 1, obs.shape) * sd[:, None]
        for k in range(obs.shape[0]):
            init = pd.Timestamp("2026-10-01") + pd.Timedelta(days=k)
            rows.append(pd.DataFrame({
                "model": model, "network": "GHCNH", "variable": "t2m", "region": "conus", "init": init,
                "lead_h": 120.0, "valid_time": init + pd.Timedelta("120h"), "member": None,
                "station_id": [f"S{i}" for i in range(n_stations)], "fcst": fc[k], "obs": obs[k]}))
    return pd.concat(rows, ignore_index=True)


def test_a_model_better_in_every_run_is_significant_and_an_equal_one_is_not():
    runs = np.random.default_rng(1).uniform(1.5, 3.0, 12)  # how hard each run's weather was
    t = significance.tests(_pairs({"A": runs * 0.8, "B": runs, "C": runs})).set_index(["model", "reference"])
    assert t.loc[("A", "B"), "p"] < 0.01 and t.loc[("A", "B"), "runs"] == 12
    assert t.loc[("C", "B"), "p"] > 0.05
    assert t.loc[("A", "B"), "p"] == pytest.approx(t.loc[("B", "A"), "p"])
    assert 0 < t.loc[("A", "B"), "runs_eff"] <= 12
    assert t.loc[("A", "B"), "runs_eff"] == pytest.approx(t.loc[("B", "A"), "runs_eff"])


def test_correlated_runs_count_for_less():
    smooth = np.cumsum(np.ones(10))            # differences that drift slowly from run to run
    assert significance.effective_runs(smooth) < 4
    assert significance.effective_runs(np.array([1.0, -1, 1, -1, 1, -1])) == 6  # never more than the runs
    assert np.isnan(significance.paired_test(np.array([0.1, 0.2])))


def test_vector_wind_is_tested_from_u_and_v():
    p = _pairs({"A": np.full(6, 1.0), "B": np.full(6, 2.0)})
    wind = pd.concat([p.assign(variable="u10"), p.assign(variable="v10")], ignore_index=True)
    t = significance.tests(wind)
    assert set(t["variable"]) == {"u10", "v10", "wind10"}


def test_common_sample_keeps_what_every_model_has():
    p = _pairs({"A": np.ones(3), "B": np.ones(3)})
    p = p.drop(p[(p["model"] == "B") & (p["station_id"] == "S0")].index)
    c = common_sample(p)
    assert "S0" not in set(c["station_id"]) and c.groupby("model").size().nunique() == 1


def test_scorecard_carries_the_p_values():
    p = _pairs({"A": np.full(5, 1.0), "B": np.full(5, 2.0)})
    table = pd.DataFrame([{"model": m, "network": "GHCNH", "variable": "t2m", "region": "conus", "lead_h": 120.0,
                           "rmse": 1.0, "n": 150} for m in "AB"])
    card = report.scorecard(table, {}, significance.tests(p))
    assert card["rows"][0]["p"]["A|B"][0] < 0.05 and set(card["rows"][0]["p"]) == {"A|B", "B|A"}
    # the paired runs, once per pair of models
    assert card["rows"][0]["runs"] == {"A|B": [5]} and set(card["rows"][0]["runs_eff"]) == {"A|B"}
    assert 0 < card["rows"][0]["runs_eff"]["A|B"][0] <= 5
