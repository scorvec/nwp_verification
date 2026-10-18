import numpy as np
import pandas as pd
import pytest

from mverify import scores


def pairs(fcst, obs, **kw):
    n = len(fcst)
    base = {"model": "M", "network": "GHCNH", "variable": "t2m", "lead_h": 24.0, "member": np.nan,
            "init": pd.Timestamp("2026-10-13"), "valid_time": pd.date_range("2026-10-14", periods=n, freq="h"),
            "station_id": [f"S{i}" for i in range(n)], "lat": 40.0, "lon": -100.0, "fcst": fcst, "obs": obs}
    base.update(kw)
    return pd.DataFrame(base)


def test_deterministic_scores():
    t = scores.deterministic(pairs([1.0, 2.0, 5.0], [0.0, 2.0, 2.0]), ["model", "variable"]).iloc[0]
    assert t["n"] == 3 and t["bias"] == pytest.approx(4 / 3) and t["mae"] == pytest.approx(4 / 3)
    assert t["rmse"] == pytest.approx(np.sqrt(10 / 3))


def test_vector_wind():
    u = pairs([3.0], [0.0], variable="u850")
    v = pairs([4.0], [0.0], variable="v850")
    t = scores.vector_wind(pd.concat([u, v]).assign(member=-1), ["model", "variable"]).iloc[0]
    assert t["variable"] == "wind850" and t["rmse"] == pytest.approx(5.0) and t["bias"] == pytest.approx(5.0)


def test_crps_of_one_member_is_the_absolute_error_and_fair_crps_rewards_spread_correctly():
    obs = np.array([0.0])
    assert scores.crps_ensemble(np.array([[2.0]]), obs)[0] == pytest.approx(2.0)
    # Two members at -1 and +1 around the truth: E|X-y| = 1, E|X-X'| (fair) = 2 -> CRPS = 1 - 1 = 0
    assert scores.crps_ensemble(np.array([[-1.0, 1.0]]), obs)[0] == pytest.approx(0.0)


def test_ensemble_scores():
    p = pd.concat([pairs([0.0, 1.0], [0.5, 0.5], member=0.0), pairs([1.0, 2.0], [0.5, 0.5], member=1.0)])
    t = scores.ensemble(p, ["model", "variable"]).iloc[0]
    assert t["n"] == 2 and t["spread"] > 0 and t["ssr"] == pytest.approx(t["spread"] / t["rmse_mean"])


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    p = pairs(rng.normal(0, 1, 300), np.zeros(300))
    ci = scores.bootstrap(p, ["model"]).iloc[0]
    rmse = np.sqrt((p["fcst"] ** 2).mean())
    assert ci["rmse_lo"] < rmse < ci["rmse_hi"]


def test_regions_copy_pairs_into_every_containing_region():
    p = pairs([1.0, 1.0], [0.0, 0.0], lat=[40.0, -40.0], lon=[-100.0, 20.0])
    r = scores.assign_regions(p, ["global", "conus", "sh"])
    assert sorted(r["region"]) == ["conus", "global", "global", "sh"]


def test_rmse_with_the_observation_error_removed():
    t = scores.deterministic(pairs([2.0, -2.0], [0.0, 0.0], sigma_o=1.5), ["model"]).iloc[0]
    assert t["rmse"] == pytest.approx(2.0) and t["rmse_oc"] == pytest.approx(np.sqrt(4 - 2.25))
    # an observation error larger than the whole error leaves nothing, never a negative variance
    assert scores.deterministic(pairs([0.1], [0.0], sigma_o=1.0), ["model"]).iloc[0]["rmse_oc"] == 0.0
    assert np.isnan(scores.deterministic(pairs([1.0], [0.0]), ["model"]).iloc[0]["rmse_oc"])


def test_vector_wind_removes_the_error_of_both_components():
    u = pairs([3.0], [0.0], variable="u850", sigma_o=2.0)
    v = pairs([4.0], [0.0], variable="v850", sigma_o=1.0)
    t = scores.vector_wind(pd.concat([u, v]).assign(member=-1), ["model", "variable"]).iloc[0]
    assert t["rmse_oc"] == pytest.approx(np.sqrt(25 - 5))


def test_fair_crps_from_sorted_members_equals_the_pairwise_definition():
    rng = np.random.default_rng(3)
    mem, obs = rng.normal(size=(50, 21)), rng.normal(size=50)
    m = mem.shape[1]
    pairwise = np.abs(mem[:, :, None] - mem[:, None, :]).sum(axis=(1, 2)) / (2 * m * (m - 1))
    expected = np.abs(mem - obs[:, None]).mean(axis=1) - pairwise
    assert scores.crps_ensemble(mem, obs) == pytest.approx(expected)


def test_a_calibrated_ensemble_looks_underdispersive_until_observation_error_is_counted():
    # Truth and members drawn from the same distribution; observations add N(0, sigma_o) error.
    rng = np.random.default_rng(0)
    n, m, spread, sigma_o = 4000, 30, 1.0, 1.0
    truth = rng.normal(0, spread, n)
    obs = truth + rng.normal(0, sigma_o, n)
    rows = [pairs(rng.normal(0, spread, n), obs, member=float(k), sigma_o=sigma_o) for k in range(m)]
    t = scores.ensemble(pd.concat(rows), ["model"]).iloc[0]
    assert t["ssr"] == pytest.approx(1 / np.sqrt(2), abs=0.05)   # raw: spread 1 against an error of sqrt(2)
    assert t["ssr_oe"] == pytest.approx(1.0, abs=0.05)
    assert t["crps_oe"] < t["crps"]
