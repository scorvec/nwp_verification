import numpy as np
import pandas as pd
import pytest

from mverify import scores, significance
from mverify import variables as V
from mverify.ensemble import CELL, summarise

INIT = pd.Timestamp("2026-10-01")


def _pairs(rng, cases=40, members=11, var="t2m", model="ENS"):
    rows = []
    for c in range(cases):
        obs = rng.normal(10, 3)
        for k in range(members):
            rows.append({"model": model, "init": INIT, "lead_h": 24.0, "valid_time": INIT + pd.Timedelta("24h"),
                         "station_id": f"S{c:03d}", "variable": var, "member": k, "fcst": obs + rng.normal(0.5, 2),
                         "obs": obs, "sigma_o": 1.4, "lat": 50.0, "lon": c * 1.0})
    return pd.DataFrame(rows)


def test_summary_rows_give_the_ensemble_scores():
    p = _pairs(np.random.default_rng(1))
    s = summarise(p)
    ref = scores.ensemble(p, ["model"]).iloc[0]
    assert len(s) == 40 and (s["n_members"] == 11).all()
    assert s["crps"].mean() == pytest.approx(ref["crps"])
    assert np.sqrt((s["spread"] ** 2).mean()) == pytest.approx(ref["spread"])
    assert np.sqrt(((s["mean"] - s["obs"]) ** 2).mean()) == pytest.approx(ref["rmse_mean"])
    assert s["crps_oe"].notna().all() and set(s.columns) >= {"lat", "lon", "sigma_o", "below", "ties", "min", "max"}


def test_ranks_thresholds_and_a_missing_member():
    base = {"model": "ENS", "init": INIT, "lead_h": 24.0, "valid_time": INIT + pd.Timedelta("24h"), "sigma_o": np.nan}
    rain = [dict(base, station_id="A", variable="pr24", member=k, fcst=f, obs=0.0)
            for k, f in enumerate([0.0, 0.0, 2.0, 6.0, 12.0])]
    temp = [dict(base, station_id="B", variable="t2m", member=k, fcst=f, obs=1.5)
            for k, f in enumerate([1.0, 2.0, 3.0, 4.0])]  # member 4 missing
    s = summarise(pd.DataFrame(rain + temp)).set_index("station_id")
    a, b = s.loc["A"], s.loc["B"]
    assert (a["below"], a["ties"], a["n_members"]) == (0, 2, 5)
    assert (a["p_ge_1"], a["p_ge_5"], a["p_ge_10"], a["p_ge_25"]) == (0.6, 0.4, 0.2, 0.0)
    assert np.isnan(a["p_ge_15"]) and np.isnan(b["p_ge_1"])
    assert b["n_members"] == 4 and b["below"] == 1
    assert b["crps"] == pytest.approx(scores.crps_ensemble(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([1.5]))[0])
    assert np.isnan(b["crps_oe"])  # no observation error known


def test_grid_cells_are_cases_by_position():
    p = _pairs(np.random.default_rng(2), cases=3, members=5, var="tp").drop(columns="station_id")
    s = summarise(p, keys=CELL)
    assert len(s) == 3 and list(s["lon"]) == [0.0, 1.0, 2.0]


def test_dew_point_inverts_the_vapour_pressure():
    t, rh = np.array([-10.0, 5.0, 25.0]), np.array([40.0, 75.0, 100.0])
    td = V.dew_point(t, rh)
    e = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))  # specific_humidity's vapour pressure
    assert td[2] == pytest.approx(25.0) and e(td) == pytest.approx(rh / 100 * e(t))
    assert V.DERIVED["td2m"][0] == ("t2m", "rh2m")


def test_ensemble_tests_find_the_better_ensemble():
    rng = np.random.default_rng(3)
    rows = []
    for d in range(20):
        for model, crps in (("A", 1.0), ("B", 1.5)):
            for c in range(30):
                rows.append({"model": model, "network": "GHCNH", "variable": "t2m", "region": "Global",
                             "lead_h": 120.0, "init": INIT + pd.Timedelta(days=d), "crps": crps + rng.normal(0, 0.3)})
    t = significance.ensemble_tests(pd.DataFrame(rows)).set_index(["model", "reference"])
    assert t.loc[("A", "B"), "p"] < 0.01 and t.loc[("A", "B"), "runs"] == 20
    assert t.loc[("B", "A"), "p"] == pytest.approx(t.loc[("A", "B"), "p"])


def test_grib_control_and_perturbed_members(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    pytest.importorskip("cfgrib")
    from mverify.io.open import open_forecast

    p = tmp_path / "ens.grib2"
    with open(p, "wb") as fh:
        for number in (0, 1, 2):  # the control (cf) and two perturbed members (pf)
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set_key_vals(h, {"Ni": 36, "Nj": 19, "latitudeOfFirstGridPointInDegrees": 90,
                                           "longitudeOfFirstGridPointInDegrees": 0,
                                           "latitudeOfLastGridPointInDegrees": -90,
                                           "longitudeOfLastGridPointInDegrees": 350,
                                           "iDirectionIncrementInDegrees": 10, "jDirectionIncrementInDegrees": 10,
                                           "dataDate": int(INIT.strftime("%Y%m%d")), "dataTime": 0,
                                           "productDefinitionTemplateNumber": 1,
                                           "typeOfProcessedData": 3 if number == 0 else 4,
                                           "perturbationNumber": number, "numberOfForecastsInEnsemble": 3,
                                           "stepUnits": 1, "endStep": 24})
            eccodes.codes_set(h, "shortName", "2t")
            eccodes.codes_set_key_vals(h, {"typeOfFirstFixedSurface": 103, "scaledValueOfFirstFixedSurface": 2,
                                           "scaleFactorOfFirstFixedSurface": 0})
            eccodes.codes_set_values(h, np.full(36 * 19, 280.0 + number))
            eccodes.codes_write(h, fh)
            eccodes.codes_release(h)
    fields = sorted(open_forecast(p).fields, key=lambda f: f.member)
    assert [f.member for f in fields] == [0, 1, 2]
    assert [round(float(f.values().mean()), 2) for f in fields] == [6.85, 7.85, 8.85]
