"""Upper-air profiles: IGRA2 ascents, humidity, and scoring many levels from pair files."""

import numpy as np
import pandas as pd
import pytest

from mverify import variables as V
from mverify.ingest import igra
from mverify.match import PAIR_COLUMNS
from mverify.verify import common_sample, score_pair_files, score_pairs, variable_group


def _level(p_pa, z, t10, dpdp10, wd, ws10, etime=-9999):
    def f(x, w):
        return f"{x:>{w}d}"
    return (f"21 {f(etime, 5)} {f(p_pa, 6)} {f(z, 5)} {f(t10, 5)} {f(-9999, 5)} {f(dpdp10, 5)} "
            f"{f(wd, 5)} {f(ws10, 5)}")


def _header(sid, y, m, d, hh, rel, n, lat, lon):
    return f"#{sid} {y:4d} {m:02d} {d:02d} {hh:02d} {rel} {n:4d} ncdc-gts ncdc-gts {lat:7d} {lon:8d}"


def test_igra_lines_are_read_by_column():
    lv = [_level(100000, 110, 150, 20, 270, 50, etime=0), _level(85000, 1500, 50, 30, 270, 100, etime=500),
          _level(50000, 5600, -200, -9999, 250, 200, etime=2730)]
    text = "\n".join([_header("USM00072520", 2026, 10, 17, 12, "1102", 3, 405167, -802167), *lv,
                      _header("USM00072520", 2026, 10, 18, 0, "2302", 0, 405167, -802167)])
    (hour, lat, lon, prof), = igra.parse(text)  # the empty second ascent has no levels to keep
    assert hour == pd.Timestamp("2026-10-17 12:00") and (lat, lon) == (40.5167, -80.2167)
    assert prof["p"].tolist() == [1000, 850, 500] and prof["t"].tolist() == [15.0, 5.0, -20.0]
    assert prof["td"].iloc[1] == pytest.approx(2.0) and np.isnan(prof["td"].iloc[2])  # dew point = T - depression
    assert prof["ws"].tolist() == [5.0, 10.0, 20.0]
    assert prof["time"].iloc[2] == pd.Timestamp("2026-10-17 11:29:30")  # launch 11:02 + 27 min 30 s
    assert igra.parse(text, since="2026-10-18") == []


def test_igra_launch_before_midnight_belongs_to_the_previous_day():
    text = "\n".join([_header("USM00072520", 2026, 10, 18, 0, "2302", 1, 405167, -802167),
                      _level(100000, 110, 150, 20, 270, 50, etime=100)])
    (_, _, _, prof), = igra.parse(text)
    assert prof["time"].iloc[0] == pd.Timestamp("2026-10-17 23:03")


def test_igra_station_ids_match_the_wmo_index():
    assert igra.station_id("USM00072520") == "RAOB:72520"
    assert igra.station_id("USW00003160") == "RAOB:USW00003160"  # not a WMO station


def test_specific_humidity_from_dew_point():
    # Saturated at 20 C and 1000 hPa: e = 23.4 hPa, q = 14.7 g/kg (standard tables).
    assert V.specific_humidity(20.0, 1000.0) == pytest.approx(14.7, abs=0.1)
    q = V.CATALOG["q850"]
    assert V.to_canonical(np.array([0.008]), "kg kg**-1", q)[0] == pytest.approx(8.0)
    assert V.to_canonical(np.array([8.0]), "g kg-1", q)[0] == 8.0
    assert "q200" not in V.CATALOG and "t100" in V.CATALOG and "z400" in V.CATALOG


def test_variables_scored_together():
    assert {variable_group(v) for v in ("z500", "t500", "u500", "v500", "q500")} == {"500"}
    assert variable_group("u10") == variable_group("v10") == "wind10"
    assert variable_group("t2m") == "t2m"


def _upper_pairs(seed=0, runs=6, stations=40):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(runs):
        init = pd.Timestamp("2026-10-01") + pd.Timedelta(days=k)
        for var in ("t500", "u500", "v500", "z850"):
            obs = rng.normal(0, 5, stations)
            for model, sd in (("A", 1.0), ("B", 1.5)):
                n = stations - (3 if model == "B" else 0)  # B misses three stations
                rows.append(pd.DataFrame({
                    "model": model, "init": init, "lead_h": 48.0, "member": np.nan,
                    "valid_time": init + pd.Timedelta("48h"), "station_id": [f"RAOB:{i}" for i in range(n)],
                    "network": "RAOB", "lat": 40.0, "lon": -100.0, "variable": var,
                    "fcst": obs[:n] + rng.normal(0, sd, n), "obs": obs[:n], "dz_m": np.nan, "dt_min": 0,
                    "sensor_h_m": np.nan, "land_frac": np.nan, "sigma_o": 1.0}))
    return pd.concat(rows, ignore_index=True)[PAIR_COLUMNS]


def test_pair_files_score_like_one_sample(tmp_path):
    pairs = _upper_pairs()
    for k, (init, g) in enumerate(pairs.groupby("init")):
        g.to_parquet(tmp_path / f"{k}.parquet")
    files = score_pair_files(sorted(tmp_path.glob("*.parquet")), regions=["conus"])
    whole = score_pairs(common_sample(pairs), regions=["conus"])
    by = ["model", "variable", "lead_h"]
    a, b = files.table.set_index(by).sort_index(), whole.table.set_index(by).sort_index()
    assert set(a.index) == set(b.index) and "wind500" in a.index.get_level_values("variable")
    assert np.allclose(a["rmse"], b.loc[a.index, "rmse"], equal_nan=True)
    assert (a["n"] == b.loc[a.index, "n"]).all()
    t = files.tests.set_index(["variable", "model", "reference"])
    assert t.loc[("t500", "A", "B"), "p"] < 0.05 and t.loc[("t500", "A", "B"), "runs"] == 6
    assert files.inits["A"][0] == "2026-10-01T00" and files.pairs.empty


def test_igra_ingest_writes_each_changed_file_once(store, monkeypatch):
    p = [100000, 92500, 85000, 70000, 50000]
    z = [110, 780, 1500, 3050, 5620]
    lv = [_level(pp, zz, 150 - 60 * k, 30, 270, 100, etime=200 * k) for k, (pp, zz) in enumerate(zip(p, z))]
    text = "\n".join([_header("USM00072520", 2026, 10, 17, 12, "1102", len(lv), 405167, -802167), *lv])
    sl = pd.DataFrame({"id": ["USM00072520"], "lat": [40.5167], "lon": [-80.2167], "elev_m": [360.0],
                       "name": ["PITTSBURGH"], "last_year": [2026]})
    files = {"USM00072520": ("USM00072520-data-beg2026.txt.zip", "2026-10-17 21:39/2815389")}
    monkeypatch.setattr(igra, "station_list", lambda cache: sl)
    monkeypatch.setattr(igra, "listing", lambda archive: files)
    monkeypatch.setattr(igra, "_read", lambda archive, name: text)
    n = igra.ingest(store, since="2026-10-01", progress=lambda *a: None)
    obs = store.con.execute("SELECT variable, value FROM obs WHERE station_id = 'RAOB:72520'").df()
    assert n == len(obs) and {"z500", "t850", "u700", "q850"} <= set(obs["variable"])
    assert obs.set_index("variable").loc["q850", "value"] == pytest.approx(V.specific_humidity(0.0, 850), rel=1e-6)  # 3 C, depression 3 C
    assert igra.ingest(store, since="2026-10-01", progress=lambda *a: None) == 0  # unchanged file: not read again
