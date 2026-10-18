"""Forecast files -> matched pairs -> scores, with known answers."""

import json

import numpy as np
import pandas as pd
import pytest

from mverify import report
from mverify.cli import main
from mverify.verify import verify


def test_perfect_model_scores_zero_after_height_correction(tmp_path, filled_store, forecast_ds):
    p = tmp_path / "perfect.nc"
    forecast_ds.to_netcdf(p)
    res = verify({"PERFECT": [p]}, filled_store, regions=["conus"], max_dz=None)
    t = res.table.set_index("lead_h")
    assert (t["n"] == 3).all()
    assert t["rmse"].max() == pytest.approx(0.0, abs=1e-9)
    # without the correction the error is the lapse rate times the station height
    raw = verify({"PERFECT": [p]}, filled_store, regions=["conus"], height_correction=False, max_dz=None)
    assert raw.table["bias"].iloc[0] == pytest.approx(0.0065 * np.mean([500, 300, 1600]))


def test_stations_far_from_the_model_terrain_are_left_out(tmp_path, filled_store, forecast_ds):
    # The model terrain is flat at 0 m; the CONUS stations sit at 500, 300 and 1600 m.
    forecast_ds.to_netcdf(tmp_path / "m.nc")
    pairs = verify({"M": [tmp_path / "m.nc"]}, filled_store, regions=["conus"]).pairs
    assert pairs.empty or (pairs["dz_m"].abs() <= 100).all()
    pairs = verify({"M": [tmp_path / "m.nc"]}, filled_store, regions=["conus"], max_dz=300).pairs
    t2m = pairs[pairs["variable"] == "t2m"]
    assert set(t2m["station_id"]) == {"GHCNH:B"}  # 300 m is the limit itself and stays in
    assert (t2m["dz_m"] == -300).all()


def test_models_are_compared_on_the_pairs_they_all_have(tmp_path, filled_store, forecast_ds):
    # Flat terrain keeps the 300 m station; terrain at 400 m keeps the 300 m and 500 m ones.
    forecast_ds.to_netcdf(tmp_path / "flat.nc")
    forecast_ds.assign(orog=forecast_ds["orog"] + 400).to_netcdf(tmp_path / "high.nc")
    models = {"FLAT": [tmp_path / "flat.nc"], "HIGH": [tmp_path / "high.nc"]}
    common = verify(models, filled_store, regions=["conus"], max_dz=300)
    assert set(common.pairs["station_id"]) == {"GHCNH:B"}
    assert (common.table["n"] == 1).all() and set(common.table["model"]) == {"FLAT", "HIGH"}
    own = verify(models, filled_store, regions=["conus"], max_dz=300, common=False).table
    assert own.groupby("model")["n"].max().to_dict() == {"FLAT": 1, "HIGH": 2}


def test_known_bias_and_the_same_answer_from_every_format(tmp_path, filled_store, forecast_ds):
    warm = forecast_ds.copy()
    warm["t2m"] = warm["t2m"] + 1.5
    warm["t2m"].attrs["units"] = "K"
    warm.to_netcdf(tmp_path / "warm.nc")
    warm.to_zarr(tmp_path / "warm.zarr")
    a = verify({"W": [tmp_path / "warm.nc"]}, filled_store, regions=["conus"], max_dz=None).table
    b = verify({"W": [tmp_path / "warm.zarr"]}, filled_store, regions=["conus"], max_dz=None).table
    assert a["bias"].to_numpy() == pytest.approx(1.5)
    pd.testing.assert_frame_equal(a, b)


def test_scorecard_json_shape(tmp_path, filled_store, forecast_ds):
    forecast_ds.to_netcdf(tmp_path / "m.nc")
    card = verify({"M": [tmp_path / "m.nc"]}, filled_store, regions=["conus", "global"], max_dz=None).scorecard()
    out = tmp_path / "card.json"
    report.write_json(card, out)
    card = json.loads(out.read_text())
    assert card["models"] == ["M"] and card["leads"] == [0, 6, 12]
    row = next(r for r in card["rows"] if r["key"] == "ghcnh:t2m:conus")
    assert row["unit"] == "degC" and len(row["scores"]["M"]["rmse"]) == 3 and row["scores"]["M"]["n"] == [3, 3, 3]


def test_cli_score(tmp_path, filled_store, forecast_ds, capsys):
    db = tmp_path / "obs.duckdb"
    filled_store.con.execute(f"ATTACH '{db}' AS f")
    filled_store.con.execute("COPY FROM DATABASE memory TO f")
    filled_store.con.execute("DETACH f")
    forecast_ds.to_netcdf(tmp_path / "gfs.nc")
    assert main(["score", str(tmp_path / "gfs.nc"), "--db", str(db), "--region", "conus", "--max-dz", "none",
                 "--out", str(tmp_path / "s.json")]) == 0
    assert "GFS" in capsys.readouterr().out
    assert json.loads((tmp_path / "s.json").read_text())["models"] == ["GFS"]
