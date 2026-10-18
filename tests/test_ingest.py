import numpy as np
import pandas as pd
import pytest

from mverify.db import qc
from mverify.ingest import ghcnh, soundings


def ghcnh_frame(rows):
    return pd.DataFrame([{c: r.get(c, np.nan) for c in ghcnh.COLUMNS} for r in rows])


def test_ghcnh_picks_the_report_nearest_the_hour_and_prefers_metar_on_ties():
    raw = ghcnh_frame([
        {"DATE": "2026-10-13T11:51:00", "temperature": 10.0, "temperature_Report_Type": "FM15"},
        {"DATE": "2026-10-13T12:00:00", "temperature": 11.0, "temperature_Report_Type": "FM12"},
        {"DATE": "2026-10-13T12:00:00", "temperature": 11.5, "temperature_Report_Type": "FM15"},
        {"DATE": "2026-10-13T12:40:00", "temperature": 99.0, "temperature_Report_Type": "FM16"},
    ])
    out = ghcnh.parse(raw, "GHCNH:X")
    t = out[out["variable"] == "t2m"].set_index("valid_time")["value"]
    assert t[pd.Timestamp("2026-10-13 12:00")] == 11.5
    assert t[pd.Timestamp("2026-10-13 13:00")] == 99.0  # 12:40 is nearest 13:00, QC handles the value


def test_ghcnh_upstream_flags():
    raw = ghcnh_frame([
        {"DATE": "2026-10-13T00:00:00", "temperature": 10.0, "temperature_Quality_Code": "s"},
        {"DATE": "2026-10-13T01:00:00", "temperature": 10.0, "temperature_Quality_Code": "1"},
        {"DATE": "2026-10-13T02:00:00", "temperature": 10.0, "temperature_Quality_Code": "7"},
        {"DATE": "2026-10-13T03:00:00", "temperature": 10.0, "temperature_Quality_Code": "A"},
    ])
    out = ghcnh.parse(raw, "GHCNH:X")
    assert out[out["variable"] == "t2m"]["qc"].tolist() == [qc.UPSTREAM, 0, qc.UPSTREAM, 0]


def test_ghcnh_wind_components_and_calm():
    raw = ghcnh_frame([
        {"DATE": "2026-10-13T00:00:00", "wind_speed": 10.0, "wind_direction": "270"},
        {"DATE": "2026-10-13T01:00:00", "wind_speed": 0.0, "wind_direction": "000"},
        {"DATE": "2026-10-13T02:00:00", "wind_speed": 5.0, "wind_direction": np.nan},
    ])
    out = ghcnh.parse(raw, "GHCNH:X").pivot(index="valid_time", columns="variable", values="value")
    assert out["u10"].iloc[0] == pytest.approx(10.0) and out["v10"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["u10"].iloc[1] == 0 and out["v10"].iloc[1] == 0
    assert np.isnan(out["u10"].iloc[2]) and out["ws10"].iloc[2] == 5.0  # variable direction: speed only


def test_ghcnh_station_metadata_ignores_missing_sentinels():
    meta = pd.Series({"LATITUDE": 49.2, "LONGITUDE": -123.2, "ELEVATION": 4.0, "NAME": " YVR ", "WMO_ID": "71892",
                      "ICAO": "CYVR", "ISO_CODE": "CA"})
    raw = ghcnh_frame([{"DATE": "2026-10-13T00:00:00", "LATITUDE": 49.195, "LONGITUDE": -123.184,
                        "ELEVATION": -999.9}])
    row = ghcnh._station_row("CAN01108395", meta, raw)
    assert row["elev_m"] == 4.0 and row["lat"] == 49.195 and row["name"] == "YVR"


def test_sounding_mandatory_levels():
    prof = pd.DataFrame({
        "pressure_hPa": [1000, 880, 830, 500, 400], "geopotential height_m": [100, 990, 1950, 5600, 7300],
        "temperature_C": [15, 9, 3, -20, -30], "wind direction_degree": [270, 270, 270, 180, 180],
        "wind speed_m/s": [5, 5, 5, 10, 10]})
    lev = soundings.mandatory_levels(prof)
    assert lev["z500"] == 5600 and lev["v500"] == pytest.approx(10.0)
    w = np.log(850 / 880) / np.log(830 / 880)
    assert lev["t850"] == pytest.approx(9 + w * (3 - 9))
    assert "u850" not in lev  # 50 hPa between wind levels is too far to interpolate
    assert "z700" not in lev  # 300 hPa gap


def test_sounding_positions_must_agree_with_the_wmo_list():
    ref = {"31770": (48.97, 140.3), "47000": (40.0, 179.8)}
    assert not soundings.position_confirmed("31770", 0.0, 140.3, ref)
    assert soundings.position_confirmed("31770", 48.9, 140.2, ref)
    assert soundings.position_confirmed("47000", 40.0, -179.9, ref)  # across the dateline
    assert soundings.position_confirmed("99999", 10.0, 10.0, ref)
