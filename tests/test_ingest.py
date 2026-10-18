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
    lev = soundings.mandatory_levels(prof)["value"]
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


def _ascent(**cols):
    """A sounding rising through 1000-200 hPa with heights from a 7.5 km scale height."""
    p = np.array([1000, 925, 850, 700, 500, 300, 250, 200], float)
    z = 7500 * np.log(1000 / p)
    prof = pd.DataFrame({"pressure_hPa": p, "geopotential height_m": z, "temperature_C": 15 - 0.0065 * z,
                         "wind direction_degree": 270.0, "wind speed_m/s": 10.0})
    for k, v in cols.items():
        prof[k] = v
    return prof


def test_drift_from_winds_and_the_standard_ascent_rate():
    # A 10 m/s westerly, no per-level times: launch 45 min before the hour, rising at 5 m/s.
    lev = soundings.mandatory_levels(_ascent(), pd.Timestamp("2026-10-13 12:00"), 0.0, 10.0)
    z500 = 7500 * np.log(2)
    flight = z500 / soundings.ASCENT_RATE
    assert lev.loc["z500", "dt_min"] == round(-45 + flight / 60)
    assert lev.loc["z500", "lat"] == pytest.approx(0.0, abs=1e-9)
    assert lev.loc["z500", "lon"] == pytest.approx(10.0 + np.rad2deg(10.0 * flight / soundings.EARTH_RADIUS))
    assert lev.loc["z1000", "lon"] == pytest.approx(10.0)


def test_drift_uses_reported_times_and_positions():
    hour = pd.Timestamp("2026-10-13 00:00")
    times = hour + pd.to_timedelta(np.linspace(-50, -10, 8), unit="min")
    lev = soundings.mandatory_levels(_ascent(time=times.astype(str)), hour, 0.0, 10.0)
    assert lev.loc["z200", "dt_min"] == -10 and lev.loc["z1000", "dt_min"] == -50
    gps = _ascent(time=times.astype(str), latitude=np.linspace(0, 0.4, 8), longitude=np.linspace(10, 9.8, 8))
    lev = soundings.mandatory_levels(gps, hour, 0.0, 10.0)  # the reported track wins over the winds
    assert lev.loc["z200", ["lat", "lon"]].tolist() == pytest.approx([0.4, 9.8])


def test_a_time_column_holding_only_the_launch_time_is_not_a_flight_time():
    hour = pd.Timestamp("2026-10-13 12:00")
    lev = soundings.mandatory_levels(_ascent(time=str(hour)), hour, 0.0, 10.0)
    assert lev.loc["z200", "dt_min"] > lev.loc["z1000", "dt_min"]  # fell back to the ascent rate


def test_drift_across_the_dateline():
    lev = soundings.mandatory_levels(_ascent(), pd.Timestamp("2026-10-13 12:00"), 0.0, 179.9)
    assert -180 <= lev.loc["z200", "lon"] < -179.5


def test_downloads_retry_a_body_cut_off_mid_transfer(monkeypatch):
    import http.client

    from mverify.ingest import http as H

    class Body:
        def __init__(self, fail):
            self.fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            if self.fail:
                raise http.client.IncompleteRead(b"x", 10)
            return b"whole"

    calls = []
    monkeypatch.setattr(H.urllib.request, "urlopen", lambda url, timeout: calls.append(url) or Body(len(calls) == 1))
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    assert H.get("https://example.invalid/f") == b"whole" and len(calls) == 2


def test_earthdata_login_follows_the_data_server_redirect(monkeypatch):
    import base64
    import email.message
    import io
    import urllib.parse
    import urllib.request
    import urllib.response

    from mverify.ingest import http as H

    seen = []

    class Servers(urllib.request.BaseHandler):
        """GES DISC redirects to Earthdata, which checks the login and sends a code back for a session cookie."""
        handler_order = 100  # ahead of the real HTTPS handler

        def https_open(self, req):
            host, auth = urllib.parse.urlsplit(req.full_url).hostname, req.get_header("Authorization")
            seen.append((host, auth is not None))
            headers, body, code = email.message.Message(), b"", 200
            if host == "data.example.gov" and "session=1" in (req.get_header("Cookie") or ""):
                body = b"precip"
            elif host == "data.example.gov" and "code=" in req.full_url:  # the server's own login callback
                code, headers["Location"] = 302, "https://data.example.gov/f.nc4"
                headers["Set-Cookie"] = "session=1; Path=/"
            elif host == "data.example.gov":
                code, headers["Location"] = 302, f"https://{H.EARTHDATA}/oauth/authorize?app=gesdisc"
            elif auth == "Basic " + base64.b64encode(b"shawn:right").decode():
                code, headers["Location"] = 302, "https://data.example.gov/login?code=abc"
            else:
                code, headers["WWW-Authenticate"] = 401, 'Basic realm="Please enter your Earthdata Login"'
                body = b"<p>Invalid username or password</p>"
            r = urllib.response.addinfourl(io.BytesIO(body), headers, req.full_url, code)
            r.msg = "test"
            return r

    def opener():
        o = H.earthdata_opener()
        o.add_handler(Servers())
        return o

    monkeypatch.setenv("EARTHDATA_USERNAME", "shawn")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "right\n")  # pasted secrets end in a newline
    assert H.get("https://data.example.gov/f.nc4", opener=opener()) == b"precip"
    assert seen == [("data.example.gov", False), (H.EARTHDATA, True),  # no 401 round trip
                    ("data.example.gov", False), ("data.example.gov", False)]  # the password stays at Earthdata

    monkeypatch.setenv("EARTHDATA_PASSWORD", "wrong")
    with pytest.raises(PermissionError, match="Invalid username or password"):
        H.get("https://data.example.gov/f.nc4", opener=opener())


def test_a_station_year_without_usable_reports_passes_qc():
    raw = pd.DataFrame({c: pd.Series([], dtype=object) for c in ghcnh.COLUMNS})
    assert qc.run_surface(ghcnh.parse(raw, "GHCNH:X")).empty


def test_a_limited_station_sample_is_spread_over_the_map():
    sl = pd.DataFrame({"GHCN_ID": ["AA1", "AA2", "AA3", "ZZ1"], "LATITUDE": [10.1, 10.5, 10.9, 50.0],
                       "LONGITUDE": [20.1, 20.3, 20.5, 5.0], "WMO_ID": ["1", None, "3", None],
                       "ICAO": ["A", None, "C", None]})
    # one per 2-degree box first (the WMO/ICAO station in a box before the others), then the rest
    assert ghcnh.select(sl, 2)["GHCN_ID"].tolist() == ["AA1", "ZZ1"]
    assert ghcnh.select(sl, 3)["GHCN_ID"].tolist() == ["AA1", "ZZ1", "AA3"]


def test_ghcnh_station_level_pressure_is_surface_pressure():
    raw = ghcnh_frame([{"DATE": "2026-10-13T12:00:00", "station_level_pressure": 951.3, "sea_level_pressure": 1012.0}])
    out = ghcnh.parse(raw, "GHCNH:X").set_index("variable")["value"]
    assert out["sp"] == 951.3 and out["msl"] == 1012.0
