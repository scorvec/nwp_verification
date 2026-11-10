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


def test_ghcnh_24_hour_precipitation_is_the_gauge_total():
    raw = ghcnh_frame([{"DATE": "2026-10-13T12:00:00", "precipitation_24_hour": 12.7}])
    out = ghcnh.parse(raw, "GHCNH:X").set_index("variable")["value"]
    assert out["pr24"] == 12.7


def metars(start, hours, wet=(), remarks="RMK AO2", minute=51):
    """Routine METARs each hour from ``start``; ``wet`` maps an hour's position to its amount in mm."""
    wet = dict(wet)
    rows = []
    for k, t in enumerate(pd.date_range(start, periods=hours, freq="h") + pd.Timedelta(minutes=minute)):
        group = f" P{round(wet[k] / 0.254):04d}" if k in wet else ""
        row = {"DATE": t.isoformat(), "temperature": 10.0, "temperature_Report_Type": "FM15",
               "REM": f"KXXX {t:%d%H%M}Z 27005KT 10SM FEW035 10/05 A3000 {remarks}{group}".strip()}
        if k in wet:
            row |= {"precipitation": wet[k], "precipitation_Report_Type": "FM15"}
        rows.append(row)
    return rows


@pytest.fixture(params=["python", "pyarrow"])
def strings(request):
    """Each string backend in turn: Arrow's comes with its own regular-expression engine."""
    with pd.option_context("mode.string_storage", request.param):
        yield


def totals(rows):
    out = ghcnh.parse(ghcnh_frame(rows), "GHCNH:X")
    return out[out["variable"] == "pr24"].set_index("valid_time")


def test_ghcnh_builds_24_hour_totals_from_the_routine_reports_of_a_station_that_measures():
    # 30 hourly reports from 06:51: the first closes the hour to 07Z, so a full day ends at 06Z and at 12Z
    rows = metars("2026-10-12T06:00", 30, wet={0: 2.5, 6: 5.1, 26: 1.3})
    # a special between two routine reports repeats a running total: not part of the sum
    rows.append({"DATE": "2026-10-12T12:58:00", "temperature": 10.0, "temperature_Report_Type": "FM16",
                 "precipitation": 4.0, "precipitation_Report_Type": "FM16",
                 "REM": "KXXX 121258Z RMK AO2 P0016"})
    out = totals(rows)
    assert out.index.tolist() == [pd.Timestamp("2026-10-13 06:00"), pd.Timestamp("2026-10-13 12:00")]
    assert out["value"].tolist() == [7.6, 6.4]  # 2.5 + 5.1, then 5.1 + 1.3 once the first hour has left
    assert set(out["source"]) == {"ghcnh:hourly"} and set(out["dt_min"]) == {-9} and not out["qc"].any()


def test_ghcnh_an_hour_without_an_amount_is_dry_only_where_the_gauge_is_known_to_work(strings):
    day = pd.Timestamp("2026-10-13 12:00")
    assert day in totals(metars("2026-10-12T06:00", 30, wet={0: 0.0})).index  # a trace proves the gauge
    assert totals(metars("2026-10-12T06:00", 30)).empty  # a station that never reports an amount
    assert totals(metars("2026-10-12T06:00", 30, wet={0: 2.5}, remarks="")).empty  # copies without remarks
    for spoiled in ("KXXX 130251Z 27005KT 10SM FEW035 10/05 A3000 RMK AO2 PNO",  # gauge out
                    "KXXX 130251Z 27005KT 3SM -RA OVC010 10/09 A3000 RMK AO2"):  # raining, no amount
        rows = metars("2026-10-12T06:00", 30, wet={0: 2.5})
        rows[20]["REM"] = spoiled
        assert day not in totals(rows).index
    rows = metars("2026-10-12T06:00", 30, wet={0: 2.5})
    del rows[20]  # an hour with no routine report
    assert day not in totals(rows).index
    # proof runs out: after 30 dry days without an amount the hours are unknown again
    long_dry = totals(metars("2026-09-01T00:00", 24 * 40, wet={0: 2.5}))
    assert long_dry.index.max() == pd.Timestamp("2026-10-01 00:00")


def test_ghcnh_hourly_feed_totals_need_every_hour():
    feed = [{"DATE": t.isoformat(), "precipitation": 0.5 if k == 2 else 0.0,
             "precipitation_Report_Type": "EnvCan"}
            for k, t in enumerate(pd.date_range("2026-10-12T01:00", periods=36, freq="h"))]
    ends = pd.to_datetime(["2026-10-13 00:00", "2026-10-13 06:00", "2026-10-13 12:00"])
    assert totals(feed)["value"].to_dict() == dict(zip(ends, [0.5, 0.0, 0.0]))
    # without 11Z only the day that starts at 13Z is whole
    assert totals(feed[:10] + feed[11:]).index.tolist() == [ends[2]]


def test_ghcnh_reported_total_stands_and_is_flagged_where_the_built_total_disagrees():
    def reported(value, code=np.nan):
        return {"DATE": "2026-10-13T12:00:00", "precipitation_24_hour": value,
                "precipitation_24_hour_Quality_Code": code, "precipitation_24_hour_Report_Type": "FM12"}

    hours = metars("2026-10-12T06:00", 30, wet={0: 0.3, 10: 12.7})
    day = pd.Timestamp("2026-10-13 12:00")
    agree = totals([*hours, reported(13.0)]).loc[day]
    assert (agree["value"], agree["qc"], agree["source"]) == (13.0, 0, "ghcnh:FM12")
    differ = totals([*hours, reported(2.0)]).loc[day]
    assert (differ["value"], differ["qc"]) == (2.0, qc.INCONSISTENT)
    # a total the provider flagged gives way to the built one
    replaced = totals([*hours, reported(2.0, "s")]).loc[day]
    assert (replaced["value"], replaced["qc"], replaced["source"]) == (12.7, 0, "ghcnh:hourly")


def test_ghcnh_a_large_total_through_hours_the_reports_show_as_dry_is_flagged(strings):
    day = pd.Timestamp("2026-10-13 12:00")

    def reports(rain_at=5):  # hourly from the 8th to the hour of the total, rain in one of them
        rows = metars("2026-10-08T00:00", 24 * 5 + 13)
        if rain_at is not None:
            at = pd.Timestamp(rows[rain_at]["DATE"])
            rows[rain_at]["REM"] = f"KXXX {at:%d%H%M}Z 27005KT 3SM -RA OVC010 10/09 A3000 RMK AO2"
        return rows

    def flag(rows, value=60.0, **amount):
        total = {"DATE": day.isoformat(), "precipitation_24_hour": value,
                 "precipitation_24_hour_Report_Type": "FM12", **amount}
        return totals([*rows, total]).loc[day, "qc"]

    assert flag(reports()) == qc.INCONSISTENT  # rain five days before: the reports are known to show it
    assert flag(reports(), value=20.0) == 0  # not a large total
    assert flag(reports(rain_at=120)) == 0  # rain in the hours of the total
    assert flag(reports(), precipitation_6_hour=4.0) == 0  # an amount for part of the day
    assert flag(reports(rain_at=None)) == 0  # a station whose reports never show precipitation
    assert flag(reports()[:108] + reports()[116:]) == 0  # reports for 16 of the 24 hours
    blind = reports()
    for row in blind[100:]:
        row["REM"] = row["REM"].replace("10SM", "10SM //")  # an automatic station without a weather sensor
    assert flag(blind) == 0


def test_ghcnh_a_failed_large_total_outside_the_window_keeps_the_station_out(monkeypatch, store, tmp_path):
    sl = pd.DataFrame({"GHCN_ID": ["X1"], "NAME": ["x"], "LATITUDE": [40.0], "LONGITUDE": [-100.0],
                       "ELEVATION": [500.0], "WMO_ID": [None], "ICAO": [None], "ISO_CODE": ["US"]})
    monkeypatch.setattr(ghcnh, "station_list", lambda cache_dir: sl)
    for first, kept in ((500.0, []), (5.0, [3.0])):  # above the range of the variable, then a sound total
        raw = ghcnh_frame([{"DATE": f"2026-10-{d:02d}T12:00:00", "precipitation_24_hour": v}
                           for d, v in ((2, first), (5, 3.0))])
        monkeypatch.setattr(ghcnh, "_fetch", lambda g, y, raw=raw: raw)
        monkeypatch.setattr(ghcnh.http, "etag", lambda url, first=first: f"e{first}")
        ghcnh.ingest(store, [2026], since="2026-10-04", until="2026-10-06", cache_dir=tmp_path,
                     progress=lambda m: None)
        obs = store.fetch_obs(["pr24"], pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-31"))
        assert obs["value"].tolist() == kept


def test_ghcnh_built_totals_reach_the_store_for_the_window(monkeypatch, store, tmp_path):
    sl = pd.DataFrame({"GHCN_ID": ["X1"], "NAME": ["x"], "LATITUDE": [40.0], "LONGITUDE": [-100.0],
                       "ELEVATION": [500.0], "WMO_ID": [None], "ICAO": ["KXXX"], "ISO_CODE": ["US"]})
    raw = ghcnh_frame(metars("2026-10-01T00:00", 24 * 8, wet={0: 2.5, 100: 5.1}))
    monkeypatch.setattr(ghcnh, "station_list", lambda cache_dir: sl)
    monkeypatch.setattr(ghcnh.http, "etag", lambda url: "e1")
    monkeypatch.setattr(ghcnh, "_fetch", lambda g, y: raw)
    ghcnh.ingest(store, [2026], since="2026-10-05", until="2026-10-06", cache_dir=tmp_path,
                 progress=lambda m: None)
    obs = store.fetch_obs(["pr24"], pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-31"))
    # the amount fell in the hour to 05Z on the 5th: the proof from the 1st, outside the window, still counts
    assert dict(zip(obs["valid_time"], obs["value"])) == dict(zip(
        pd.to_datetime(["2026-10-05 00:00", "2026-10-05 06:00", "2026-10-05 12:00", "2026-10-05 18:00",
                        "2026-10-06 00:00"]), [0.0, 5.1, 5.1, 5.1, 5.1]))


def test_ghcnh_window_stores_only_its_hours_of_each_station_year(monkeypatch, store, tmp_path):
    sl = pd.DataFrame({"GHCN_ID": ["X1"], "NAME": ["x"], "LATITUDE": [40.0], "LONGITUDE": [-100.0],
                       "ELEVATION": [500.0], "WMO_ID": [None], "ICAO": [None], "ISO_CODE": ["US"]})
    raw = ghcnh_frame([{"DATE": f"2026-10-{d:02d}T12:00:00", "temperature": 10.0 + d} for d in (1, 5, 9)])
    monkeypatch.setattr(ghcnh, "station_list", lambda cache_dir: sl)
    monkeypatch.setattr(ghcnh.http, "etag", lambda url: "e1")
    monkeypatch.setattr(ghcnh, "_fetch", lambda g, y: raw)
    ghcnh.ingest(store, [2026], since="2026-10-04", until="2026-10-06", cache_dir=tmp_path,
                 progress=lambda m: None)
    obs = store.fetch_obs(["t2m"], pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-31"))
    assert obs["valid_time"].tolist() == [pd.Timestamp("2026-10-05 12:00")]
    # the version logged names the window, so another window reads the file again
    assert store.ingested_version("ghcnh", "X1/2026") == "e1@2026-10-04..2026-10-06"
