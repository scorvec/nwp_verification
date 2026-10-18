import duckdb
import numpy as np
import pandas as pd
import pytest

from mverify.db import qc
from mverify.db.store import ObsStore, SchemaError, ValidationError


def obs(**kw):
    row = {"station_id": "GHCNH:A", "valid_time": "2026-10-13 00:00", "variable": "t2m", "value": 10.0,
           "source": "test"}
    row.update(kw)
    return pd.DataFrame([row])


def test_reference_tables_describe_the_catalogue(store):
    v = store.con.execute("SELECT name, unit FROM variables WHERE name IN ('t2m', 'z500') ORDER BY 1").fetchall()
    assert v == [("t2m", "degC"), ("z500", "m")]
    bits = store.con.execute("SELECT sum(bit) FROM qc_flags").fetchone()[0]
    assert bits == sum(qc.FLAGS)


def test_create_is_idempotent(tmp_path):
    p = tmp_path / "o.duckdb"
    ObsStore.open(p).close()
    with ObsStore.open(p) as s:
        assert s.con.execute("SELECT count(*) FROM variables").fetchone()[0] > 10


def test_schema_version_mismatch_is_refused(tmp_path):
    p = tmp_path / "o.duckdb"
    with ObsStore.open(p) as s:
        s.con.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(SchemaError):
        ObsStore.open(p)
    with pytest.raises(SchemaError):
        ObsStore.open(p, read_only=True)


@pytest.mark.parametrize("bad, message", [
    ({"variable": "nonsense"}, "outside the catalogue"),
    ({"value": np.nan}, "non-finite"),
    ({"valid_time": "2026-10-13 00:51"}, "whole hour"),
    ({"dt_min": 75}, "dt_min"),
    ({"qc": 1 << 12}, "qc bits"),
])
def test_bad_rows_are_rejected_with_a_reason(store, stations, bad, message):
    store.upsert_stations(stations)
    with pytest.raises(ValidationError, match=message):
        store.upsert_obs(obs(**bad))


def test_duplicate_keys_in_a_batch_are_rejected(store, stations):
    store.upsert_stations(stations)
    with pytest.raises(ValidationError, match="repeat"):
        store.upsert_obs(pd.concat([obs(), obs(value=11.0)]))


def test_unknown_station_is_rejected(store):
    with pytest.raises(ValidationError, match="not in stations"):
        store.upsert_obs(obs())


def test_station_validation(store):
    good = {"station_id": "RAOB:1", "network": "RAOB", "lat": 10, "lon": 200}
    store.upsert_stations(pd.DataFrame([good]))
    assert store.con.execute("SELECT lon FROM stations").fetchone()[0] == pytest.approx(-160)
    with pytest.raises(ValidationError, match="prefixed"):
        store.upsert_stations(pd.DataFrame([{**good, "station_id": "X:1"}]))
    with pytest.raises(ValidationError, match="elevation"):
        store.upsert_stations(pd.DataFrame([{**good, "elev_m": -999.9}]))


def test_upsert_replaces_same_key(store, stations):
    store.upsert_stations(stations)
    store.upsert_obs(obs(value=10.0))
    store.upsert_obs(obs(value=12.0))
    assert store.con.execute("SELECT count(*), max(value) FROM obs").fetchone() == (1, 12.0)


def test_replace_drops_withdrawn_rows_inside_the_window_only(store, stations):
    store.upsert_stations(stations)
    store.upsert_obs(pd.concat([obs(valid_time="2026-10-13 00:00"), obs(valid_time="2026-10-13 01:00"),
                                obs(valid_time="2026-10-14 00:00")]))
    store.replace_obs(obs(valid_time="2026-10-13 00:00", value=5.0), ["GHCNH:A"],
                      "2026-10-13 00:00", "2026-10-13 23:00")
    rows = store.con.execute("SELECT valid_time, value FROM obs ORDER BY 1").fetchall()
    assert [(str(t), v) for t, v in rows] == [("2026-10-13 00:00:00", 5.0), ("2026-10-14 00:00:00", 10.0)]
    with pytest.raises(ValidationError, match="outside the replace window"):
        store.replace_obs(obs(valid_time="2026-10-15 00:00"), ["GHCNH:A"], "2026-10-13", "2026-10-13 23:00")


def test_ingest_batch_is_atomic(store, stations):
    st = stations.iloc[:1]
    with pytest.raises(ValidationError):
        store.ingest_batch("t", [("k", "v1", 1)], st, obs(value=np.inf), (["GHCNH:A"], "2026-10-13", "2026-10-14"))
    assert store.con.execute("SELECT count(*) FROM stations").fetchone()[0] == 0
    assert store.ingested_version("t", "k") is None
    store.ingest_batch("t", [("k", "v1", 1)], st, obs(), (["GHCNH:A"], "2026-10-13", "2026-10-14"))
    assert store.ingested_version("t", "k") == "v1"
    assert store.check_integrity() == dict.fromkeys(store.check_integrity(), 0)


def test_integrity_audit_finds_problems_written_behind_the_stores_back(store, stations):
    store.upsert_stations(stations)
    store.upsert_obs(obs())
    store.con.execute("INSERT INTO obs (station_id, valid_time, variable, value, source) "
                      "VALUES ('GHCNH:A', '2026-10-13 00:00', 't2m', 99, 'x'), ('GHCNH:Z', '2026-10-13', 'zz', 1, 'x')")
    problems = store.check_integrity()
    assert problems["duplicate_keys"] == 1
    assert problems["orphan_station"] == 1
    assert problems["unknown_variable"] == 1
    assert problems["range_not_flagged"] == 1


def test_fetch_obs_filters(store, stations):
    store.upsert_stations(stations)
    store.upsert_obs(pd.concat([obs(), obs(station_id="GHCNH:B"), obs(station_id="GHCNH:C", qc=qc.SPIKE),
                                obs(station_id="RAOB:72520", variable="z500", value=5600.0)]))
    got = store.fetch_obs(["t2m"], "2026-10-13", "2026-10-13 01:00")
    assert sorted(got["station_id"]) == ["GHCNH:A", "GHCNH:B"]
    got = store.fetch_obs(["t2m"], "2026-10-13", "2026-10-13", clean_only=False, bbox=(-110, 35, -99, 45))
    assert sorted(got["station_id"]) == ["GHCNH:A", "GHCNH:C"]
    got = store.fetch_obs(["t2m", "z500"], "2026-10-13", "2026-10-13", networks=["RAOB"])
    assert got["station_id"].tolist() == ["RAOB:72520"]


def test_fetch_obs_bbox_across_the_dateline(store):
    store.upsert_stations(pd.DataFrame([
        {"station_id": "RAOB:E", "network": "RAOB", "lat": 50, "lon": 175},
        {"station_id": "RAOB:W", "network": "RAOB", "lat": 50, "lon": -175},
        {"station_id": "RAOB:F", "network": "RAOB", "lat": 50, "lon": 0}]))
    store.upsert_obs(pd.concat([obs(station_id=s, variable="t500", value=-20.0) for s in ("RAOB:E", "RAOB:W", "RAOB:F")]))
    got = store.fetch_obs(["t500"], "2026-10-13", "2026-10-13", bbox=(170, 40, -170, 60))
    assert sorted(got["station_id"]) == ["RAOB:E", "RAOB:W"]


def test_snapshot_round_trip(store, stations, tmp_path):
    store.upsert_stations(stations)
    store.upsert_obs(pd.concat([obs(), obs(valid_time="2026-11-01 00:00"),
                                obs(station_id="RAOB:72520", variable="z500", value=5600.0)]))
    store.export_parquet(tmp_path / "snap")
    assert (tmp_path / "snap" / "obs" / "variable=t2m" / "year=2026" / "month=11").is_dir()
    with ObsStore.snapshot(str(tmp_path / "snap")) as snap:
        a = store.fetch_obs(["t2m", "z500"], "2026-01-01", "2026-12-31")
        b = snap.fetch_obs(["t2m", "z500"], "2026-01-01", "2026-12-31")
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        with pytest.raises(PermissionError):
            snap.upsert_obs(obs())


def test_read_only_store_refuses_writes(tmp_path, stations):
    p = tmp_path / "o.duckdb"
    ObsStore.open(p).close()
    with ObsStore.open(p, read_only=True) as s:
        with pytest.raises(PermissionError):
            s.upsert_stations(stations)
        with pytest.raises(duckdb.Error):
            s.con.execute("DELETE FROM obs")
