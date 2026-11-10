import numpy as np
import pandas as pd

from mverify.db import qc


def series(var, values, start="2026-10-13", station="GHCNH:A", freq="h"):
    t = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"station_id": station, "valid_time": t, "variable": var, "value": values, "qc": 0})


def test_range():
    df = series("t2m", [10.0, 75.0, -95.0])
    assert qc.range_check(df).tolist() == [0, qc.RANGE, qc.RANGE]


def test_spike_flags_only_isolated_jumps():
    df = series("t2m", [10, 10.5, 25, 11, 11.5, 30, 30.5, 31])
    flags = qc.spike_check(df)
    assert flags.tolist() == [0, 0, qc.SPIKE, 0, 0, 0, 0, 0]  # the step at index 5 persists: real change


def test_spike_needs_hourly_neighbours():
    df = series("t2m", [10, 25, 11])
    df.loc[2, "valid_time"] += pd.Timedelta("5h")
    assert qc.spike_check(df).sum() == 0


def test_stuck_sensor():
    vals = [12.3] * 13 + [12.0, 12.5]
    flags = qc.stuck_check(series("t2m", vals))
    assert (flags[:13] == qc.STUCK).all() and (flags[13:] == 0).all()


def test_stuck_runs_do_not_span_stations():
    df = pd.concat([series("t2m", [5.0] * 8), series("t2m", [5.0] * 8, station="GHCNH:B")], ignore_index=True)
    assert qc.stuck_check(df).sum() == 0


def test_dew_point_above_temperature():
    df = pd.concat([series("t2m", [10.0, 10.0]), series("td2m", [9.0, 12.0])], ignore_index=True)
    assert qc.dewpoint_check(df).tolist() == [0, 0, 0, qc.INCONSISTENT]


def test_run_surface_keeps_upstream_bits():
    df = series("t2m", [10.0, 99.0])
    df.loc[0, "qc"] = qc.UPSTREAM
    assert qc.run_surface(df)["qc"].tolist() == [qc.UPSTREAM, qc.RANGE]


def test_a_large_gauge_total_must_show_in_the_totals_12_hours_either_side():
    df = series("pr24", [0.0, 60.0, 0.0, 60.0, 55.0, 2.0], freq="12h")
    assert qc.total_spike_check(df).tolist() == [0, qc.SPIKE, 0, 0, 0, 0]  # the second 60 shows 12 h on
    assert qc.total_spike_check(series("pr24", [0.0, 20.0, 0.0], freq="12h")).sum() == 0  # not a large total
    assert qc.total_spike_check(series("pr24", [0.0, 60.0], freq="12h")).sum() == 0  # no total 12 h after
    df.loc[0, "qc"] = qc.UPSTREAM  # a flagged total is no neighbour
    assert qc.total_spike_check(df).sum() == 0


def test_a_false_low_gauge_total_between_true_ones_takes_the_flag():
    values = [0.0] * 14 + [30.0] * 24 + [0.0] * 4
    values[26] = 0.0  # a shorter period's amount, 12 h after the day's rain first shows
    flags = qc.total_spike_check(series("pr24", values))
    assert flags[26] == qc.SPIKE and flags.sum() == qc.SPIKE


def test_a_failed_large_gauge_total_makes_the_other_totals_of_the_station_suspect():
    small = series("pr24", [3.0, 12.0], freq="12h", station="GHCNH:B")
    small.loc[0, "qc"] = qc.INCONSISTENT  # a failed total below the large ones does not count
    df = pd.concat([series("pr24", [0.0, 60.0, 0.0, 5.0], freq="12h"), series("t2m", [10.0, 11.0]), small,
                    series("pr24", [450.0, 1.0], freq="12h", station="GHCNH:C")], ignore_index=True)
    out = qc.run_surface(df).groupby(["station_id", "variable"], sort=False)["qc"].apply(list)
    assert out["GHCNH:A", "pr24"] == [qc.SUSPECT, qc.SPIKE | qc.SUSPECT, qc.SUSPECT, qc.SUSPECT]
    assert out["GHCNH:A", "t2m"] == [0, 0]
    assert out["GHCNH:B", "pr24"] == [qc.INCONSISTENT, 0]
    assert out["GHCNH:C", "pr24"] == [qc.RANGE | qc.SUSPECT, qc.SUSPECT]


def _ascent(z850=1500.0, t=(15.0, 10.0, 0.0, -20.0)):
    levels = [(1000, 110.0), (850, z850), (700, 3050.0), (500, 5700.0)]
    rows = []
    for (p, z), tt in zip(levels, t):
        rows += [{"variable": f"z{p}", "value": z}, {"variable": f"t{p}", "value": tt}]
    return pd.DataFrame(rows).assign(station_id="RAOB:1", valid_time=pd.Timestamp("2026-10-13"), qc=0)


def test_hydrostatic_check_passes_a_consistent_ascent():
    assert (qc.run_sounding_levels(_ascent())["qc"] == 0).all()


def test_hydrostatic_check_flags_a_corrupt_height():
    out = qc.run_sounding_levels(_ascent(z850=1900.0)).set_index("variable")["qc"]
    assert out["z850"] & qc.VERTICAL
    assert out["t850"] == 0 and out["z500"] == 0


def test_flags_have_unique_bits():
    bits = list(qc.FLAGS)
    assert all(b & (b - 1) == 0 for b in bits) and len(set(bits)) == len(bits)
    assert np.uint16(sum(bits)) == sum(bits)
