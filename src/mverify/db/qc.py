"""Observation quality control. Every check sets one bit of ``obs.qc``; 0 means clean.

Flagged rows are kept in the database (so a check can be revisited and the
flag rates audited) but are excluded from scoring by default.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from mverify import variables as V

UPSTREAM = 1
RANGE = 2
SPIKE = 4
STUCK = 8
INCONSISTENT = 16
VERTICAL = 32

# bit -> (name, description); written to the qc_flags table so the database documents itself.
FLAGS = {
    UPSTREAM: ("upstream", "the data provider's own quality control flagged the value"),
    RANGE: ("range", "outside the physical limits of the variable (variables.lo / variables.hi)"),
    SPIKE: ("spike", "isolated jump away from both hourly neighbours in the same direction"),
    STUCK: ("stuck", "identical value repeated longer than plausible (frozen sensor)"),
    INCONSISTENT: ("inconsistent", "cross-variable check failed, e.g. dew point above temperature"),
    VERTICAL: ("vertical", "radiosonde layer thickness disagrees with the hypsometric thickness"),
}

# Largest plausible one-hour departure of a value from both of its neighbours.
SPIKE_LIMIT = {"t2m": 8.0, "td2m": 10.0, "msl": 6.0, "u10": 25.0, "v10": 25.0, "ws10": 25.0}
# Hours a value may repeat exactly before the sensor is presumed stuck.
STUCK_HOURS = {"t2m": 12, "td2m": 18, "msl": 12}


def range_check(df: pd.DataFrame) -> pd.Series:
    lo = df["variable"].map(lambda v: V.CATALOG[v].lo if v in V.CATALOG else -np.inf)
    hi = df["variable"].map(lambda v: V.CATALOG[v].hi if v in V.CATALOG else np.inf)
    return ((df["value"] < lo) | (df["value"] > hi)).astype("uint16") * RANGE


def spike_check(df: pd.DataFrame) -> pd.Series:
    """A value that departs from BOTH hourly neighbours, in opposite-sign agreement, by more than the limit."""
    out = pd.Series(0, index=df.index, dtype="uint16")
    for var, lim in SPIKE_LIMIT.items():
        sub = df[df["variable"] == var].sort_values(["station_id", "valid_time"])
        if sub.empty:
            continue
        g = sub.groupby("station_id")
        prev_v, next_v = g["value"].shift(1), g["value"].shift(-1)
        prev_t, next_t = g["valid_time"].shift(1), g["valid_time"].shift(-1)
        hourly = ((sub["valid_time"] - prev_t) <= pd.Timedelta("1h")) & (
            (next_t - sub["valid_time"]) <= pd.Timedelta("1h"))
        d1, d2 = sub["value"] - prev_v, sub["value"] - next_v
        bad = hourly & (d1.abs() > lim) & (d2.abs() > lim) & (np.sign(d1) == np.sign(d2))
        out.loc[sub.index[bad.to_numpy()]] = SPIKE
    return out


def stuck_check(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(0, index=df.index, dtype="uint16")
    for var, hours in STUCK_HOURS.items():
        sub = df[df["variable"] == var].sort_values(["station_id", "valid_time"])
        if sub.empty:
            continue
        change = (sub["value"] != sub.groupby("station_id")["value"].shift()) | (
            sub["station_id"] != sub["station_id"].shift())
        run_id = change.cumsum()
        g = sub.groupby(run_id)["valid_time"]
        span = g.transform("max") - g.transform("min")
        bad = span >= pd.Timedelta(hours=hours)
        out.loc[sub.index[bad.to_numpy()]] = STUCK
    return out


def dewpoint_check(df: pd.DataFrame) -> pd.Series:
    """Dew point more than 0.5 K above the simultaneous temperature flags the dew point."""
    out = pd.Series(0, index=df.index, dtype="uint16")
    t = df[df["variable"] == "t2m"].set_index(["station_id", "valid_time"])["value"]
    td = df[df["variable"] == "td2m"]
    if td.empty or t.empty:
        return out
    tt = t.reindex(pd.MultiIndex.from_frame(td[["station_id", "valid_time"]])).to_numpy()
    bad = td["value"].to_numpy() > tt + 0.5
    out.loc[td.index[bad]] = INCONSISTENT
    return out


def run_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every surface check; ``df`` has station_id, valid_time, variable, value, qc."""
    df = df.copy()
    df["qc"] = (df["qc"].astype("uint16") | range_check(df) | spike_check(df) | stuck_check(df)
                | dewpoint_check(df)).astype("uint16")
    return df


def run_sounding_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Range check plus a hydrostatic check of the mandatory-level heights.

    For each pair of adjacent levels with heights and temperatures, the observed
    thickness must agree with the hypsometric thickness from the layer-mean
    temperature to within 2% + 15 m. A height is flagged when every layer it
    bounds fails, so one corrupt interior level fails both its layers and only
    it is flagged; an end level (one layer) is flagged only when its neighbour
    does not already explain the failure.
    """
    df = df.copy()
    df["qc"] = (df["qc"].astype("uint16") | range_check(df)).astype("uint16")
    key = ["station_id", "valid_time"]
    bad: set[tuple] = set()
    for k, asc in df.groupby(key, sort=False):
        bad |= {(*k, f"z{p}") for p in _hydrostatic_failures(asc)}
    if bad:
        m = pd.MultiIndex.from_frame(df[key + ["variable"]]).isin(list(bad))
        df.loc[m, "qc"] = df.loc[m, "qc"] | VERTICAL
    return df


def _hydrostatic_failures(asc: pd.DataFrame) -> list[int]:
    vals = asc.set_index("variable")["value"]
    levels = sorted((int(v[1:]) for v in vals.index if v[0] == "z" and f"t{v[1:]}" in vals.index), reverse=True)
    if len(levels) < 2:
        return []
    fails = []
    for p1, p2 in pairwise(levels):
        tm = (vals[f"t{p1}"] + vals[f"t{p2}"]) / 2 + 273.15
        hyps = 287.05 * tm / V.G0 * np.log(p1 / p2)
        fails.append(abs((vals[f"z{p2}"] - vals[f"z{p1}"]) - hyps) > 0.02 * hyps + 15.0)
    n = len(levels)
    layers_of = [[j for j in (i - 1, i) if 0 <= j < n - 1] for i in range(n)]  # layer j spans levels j, j+1
    flagged = [bool(ls) and all(fails[j] for j in ls) for ls in layers_of]
    for end, nb in ((0, 1), (n - 1, n - 2)):
        if flagged[end] and 0 < nb < n - 1 and flagged[nb]:
            flagged[end] = False
    return [levels[i] for i in range(n) if flagged[i]]
