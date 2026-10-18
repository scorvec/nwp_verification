"""Scorecard JSON for web pages, plus plain-text tables.

The JSON is shaped for a static page to fetch and draw without any further
processing: one row per truth x variable x region, each holding a score array
per model aligned with the top-level ``leads`` list (null where a model has no
value at that lead).

    {
      "generated": "2026-10-18 10:00 UTC", "package": "mverify 0.1.0",
      "models": ["GFS", "IFS"], "leads": [0, 6, 12, ...],
      "inits": {"GFS": ["2026-10-13T00", ...]},
      "rows": [{"key": "ghcnh:t2m:conus", "var": "t2m", "label": "2 m temperature",
                "truth": "surface stations (GHCNh)", "region": "CONUS", "unit": "degC",
                "scores": {"GFS": {"rmse": [...], "bias": [...], "mae": [...], "n": [...]}},
                "p": {"IFS|GFS": [...]}, "runs": {"GFS|IFS": [...]}, "runs_eff": {"GFS|IFS": [...]}}]
    }

``p`` holds, for each ordered pair of models, the p value per lead of the paired test that
their mean squared errors differ (``mverify.significance``); it is absent with one model.
``runs`` and ``runs_eff`` hold, per lead, the runs the test paired and the effective runs it
counted after allowing for correlated errors on consecutive runs, once per pair of models,
under the names in alphabetical order ("GFS|IFS").
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mverify import __version__
from mverify import variables as V
from mverify.scores import REGION_LABELS

TRUTH = {"GHCNH": ("ghcnh", "surface stations (GHCNh)"), "RAOB": ("raob", "radiosondes"),
         "STAGEIV": ("stageiv", "Stage IV analysis"), "GPCP": ("gpcp", "GPCP analysis"),
         "IMERG_LATE": ("imerg_late", "IMERG Late analysis")}
STATS = ("rmse", "rmse_oc", "bias", "mae", "corr", "crps", "crps_oe", "spread", "spread_oe", "ssr", "ssr_oe",
         "ets", "fbias", "pod", "far", "fss_50km", "fss_100km", "fss_200km", "fss_400km", "fcst_mean", "obs_mean",
         "base_rate", "n")


def _label(var: str) -> tuple[str, str]:
    if var in V.CATALOG:
        return V.CATALOG[var].label, V.CATALOG[var].unit
    if m := re.fullmatch(r"tp(\d+)(?:>=([\d.]+)mm)?", var):
        hours, thr = m.groups()
        return (f"{hours} h precipitation" + (f" >= {thr} mm" if thr else "")), ("mm" if not thr else "")
    if var.startswith("wind"):
        lev = var[4:]
        return ("10 m vector wind" if lev == "10" else f"{lev} hPa vector wind"), "m s-1"
    return var, ""


def _clean(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), 4) if not isinstance(x, (int,)) else x


def scorecard(table: pd.DataFrame, inits: dict[str, list[str]], tests: pd.DataFrame | None = None) -> dict:
    """``table``: one row per model x network x variable x region x lead_h with score columns;
    ``tests``: the output of ``mverify.significance.tests`` for the same pairs."""
    leads = sorted(table["lead_h"].unique().tolist())
    by_row = ({} if tests is None or tests.empty
              else {k: g for k, g in tests.groupby(["network", "variable", "region"], sort=False)})
    rows = []
    for (net, var, region), g in table.groupby(["network", "variable", "region"], sort=True):
        key, truth = TRUTH.get(net, (net.lower(), net))
        label, unit = _label(var)
        scores = {}
        for model, m in g.groupby("model"):
            m = m.set_index("lead_h").reindex(leads)
            scores[model] = {s: [_clean(v) if s != "n" else (None if pd.isna(v) else int(v)) for v in m[s]]
                             for s in STATS if s in m and m[s].notna().any()}
        row = {"key": f"{key}:{var}:{region}", "var": var, "label": label, "truth": truth,
               "region": REGION_LABELS.get(region, region), "unit": unit, "scores": scores}
        if (t := by_row.get((net, var, region))) is not None:
            pairs = {ab: g.set_index("lead_h").reindex(leads) for ab, g in t.groupby(["model", "reference"])}
            row["p"] = {f"{a}|{b}": [_clean(v) for v in g["p"]] for (a, b), g in pairs.items()}
            # the paired runs behind each test and the effective runs it counts, the same for a|b and b|a:
            # stored once, under the pair in alphabetical order
            once = {f"{a}|{b}": g for (a, b), g in pairs.items() if a < b}
            row["runs"] = {k: [None if pd.isna(v) else int(v) for v in g["runs"]] for k, g in once.items()}
            row["runs_eff"] = {k: [None if pd.isna(v) else round(float(v), 1) for v in g["runs_eff"]]
                               for k, g in once.items()}
        rows.append(row)
    return {"generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "package": f"mverify {__version__}", "models": sorted(table["model"].unique().tolist()),
            "leads": [int(x) if float(x).is_integer() else x for x in leads], "inits": inits, "rows": rows}


def station_map(stations: pd.DataFrame, leads: list[float] | None = None, variables: list[str] | None = None,
                min_n: int = 5) -> dict:
    """Per-station scores (``Result.stations``) as JSON for a map: one layer per network and variable.

        {"models": [...], "leads": [24, 72, ...],
         "layers": [{"var": "t2m", "label": ..., "unit": ..., "truth": ..., "ids": [...], "lat": [...],
                     "lon": [...], "scores": {"GFS": {"rmse": [[per station] per lead], "bias": ..., "n": ...}}}]}

    Only stations where every model has at least ``min_n`` pairs at a lead get a value there, so
    the models' maps show the same stations.
    """
    st = stations
    if leads is not None:
        st = st[st["lead_h"].isin(leads)]
    if variables is not None:
        st = st[st["variable"].isin(variables)]
    models = sorted(st["model"].unique().tolist())
    leads = sorted(st["lead_h"].unique().tolist())
    layers = []
    for (net, var), g in st.groupby(["network", "variable"], sort=True):
        g = g[g["n"] >= min_n]
        full = g.groupby(["station_id", "lead_h"])["model"].transform("nunique") == len(models)
        g = g[full]
        if g.empty:
            continue
        pos = g.groupby("station_id")[["lat", "lon"]].median()
        label, unit = _label(var)
        scores = {}
        for model, m in g.groupby("model"):
            scores[model] = {}
            for s in ("rmse", "bias", "n"):
                grid = m.pivot(index="station_id", columns="lead_h", values=s).reindex(index=pos.index, columns=leads)
                scores[model][s] = [[None if pd.isna(v) else (int(v) if s == "n" else round(float(v), 3))
                                     for v in grid[c]] for c in leads]
        layers.append({"var": var, "label": label, "unit": unit, "truth": TRUTH.get(net, (net, net))[1],
                       "ids": [i.split(":", 1)[1] for i in pos.index], "lat": [round(x, 3) for x in pos["lat"]],
                       "lon": [round(x, 3) for x in pos["lon"]], "scores": scores})
    return {"generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "package": f"mverify {__version__}", "models": models,
            "leads": [int(x) if float(x).is_integer() else x for x in leads], "layers": layers}


def precip_maps(maps: dict) -> dict:
    """Per-cell precipitation scores (``mverify.precip.maps.cell_scores``, one Dataset per model) as JSON for a map:

        {"product": "stageiv", "truth": ..., "accum_hours": 24, "fdr_q": 0.1, "min_days": 8,
         "groups": [{"label": "days 1-2", "ends_h": [36, 60]}, ...],
         "models": {"GFS": {"lat": [...], "lon": [...], "bias": [[per cell] per group], "rmse": ..., "obs": ...,
                            "days": ..., "significant": ...}}}

    Amounts are tenths of a millimetre per window as integers, cells row by row from the first latitude, None where
    the cell has no score (the model's mean is obs + bias); ``significant`` is 1 where the bias passes the
    false-discovery-rate procedure."""
    first = next(iter(maps.values()))
    out = {"product": first.attrs["product"], "truth": TRUTH.get(first.attrs["product"].upper(), ("", ""))[1],
           "accum_hours": first.attrs["accum_hours"], "fdr_q": first.attrs["fdr_q"],
           "min_days": first.attrs["min_days"],
           "groups": [{"label": str(g), "ends_h": e} for g, e in zip(first.group.values, first.attrs["window_ends_h"],
                                                                      strict=True)],
           "models": {}}

    def tenths(a):
        return [[None if not np.isfinite(v) else int(np.rint(10 * v)) for v in g.ravel()] for g in a]

    for model, m in maps.items():
        out["models"][model] = {
            "lat": [round(float(x), 4) for x in m.lat.values], "lon": [round(float(x), 4) for x in m.lon.values],
            **{k: tenths(m[k].values) for k in ("bias", "rmse", "obs")},
            "days": [g.ravel().tolist() for g in m["days"].values],
            "significant": [g.ravel().astype(int).tolist() for g in m["significant"].values],
        }
    return out


def write_json(card: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(card, separators=(",", ":"), allow_nan=False))


def text_table(table: pd.DataFrame, stat: str = "rmse") -> str:
    if table.empty:
        return "(no matched pairs)"
    t = table.pivot_table(index=["network", "variable", "region", "model"], columns="lead_h", values=stat)
    t.columns = [f"{int(c)}h" if float(c).is_integer() else f"{c}h" for c in t.columns]
    return t.round(2).to_string()
