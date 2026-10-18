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
                "scores": {"GFS": {"rmse": [...], "bias": [...], "mae": [...], "n": [...]}}}]
    }
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from mverify import __version__
from mverify import variables as V
from mverify.scores import REGION_LABELS

TRUTH = {"GHCNH": ("ghcnh", "surface stations (GHCNh)"), "RAOB": ("raob", "radiosondes"),
         "STAGEIV": ("stageiv", "Stage IV analysis"), "GPCP": ("gpcp", "GPCP analysis")}
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


def scorecard(table: pd.DataFrame, inits: dict[str, list[str]]) -> dict:
    """``table``: one row per model x network x variable x region x lead_h with score columns."""
    leads = sorted(table["lead_h"].unique().tolist())
    rows = []
    for (net, var, region), g in table.groupby(["network", "variable", "region"], sort=True):
        key, truth = TRUTH.get(net, (net.lower(), net))
        label, unit = _label(var)
        scores = {}
        for model, m in g.groupby("model"):
            m = m.set_index("lead_h").reindex(leads)
            scores[model] = {s: [_clean(v) if s != "n" else (None if pd.isna(v) else int(v)) for v in m[s]]
                             for s in STATS if s in m and m[s].notna().any()}
        rows.append({"key": f"{key}:{var}:{region}", "var": var, "label": label, "truth": truth,
                     "region": REGION_LABELS.get(region, region), "unit": unit, "scores": scores})
    return {"generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "package": f"mverify {__version__}", "models": sorted(table["model"].unique().tolist()),
            "leads": [int(x) if float(x).is_integer() else x for x in leads], "inits": inits, "rows": rows}


def write_json(card: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(card, separators=(",", ":"), allow_nan=False))


def text_table(table: pd.DataFrame, stat: str = "rmse") -> str:
    if table.empty:
        return "(no matched pairs)"
    t = table.pivot_table(index=["network", "variable", "region", "model"], columns="lead_h", values=stat)
    t.columns = [f"{int(c)}h" if float(c).is_integer() else f"{c}h" for c in t.columns]
    return t.round(2).to_string()
