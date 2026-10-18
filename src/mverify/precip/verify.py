"""Score a forecast's precipitation against the analyses in the database.

For each product (Stage IV, GPCP) and each of its windows the database holds
inside the forecast's lead range, the model total over the window is built from
the file's accumulations (``accum.py``), put on the analysis grid
(``regrid.py``) and summed into the additive scores of ``scores.py`` per region.
An ensemble is scored through its member mean; probabilistic precipitation
scores (Brier, CRPS of the amounts) come later.

Rows of the result use the station table's columns, so they join it in
scorecards: ``network`` is the product (STAGEIV, GPCP), ``variable`` is
``tp24`` for amounts and ``tp24>=10mm`` for a threshold, and ``lead_h`` is the
lead at the end of the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mverify.db.store import ObsStore
from mverify.io.open import Forecast
from mverify.precip import scores as P
from mverify.precip.accum import by_run
from mverify.precip.analyses import PRODUCTS
from mverify.precip.regrid import Regridder, analysis_grid, spacing_km
from mverify.scores import REGIONS

BY = ["model", "network", "variable", "region", "lead_h"]


def _region_mask(lat: np.ndarray, lon: np.ndarray, region: str) -> np.ndarray:
    lo0, la0, lo1, la1 = REGIONS[region]
    return (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)


def verify_precip(fc: Forecast, store: ObsStore, products: list[str] | None = None,
                  regions: list[str] | None = None) -> pd.DataFrame:
    runs = by_run([f for f in fc.fields if f.var == "tp"])
    if not runs:
        return pd.DataFrame(columns=BY)
    # Members of one init and grid are averaged into the ensemble mean before scoring.
    groups: dict[tuple, list] = {}
    for (init, _member, gkey), acc in runs.items():
        groups.setdefault((init, gkey), []).append(acc)
    rows = []
    for name in products or list(PRODUCTS):
        product = PRODUCTS[name]
        span = pd.Timedelta(hours=product.accum_hours)
        first = min(i for i, _ in groups) + span
        last = max(i + max(f.lead for f in fc.fields if f.var == "tp") for i, _ in groups)
        ends = store.grid_times(name, product.accum_hours, first, last)
        regridders: dict[tuple, Regridder] = {}
        for end in ends:
            truth = store.get_grid(name, product.accum_hours, end)
            grid = analysis_grid(truth)
            lat, lon = (np.meshgrid(grid.lat, grid.lon, indexing="ij") if grid.regular else (grid.lat, grid.lon))
            weights = np.cos(np.deg2rad(lat)) if grid.regular else np.ones_like(lat)
            periodic = grid.regular and abs(grid.lon.size * np.median(np.diff(grid.lon)) - 360) < 1e-3
            cell_km = spacing_km(grid)
            masks = {r: _region_mask(lat, lon, r) for r in regions or product.regions}
            obs = truth.values
            for (init, gkey), accs in groups.items():
                a, b = end - span - init, end - init
                if a < pd.Timedelta(0):
                    continue
                totals = [t for t in (acc.total(a, b) for acc in accs) if t is not None]
                if not totals:
                    continue
                key = (gkey, grid.lat.shape, float(grid.lat.flat[0]), float(grid.lon.flat[0]))
                if key not in regridders:
                    regridders[key] = Regridder(accs[0].grid, grid)
                fcst = regridders[key](np.mean(totals, axis=0))
                for region, mask in masks.items():
                    if not mask.any():
                        continue
                    sums = P.case_sums(fcst, obs, mask, weights, cell_km, periodic)
                    base = {"model": fc.name, "network": name.upper(), "region": region,
                            "lead_h": b / pd.Timedelta("1h"), "n_cases": 1}
                    rows.append({**base, "variable": f"tp{product.accum_hours}", "threshold": np.nan,
                                 **sums.pop("amount")})
                    rows += [{**base, "variable": f"tp{product.accum_hours}>={t:g}mm", "threshold": t, **c}
                             for t, c in sums.items()]
    if not rows:
        return pd.DataFrame(columns=BY)
    table = P.finish(pd.DataFrame(rows), BY)
    order = ["model", "network", "region", "lead_h", "threshold"]
    return table.sort_values(order, na_position="first").drop(columns="threshold").reset_index(drop=True)
