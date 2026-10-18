"""Precipitation scores per grid cell, for maps: a forecast against an analysis on the forecast's own grid.

Each analysis window (24 h ending 12Z for Stage IV) is averaged onto the forecast grid - the common grid of
mverify.coarsen, say 0.5 deg - where it covers at least half a cell, and every run's total over the same window is
compared with it cell by cell. Windows are pooled into lead groups (days 1-2, 3-5, 6-9 by default) to give the
bias, RMSE and mean rain of each cell.

Whether a cell's bias is more than chance is tested over the valid days: the day's error (averaged over the runs
of the group that verify it) is put through the paired t-test of mverify.significance, with its allowance for
errors that persist from day to day, and the p values of all the cells of a map through the Benjamini-Hochberg
procedure at a false-discovery rate of 10 %, so a map shades only cells whose bias is unlikely to be noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from mverify.db.store import ObsStore
from mverify.io.open import Forecast
from mverify.precip.accum import by_run
from mverify.precip.analyses import PRODUCTS
from mverify.precip.regrid import Regridder, analysis_grid
from mverify.significance import fdr, paired_test

# (label, first and last window end in hours) of the lead groups
LEAD_GROUPS = (("days 1-2", 36, 60), ("days 3-5", 84, 132), ("days 6-9", 156, 228))
MIN_COVER = 0.5   # an analysis cell needs values over at least this share of its source points
MIN_DAYS = 8      # valid days a cell needs before its bias is tested
FDR_Q = 0.10


def _group(end_h: float, groups) -> int | None:
    return next((k for k, (_, a, b) in enumerate(groups) if a <= end_h <= b), None)


def cell_scores(fc: Forecast, store: ObsStore, product: str = "stageiv", groups=LEAD_GROUPS,
                min_days: int = MIN_DAYS, q: float = FDR_Q) -> xr.Dataset | None:
    """Per-cell scores of ``fc``'s precipitation against ``product``, on ``fc``'s grid: variables ``bias``,
    ``rmse``, ``fcst`` and ``obs`` (mean mm per window), ``n`` (windows), ``days`` (valid days), ``p`` (bias
    test, NaN where not tested) and ``significant`` (passes the FDR procedure), over dims (group, lat, lon).
    None when the analysis has no window inside the forecast's leads."""
    runs = by_run([f for f in fc.fields if f.var == "tp"])
    if not runs:
        return None
    members: dict[tuple, list] = {}
    for (init, _member, gkey), acc in runs.items():
        members.setdefault((init, gkey), []).append(acc)
    if len({gkey for _, gkey in members}) > 1:
        raise ValueError("cell_scores needs one grid per forecast (mverify.coarsen puts them on one)")
    grid = next(iter(members.values()))[0].grid
    if not grid.regular:
        raise ValueError("cell_scores needs a regular grid")
    spec = PRODUCTS[product]
    span = pd.Timedelta(hours=spec.accum_hours)
    inits = sorted({i for i, _ in members})
    max_lead = max(f.lead for f in fc.fields if f.var == "tp")
    ends = store.grid_times(product, spec.accum_hours, inits[0] + span, inits[-1] + max_lead)
    if not ends:
        return None
    regridders: dict[tuple, Regridder] = {}

    def analysis(end):
        """The window's analysis averaged onto the forecast grid (NaN where it covers under MIN_COVER), and the
        share of each cell it covers."""
        truth = store.get_grid(product, spec.accum_hours, end)
        tgrid = analysis_grid(truth)
        key = (tgrid.lat.shape, float(tgrid.lat.flat[0]), float(tgrid.lon.flat[0]))
        if key not in regridders:
            regridders[key] = Regridder(tgrid, grid)
        rg = regridders[key]
        if not rg.coarsen:
            raise ValueError(f"{product} is not finer than the forecast grid; cell maps need a coarser forecast grid")
        cover = rg.coverage(truth.values)
        return np.where(cover >= MIN_COVER, rg(truth.values), np.nan), cover

    # every array is cut to the rows and columns the analysis reaches (Stage IV covers a sliver of a global grid)
    reach = analysis(ends[0])[1] > 0
    rows, cols = np.flatnonzero(reach.any(1)), np.flatnonzero(reach.any(0))
    cut = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    shape, ng = reach[cut].shape, len(groups)
    sums = {k: np.zeros((ng, *shape)) for k in ("err", "sq", "fcst", "obs", "n")}
    daily: list[dict] = [{} for _ in groups]  # per group: valid end -> (sum of errors, count)
    for end in ends:
        obs = analysis(end)[0][cut]
        for (init, _), accs in members.items():
            a, b = end - span - init, end - init
            g = _group(b / pd.Timedelta("1h"), groups)
            if a < pd.Timedelta(0) or g is None:
                continue
            totals = [t for t in (acc.total(a, b) for acc in accs) if t is not None]
            if not totals:
                continue
            fcst = np.mean(totals, axis=0)[cut]
            err = fcst - obs
            ok = np.isfinite(err)
            for name, v in (("err", err), ("sq", err**2), ("fcst", fcst), ("obs", obs), ("n", 1.0)):
                sums[name][g] += np.where(ok, v, 0.0)
            s, c = daily[g].get(end, (np.zeros(shape), np.zeros(shape)))
            daily[g][end] = (s + np.where(ok, err, 0.0), c + ok)
    n = sums["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = {k: np.where(n > 0, sums[k] / n, np.nan) for k in ("err", "sq", "fcst", "obs")}
    p = np.full((ng, *shape), np.nan)
    days = np.zeros((ng, *shape), int)
    for g in range(ng):
        if not daily[g]:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.stack([np.where(c > 0, s / c, np.nan) for _, (s, c) in sorted(daily[g].items())])
        days[g] = np.isfinite(d).sum(0)
        for j, i in zip(*np.nonzero(days[g] >= min_days), strict=True):
            p[g, j, i] = paired_test(d[:, j, i])
    significant = np.stack([fdr(p[g], q) for g in range(ng)])
    dims = ("group", "lat", "lon")
    return xr.Dataset(
        {"bias": (dims, mean["err"]), "rmse": (dims, np.sqrt(mean["sq"])), "fcst": (dims, mean["fcst"]),
         "obs": (dims, mean["obs"]), "n": (dims, n.astype(int)), "days": (dims, days), "p": (dims, p),
         "significant": (dims, significant)},
        coords={"group": [g[0] for g in groups], "lat": grid.lat[cut[0]], "lon": grid.lon[cut[1]]},
        attrs={"model": fc.name, "product": product, "accum_hours": spec.accum_hours, "fdr_q": q,
               "min_days": min_days, "window_ends_h": [list(g[1:]) for g in groups]},
    )
