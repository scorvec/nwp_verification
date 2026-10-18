"""Model precipitation totals over an analysis window, from whatever accumulations a file holds.

A model writes accumulated precipitation as running totals from the start of the
forecast (ECMWF, ERA5, GFS's 0-N h field), as buckets (GFS's 6 h fields, HRRR's
1 h fields), or both. ``Accumulations`` collects the fields of one init and member
by their lead-time window and builds any requested window from them: directly,
as the difference of two totals with the same start, or by chaining contiguous
windows. Differences can come out slightly negative from GRIB packing and are
clipped at zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mverify.io.open import Field

H = pd.Timedelta("1h")


class Accumulations:
    def __init__(self, fields: list[Field]):
        self.fields = {f.accum: f for f in fields if f.accum is not None}
        self.grid = fields[0].grid

    def total(self, start: pd.Timedelta, end: pd.Timedelta) -> np.ndarray | None:
        """Precipitation (mm) between leads ``start`` and ``end``, or None if the file cannot give it."""
        recipe = self._recipe(start, end, depth=0)
        if recipe is None:
            return None
        out = sum(sign * self.fields[w].values() for sign, w in recipe)
        return np.maximum(out, 0.0) if len(recipe) > 1 else out

    def _recipe(self, a, b, depth) -> list[tuple[int, tuple]] | None:
        if (a, b) in self.fields:
            return [(1, (a, b))]
        if depth > 48:  # a chain of hourly buckets over two days at most
            return None
        starts = sorted({s for s, e in self.fields if e == b and s < a}, reverse=True)
        for s in starts:  # (s, b) - (s, a)
            if (s, a) in self.fields:
                return [(1, (s, b)), (-1, (s, a))]
        for e in sorted({e for s, e in self.fields if s == a and e < b}, reverse=True):
            rest = self._recipe(e, b, depth + 1)  # (a, e) + (e, b), longest first piece first
            if rest is not None:
                return [(1, (a, e)), *rest]
        if a == pd.Timedelta(0):
            return None
        for s in sorted({s for s, e in self.fields if e == b and s < a}, reverse=True):
            head = self._recipe(s, a, depth + 1)  # (s, b) - (s, a) with (s, a) itself built
            if head is not None:
                return [(1, (s, b)), *((-sign, w) for sign, w in head)]
        return None


def by_run(fields: list[Field]) -> dict[tuple, Accumulations]:
    """Accumulations per (init, member, grid) of the precipitation fields of a forecast."""
    runs: dict[tuple, list[Field]] = {}
    for f in fields:
        if f.accum is not None:
            runs.setdefault((f.init, f.member, f.grid.key), []).append(f)
    return {k: Accumulations(v) for k, v in runs.items()}
