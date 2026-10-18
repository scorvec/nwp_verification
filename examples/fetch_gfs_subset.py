#!/usr/bin/env python3
"""Download a small GRIB2 subset of GFS 0.25° from the NOAA AWS bucket.

Uses the .idx byte offsets so only the requested messages are fetched (~10 MB
per lead instead of ~500 MB). Handy for trying mverify on real data.

    python fetch_gfs_subset.py 2026101300 0 6 12 18 24 --out data/gfs
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
DEFAULT_FIELDS = [
    ":TMP:2 m above ground:", ":DPT:2 m above ground:", ":PRMSL:mean sea level:",
    ":UGRD:10 m above ground:", ":VGRD:10 m above ground:", ":HGT:surface:", ":LAND:surface:",
    ":SFCR:surface:",
    ":HGT:500 mb:", ":TMP:850 mb:", ":UGRD:250 mb:", ":VGRD:250 mb:",
]


def _get(url: str, rng: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch(init: str, lead: int, out: Path, fields=DEFAULT_FIELDS) -> Path:
    ymd, hh = init[:8], init[8:10]
    url = f"{BUCKET}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{lead:03d}"
    lines = _get(url + ".idx").decode().splitlines()
    offsets = [int(l.split(":")[1]) for l in lines]
    dest = out / f"gfs.t{hh}z.{ymd}.pgrb2.0p25.f{lead:03d}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        for i, line in enumerate(lines):
            if any(f in line for f in fields):
                end = offsets[i + 1] - 1 if i + 1 < len(offsets) else ""
                fh.write(_get(url, f"{offsets[i]}-{end}"))
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("init", help="YYYYMMDDHH")
    ap.add_argument("leads", nargs="+", type=int)
    ap.add_argument("--out", type=Path, default=Path("data/gfs"))
    a = ap.parse_args()
    assert re.fullmatch(r"\d{10}", a.init)
    for ld in a.leads:
        print(fetch(a.init, ld, a.out))
