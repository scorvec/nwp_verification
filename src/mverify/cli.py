"""Command line: ``mverify score``, ``mverify inspect`` and ``mverify db ...``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mverify import __version__
from mverify.match import MAX_DZ


def pd_now():
    import pandas as pd
    return pd.Timestamp.now("UTC").tz_localize(None)


def _optional_float(s: str) -> float | None:
    return None if s.lower() in ("none", "off") else float(s)


def _bbox(s: str | None):
    if not s:
        return None
    from mverify.scores import REGIONS
    if s in REGIONS:
        return REGIONS[s]
    vals = tuple(float(x) for x in s.split(","))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("bbox is lon_min,lat_min,lon_max,lat_max or a region name")
    return vals


def _models(items: list[str]) -> dict[str, list[str]] | list[str]:
    """``NAME=PATH`` items group into models; bare paths form one model named from the files."""
    named: dict[str, list[str]] = {}
    bare: list[str] = []
    for it in items:
        if "=" in it and not Path(it).exists():
            name, path = it.split("=", 1)
            named.setdefault(name, []).append(path)
        else:
            bare.append(it)
    if named and bare:
        raise SystemExit("mix of NAME=PATH and bare paths; name every model or none")
    return named or bare


def cmd_inspect(a) -> int:
    from mverify.io.open import open_forecast
    fc = open_forecast(a.paths, init=a.init)
    s = fc.summary()
    if s.empty:
        print("no recognised variables")
        return 1
    has = {name: "yes" if any(getattr(f.grid, attr) is not None for f in fc.fields) else "no"
           for name, attr in (("orography", "orog"), ("land-sea mask", "lsm"), ("roughness", "z0"))}
    print(f"model: {fc.name}   fields: {len(s)}   " + "   ".join(f"{k}: {v}" for k, v in has.items()))
    agg = s.groupby("var").agg(inits=("init", "nunique"), first_init=("init", "min"),
                               leads=("lead_h", lambda x: ",".join(f"{v:g}" for v in sorted(set(x)))),
                               members=("member", "nunique"), grid=("grid", "first"), shape=("shape", "first"))
    print(agg.to_string())
    return 0


def cmd_score(a) -> int:
    from mverify import report
    from mverify.db.store import ObsStore
    from mverify.match import T_HEIGHT_RANGE
    from mverify.verify import verify
    store = ObsStore.snapshot(a.snapshot) if a.snapshot else ObsStore.open(a.db, read_only=True)
    with store:
        res = verify(_models(a.paths), store, regions=a.region, variables=a.var, networks=a.network,
                     height_correction=not a.no_height_correction, init=a.init, max_dz=a.max_dz,
                     t_height_range=None if a.any_thermometer_height else T_HEIGHT_RANGE)
    if res.table.empty:
        print("no forecast/observation pairs: check the valid times and region against `mverify db status`")
        return 1
    print(report.text_table(res.table, a.stat))
    if a.out:
        report.write_json(res.scorecard(), a.out)
        print(f"scorecard -> {a.out}")
    if a.pairs:
        res.pairs.to_parquet(a.pairs, index=False)
        print(f"pairs -> {a.pairs}")
    return 0


def cmd_db(a) -> int:
    from mverify.db.store import ObsStore
    if a.db_cmd == "status":
        with ObsStore.open(a.db, read_only=True) as s:
            print(s.summary().to_string(index=False))
            q = s.qc_report()
            if not q.empty:
                print("\nQC flags\n" + q.to_string(index=False))
        return 0
    with ObsStore.open(a.db) as s:
        if a.db_cmd == "init":
            print(f"database ready: {s.con.execute('PRAGMA database_list').fetchall()[0][2]}")
        elif a.db_cmd == "ghcnh":
            from mverify.ingest import ghcnh
            n = ghcnh.ingest(s, a.year, bbox=_bbox(a.bbox), ids=a.id, max_stations=a.max_stations,
                             workers=a.workers)
            print(f"{n} rows written")
        elif a.db_cmd == "soundings":
            from mverify.ingest import soundings
            n = soundings.ingest(s, root=a.root, since=a.since, bbox=_bbox(a.bbox))
            print(f"{n} rows written")
        elif a.db_cmd == "precip":
            from mverify.db.store import db_path
            from mverify.precip import analyses
            end = a.end or pd_now()
            for product in a.product:
                n = analyses.ingest(s, product, a.start, end, db_path(a.db).parent / "cache")
                print(f"{product}: {n} windows added")
        elif a.db_cmd == "sensors":
            import pandas as pd
            n, unmatched = s.set_sensor_heights(pd.read_csv(a.file, dtype=str))
            print(f"sensor heights set for {n} stations")
            if unmatched:
                print(f"{len(unmatched)} rows matched no station, e.g. {unmatched[:5]}")
        elif a.db_cmd == "export":
            s.export_parquet(a.out)
            print(f"snapshot -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mverify", description=__doc__)
    ap.add_argument("--version", action="version", version=f"mverify {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="show what mverify recognises in forecast files")
    p.add_argument("paths", nargs="+")
    p.add_argument("--init", help="init time, when the files do not carry one")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("score", help="verify forecast files against the observation database")
    p.add_argument("paths", nargs="+", help="files, globs or directories; NAME=PATH to name and compare models")
    p.add_argument("--db", help="observation database (default $MVERIFY_DB or ~/.cache/mverify/obs.duckdb)")
    p.add_argument("--snapshot", help="published Parquet snapshot (directory or URL) instead of --db")
    p.add_argument("--region", nargs="+", help="regions to score (default: all that hold pairs)")
    p.add_argument("--var", nargs="+", help="variables to score (default: all recognised)")
    p.add_argument("--network", nargs="+", help="observation networks (GHCNH, RAOB)")
    p.add_argument("--init", help="init time, when the files do not carry one")
    p.add_argument("--stat", default="rmse", help="score shown in the table (rmse, rmse_oc, bias, mae, corr, "
                   "crps, crps_oe, spread, ssr, ssr_oe)")
    p.add_argument("--out", help="write the scorecard JSON here")
    p.add_argument("--pairs", help="write the matched pairs (Parquet) here")
    p.add_argument("--no-height-correction", action="store_true",
                   help="do not move 2 m temperature to station height")
    p.add_argument("--max-dz", type=_optional_float, default=MAX_DZ, metavar="M",
                   help=f"drop surface pairs where model terrain and station differ by more than M m "
                        f"(default {MAX_DZ:g}; 'none' keeps all)")
    p.add_argument("--any-thermometer-height", action="store_true",
                   help="keep t2m/td2m from thermometers outside the 1.25-2.5 m siting range")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("db", help="build and inspect the observation database")
    p.add_argument("--db", help="database path")
    dsub = p.add_subparsers(dest="db_cmd", required=True)
    dsub.add_parser("init", help="create an empty database")
    dsub.add_parser("status", help="rows, stations and time span per network and variable, and QC rates")
    g = dsub.add_parser("ghcnh", help="ingest GHCNh hourly surface observations")
    g.add_argument("--year", type=int, nargs="+", required=True)
    g.add_argument("--bbox", help="lon_min,lat_min,lon_max,lat_max or a region name")
    g.add_argument("--id", nargs="+", help="GHCN station ids")
    g.add_argument("--max-stations", type=int)
    g.add_argument("--workers", type=int, default=8)
    r = dsub.add_parser("soundings", help="ingest radiosonde mandatory levels from the scorvec.com mirror")
    r.add_argument("--root", default=None, help="mirror URL or local checkout of the skewt-data branch")
    r.add_argument("--since", help="only ascents at or after this time")
    r.add_argument("--bbox")
    pr = dsub.add_parser("precip", help="add precipitation analyses (Stage IV over CONUS; GPCP and IMERG Late "
                         "globally, which need EARTHDATA_USERNAME / EARTHDATA_PASSWORD)")
    pr.add_argument("--product", nargs="+", default=["stageiv", "gpcp", "imerg_late"],
                    choices=["stageiv", "gpcp", "imerg_late"])
    pr.add_argument("--start", required=True, help="first window end, e.g. 2026-10-01")
    pr.add_argument("--end", help="last window end (default now)")
    sn = dsub.add_parser("sensors", help="load thermometer and anemometer heights from a CSV")
    sn.add_argument("file", help="CSV with one of station_id / wmo / icao, and t_height_m and/or wind_height_m")
    e = dsub.add_parser("export", help="write the Parquet snapshot")
    e.add_argument("out")
    p.set_defaults(fn=cmd_db)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if getattr(a, "db_cmd", None) == "soundings" and a.root is None:
        from mverify.ingest.soundings import MIRROR
        a.root = MIRROR
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
