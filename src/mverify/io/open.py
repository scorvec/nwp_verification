"""Open forecast files of any common format and normalise them to canonical fields.

``open_forecast(paths)`` accepts GRIB 1/2, netCDF 3/4 and Zarr (files, globs,
directories of files, or a Zarr store) and returns a ``Forecast``: a flat list
of 2-D ``Field`` records, one per canonical variable x init x lead x member,
each knowing its grid. Values load lazily, one field at a time, so a season of
global 0.25 deg output never has to sit in memory.

What is detected automatically:
  * format, from magic bytes (not the extension): GRIB, CDF, HDF5, Zarr metadata
  * variables, via mverify.variables (GRIB shortName + level type, CF
    standard_name, or common netCDF/zarr aliases, including names with the
    level embedded such as ``TMP_850mb`` or ``z500``)
  * units, from the ``units`` attribute or, failing that, the magnitude
  * init time and lead: cfgrib time/step; netCDF/zarr init+lead, init+valid,
    or valid alone with the init taken from attributes or the file name
  * grid: regular lat/lon (1-D coordinates, either longitude convention, either
    latitude order) or curvilinear 2-D lat/lon (Lambert, polar stereographic,
    rotated, WRF)
  * ensemble members, pressure levels in hPa or Pa, model orography
"""

from __future__ import annotations

import glob
import logging
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from mverify import variables as V

log = logging.getLogger(__name__)

LAT = ("latitude", "lat", "xlat", "nav_lat", "gridlat_0", "lat_0", "y_lat")
LON = ("longitude", "lon", "xlong", "xlon", "nav_lon", "gridlon_0", "lon_0", "x_lon")
LEVEL = ("isobaricinhpa", "level", "plev", "pressure", "lev", "isobaric", "pressure_level", "air_pressure",
         "lv_isbl0", "levels")
MEMBER = ("number", "member", "realization", "ensemble", "ens", "mem", "ensemble_member")
INIT = ("init_time", "forecast_reference_time", "reftime", "initial_time", "time_ref", "run", "init",
        "analysis_time", "start_time")
LEAD = ("step", "lead_time", "forecast_period", "leadtime", "fhr", "forecast_hour", "prediction_timedelta",
        "lead", "fcst_hour")
VALID = ("valid_time", "time", "times", "xtime", "t")

MODEL_NAMES = ("aifs-ens", "aifs", "ifs-ens", "ifs", "ecmwf", "gefs", "gfs", "hrrr", "rrfs", "nam", "rap",
               "hrdps", "rdps", "gdps", "geps", "icon", "arpege", "arome", "ukmo", "graphcast", "pangu",
               "fourcastnet", "gencast", "aurora", "fuxi", "wrf", "era5")


@dataclass
class Grid:
    lat: np.ndarray            # 1-D (regular) or 2-D (curvilinear)
    lon: np.ndarray            # degrees, any convention
    orog: np.ndarray | None = None

    @property
    def regular(self) -> bool:
        return self.lat.ndim == 1

    @property
    def key(self) -> tuple:
        return (self.lat.shape, self.lon.shape, float(self.lat.flat[0]), float(self.lat.flat[-1]),
                float(self.lon.flat[0]), float(self.lon.flat[-1]))


@dataclass
class Field:
    var: str
    init: pd.Timestamp
    lead: pd.Timedelta
    member: int | None
    grid: Grid
    _da: xr.DataArray = field(repr=False)
    _unit: str | None = field(default=None, repr=False)
    source: str = ""

    @property
    def valid(self) -> pd.Timestamp:
        return self.init + self.lead

    @property
    def lead_h(self) -> float:
        return self.lead / pd.Timedelta("1h")

    def values(self) -> np.ndarray:
        a = np.asarray(self._da.values, dtype="float64")
        return np.asarray(V.to_canonical(a, self._unit, V.CATALOG[self.var]))


@dataclass
class Forecast:
    name: str
    fields: list[Field]

    def variables(self) -> list[str]:
        return sorted({f.var for f in self.fields})

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"var": f.var, "init": f.init, "lead_h": f.lead_h, "member": f.member,
                              "grid": "regular" if f.grid.regular else "curvilinear",
                              "shape": f.grid.lat.shape if not f.grid.regular else
                              (f.grid.lat.size, f.grid.lon.size), "source": f.source} for f in self.fields])


# --------------------------------------------------------------------------- format detection

def detect_format(path: Path) -> str:
    p = Path(path)
    if p.is_dir():
        if any((p / m).exists() for m in (".zgroup", ".zarray", "zarr.json", ".zmetadata")):
            return "zarr"
        raise ValueError(f"{p} is a directory but not a Zarr store")
    if p.suffix == ".zip" and ".zarr" in p.name:
        return "zarr"
    with open(p, "rb") as fh:
        head = fh.read(4096)
    if head.startswith((b"CDF", b"\x89HDF")):
        return "netcdf"
    if b"GRIB" in head:
        return "grib"
    raise ValueError(f"{p}: not GRIB, netCDF or Zarr (first bytes {head[:8]!r})")


def expand(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        s = str(p)
        if any(ch in s for ch in "*?["):
            out += [Path(x) for x in sorted(glob.glob(s))]
        elif Path(s).is_dir() and detect_dir_is_collection(Path(s)):
            out += sorted(q for q in Path(s).iterdir() if q.is_file() and not q.name.startswith(".")
                          and not q.name.endswith((".idx", ".ncx", ".gbx9")))
        else:
            out.append(Path(s))
    return out


def detect_dir_is_collection(p: Path) -> bool:
    return not any((p / m).exists() for m in (".zgroup", ".zarray", "zarr.json", ".zmetadata"))


def _open(path: Path, fmt: str) -> list[xr.Dataset]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt == "grib":
            import cfgrib
            return cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
        if fmt == "netcdf":
            try:
                return [xr.open_dataset(path, decode_timedelta=True)]
            except ValueError:
                return [xr.open_dataset(path, decode_times=False)]
        return [xr.open_zarr(path, consolidated=None, decode_timedelta=True)]


# --------------------------------------------------------------------------- coordinates

def _find(ds: xr.Dataset | xr.DataArray, names: tuple[str, ...]) -> str | None:
    lower = {str(k).lower(): str(k) for k in list(ds.coords) + list(getattr(ds, "data_vars", {}))}
    lower.update({str(k).lower(): str(k) for k in ds.dims})
    for n in names:
        if n in lower:
            return lower[n]
    return None


def _grid(ds: xr.Dataset) -> tuple[Grid, tuple[str, ...]]:
    la, lo = _find(ds, LAT), _find(ds, LON)
    if la is None or lo is None:
        for k in ds.coords:
            sn = ds[k].attrs.get("standard_name", "")
            la = k if sn == "latitude" else la
            lo = k if sn == "longitude" else lo
    if la is None or lo is None:
        raise ValueError("no latitude/longitude coordinates found")
    lat, lon = ds[la], ds[lo]
    while lat.ndim > 2:  # e.g. WRF XLAT(Time, south_north, west_east)
        lat = lat.isel({lat.dims[0]: 0})
    while lon.ndim > 2:
        lon = lon.isel({lon.dims[0]: 0})
    if lat.ndim == 1 and lon.ndim == 1:
        return Grid(lat.values.astype(float), lon.values.astype(float)), (lat.dims[0], lon.dims[0])
    lat2 = lat.values.astype(float)
    lon2 = (lon.values if lon.ndim == 2 else lon.values[None, :]).astype(float)
    return Grid(lat2, lon2), tuple(lat.dims[-2:])


def _pressure_hpa(vals: np.ndarray, units: str | None) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if (units or "").lower() == "pa" or np.nanmax(vals) > 2000:
        return vals / 100.0
    return vals


def _time_from_name(path: Path) -> pd.Timestamp | None:
    s = str(path)
    m = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)[T_.-]?([0-2]\d)(?:z|Z|00)?(?!\d)", s)
    if m:
        y, mo, d, h = map(int, m.groups())
        try:
            return pd.Timestamp(y, mo, d, h)
        except ValueError:
            pass
    m = re.search(r"(20\d{2})([01]\d)([0-3]\d)", s)
    if m:
        hh = re.search(r"[tT]([0-2]\d)[zZ]", s)
        return pd.Timestamp(*map(int, m.groups()), int(hh.group(1)) if hh else 0)
    return None


def _lead_from_name(path: Path) -> pd.Timedelta | None:
    m = re.search(r"(?:\.f|_f|\+|fh|\bf)(\d{2,4})(?:\D|$)", path.name)
    return pd.Timedelta(hours=int(m.group(1))) if m else None


def _as_timedelta(da: xr.DataArray) -> np.ndarray:
    v = da.values
    if np.issubdtype(v.dtype, np.timedelta64):
        return v
    units = da.attrs.get("units", "hours").lower()
    unit = "s" if units.startswith("s") else "m" if units.startswith("min") else "D" if units.startswith("d") else "h"
    return pd.to_timedelta(np.asarray(v, dtype=float), unit=unit).values


def _times(ds: xr.Dataset, is_grib: bool, path: Path, init_hint):
    """Return (init_dim, lead_dim, init values, lead values, mode) describing the time axes."""
    if is_grib:
        return "time", "step", None, None, "grib"
    i, ld, vt = _find(ds, INIT), _find(ds, LEAD), _find(ds, VALID)
    if i and i == vt:
        vt = None
    if i and ld:
        return i, ld, None, None, "init+lead"
    if i and vt:
        return i, vt, None, None, "init+valid"
    if vt:
        init = pd.Timestamp(init_hint) if init_hint is not None else None
        if init is None:
            for key in ("init_time", "initial_time", "forecast_reference_time", "SIMULATION_START_DATE",
                        "START_DATE", "start_date", "base_time"):
                if key in ds.attrs:
                    try:
                        init = pd.Timestamp(str(ds.attrs[key]).replace("_", " "))
                        break
                    except ValueError:
                        pass
        if init is None:
            init = _time_from_name(path)
        if init is None:
            init = pd.Timestamp(np.min(ds[vt].values))
            log.warning("%s: init time not found, assuming the first valid time %s", path, init)
        return None, vt, init, None, "valid"
    init = pd.Timestamp(init_hint) if init_hint is not None else _time_from_name(path)
    lead = _lead_from_name(path)
    if init is None or lead is None:
        raise ValueError(f"{path}: no time coordinate; pass init= explicitly")
    return None, None, init, lead, "static"


# --------------------------------------------------------------------------- variables

def _level_of(da: xr.DataArray, level_name: str | None):
    """(level values in hPa or None, dim name or None) for a variable."""
    if level_name and level_name in da.coords:
        c = da[level_name]
        if level_name.lower() in ("heightaboveground", "surface", "meansea"):
            return None, None
        vals = _pressure_hpa(np.atleast_1d(c.values), c.attrs.get("units"))
        return vals, (level_name if level_name in da.dims else None)
    for cand in LEVEL:
        k = _find(da, (cand,))
        if k and k in da.coords:
            c = da[k]
            vals = _pressure_hpa(np.atleast_1d(c.values), c.attrs.get("units"))
            return vals, (k if k in da.dims else None)
    return None, None


def identify(name: str, da: xr.DataArray, has_level: bool) -> str | None:
    """Canonical surface name or upper-air base ('z', 't', 'u', 'v') for one data variable."""
    low = name.lower()
    attrs = da.attrs
    short = str(attrs.get("GRIB_shortName", "")).lower()
    tol = attrs.get("GRIB_typeOfLevel")
    std = attrs.get("standard_name", "")
    if has_level:
        m = None
        for base, al in V.UPPER_ALIASES.items():
            if low in al or short in al or std in al:
                m = base
        return m
    if tol == "surface" and (short in ("orog", "z", "hgt", "gh") or low in ("orog", "z", "hgt")):
        return "orog"
    for v in V.SURFACE:
        if v.name == "orog":
            if low in (a.lower() for a in v.aliases if a not in ("HGT",)) or std in v.standard_names:
                return "orog"
            continue
        if low in (a.lower() for a in v.aliases) or (short and short in v.grib and (tol in (None, v.grib_level))):
            return v.name
    if std:
        hag = da.coords.get("heightAboveGround", da.coords.get("height"))
        h = float(np.atleast_1d(hag.values)[0]) if hag is not None else None
        for v in V.SURFACE:
            if std in v.standard_names and v.name != "orog":
                if v.name in ("t2m", "td2m") and h not in (None, 2.0):
                    continue
                if v.name in ("u10", "v10") and h not in (None, 10.0):
                    continue
                return v.name
    return None


def _embedded_level(name: str) -> tuple[str, float] | None:
    m = re.fullmatch(r"([A-Za-z]+)[_]?(\d{3,4})(?:mb|hpa|_?mb|_hpa)?", name, flags=re.IGNORECASE)
    if not m:
        return None
    base, lev = m.group(1).lower(), float(m.group(2))
    for b, al in V.UPPER_ALIASES.items():
        if base in al:
            return b, lev
    return None


# --------------------------------------------------------------------------- main entry

def _iter_dataset(ds: xr.Dataset, path: Path, init_hint, fmt: str, orog: dict) -> list[Field]:
    """Fields of one dataset; orography found on the way is stored in ``orog`` by grid key."""
    is_grib = any("GRIB_shortName" in ds[v].attrs for v in ds.data_vars)
    grid, (ydim, xdim) = _grid(ds)
    idim, ldim, init_c, lead_c, mode = _times(ds, is_grib, path, init_hint)
    mdim = _find(ds, MEMBER)
    level_name = _find(ds, LEVEL)
    skip = {_find(ds, LAT), _find(ds, LON)}
    found: list[tuple[str, xr.DataArray]] = []
    for name, da in ds.data_vars.items():
        if name in skip or ydim not in da.dims or xdim not in da.dims:
            continue
        levels, ldim_v = _level_of(da, level_name)
        emb = None if levels is not None else _embedded_level(name)
        if emb:
            base, lev = emb
            if f"{base}{int(lev)}" in V.CATALOG:
                found.append((f"{base}{int(lev)}", da))
            continue
        canon = identify(name, da, levels is not None)
        if canon is None:
            continue
        if levels is None:
            if canon in V.CATALOG:
                found.append((canon, da))
            continue
        for k, lev in enumerate(levels):
            lv = round(float(lev))
            vname = f"{canon}{lv}"
            if vname in V.CATALOG and abs(lev - lv) < 1e-6:
                found.append((vname, da.isel({ldim_v: k}) if ldim_v else da))
    fields: list[Field] = []
    for vname, da in found:
        if vname == "orog":
            o = da
            for d in [d for d in o.dims if d not in (ydim, xdim)]:
                o = o.isel({d: 0})
            orog[grid.key] = V.to_canonical(np.asarray(o.values, float), o.attrs.get("units"),
                                            V.CATALOG["orog"])
            continue
        unit = da.attrs.get("units")
        other = [d for d in da.dims if d not in (ydim, xdim)]
        sizes = [da.sizes[d] for d in other]
        for idx in np.ndindex(*sizes) if sizes else [()]:
            sel = dict(zip(other, idx))
            sub = da.isel(sel) if sel else da
            init, lead = _field_time(sub, mode, idim, ldim, init_c, lead_c)
            if init is None:
                continue
            member = None
            if mdim and mdim in sub.coords and (mdim in da.dims):
                member = int(sub[mdim].values)
            fields.append(Field(vname, init, lead, member, grid, sub.transpose(ydim, xdim), unit,
                                source=f"{path.name}:{fmt}"))
    return fields


def _field_time(sub: xr.DataArray, mode, idim, ldim, init_c, lead_c):
    def scalar(name):
        if name not in sub.coords:
            return None
        return sub[name].values.item() if sub[name].ndim == 0 else None
    if mode == "grib":
        t, s = scalar("time"), scalar("step")
        if t is None:
            return None, None
        return pd.Timestamp(t), pd.Timedelta(s if s is not None else 0)
    if mode == "init+lead":
        return pd.Timestamp(scalar(idim)), pd.Timedelta(_as_timedelta(sub[ldim]).item())
    if mode == "init+valid":
        i, v = pd.Timestamp(scalar(idim)), pd.Timestamp(scalar(ldim))
        return i, v - i
    if mode == "valid":
        v = scalar(ldim)
        if v is None:
            return None, None
        return init_c, pd.Timestamp(v) - init_c
    return init_c, lead_c


def guess_model_name(paths: list[Path]) -> str:
    s = " ".join(str(p).lower() for p in paths[:20])
    for n in MODEL_NAMES:
        if re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", s):
            return n.upper()
    return paths[0].stem.split(".")[0] if paths else "model"


def open_forecast(paths, name: str | None = None, init=None) -> Forecast:
    """Open any mix of GRIB / netCDF / Zarr paths (globs and directories allowed) as one Forecast."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = expand(paths)
    if not files:
        raise FileNotFoundError(f"nothing matched {paths}")
    fields: list[Field] = []
    orog: dict[tuple, np.ndarray] = {}
    for f in files:
        fmt = detect_format(f)
        for ds in _open(f, fmt):
            try:
                fields += _iter_dataset(ds, f, init, fmt, orog)
            except ValueError as e:
                log.warning("%s: skipped a dataset (%s)", f, e)
    # Orography found in any file applies to every field on the same grid.
    for f in fields:
        if f.grid.orog is None and f.grid.key in orog:
            f.grid.orog = orog[f.grid.key]
    return Forecast(name or guess_model_name(files), fields)
