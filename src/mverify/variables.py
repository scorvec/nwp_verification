"""Canonical variable catalogue.

Every forecast field and every observation is mapped onto one canonical name
(``t2m``, ``msl``, ``z500``, ``t850`` ...) in one canonical unit, so matching
never has to care where a number came from. A forecast variable is recognised
by any of: its GRIB shortName + level, its CF ``standard_name``, or a common
netCDF/zarr alias (WRF, CMIP, ERA5, ECMWF open data, GRIB2-to-netCDF names).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G0 = 9.80665


@dataclass(frozen=True)
class Var:
    name: str
    label: str
    unit: str                       # canonical unit
    kind: str                       # "surface" or "upper"
    level: float | None = None      # hPa for upper-air variables
    grib: tuple[str, ...] = ()      # cfgrib shortNames
    grib_level: str | None = None   # cfgrib typeOfLevel
    standard_names: tuple[str, ...] = ()
    aliases: tuple[str, ...] = field(default=())
    lo: float = -1e9                # physical range for QC, canonical unit
    hi: float = 1e9
    sigma_o: float | None = None    # observation error SD, canonical unit (see SIGMA_O_UPPER)
    grib2: tuple[tuple[int, int, int], ...] = ()  # WMO (discipline, category, number), for messages eccodes cannot name


LEVELS = (1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100)
# Radiosonde humidity is unreliable above about 300 hPa (slow, dry-biased sensors in the cold), so
# specific humidity is verified only below it.
HUMIDITY_LEVELS = (1000, 925, 850, 700, 500, 400, 300)

# Observation error. sigma_o is the standard deviation of the difference between an
# observation and the grid-scale truth it is compared with: instrument error plus
# representativeness error, which dominates for a 10-25 km model. The values below are
# approximate starting points of the size global data assimilation assigns to these
# observation types (ECMWF and NCEP observation-error tables), not measured here; they are
# meant to be replaced by estimates from the database itself (Desroziers et al. 2005).
_SIGMA_O = {
    "z": (5.0, 5.5, 6.0, 7.0, 9.0, 10.5, 12.0, 13.0, 14.0, 15.5, 17.0),     # m
    "t": (1.2, 1.1, 1.0, 0.9, 0.8, 0.85, 0.9, 1.0, 1.1, 1.1, 1.2),           # K
    "u": (1.8, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.6, 2.5, 2.4, 2.3),           # m s-1, each component
    "v": (1.8, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.6, 2.5, 2.4, 2.3),
    "q": (1.0, 0.9, 0.8, 0.6, 0.35, 0.2, 0.1),                              # g kg-1, HUMIDITY_LEVELS
}
SIGMA_O_UPPER = {b: dict(zip(HUMIDITY_LEVELS if b == "q" else LEVELS, s)) for b, s in _SIGMA_O.items()}


def _upper(base: str, level: int, label: str, unit: str, grib: tuple[str, ...], std: tuple[str, ...],
           lo: float, hi: float) -> Var:
    return Var(f"{base}{level}", f"{level} hPa {label}", unit, "upper", float(level), grib,
               "isobaricInhPa", std, (f"{base}{level}", f"{base.upper()}{level}"), lo, hi,
               SIGMA_O_UPPER[base][level])


SURFACE = [
    Var("t2m", "2 m temperature", "degC", "surface", grib=("2t", "t2m"), grib_level="heightAboveGround",
        standard_names=("air_temperature",),
        aliases=("t2m", "2t", "T2", "tas", "TMP_2maboveground", "air_temperature_2m", "temperature_2m"),
        lo=-90, hi=60, sigma_o=1.4),
    Var("td2m", "2 m dew point", "degC", "surface", grib=("2d", "d2m"), grib_level="heightAboveGround",
        standard_names=("dew_point_temperature",),
        aliases=("d2m", "2d", "td2m", "TD2", "DPT_2maboveground", "dewpoint_temperature_2m"),
        lo=-90, hi=40, sigma_o=1.6),
    Var("msl", "mean sea-level pressure", "hPa", "surface", grib=("prmsl", "msl", "mslet"),
        grib_level="meanSea", standard_names=("air_pressure_at_mean_sea_level", "air_pressure_at_sea_level"),
        aliases=("msl", "prmsl", "PRMSL_meansealevel", "MSLP", "psl", "slp", "mslp"), lo=870, hi=1090,
        sigma_o=1.0),
    # Station-level pressure, against the model's surface pressure with no reduction: only where the model
    # terrain is within match.MAX_DZ_PRESSURE of the station (mean sea-level pressure is a reduction below
    # ground that each model and station does its own way).
    Var("sp", "surface pressure", "hPa", "surface", grib=("sp", "pres"), grib_level="surface",
        standard_names=("surface_air_pressure",),
        aliases=("sp", "ps", "PSFC", "PRES_surface", "surface_pressure"), lo=400, hi=1100, sigma_o=0.5),
    Var("u10", "10 m zonal wind", "m s-1", "surface", grib=("10u", "u10"), grib_level="heightAboveGround",
        standard_names=("eastward_wind",), aliases=("u10", "10u", "U10", "uas", "UGRD_10maboveground"),
        lo=-75, hi=75, sigma_o=1.8),
    Var("v10", "10 m meridional wind", "m s-1", "surface", grib=("10v", "v10"),
        grib_level="heightAboveGround", standard_names=("northward_wind",),
        aliases=("v10", "10v", "V10", "vas", "VGRD_10maboveground"), lo=-75, hi=75,
        sigma_o=1.8),
    Var("ws10", "10 m wind speed", "m s-1", "surface", grib=("10si", "si10"), grib_level="heightAboveGround",
        standard_names=("wind_speed",), aliases=("si10", "10si", "ws10", "wind_speed_10m", "sfcWind"),
        lo=0, hi=75, sigma_o=1.6),
    Var("orog", "model orography", "m", "surface", grib=("orog", "z", "hgt"), grib_level="surface",
        standard_names=("surface_altitude", "surface_geopotential"),
        aliases=("orog", "HGT_surface", "HGT", "z_sfc", "zs", "terrain", "orography"), lo=-500, hi=9000),
    Var("fsr", "model surface roughness", "m", "surface", grib=("fsr", "sr"), grib_level="surface",
        standard_names=("surface_roughness_length",),
        aliases=("fsr", "sr", "SFCR", "SFCR_surface", "znt", "ZNT", "z0", "surface_roughness"), lo=0, hi=10),
    Var("lsm", "model land fraction", "1", "surface", grib=("lsm", "land"), grib_level="surface",
        standard_names=("land_binary_mask", "land_area_fraction"),
        aliases=("lsm", "LAND", "LAND_surface", "landmask", "LANDMASK", "sftlf", "land_sea_mask"), lo=0, hi=1),
]

# Fields describing the model surface rather than the atmosphere: read from any file of a
# forecast, attached to every field on the same grid, never scored.
STATIC = ("orog", "fsr", "lsm")

# Accumulated precipitation is verified grid-to-grid against analyses (mverify.precip), not
# against station rows. A field of it covers the lead-time window Field.accum.
# (0, 1, 8) is total precipitation in kg m-2; eccodes calls it "unknown" in ECCC's GDPS/RDPS files.
PRECIP = Var("tp", "accumulated precipitation", "mm", "surface", grib=("tp", "apcp"), grib_level="surface",
             grib2=((0, 1, 8),),
             standard_names=("precipitation_amount", "lwe_thickness_of_precipitation_amount"),
             aliases=("tp", "apcp", "APCP", "APCP_surface", "total_precipitation", "precip_acc"), lo=0, hi=2000)
GRIDDED = ("tp",)
# 24 h totals reported by station gauges (GHCNh precipitation_24_hour), matched to each run's total over the same
# 24 h (mverify.match): never read from a file, so no file names
GAUGE = Var("pr24", "24 h precipitation at gauges", "mm", "surface", lo=0, hi=1000)

UPPER = [
    *(_upper("z", lv, "geopotential height", "m", ("gh", "z", "hgt"),
             ("geopotential_height", "geopotential"), lo, hi)
      # Limits sit outside the observed extremes (deep polar lows, subtropical highs); gross errors
      # inside them are left to the hydrostatic check.
      for lv, lo, hi in [(1000, -700, 700), (925, 150, 1300), (850, 700, 1900), (700, 2150, 3500),
                         (500, 4400, 6200), (400, 6000, 8000), (300, 7600, 10000), (250, 8800, 11300),
                         (200, 10000, 12800), (150, 12300, 14900), (100, 14800, 17300)]),
    *(_upper("t", lv, "temperature", "degC", ("t", "tmp"), ("air_temperature",), -100, 50)
      for lv in LEVELS),
    *(_upper("u", lv, "zonal wind", "m s-1", ("u", "ugrd"), ("eastward_wind",), -120, 120)
      for lv in LEVELS),
    *(_upper("v", lv, "meridional wind", "m s-1", ("v", "vgrd"), ("northward_wind",), -120, 120)
      for lv in LEVELS),
    *(_upper("q", lv, "specific humidity", "g kg-1", ("q", "spfh"), ("specific_humidity",), 0, 40)
      for lv in HUMIDITY_LEVELS),
]

CATALOG: dict[str, Var] = {v.name: v for v in [*SURFACE, *UPPER, PRECIP, GAUGE]}

# Quantities a model may only provide through components: name -> (components, function).
DERIVED = {"ws10": (("u10", "v10"), lambda u, v: (u**2 + v**2) ** 0.5)}

# Base names of pressure-level fields as they appear in netCDF / zarr files.
UPPER_ALIASES = {
    "z": ("z", "gh", "hgt", "zg", "geopotential", "geopotential_height", "height", "phi", "gph"),
    "t": ("t", "tmp", "ta", "temp", "temperature", "air_temperature", "air"),
    "u": ("u", "ugrd", "ua", "uwnd", "u_component_of_wind", "eastward_wind"),
    "v": ("v", "vgrd", "va", "vwnd", "v_component_of_wind", "northward_wind"),
    "q": ("q", "spfh", "hus", "shum", "qv", "specific_humidity"),
}


def specific_humidity(td, p):
    """Specific humidity in g kg-1 from dew point (degC) and pressure (hPa).

    Vapour pressure over water (Bolton 1980), as radiosonde dew points are reported
    with respect to water at every temperature (WMO No. 8).
    """
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    return 622.0 * e / (p - 0.378 * e)


def get(name: str) -> Var:
    return CATALOG[name]


def to_canonical(values, unit: str | None, var: Var):
    """Convert ``values`` (numpy/xarray) from ``unit`` into ``var.unit``.

    When the unit is missing or unrecognised the magnitude decides, which
    catches the common unlabelled cases (Kelvin, Pa, geopotential in m2 s-2).
    """
    u = (unit or "").strip().lower().replace("**", "").replace("^", "")
    if var.unit == "degC":
        if u in ("k", "kelvin") or (not u and float(abs(values).max()) > 150):
            return values - 273.15
        if u in ("degf", "f", "fahrenheit"):
            return (values - 32.0) / 1.8
        return values
    if var.unit == "hPa":
        if u in ("pa",) or (not u and float(values.max()) > 5000):
            return values / 100.0
        return values
    if var.unit == "m":
        if u in ("m2 s-2", "m2/s2", "m2s-2", "m+2 s-2") or ("s-2" in u):
            return values / G0
        if u in ("dam", "gpdm"):
            return values * 10.0
        if not u and var.level is not None and var.level <= 700 and float(values.max()) > 30000:
            return values / G0
        return values
    if var.unit == "1":  # fractions, sometimes given in percent (CMIP sftlf)
        if u in ("%", "percent") or float(values.max()) > 1.5:
            return values / 100.0
        return values
    if var.unit == "mm":  # kg m-2 is mm of water; ECMWF accumulates in metres
        if u in ("m", "m of water equivalent"):
            return values * 1000.0
        return values
    if var.unit == "g kg-1":  # models write kg kg-1; without a unit, no real g kg-1 field stays below 0.1
        if u.startswith("g"):
            return values
        if "kg" in u or u == "1" or (not u and float(values.max()) < 0.1):
            return values * 1000.0
        return values
    if var.unit == "m s-1":
        if u in ("kt", "knots", "kts"):
            return values * 0.514444
        if u in ("km h-1", "km/h"):
            return values / 3.6
        return values
    return values
