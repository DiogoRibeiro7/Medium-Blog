"""E-OBS daily weather aggregated to NUTS-2 regions (METHODS.md, section 3.2).

Nonlinear quantities (degree days, time in temperature bins, potential
evapotranspiration) are computed per grid cell and day, then averaged over each region
with weights proportional to the area of the cell inside the region times the cosine
of the cell's latitude. A region-day is missing when less than half of the region's
weight is observed; a region-season is missing when any of its days is.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from eu_wheat_weather.contract import EOBS_FILES, DataError

Array = NDArray[np.float64]
STEP = 0.25
EPOCH = dt.date(1950, 1, 1)
MIN_COVERAGE = 0.5
ELEVATION_MAX = 800.0
THRESHOLDS: tuple[float, ...] = (27.0, 30.0, 33.0)
BIN_EDGES: tuple[float, ...] = tuple(float(x) for x in range(0, 37, 3))   # 0, 3, ..., 36
# Windows within the season read (1 March to 31 October of the harvest year).
WINDOWS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "mar_jul": ((3, 1), (7, 31)),    # primary
    "apr_jul": ((4, 1), (7, 31)),    # sensitivity
    "aug_sep": ((8, 1), (9, 30)),    # placebo (E5)
    "jul": ((7, 1), (7, 31)),        # descriptive, post hoc: when the heat falls (change log)
}
# Amendment A1 (section 8.1): windows that end at each region's maturity date.
REGIONAL_WINDOWS: tuple[str, ...] = ("harvest", "harvest_m14", "post_harvest")
EARLY_END_DAYS = 14        # the sensitivity: the window ends 14 days before maturity
PLACEBO_DAYS = 61          # E5-A: the 61 days after the maturity date
CALENDAR_YEAR = 2001       # a non-leap year, in which the calendar's day of year is read
READ_FROM, READ_TO = (3, 1), (10, 31)   # to 31 October, so that E5-A fits (section 8.1)
WEIGHTINGS: tuple[str, ...] = ("all", "below_800m")


def bin_names() -> list[str]:
    edges = [int(e) for e in BIN_EDGES]
    names = ["t_lt00"] + [f"t_{a:02d}_{b:02d}" for a, b in zip(edges, edges[1:], strict=False)]
    return [*names, f"t_ge{edges[-1]:02d}"]


VARIABLES: tuple[str, ...] = (
    "gdd_0_27", "gdd_0_30", "gdd_0_33", "edd_27", "edd_30", "edd_33",
    "prec", "pet", "wb", *bin_names(),
)


# ----------------------------------------------------------------------------------
# Daily formulas, per cell
# ----------------------------------------------------------------------------------


def _sine_terms(tn: Array, tx: Array, base: float) -> tuple[Array, Array, Array]:
    lo, hi = np.minimum(tn, tx), np.maximum(tn, tx)
    mean, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arcsin(np.clip((base - mean) / np.where(half > 0, half, 1.0), -1.0, 1.0))
    return mean, half, theta


def degree_days(tn: Array, tx: Array, base: float) -> Array:
    """Degree days above ``base`` with a sine curve between ``tn`` and ``tx`` (Allen 1976)."""
    mean, half, theta = _sine_terms(tn, tx, base)
    partial = ((mean - base) * (np.pi / 2 - theta) + half * np.cos(theta)) / np.pi
    hi, lo = np.maximum(tn, tx), np.minimum(tn, tx)
    out = np.where(hi <= base, 0.0, np.where(lo >= base, mean - base, partial))
    return np.asarray(out, dtype=np.float64)


def time_above(tn: Array, tx: Array, base: float) -> Array:
    """Fraction of the day above ``base`` under the same sine curve."""
    _, _, theta = _sine_terms(tn, tx, base)
    hi, lo = np.maximum(tn, tx), np.minimum(tn, tx)
    out = np.where(hi <= base, 0.0, np.where(lo >= base, 1.0, (np.pi / 2 - theta) / np.pi))
    return np.asarray(out, dtype=np.float64)


def extraterrestrial_radiation(latitude: Array, day_of_year: int) -> Array:
    """Daily extraterrestrial radiation, MJ m-2 day-1 (FAO-56, equations 21-25)."""
    phi = np.radians(latitude)
    dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
    delta = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * math.tan(delta), -1.0, 1.0))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * math.sin(delta) + np.cos(phi) * math.cos(delta) * np.sin(ws))
    return np.asarray(ra, dtype=np.float64)


def hargreaves(tn: Array, tx: Array, ra: Array) -> Array:
    """Potential evapotranspiration, mm/day (Hargreaves and Samani 1985)."""
    mean = (tn + tx) / 2.0
    out = 0.0023 * 0.408 * ra * (mean + 17.8) * np.sqrt(np.maximum(tx - tn, 0.0))
    return np.asarray(out, dtype=np.float64)


# ----------------------------------------------------------------------------------
# Cells to regions
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionWeights:
    """Weights of the grid cells (flat indices into lat x lon) in each region."""

    codes: tuple[str, ...]
    cells: NDArray[np.int64]
    matrix: Array            # regions x cells
    cell_lat: Array

    def masked(self, keep: NDArray[np.bool_]) -> RegionWeights:
        return RegionWeights(self.codes, self.cells, self.matrix * keep[None, :], self.cell_lat)


def cell_weights(regions: Mapping[str, BaseGeometry], lat: Array, lon: Array,
                 ) -> RegionWeights:
    """Cell area inside each region times the cosine of the cell's latitude."""
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
    flat_lat, flat_lon = grid_lat.ravel(), grid_lon.ravel()
    boxes = shapely.box(flat_lon - STEP / 2, flat_lat - STEP / 2, flat_lon + STEP / 2,
                        flat_lat + STEP / 2)
    tree = shapely.STRtree(boxes)
    codes = tuple(sorted(regions))
    entries: list[tuple[int, NDArray[np.int64], Array]] = []
    for r, code in enumerate(codes):
        idx = np.asarray(tree.query(regions[code], predicate="intersects"), dtype=np.int64)
        if idx.size == 0:
            continue
        area = shapely.area(shapely.intersection(boxes[idx], regions[code]))
        weight = np.asarray(area, dtype=np.float64) * np.cos(np.radians(flat_lat[idx]))
        keep = weight > 0
        entries.append((r, idx[keep], weight[keep]))
    used = np.unique(np.concatenate([e[1] for e in entries])) if entries else \
        np.zeros(0, dtype=np.int64)
    position = {int(c): i for i, c in enumerate(used)}
    matrix = np.zeros((len(codes), used.size))
    for r, idx, weight in entries:
        matrix[r, [position[int(c)] for c in idx]] = weight
    return RegionWeights(codes, used, matrix, flat_lat[used])


def aggregate(weights: Array, values: Array, observed: NDArray[np.bool_]) -> Array:
    """Region-day means (regions x days); NaN below the coverage threshold.

    ``values`` and ``observed`` are days x cells.
    """
    obs = observed.astype(np.float64)
    numerator = weights @ np.where(observed, values, 0.0).T
    denominator = weights @ obs.T
    total = weights.sum(axis=1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    out[(denominator < MIN_COVERAGE * total) | (total == 0)] = np.nan
    return np.asarray(out, dtype=np.float64)


# ----------------------------------------------------------------------------------
# Seasons
# ----------------------------------------------------------------------------------


def season_days(year: int, read_to: tuple[int, int] = READ_TO) -> list[dt.date]:
    start, end = dt.date(year, *READ_FROM), dt.date(year, *read_to)
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def daily_region_variables(tn: Array, tx: Array, rr: Array, obs_t: NDArray[np.bool_],
                           obs_r: NDArray[np.bool_], days: Sequence[dt.date],
                           weights: RegionWeights, gdd_base: float = 0.0,
                           ) -> dict[str, Array]:
    """Region-day values of every variable (regions x days). Growing degree days count
    from ``gdd_base``: 0 °C for wheat, 10 °C for maize (section 9.1)."""
    w = weights.matrix
    out: dict[str, Array] = {}
    dd0 = degree_days(tn, tx, gdd_base)
    for threshold in THRESHOLDS:
        above = degree_days(tn, tx, threshold)
        out[f"gdd_{int(gdd_base)}_{int(threshold)}"] = aggregate(w, dd0 - above, obs_t)
        out[f"edd_{int(threshold)}"] = aggregate(w, above, obs_t)
    names = bin_names()
    fractions = [time_above(tn, tx, e) for e in BIN_EDGES]
    out[names[0]] = aggregate(w, 1.0 - fractions[0], obs_t)
    for k in range(1, len(BIN_EDGES)):
        out[names[k]] = aggregate(w, fractions[k - 1] - fractions[k], obs_t)
    out[names[-1]] = aggregate(w, fractions[-1], obs_t)
    ra = np.stack([extraterrestrial_radiation(weights.cell_lat, d.timetuple().tm_yday)
                   for d in days])
    out["pet"] = aggregate(w, hargreaves(tn, tx, ra), obs_t)
    out["prec"] = aggregate(w, rr, obs_r)
    out["wb"] = out["prec"] - out["pet"]
    return out


def window_sums(daily: Mapping[str, Array], days: Sequence[dt.date],
                windows: Mapping[str, tuple[tuple[int, int], tuple[int, int]]] | None = None,
                variables: Sequence[str] = VARIABLES) -> dict[str, dict[str, Array]]:
    """Per window, the sum over days of each variable (NaN if any day is missing)."""
    out: dict[str, dict[str, Array]] = {}
    for name, ((m0, d0), (m1, d1)) in (WINDOWS if windows is None else windows).items():
        keep = np.array([(d.month, d.day) >= (m0, d0) and (d.month, d.day) <= (m1, d1)
                         for d in days])
        out[name] = {v: daily[v][:, keep].sum(axis=1) for v in variables}
        out[name]["missing_temp"] = np.isnan(daily["edd_30"][:, keep]).sum(axis=1)
        out[name]["missing_prec"] = np.isnan(daily["prec"][:, keep]).sum(axis=1)
    return out


def regional_window_sums(daily: Mapping[str, Array], days: Sequence[dt.date],
                         ends: Sequence[tuple[int, int] | None],
                         ) -> dict[str, dict[str, Array]]:
    """Amendment A1 (section 8.1): per region, sums over 1 March to its maturity date,
    over 1 March to 14 days before it, and over the 61 days after it.

    ``ends`` gives each region's maturity date as (month, day), in the order of the
    daily arrays' rows; a region without one gets missing values.
    """
    year = days[0].year
    ordinal = np.array([d.toordinal() for d in days])
    start = dt.date(year, 3, 1).toordinal()
    masks = {name: np.zeros((len(ends), len(days)), dtype=bool) for name in REGIONAL_WINDOWS}
    for r, end in enumerate(ends):
        if end is None:
            continue
        last = dt.date(year, *end).toordinal()
        if last + PLACEBO_DAYS > ordinal[-1]:
            raise DataError(f"The placebo window after {end} goes past the days read.")
        masks["harvest"][r] = (ordinal >= start) & (ordinal <= last)
        masks["harvest_m14"][r] = (ordinal >= start) & (ordinal <= last - EARLY_END_DAYS)
        masks["post_harvest"][r] = (ordinal > last) & (ordinal <= last + PLACEBO_DAYS)
    out: dict[str, dict[str, Array]] = {}
    for name, mask in masks.items():
        absent = ~mask.any(axis=1)
        sums = {v: np.where(mask, daily[v], 0.0).sum(axis=1) for v in VARIABLES}
        sums["missing_temp"] = (np.isnan(daily["edd_30"]) & mask).sum(axis=1).astype(float)
        sums["missing_prec"] = (np.isnan(daily["prec"]) & mask).sum(axis=1).astype(float)
        for values in sums.values():
            values[absent] = np.nan
        out[name] = sums
    return out


# ----------------------------------------------------------------------------------
# The crop calendar (Amendment A1, section 8.1)
# ----------------------------------------------------------------------------------


def read_calendar(path: Path, variable: str = "maturity_day",
                  ) -> tuple[Array, Array, Array]:
    """Latitudes, longitudes and a day of the year (NaN where missing): maturity unless
    stated, or planting for maize (section 9.1)."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        day = np.ma.filled(ds.variables[variable][:].astype(np.float64), np.nan)
    return lat, lon, np.asarray(day, dtype=np.float64)


def region_maturity(weights: RegionWeights, lon: Array,
                    calendar: tuple[Array, Array, Array], name: str = "maturity",
                    ) -> list[dict[str, object]]:
    """Each region's maturity day: every E-OBS cell takes the calendar cell containing its
    centre, and the region's day is the weighted mean over its cells that have a value,
    rounded to the nearest day and read as a date in a non-leap year."""
    cal_lat, cal_lon, day = calendar
    cell_lon = lon[weights.cells % lon.size]
    rows = np.abs(weights.cell_lat[:, None] - cal_lat[None, :]).argmin(axis=1)
    cols = np.abs(cell_lon[:, None] - cal_lon[None, :]).argmin(axis=1)
    step = abs(float(cal_lat[1] - cal_lat[0]))
    if (np.abs(cal_lat[rows] - weights.cell_lat).max() > step / 2
            or np.abs(cal_lon[cols] - cell_lon).max() > step / 2):
        raise DataError("An E-OBS cell lies outside the crop calendar's grid.")
    values = day[rows, cols]
    have = np.isfinite(values)
    out: list[dict[str, object]] = []
    for r, code in enumerate(weights.codes):
        w = weights.matrix[r]
        total, covered = float(w.sum()), float(w[have].sum())
        if covered == 0:
            out.append({"geo": code, f"{name}_day": "", f"{name}_date": "",
                        "calendar_share": 0.0})
            continue
        mean = float(w[have] @ values[have]) / covered
        doy = int(math.floor(mean + 0.5))
        date = dt.date(CALENDAR_YEAR, 1, 1) + dt.timedelta(days=doy - 1)
        out.append({"geo": code, f"{name}_day": doy, f"{name}_date": date.strftime("%m-%d"),
                    "calendar_share": round(covered / total, 4)})
    return out


def write_maturity(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("geo", "maturity_day", "maturity_date",
                                                    "calendar_share"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------------
# E-OBS files
# ----------------------------------------------------------------------------------


def _read(path: Path, variable: str, first: int, last: int, cells: NDArray[np.int64],
          ) -> tuple[Array, NDArray[np.bool_]]:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        data = ds.variables[variable][first:last + 1, :, :]
    flat = np.ma.reshape(data, (data.shape[0], -1))[:, cells]
    mask = np.ma.getmaskarray(flat)
    values = np.asarray(np.ma.filled(flat.astype(np.float64), np.nan))
    return values, ~mask & np.isfinite(values)


def read_grid(directory: Path) -> tuple[Array, Array, Array]:
    """Latitudes, longitudes and elevation (NaN where missing) of the E-OBS grid."""
    import netCDF4

    with netCDF4.Dataset(directory / EOBS_FILES["elev"]) as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        elev = np.ma.filled(ds.variables["elevation"][:].astype(np.float64), np.nan)
    with netCDF4.Dataset(directory / EOBS_FILES["tx"]) as ds:
        if not (np.array_equal(ds.variables["latitude"][:], lat)
                and np.array_equal(ds.variables["longitude"][:], lon)):
            raise DataError("The E-OBS files are on different grids.")
        units = ds.variables["time"].units
        if units != "days since 1950-01-01 00:00":
            raise DataError(f"Unexpected E-OBS time units: {units}.")
    return lat, lon, np.asarray(elev, dtype=np.float64)


def build_weather(eobs_dir: Path, regions: Mapping[str, BaseGeometry],
                  years: Sequence[int], calendar: Path | None = None,
                  ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """One row per region, harvest year, window and weighting; and, with a crop calendar,
    each region's maturity day, which adds the windows of Amendment A1."""
    lat, lon, elevation = read_grid(eobs_dir)
    base = cell_weights(regions, lat, lon)
    elev_cells = elevation.ravel()[base.cells]
    weightings = {"all": base,
                  "below_800m": base.masked(np.isfinite(elev_cells)
                                            & (elev_cells < ELEVATION_MAX))}
    maturity = [] if calendar is None else region_maturity(base, lon, read_calendar(calendar))
    ends: list[tuple[int, int] | None] = [
        None if m["maturity_date"] == "" else
        (int(str(m["maturity_date"])[:2]), int(str(m["maturity_date"])[3:]))
        for m in maturity]
    rows: list[dict[str, object]] = []
    for year in years:
        days = season_days(year)
        first, last = (days[0] - EPOCH).days, (days[-1] - EPOCH).days
        tx, obs_x = _read(eobs_dir / EOBS_FILES["tx"], "tx", first, last, base.cells)
        tn, obs_n = _read(eobs_dir / EOBS_FILES["tn"], "tn", first, last, base.cells)
        rr, obs_r = _read(eobs_dir / EOBS_FILES["rr"], "rr", first, last, base.cells)
        obs_t = obs_x & obs_n
        tn, tx = np.where(obs_t, tn, 0.0), np.where(obs_t, tx, 0.0)
        rr = np.where(obs_r, rr, 0.0)
        for label, weights in weightings.items():
            daily = daily_region_variables(tn, tx, rr, obs_t, obs_r, days, weights)
            windows = window_sums(daily, days)
            if ends:
                windows.update(regional_window_sums(daily, days, ends))
            for window, sums in windows.items():
                for r, code in enumerate(weights.codes):
                    row: dict[str, object] = {"geo": code, "year": year, "window": window,
                                              "weighting": label}
                    for v in ("missing_temp", "missing_prec", *VARIABLES):
                        x = float(sums[v][r])
                        row[v] = "" if math.isnan(x) else (int(x) if v.startswith("missing")
                                                           else round(x, 6))
                    rows.append(row)
    return rows, maturity


def region_latitudes(regions: Mapping[str, BaseGeometry]) -> list[dict[str, object]]:
    """Latitude of each region's centroid in degrees north (for descriptive splits)."""
    return [{"geo": code, "latitude": round(float(geometry.centroid.y), 4)}
            for code, geometry in sorted(regions.items())]


def write_latitudes(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("geo", "latitude"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


COLUMNS = ("geo", "year", "window", "weighting", "missing_temp", "missing_prec", *VARIABLES)


def write_weather(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------------
# Grain maize (METHODS.md, section 9.1)
# ----------------------------------------------------------------------------------

MAIZE_GDD_BASE = 10.0
MAIZE_VARIABLES: tuple[str, ...] = tuple(v.replace("gdd_0_", "gdd_10_") for v in VARIABLES)
MAIZE_WINDOWS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "may_sep": ((5, 1), (9, 30)),        # sensitivity: a fixed season
}
MAIZE_READ_TO = (12, 31)                 # so that the placebo after maturity fits
MAIZE_COLUMNS = ("geo", "year", "window", "weighting", "missing_temp", "missing_prec",
                 *MAIZE_VARIABLES)
SEASON_COLUMNS = ("geo", "planting_day", "planting_date", "maturity_day", "maturity_date",
                  "calendar_share")


def season_window_sums(daily: Mapping[str, Array], days: Sequence[dt.date],
                       seasons: Sequence[tuple[tuple[int, int], tuple[int, int]] | None],
                       variables: Sequence[str] = MAIZE_VARIABLES,
                       ) -> dict[str, dict[str, Array]]:
    """Per region, sums over its season (planting to maturity, starting no earlier than
    1 March) and over the 61 days after maturity, the placebo of section 9.2.

    ``seasons`` gives each region's (planting, maturity) dates as (month, day) pairs, in
    the order of the daily arrays' rows; a region without them gets missing values.
    """
    year = days[0].year
    ordinal = np.array([d.toordinal() for d in days])
    earliest = dt.date(year, *READ_FROM).toordinal()
    masks = {name: np.zeros((len(seasons), len(days)), dtype=bool)
             for name in ("season", "post_maturity")}
    for r, season in enumerate(seasons):
        if season is None:
            continue
        first = max(dt.date(year, *season[0]).toordinal(), earliest)
        last = dt.date(year, *season[1]).toordinal()
        if last + PLACEBO_DAYS > ordinal[-1] or last < first:
            raise DataError(f"The season {season} does not fit the days read.")
        masks["season"][r] = (ordinal >= first) & (ordinal <= last)
        masks["post_maturity"][r] = (ordinal > last) & (ordinal <= last + PLACEBO_DAYS)
    out: dict[str, dict[str, Array]] = {}
    for name, mask in masks.items():
        absent = ~mask.any(axis=1)
        sums = {v: np.where(mask, daily[v], 0.0).sum(axis=1) for v in variables}
        sums["missing_temp"] = (np.isnan(daily["edd_30"]) & mask).sum(axis=1).astype(float)
        sums["missing_prec"] = (np.isnan(daily["prec"]) & mask).sum(axis=1).astype(float)
        for values in sums.values():
            values[absent] = np.nan
        out[name] = sums
    return out


def region_seasons(weights: RegionWeights, lon: Array, calendar: Path,
                   ) -> list[dict[str, object]]:
    """Each region's planting and maturity dates, both computed as in section 8.1."""
    planting = region_maturity(weights, lon, read_calendar(calendar, "planting_day"),
                               "planting")
    maturity = region_maturity(weights, lon, read_calendar(calendar, "maturity_day"))
    return [{**p, **m, "calendar_share": min(float(str(p["calendar_share"])),
                                             float(str(m["calendar_share"])))}
            for p, m in zip(planting, maturity, strict=True)]


def _month_day(text: object) -> tuple[int, int]:
    return int(str(text)[:2]), int(str(text)[3:])


def build_maize_weather(eobs_dir: Path, regions: Mapping[str, BaseGeometry],
                        years: Sequence[int], calendar: Path,
                        ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """One row per region, harvest year and window (all cells), and each region's
    season (section 9.1)."""
    lat, lon, _ = read_grid(eobs_dir)
    weights = cell_weights(regions, lat, lon)
    seasons = region_seasons(weights, lon, calendar)
    dates: list[tuple[tuple[int, int], tuple[int, int]] | None] = [
        None if s["maturity_date"] == "" or s["planting_date"] == "" else
        (_month_day(s["planting_date"]), _month_day(s["maturity_date"])) for s in seasons]
    rows: list[dict[str, object]] = []
    for year in years:
        days = season_days(year, MAIZE_READ_TO)
        first, last = (days[0] - EPOCH).days, (days[-1] - EPOCH).days
        tx, obs_x = _read(eobs_dir / EOBS_FILES["tx"], "tx", first, last, weights.cells)
        tn, obs_n = _read(eobs_dir / EOBS_FILES["tn"], "tn", first, last, weights.cells)
        rr, obs_r = _read(eobs_dir / EOBS_FILES["rr"], "rr", first, last, weights.cells)
        obs_t = obs_x & obs_n
        tn, tx = np.where(obs_t, tn, 0.0), np.where(obs_t, tx, 0.0)
        rr = np.where(obs_r, rr, 0.0)
        daily = daily_region_variables(tn, tx, rr, obs_t, obs_r, days, weights,
                                       MAIZE_GDD_BASE)
        windows = season_window_sums(daily, days, dates)
        windows.update(window_sums(daily, days, MAIZE_WINDOWS, MAIZE_VARIABLES))
        for window, sums in windows.items():
            for r, code in enumerate(weights.codes):
                row: dict[str, object] = {"geo": code, "year": year, "window": window,
                                          "weighting": "all"}
                for v in ("missing_temp", "missing_prec", *MAIZE_VARIABLES):
                    x = float(sums[v][r])
                    row[v] = "" if math.isnan(x) else (int(x) if v.startswith("missing")
                                                       else round(x, 6))
                rows.append(row)
    return rows, seasons


def write_table(rows: Sequence[Mapping[str, object]], path: Path,
                columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
