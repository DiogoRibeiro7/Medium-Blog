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
# Windows within the season read (1 March to 30 September of the harvest year).
WINDOWS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "mar_jul": ((3, 1), (7, 31)),    # primary
    "apr_jul": ((4, 1), (7, 31)),    # sensitivity
    "aug_sep": ((8, 1), (9, 30)),    # placebo (E5)
    "jul": ((7, 1), (7, 31)),        # descriptive, post hoc: when the heat falls (change log)
}
READ_FROM, READ_TO = (3, 1), (9, 30)
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


def season_days(year: int) -> list[dt.date]:
    start, end = dt.date(year, *READ_FROM), dt.date(year, *READ_TO)
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]


def daily_region_variables(tn: Array, tx: Array, rr: Array, obs_t: NDArray[np.bool_],
                           obs_r: NDArray[np.bool_], days: Sequence[dt.date],
                           weights: RegionWeights) -> dict[str, Array]:
    """Region-day values of every variable (regions x days)."""
    w = weights.matrix
    out: dict[str, Array] = {}
    dd0 = degree_days(tn, tx, 0.0)
    for threshold in THRESHOLDS:
        above = degree_days(tn, tx, threshold)
        out[f"gdd_0_{int(threshold)}"] = aggregate(w, dd0 - above, obs_t)
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
                ) -> dict[str, dict[str, Array]]:
    """Per window, the sum over days of each variable (NaN if any day is missing)."""
    out: dict[str, dict[str, Array]] = {}
    for name, ((m0, d0), (m1, d1)) in WINDOWS.items():
        keep = np.array([(d.month, d.day) >= (m0, d0) and (d.month, d.day) <= (m1, d1)
                         for d in days])
        out[name] = {v: daily[v][:, keep].sum(axis=1) for v in VARIABLES}
        out[name]["missing_temp"] = np.isnan(daily["edd_30"][:, keep]).sum(axis=1)
        out[name]["missing_prec"] = np.isnan(daily["prec"][:, keep]).sum(axis=1)
    return out


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
                  years: Sequence[int]) -> list[dict[str, object]]:
    """One row per region, harvest year, window and weighting."""
    lat, lon, elevation = read_grid(eobs_dir)
    base = cell_weights(regions, lat, lon)
    elev_cells = elevation.ravel()[base.cells]
    weightings = {"all": base,
                  "below_800m": base.masked(np.isfinite(elev_cells)
                                            & (elev_cells < ELEVATION_MAX))}
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
            for window, sums in window_sums(daily, days).items():
                for r, code in enumerate(weights.codes):
                    row: dict[str, object] = {"geo": code, "year": year, "window": window,
                                              "weighting": label}
                    for v in ("missing_temp", "missing_prec", *VARIABLES):
                        x = float(sums[v][r])
                        row[v] = "" if math.isnan(x) else (int(x) if v.startswith("missing")
                                                           else round(x, 6))
                    rows.append(row)
    return rows


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
