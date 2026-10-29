"""Amendment A2 (METHODS.md, section 10.1): weather over where maize grows.

The MIRCA-OS map gives the 2015 harvested area of irrigated and of rainfed maize, in
hectares, on a global 5-arcminute grid. Each 0.25° E-OBS cell holds 3 x 3 of its cells,
and the cell's maize is their sum. A cell's weight in a region is the share of the cell
inside the region times the cell's maize: maize is taken as spread evenly within a cell.
The season dates are the registered ones (``docs/maize/season.csv``), so only the
weighting changes.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from eu_wheat_weather.contract import DataError
from eu_wheat_weather.weather import (
    ELEVATION_MAX,
    MAIZE_READ_TO,
    STEP,
    Array,
    RegionWeights,
    cell_weights,
    maize_daily,
    maize_row,
    read_grid,
    season_days,
    season_window_sums,
)

RESOLUTION = 1 / 12                     # degrees: the map's 5 arcminutes
PER_CELL = round(STEP / RESOLUTION)     # 3 map cells along each side of an E-OBS cell
SHAPE = (2160, 4320)                    # the global grid, from 90°N and from 180°W
NODATA = -9.0
WEIGHTINGS_A2: tuple[str, ...] = ("all", "maize", "below_800m")
WINDOWS_A2: tuple[str, ...] = ("season", "post_maturity")
MAP_COLUMNS = ("geo", "maize_ha", "irrigated_ha", "irrigated_share", "maize_cells")


def read_layer(path: Path) -> Array:
    """A MIRCA-OS layer, in hectares per map cell; no data counts as no maize."""
    from PIL import Image, TiffImagePlugin

    with Image.open(path) as image:
        if not isinstance(image, TiffImagePlugin.TiffImageFile):
            raise DataError(f"{path.name} is not a TIFF file.")
        tags = image.tag_v2
        scale, tie = tags.get(33550), tags.get(33922)
        if (image.size != SHAPE[::-1] or scale is None or tie is None
                or not np.allclose(scale[:2], RESOLUTION)
                or not np.allclose(tie[3:5], (-180.0, 90.0), atol=1e-6)):
            raise DataError(f"{path.name} is not the global 5-arcminute grid expected.")
        grid = np.asarray(image, dtype=np.float64)
    grid[(grid == NODATA) | ~np.isfinite(grid)] = 0.0
    if (grid < 0).any():
        raise DataError(f"{path.name} has negative areas.")
    return grid


def eobs_cells(grid: Array, lat: Array, lon: Array, cells: NDArray[np.int64]) -> Array:
    """Each E-OBS cell's maize (``cells`` are flat indices into lat x lon): the sum of the
    3 x 3 map cells it holds."""
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing="ij")
    top = (90.0 - (grid_lat.ravel()[cells] + STEP / 2)) / RESOLUTION
    left = (grid_lon.ravel()[cells] - STEP / 2 + 180.0) / RESOLUTION
    if not (np.allclose(top, np.round(top), atol=1e-6)
            and np.allclose(left, np.round(left), atol=1e-6)):
        raise DataError("The E-OBS cells do not align with the 5-arcminute grid.")
    row, col = np.round(top).astype(np.int64), np.round(left).astype(np.int64)
    out = np.zeros(cells.size)
    for i in range(PER_CELL):
        for j in range(PER_CELL):
            out += grid[row + i, col + j]
    return out


def shares(base: RegionWeights) -> Array:
    """The share of each cell's area inside each region (regions x cells). The base
    weights are the area inside the region, in square degrees, times the cosine of the
    latitude, which also scales the cell's own area."""
    return base.matrix / (np.cos(np.radians(base.cell_lat)) * STEP ** 2)[None, :]


def maize_weights(base: RegionWeights, maize: Array) -> RegionWeights:
    """Section 10.1: the share of each cell inside the region times the cell's maize."""
    return RegionWeights(base.codes, base.cells, shares(base) * maize[None, :], base.cell_lat)


def region_map(base: RegionWeights, irrigated: Array, rainfed: Array,
               ) -> list[dict[str, object]]:
    """Per region: its maize and irrigated maize on the map, the irrigated share (section
    10.3) and the number of cells with maize."""
    share = shares(base)
    total, irr = share @ (irrigated + rainfed), share @ irrigated
    cells = ((share > 0) & ((irrigated + rainfed) > 0)[None, :]).sum(axis=1)
    return [{"geo": code, "maize_ha": round(float(total[r]), 3),
             "irrigated_ha": round(float(irr[r]), 3),
             "irrigated_share": "" if total[r] == 0 else round(float(irr[r] / total[r]), 6),
             "maize_cells": int(cells[r])} for r, code in enumerate(base.codes)]


def _dates(code: str, seasons: Mapping[str, Mapping[str, str]],
           ) -> tuple[tuple[int, int], tuple[int, int]] | None:
    s = seasons.get(code)
    if s is None or s["planting_date"] == "" or s["maturity_date"] == "":
        return None
    return ((int(s["planting_date"][:2]), int(s["planting_date"][3:])),
            (int(s["maturity_date"][:2]), int(s["maturity_date"][3:])))


def build_weather(eobs_dir: Path, regions: Mapping[str, BaseGeometry], years: Sequence[int],
                  seasons: Mapping[str, Mapping[str, str]], irrigated: Array, rainfed: Array,
                  ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """The season and placebo windows of section 9.1 under three weightings (all cells, as
    registered; the maize map; cells below 800 m), in one pass over the E-OBS files; and
    the regions' maize on the map."""
    lat, lon, elevation = read_grid(eobs_dir)
    base = cell_weights(regions, lat, lon)
    elev = elevation.ravel()[base.cells]
    maize = eobs_cells(irrigated + rainfed, lat, lon, base.cells)
    sets = {"all": base, "maize": maize_weights(base, maize),
            "below_800m": base.masked(np.isfinite(elev) & (elev < ELEVATION_MAX))}
    stacked = RegionWeights(tuple(c for _ in sets for c in base.codes), base.cells,
                            np.vstack([s.matrix for s in sets.values()]), base.cell_lat)
    dates = [_dates(c, seasons) for c in stacked.codes]
    labels = [w for w in sets for _ in base.codes]
    rows: list[dict[str, object]] = []
    for year in years:
        days: list[dt.date] = season_days(year, MAIZE_READ_TO)
        windows = season_window_sums(maize_daily(eobs_dir, days, stacked), days, dates)
        for window in WINDOWS_A2:
            rows += [maize_row(code, year, window, labels[r], windows[window], r)
                     for r, code in enumerate(stacked.codes)]
    irrigated_cells = eobs_cells(irrigated, lat, lon, base.cells)
    return rows, region_map(base, irrigated_cells, maize - irrigated_cells)


def write_map(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MAP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
