"""Amendment A2, section 10.1: the map of maize and the weather over it."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from eu_wheat_weather.contract import DataError
from eu_wheat_weather.maize_map import (
    SHAPE,
    WEIGHTINGS_A2,
    WINDOWS_A2,
    eobs_cells,
    maize_weights,
    region_map,
    shares,
)
from eu_wheat_weather.weather import STEP, RegionWeights

ROOT = Path(__file__).resolve().parents[1]
AMENDED = ROOT / "docs" / "maize" / "amendment"
REGISTERED = ROOT / "docs" / "maize" / "weather.csv"


def test_each_eobs_cell_sums_the_three_by_three_map_cells_it_holds() -> None:
    grid = np.zeros(SHAPE)
    lat, lon = np.array([25.125, 25.375]), np.array([-24.875, -24.625])
    top, left = round((90 - 25.5) * 12), round((180 - 25.0) * 12)   # the cell at 25.375, -24.875
    grid[top:top + 3, left:left + 3] = 1.0
    grid[top + 3, left] = 7.0                                         # the cell to its south
    cells = np.arange(4)                                              # lat x lon, row-major
    assert eobs_cells(grid, lat, lon, cells).tolist() == [7.0, 0.0, 9.0, 0.0]
    with pytest.raises(DataError, match="align"):
        eobs_cells(grid, lat, lon + 0.01, cells)


def test_weights_are_the_share_inside_the_region_times_the_maize() -> None:
    lat = np.array([40.125, 60.125])
    share = np.array([[1.0, 0.25], [0.0, 0.75]])           # two regions, two cells
    base = RegionWeights(("AA11", "AA12"), np.array([0, 1]),
                         share * np.cos(np.radians(lat))[None, :] * STEP ** 2, lat)
    assert np.allclose(shares(base), share)
    maize = np.array([100.0, 40.0])
    assert np.allclose(maize_weights(base, maize).matrix, share * maize[None, :])
    rows = region_map(base, np.array([50.0, 0.0]), np.array([50.0, 40.0]))
    assert rows[0]["maize_ha"] == 110.0 and rows[0]["irrigated_ha"] == 50.0
    assert math.isclose(float(str(rows[0]["irrigated_share"])), 50 / 110, rel_tol=1e-5)
    assert rows[1]["irrigated_share"] == 0.0


@pytest.mark.skipif(not (AMENDED / "weather.csv").exists(), reason="No amended weather.")
def test_the_all_cells_weighting_reproduces_the_registered_weather() -> None:
    with REGISTERED.open(encoding="utf-8") as stream:
        registered = {(r["geo"], r["year"], r["window"]): r for r in csv.DictReader(stream)
                      if r["window"] in WINDOWS_A2}
    with (AMENDED / "weather.csv").open(encoding="utf-8") as stream:
        amended = list(csv.DictReader(stream))
    assert {r["weighting"] for r in amended} == set(WEIGHTINGS_A2)
    mine = {(r["geo"], r["year"], r["window"]): r for r in amended if r["weighting"] == "all"}
    assert mine == registered


@pytest.mark.skipif(not (AMENDED / "maize_map.csv").exists(), reason="No map table.")
def test_the_map_table() -> None:
    with (AMENDED / "maize_map.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "docs" / "maize" / "panel.csv").open(encoding="utf-8") as stream:
        panel_regions = {r["geo"] for r in csv.DictReader(stream)}
    assert all(0 <= float(r["irrigated_share"]) <= 1 for r in rows if r["irrigated_share"])
    without = {r["geo"] for r in rows if r["irrigated_share"] == ""}
    assert all(float(r["maize_ha"]) == 0 for r in rows if r["geo"] in without)
    assert not without & panel_regions          # every registered region has maize on it
