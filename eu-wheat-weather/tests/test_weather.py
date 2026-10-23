"""Tests of the daily weather formulas and the cell-to-region aggregation."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
import shapely

from eu_wheat_weather.weather import (
    STEP,
    VARIABLES,
    aggregate,
    bin_names,
    cell_weights,
    daily_region_variables,
    degree_days,
    extraterrestrial_radiation,
    hargreaves,
    time_above,
    window_sums,
)

CASES = [(5.0, 20.0), (10.0, 35.0), (28.0, 36.0), (-3.0, 4.0), (31.0, 31.0), (12.0, 12.0),
         (35.0, 25.0)]   # the last has tn > tx: the curve uses min and max


def sine_day(tn: float, tx: float, n: int = 200_001) -> np.ndarray:
    lo, hi = min(tn, tx), max(tn, tx)
    t = np.linspace(0.0, 1.0, n)
    return (lo + hi) / 2 + (hi - lo) / 2 * np.sin(2 * np.pi * t)


@pytest.mark.parametrize(("tn", "tx"), CASES)
@pytest.mark.parametrize("base", [0.0, 27.0, 30.0, 33.0])
def test_degree_days_and_time_above_match_numerical_integration(
        tn: float, tx: float, base: float) -> None:
    curve = sine_day(tn, tx)
    expected_dd = float(np.trapezoid(np.maximum(curve - base, 0.0), dx=1 / (curve.size - 1)))
    expected_time = float(np.mean(curve > base))
    got_dd = float(degree_days(np.array([tn]), np.array([tx]), base)[0])
    got_time = float(time_above(np.array([tn]), np.array([tx]), base)[0])
    assert got_dd == pytest.approx(expected_dd, abs=1e-4)
    assert got_time == pytest.approx(expected_time, abs=1e-4)


def test_extraterrestrial_radiation_matches_fao56_example_8() -> None:
    # 20 degrees south, 3 September (day 246): Ra = 32.2 MJ m-2 day-1.
    ra = extraterrestrial_radiation(np.array([-20.0]), 246)
    assert float(ra[0]) == pytest.approx(32.2, abs=0.1)


def test_hargreaves_formula_and_no_negative_range() -> None:
    ra = np.array([40.0])
    expected = 0.0023 * 0.408 * 40.0 * (20.0 + 17.8) * math.sqrt(10.0)
    assert float(hargreaves(np.array([15.0]), np.array([25.0]), ra)[0]) == pytest.approx(expected)
    assert float(hargreaves(np.array([20.0]), np.array([19.0]), ra)[0]) == 0.0


def test_cell_weights_are_area_inside_times_cosine_latitude() -> None:
    lat = np.array([10.125, 10.375])
    lon = np.array([20.125, 20.375, 20.625])
    # The lower row's first cell entirely, and half of the second.
    region = shapely.box(20.0, 10.0, 20.375, 10.25)
    weights = cell_weights({"AA11": region, "BB22": shapely.box(50.0, 50.0, 51.0, 51.0)},
                           lat, lon)
    assert weights.codes == ("AA11", "BB22")
    row = dict(zip(weights.cells.tolist(), weights.matrix[0].tolist(), strict=True))
    cos = math.cos(math.radians(10.125))
    assert row == pytest.approx({0: STEP * STEP * cos, 1: STEP * STEP / 2 * cos})
    assert weights.matrix[1].sum() == 0.0          # outside the grid: no cells


def test_aggregate_applies_the_coverage_rule() -> None:
    weights = np.array([[1.0, 1.0], [0.3, 0.7]])
    values = np.array([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
    observed = np.array([[True, True], [True, False], [False, False]])
    out = aggregate(weights, values, observed)
    assert out[0, 0] == pytest.approx(3.0) and out[0, 1] == pytest.approx(2.0)
    assert math.isnan(out[0, 2])
    assert out[1, 0] == pytest.approx(0.3 * 2 + 0.7 * 4)
    assert math.isnan(out[1, 1])                    # only 30% of the weight observed


def test_bins_sum_to_the_days_and_missing_days_propagate() -> None:
    days = [dt.date(2020, 3, 1) + dt.timedelta(days=i) for i in range(214)]
    rng = np.random.default_rng(1)
    tn = rng.uniform(-5, 20, (len(days), 2))
    tx = tn + rng.uniform(0, 18, (len(days), 2))
    rr = rng.uniform(0, 10, (len(days), 2))
    obs = np.ones((len(days), 2), dtype=bool)
    lat = np.array([45.125])
    lon = np.array([5.125, 5.375])
    weights = cell_weights({"AA11": shapely.box(5.0, 45.0, 5.5, 45.25)}, lat, lon)
    daily = daily_region_variables(tn, tx, rr, obs, obs, days, weights)
    total = sum(daily[name] for name in bin_names())
    assert np.allclose(total, 1.0)
    sums = window_sums(daily, days)
    assert sums["mar_jul"]["missing_temp"][0] == 0
    assert float(sums["mar_jul"]["edd_30"][0]) >= 0 and set(VARIABLES) <= set(sums["mar_jul"])
    obs_gap = obs.copy()
    obs_gap[40, :] = False                           # 10 April: in mar_jul, apr_jul
    daily = daily_region_variables(tn, tx, rr, obs_gap, obs, days, weights)
    sums = window_sums(daily, days)
    assert math.isnan(float(sums["mar_jul"]["edd_30"][0]))
    assert math.isnan(float(sums["apr_jul"]["edd_30"][0]))
    assert not math.isnan(float(sums["aug_sep"]["edd_30"][0]))
    assert not math.isnan(float(sums["mar_jul"]["prec"][0]))
    assert sums["mar_jul"]["missing_temp"][0] == 1
