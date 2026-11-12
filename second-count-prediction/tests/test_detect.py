import math

import numpy as np
import pytest

from evaluate import TLC_FIRST_EVAL, tlc_pairs
from feed import detect as D


def series(n=80, sd=0.01, seed=1):
    rng = np.random.default_rng(seed)
    ms = [f"{2015 + i // 12:04d}-{i % 12 + 1:02d}" for i in range(n)]
    return {m: float(x) for m, x in zip(ms, rng.normal(0, sd, n), strict=True)}


def test_grid_spans_tenth_of_a_percent_to_half():
    assert D.GRID[0] == pytest.approx(0.0001) and D.GRID[-1] == pytest.approx(0.501, rel=1e-3)
    assert all(b / a == pytest.approx(10**0.1, rel=1e-3) for a, b in zip(D.GRID, D.GRID[1:], strict=False))


def test_detection_grows_with_the_loss_and_stops_at_the_budget_ceiling():
    r = series(sd=0.02)
    cur = D.curve(r)
    assert cur[D.GRID[0]] < 0.2
    assert max(cur.values()) <= 1 - 0  # never above one
    assert cur[D.GRID[-1]] >= 0.9


def test_observed_l90_needs_the_target_to_hold_from_there_on():
    cur = {0.01: 0.5, 0.02: 0.91, 0.04: 0.89, 0.08: 0.95}
    assert D.observed_l90(cur) == 0.08
    assert D.observed_l90({0.01: 0.1, 0.02: 0.5}) is None


def test_prediction_scales_with_the_spread():
    a = D.predicted(series(sd=0.005))["p95_rule"]
    b = D.predicted(series(sd=0.02))["p95_rule"]
    assert 3 < b / a < 5


def test_factor_two_rule():
    assert D.within_factor_two(0.04, 0.05) and D.within_factor_two(0.04, 0.02) and D.within_factor_two(0.04, 0.08)
    assert not D.within_factor_two(0.04, 0.0199) and not D.within_factor_two(0.04, 0.081)
    assert D.within_factor_two(0.3, None) and not D.within_factor_two(0.2, None)
    assert D.within_factor_two(0.0002, D.GRID[0]) and not D.within_factor_two(0.00021, D.GRID[0])
    assert D.log_error(0.04, 0.08) == pytest.approx(math.log(2))


def test_detector_reproduces_part4_on_the_committed_tlc_data():
    """Part 4, results/summary.txt: outside count at the matched budget caught 92% of 5% losses in yellow."""
    pairs = tlc_pairs(__import__("pathlib").Path(__file__).resolve().parents[2] / "data-tests-limits")
    d, o = pairs["TLC yellow / monthly report"]
    r = D.ratio_series(d, o)
    assert D.detection(r, 0.05, TLC_FIRST_EVAL) == pytest.approx(60 / 65, abs=0.001)
    d, o = pairs["TLC fhv / monthly report"]
    assert D.detection(D.ratio_series(d, o), 0.05, TLC_FIRST_EVAL) == pytest.approx(14 / 64, abs=0.001)
