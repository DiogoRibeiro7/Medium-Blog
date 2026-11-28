import math

import pytest

from airline_ops.experiments.stats import mean_ci, paired_difference, pareto_front


def test_mean_ci() -> None:
    r = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert r["mean"] == 2.5
    assert r["lo"] < 2.5 < r["hi"]
    assert math.isnan(mean_ci([])["mean"])


def test_paired_difference() -> None:
    r = paired_difference([3.0, 4.0, 5.0], [1.0, 2.0, 3.0])
    assert r["mean"] == 2.0
    assert r["sd"] == 0.0
    with pytest.raises(ValueError, match="same length"):
        paired_difference([1.0], [1.0, 2.0])


def test_pareto_front() -> None:
    pts = [(1.0, 5.0), (2.0, 3.0), (3.0, 4.0), (4.0, 1.0), (2.0, 3.0)]
    front = pareto_front(pts)
    assert 2 not in front
    assert 0 in front
    assert 3 in front
