"""Checks on the pieces of ``ceiling.py`` that the numbers depend on."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ceiling  # noqa: E402


def test_one_floor_area_reduces_to_a_log_normal():
    # With every home the same size, rent > cap iff rent per m2 > cap / area.
    p25, p50, p75, area, cap = 8.0, 10.0, 12.5, 80.0, 1000.0
    sigma = (math.log(p75) - math.log(p25)) / (2 * norm.ppf(0.75))
    expected = norm.sf((math.log(cap / area) - math.log(p50)) / sigma)
    got = ceiling.share_above(p50, p25, p75, np.array([area]), np.array([1.0]), cap, beta=-0.4)
    assert got == pytest.approx(expected, rel=1e-9)


def test_share_falls_as_the_cap_rises():
    areas, w = np.array([40.0, 70.0, 110.0]), np.array([1.0, 2.0, 1.0])
    shares = [ceiling.share_above(10, 8, 13, areas, w, c, beta=-0.4) for c in (500, 1000, 2000, 4000)]
    assert all(a > b for a, b in zip(shares, shares[1:]))


def test_rent_per_m2_falling_with_size_thins_the_upper_tail():
    areas, w = np.array([30.0, 60.0, 90.0, 150.0]), np.ones(4)
    independent = ceiling.share_above(10, 8, 13, areas, w, 2300, beta=0.0)
    elastic = ceiling.share_above(10, 8, 13, areas, w, 2300, beta=-0.4)
    assert elastic < independent


def test_cdf_and_share_agree():
    areas, w = np.array([45.0, 75.0, 120.0]), np.array([3.0, 4.0, 1.0])
    grid = np.array([600.0, 1200.0, 2300.0])
    cdf = ceiling.rent_cdf(grid, 9.3, 5.8, 13.9, areas, w, beta=-0.43)
    for r, c in zip(grid, cdf):
        assert 1 - c == pytest.approx(ceiling.share_above(9.3, 5.8, 13.9, areas, w, r, beta=-0.43), abs=1e-12)


def test_weighted_median():
    assert ceiling.wmedian(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0
    assert ceiling.wmedian(np.array([5.0, 1.0, 3.0]), np.array([1.0, 1.0, 1.0])) == 3.0


def test_rule_ceiling_matches_the_2026_value():
    assert ceiling.RULE_MULTIPLE * ceiling.RMMG[2026] == ceiling.CEILING


def test_summary_has_every_section():
    path = Path(__file__).resolve().parents[1] / "results" / "summary.txt"
    text = path.read_text()
    for head in ("## 1.", "## 2.", "## 2b.", "## 3.", "## 4.", "## 5."):
        assert head in text
