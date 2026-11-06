"""Tests of the concentration bounds and the supply-based exposure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def hhi_and_top(shares: np.ndarray) -> tuple[float, float]:
    s = shares / shares.sum()
    return float((s**2).sum()), float(s.max())


def test_hhi_is_between_top_share_squared_and_top_share() -> None:
    rng = np.random.default_rng(7)
    for _ in range(2000):
        n = rng.integers(1, 40)
        h, top = hhi_and_top(rng.pareto(1.2, n) + 1e-9)
        assert top**2 - 1e-12 <= h <= top + 1e-12


def test_hhi_threshold_implies_top_share_threshold() -> None:
    # HHI > 0.4 forces the largest share above 0.4, and a largest share above sqrt(0.4) forces HHI above 0.4
    rng = np.random.default_rng(3)
    for _ in range(2000):
        h, top = hhi_and_top(rng.random(rng.integers(1, 30)))
        if h > 0.4:
            assert top > 0.4
        if top > 0.4**0.5:
            assert h > 0.4


def test_dominant_supplier_with_spread_fringe_escapes_the_hhi_criterion() -> None:
    # 45% from one supplier, the rest spread over eleven others of 5% each: HHI about 0.23
    h, top = hhi_and_top(np.array([0.45] + [0.05] * 11))
    assert top == pytest.approx(0.45)
    assert h == pytest.approx(0.45**2 + 11 * 0.05**2)
    assert h < 0.4


def test_supply_exposure_is_bounded_and_ignores_exports() -> None:
    production, imports, top_share = 100.0, 300.0, 0.8
    for exports in (0.0, 350.0, 399.0):
        exposure = top_share * imports / (production + imports)
        assert exposure == pytest.approx(0.6)
        consumption = production + imports - exports
        assert consumption > 0
    # the consumption-based version explodes as exports approach production + imports
    assert top_share * imports / (production + imports - 399.0) > 1


def test_correspondence_aggregation() -> None:
    from supply import extra_eu

    partners = pd.Series(["CN", "DE", "QU", "EXT_EU27_2020", "US", "EL"])
    assert extra_eu(partners).tolist() == [True, False, False, False, True, False]
