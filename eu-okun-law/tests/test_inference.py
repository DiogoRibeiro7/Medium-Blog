"""Checks for multiplicity, covariance, and synchronized common-block sampling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eu_okun.dependence import paired_block_comparisons
from eu_okun.stability import chi_square_tail, holm_adjust


def test_holm() -> None:
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_chi_square() -> None:
    assert chi_square_tail(0.0, 4) == pytest.approx(1.0)
    assert chi_square_tail(0.0, 3) == pytest.approx(1.0)
    assert chi_square_tail(2.0, 4) == pytest.approx(2 * math.exp(-1))


def test_identical_countries_have_zero_paired_difference() -> None:
    rng = np.random.default_rng(4)
    x = np.column_stack([np.ones(70), rng.normal(size=(70, 3))])
    y = 0.2 - 0.3 * x[:, 1] + rng.normal(size=70)
    result = paired_block_comparisons(
        {"PT": (x, y), "ES": (x.copy(), y.copy())},
        replicates=199, block_length=4,
    )
    assert len(result) == 1
    assert result[0].estimate_a_minus_b == pytest.approx(0)
    assert result[0].ci95_low == pytest.approx(0)
    assert result[0].ci95_high == pytest.approx(0)


def test_too_short_sample_rejected() -> None:
    x = np.column_stack([np.ones(20), np.arange(20.), np.arange(20.) ** 2,
                         np.arange(20.) ** 3])
    with pytest.raises(ValueError):
        paired_block_comparisons({"PT": (x, x[:, 1]), "ES": (x, x[:, 1])})
