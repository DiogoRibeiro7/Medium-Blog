"""Independent tests of coefficient contrasts and common-sample construction."""

from __future__ import annotations

import numpy as np
import pytest

from eu_okun.baseline import fit_hac_covariance
from eu_okun.comparison import common_sample_designs, cumulative_gdp


def test_cumulative_variance_includes_off_diagonal_covariance() -> None:
    beta = np.array([0.4, -0.3, -0.2])
    covariance = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.04, 0.03],
        [0.0, 0.03, 0.09],
    ])
    effect, standard_error, lower, upper = cumulative_gdp(beta, covariance, 2)
    assert effect == pytest.approx(-0.5)
    assert standard_error == pytest.approx(np.sqrt(0.19))
    assert (lower + upper) / 2 == pytest.approx(effect)


def test_common_sample_across_lags_controls_and_countries() -> None:
    panel = {}
    countries = ("PT", "ES")
    quarters = [f"2020-Q{quarter}" for quarter in range(1, 5)]
    quarters += [f"2021-Q{quarter}" for quarter in range(1, 5)]
    quarters += [f"2022-Q{quarter}" for quarter in range(1, 5)]
    quarters += [f"2023-Q{quarter}" for quarter in range(1, 5)]
    quarters += [f"2024-Q{quarter}" for quarter in range(1, 5)]
    quarters += [f"2025-Q{quarter}" for quarter in range(1, 5)]
    quarters += [f"2026-Q{quarter}" for quarter in range(1, 5)]
    for country in countries:
        for i, quarter in enumerate(quarters):
            panel[(country, quarter)] = {
                "gdp": float(i % 7),
                "unemployment": float(6 + i / 10),
                "participation": float(70 + i / 20),
            }
    # With fewer than 30 overlapping quarters, the contract rejects the sample.
    with pytest.raises(ValueError, match="30 common quarters"):
        common_sample_designs(panel, quarters, countries, (0, 1, 2, 4))


def test_calendar_gap_hac_ignores_unobserved_quarter() -> None:
    rng = np.random.default_rng(11)
    x = np.column_stack([np.ones(40), rng.normal(size=40)])
    y = 0.3 + 0.8 * x[:, 1] + rng.normal(size=40)
    quarters = [f"{2010 + i // 4}-Q{(i % 4) + 1}" for i in range(41)]
    quarters.pop(20)
    beta, covariance, _, _ = fit_hac_covariance(x, y, maxlags=1, periods=quarters)
    residual = y - x @ beta
    scores = x * residual[:, None]
    quarter_to_index = {quarter: i for i, quarter in enumerate(quarters)}
    cross = np.zeros((2, 2))
    for i, quarter in enumerate(quarters):
        year, q = int(quarter[:4]), int(quarter[-1])
        previous = f"{year - 1}-Q4" if q == 1 else f"{year}-Q{q - 1}"
        if previous in quarter_to_index:
            cross += np.outer(scores[i], scores[quarter_to_index[previous]])
    bread = np.linalg.inv(x.T @ x)
    expected = bread @ (scores.T @ scores + 0.5 * (cross + cross.T)) @ bread
    np.testing.assert_allclose(covariance, expected, rtol=1e-10)


def test_bad_contrast_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid contrast"):
        cumulative_gdp(np.array([1.0]), np.eye(1), 1)
