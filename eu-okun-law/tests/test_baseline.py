"""Verify calendar-lag design and HAC OLS against independent calculations."""

from __future__ import annotations

import numpy as np
import pytest

from eu_okun.baseline import design_matrix, fit_hac


def test_lags_do_not_bridge_missing_calendar_quarters() -> None:
    panel = {
        ("PT", "2022-Q1"): {"gdp": 1.0, "unemployment": 7.0, "participation": 70.0},
        ("PT", "2022-Q2"): {"gdp": None, "unemployment": 6.8, "participation": 70.1},
        ("PT", "2022-Q3"): {"gdp": 0.5, "unemployment": 6.7, "participation": 70.2},
        ("PT", "2022-Q4"): {"gdp": 0.8, "unemployment": 6.5, "participation": 70.3},
    }
    x, y, quarters = design_matrix(
        panel, "PT", ["2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4"], 1, True
    )
    assert quarters == ["2022-Q4"]
    np.testing.assert_allclose(x[0], [1.0, 0.8, 0.5, 0.1])
    np.testing.assert_allclose(y, [-0.2])


def test_hac_white_covariance_when_zero_lags() -> None:
    rng = np.random.default_rng(42)
    x = np.column_stack([np.ones(100), rng.normal(size=100)])
    y = 0.4 - 0.6 * x[:, 1] + rng.normal(scale=0.3, size=100)
    beta, se, r2, _ = fit_hac(x, y, maxlags=0)
    expected_beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ expected_beta
    bread = np.linalg.inv(x.T @ x)
    expected_cov = bread @ (x * residual[:, None]).T @ (x * residual[:, None]) @ bread
    np.testing.assert_allclose(beta, expected_beta, rtol=1e-11)
    np.testing.assert_allclose(se, np.sqrt(np.diag(expected_cov)), rtol=1e-11)
    assert 0 < r2 < 1


def test_hac_bartlett_one_lag() -> None:
    rng = np.random.default_rng(7)
    x = np.column_stack([np.ones(120), rng.normal(size=120)])
    y = 0.5 + x[:, 1] + rng.normal(size=120)
    beta, se, _, _ = fit_hac(x, y, maxlags=1)
    scores = x * (y - x @ beta)[:, None]
    meat = scores.T @ scores
    cross = scores[1:].T @ scores[:-1]
    meat += 0.5 * (cross + cross.T)
    inv = np.linalg.inv(x.T @ x)
    np.testing.assert_allclose(se**2, np.diag(inv @ meat @ inv), rtol=1e-11)


def test_rank_deficiency_rejected() -> None:
    x = np.column_stack([np.ones(40), np.ones(40)])
    with pytest.raises(ValueError, match="rank deficient"):
        fit_hac(x, np.arange(40, dtype=float))


def test_short_samples_rejected() -> None:
    with pytest.raises(ValueError, match="Insufficient"):
        fit_hac(np.ones((5, 1)), np.arange(5, dtype=float))
