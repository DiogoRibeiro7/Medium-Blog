"""Invariants for independent split-conformal calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibration_analysis import (
    aggregate_coverage,
    decision_evaluation,
    fit_calibration_models,
    gaussian_plugin_radius,
    make_calibration_split,
    run_calibration_experiment,
    split_conformal_radius,
)


def test_corrected_quantile_and_infinite_threshold() -> None:
    y = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    pred = np.zeros_like(y)
    assert split_conformal_radius(y, pred, alpha=0.20) == 8.0
    assert np.isinf(split_conformal_radius(y, pred, alpha=0.05))
    with pytest.raises(ValueError):
        split_conformal_radius(y, pred, alpha=1.0)


def test_fit_calibration_test_are_disjoint_and_deterministic() -> None:
    a = make_calibration_split(
        seed=17, scenario="low_variance_signal", n_fit=80, n_cal=30, n_test=40
    )
    b = make_calibration_split(
        seed=17, scenario="low_variance_signal", n_fit=80, n_cal=30, n_test=40
    )
    assert a.x_fit.shape == (80, 16)
    assert a.x_cal.shape == (30, 16)
    assert a.x_test.shape == (40, 16)
    np.testing.assert_array_equal(a.x_cal, b.x_cal)
    assert not np.array_equal(a.x_fit[:30], a.x_cal)
    assert not np.array_equal(a.x_test[:30], a.x_cal)


def test_selection_uses_fit_partition_only() -> None:
    a = make_calibration_split(
        seed=22, scenario="high_variance_signal",
        stress="mar", n_fit=80, n_cal=30, n_test=40,
    )
    _, winner_a = fit_calibration_models(a, seed=3)
    # Deliberately mutate calibration labels without changing fitting data.
    altered = type(a)(
        x_fit=a.x_fit, y_fit=a.y_fit,
        x_cal=a.x_cal, y_cal=a.y_cal + 1000.0,
        x_test=a.x_test, y_test=a.y_test, mean_test=a.mean_test,
    )
    _, winner_b = fit_calibration_models(altered, seed=3)
    assert winner_a == winner_b


def test_gaussian_radius_uses_fit_residuals() -> None:
    actual = np.array([1., 2., 3., 4.])
    predicted = np.array([1., 1., 1., 1.])
    assert gaussian_plugin_radius(actual, predicted, 0.1) > 0


def test_economic_regret_and_review_cost_identity() -> None:
    mean = np.array([-1., 0., 1., 2.])
    pred = np.array([-1., 0., 1., 2.])
    radii = {"Gaussian plug-in": 0.2, "Split conformal": 0.5}
    result = decision_evaluation(mean, pred, radii, costs=(0.5, 1.5))
    assert len(result) == 2 * 5
    np.testing.assert_allclose(
        result["oracle_profit"] - result["expected_profit"], result["regret"]
    )
    assert (result["regret"] >= -1e-12).all()


def test_calibration_experiment_smoke_shapes() -> None:
    intervals, decisions = run_calibration_experiment(
        repeats=1, n_fit=80, n_cal_values=(30,), n_test=40,
        alphas=(0.1,), scenarios=("low_variance_signal",),
        stresses=("clean",), seed=5,
    )
    assert len(intervals) == 3 * 2 * 2
    # 3 fitted models, 4 costs, and 5 rules at each cost:
    # one point-threshold rule plus conservative/review for each interval method.
    assert len(decisions) == 3 * 4 * (1 + 2 * 2)
    assert set(decisions["rule"]) == {
        "Point threshold",
        "Conservative",
        "Ideal review",
    }
    assert set(decisions["method"]) == {
        "Point estimate",
        "Gaussian plug-in",
        "Split conformal",
    }
    assert not decisions.duplicated(
        ["model", "method", "rule", "cost"]
    ).any()
    assert intervals["coverage"].between(0, 1).all()
    assert set(intervals["distribution"]) == {"IID", "Noise inflation"}
    assert np.isfinite(decisions["regret"]).all()
    # One independent simulation cannot support replicate-level CIs.
    summary = aggregate_coverage(intervals, bootstrap_draws=10)
    assert summary["bootstrap_lower_95"].isna().all()


def test_no_duplicate_dataset_indices_in_coverage_aggregation() -> None:
    intervals, _ = run_calibration_experiment(
        repeats=1, n_fit=80, n_cal_values=(30,), n_test=40,
        alphas=(0.1,), scenarios=("high_variance_signal",),
        stresses=("clean",), seed=9,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_coverage(pd.concat([intervals, intervals], ignore_index=True))
