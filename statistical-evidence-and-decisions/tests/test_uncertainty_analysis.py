"""Checks for bootstrap selection, variance identities and decision accounting."""

from __future__ import annotations

import numpy as np
import pytest

from analysis import simulate_data
from uncertainty_analysis import (
    bootstrap_oob_indices,
    bootstrap_predictive_draws,
    coefficient_uncertainty,
    decision_policy_audit,
    interval_audit,
    run_uncertainty_study,
    variance_decomposition,
)


def test_oob_never_contains_inbag_sampled_positions() -> None:
    inbag = np.array([0, 0, 1, 1, 1, 3], dtype=np.int64)
    oob = bootstrap_oob_indices(inbag, 6)
    np.testing.assert_array_equal(oob, np.array([2, 4, 5]))
    assert np.intersect1d(inbag, oob).size == 0
    with pytest.raises(ValueError):
        bootstrap_oob_indices(inbag, 5)


def test_bootstrap_deterministic_and_no_test_response_interface() -> None:
    data = simulate_data(seed=11, n_train=80, n_test=40)
    a = bootstrap_predictive_draws(
        data.x_train, data.y_train, data.x_test,
        bootstrap_replicates=6, seed=32,
    )
    b = bootstrap_predictive_draws(
        data.x_train, data.y_train, data.x_test,
        bootstrap_replicates=6, seed=32,
    )
    assert set(a.predictions) == {
        "Fixed OLS", "Fixed Ridge", "Selected RMSE", "Selected utility"
    }
    for label in a.predictions:
        assert a.predictions[label].shape == (6, 40)
        np.testing.assert_allclose(a.predictions[label], b.predictions[label])
        assert np.isfinite(a.predictions[label]).all()
    assert np.all(a.oob_sizes >= 8)
    assert all(len(v) == 6 for v in a.selected_models.values())


def test_variance_identity_including_single_label() -> None:
    predictions = np.array(
        [[1.0, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 9.0]]
    )
    report = variance_decomposition(predictions, ("A", "A", "B", "B"))
    np.testing.assert_allclose(
        report["total_variance"],
        report["within_model_label"] + report["between_model_labels"],
    )
    assert (report["between_model_labels"] > 0).all()
    only_one = variance_decomposition(predictions, ("A", "A", "A", "A"))
    np.testing.assert_allclose(only_one["between_model_labels"], 0)


def test_decision_regret_and_profit_identity() -> None:
    truth = np.array([-0.5, 0.2, 1.2, 2.1], dtype=np.float64)
    predictions = {
        "test": np.tile(truth, (6, 1))
    }
    result = decision_policy_audit(predictions, truth, costs=(0.5, 1.5))
    assert len(result) == 6
    assert (result["regret"] >= -1e-12).all()
    np.testing.assert_allclose(
        result["oracle_profit"] - result["expected_profit"],
        result["regret"],
    )
    assert result["ambiguous_case_rate"].eq(0).all()


def test_intervals_distinguish_conditional_mean_and_observations() -> None:
    truth = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    observed = np.array([0.3, 1.2, 2.1], dtype=np.float64)
    predictions = {"A": np.tile(truth, (8, 1))}
    result = interval_audit(
        predictions, truth, observed, noise_sd=0.65,
        noise_replicates=4, seed=44
    )
    assert result.loc[0, "conditional_mean_90pct_coverage"] == 1.0
    assert result.loc[0, "mean_interval_width"] == 0.0
    assert result.loc[0, "predictive_interval_width"] > 0.0
    assert result.loc[0, "observation_90pct_coverage"] <= 1.0


def test_full_model_coefficient_intervals() -> None:
    data = simulate_data(seed=35, n_train=80, n_test=25)
    result = coefficient_uncertainty(
        data.x_train, data.y_train,
        bootstrap_replicates=12, seed=12,
    )
    assert len(result) == data.x_train.shape[1] + 1
    assert np.isfinite(
        result[["estimate", "analytic_lower", "bootstrap_lower", "bootstrap_sd"]]
    ).all().all()


def test_small_end_to_end_clean_and_mar() -> None:
    clean = run_uncertainty_study(
        n_train=80, n_test=40, bootstrap_replicates=6,
        stress="clean", seed=5,
    )
    assert clean.coefficient_table is not None
    assert len(clean.decision_table) == 4 * 3 * 4
    assert len(clean.variance_table) == 2
    assert len(clean.coverage_table) == 4
    stressed = run_uncertainty_study(
        n_train=80, n_test=40, bootstrap_replicates=6,
        stress="mar", seed=5,
    )
    assert stressed.coefficient_table is None
    assert np.isfinite(stressed.decision_table["regret"]).all()
