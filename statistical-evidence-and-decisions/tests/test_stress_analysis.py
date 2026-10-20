"""Controlled checks for paired robustness and leakage boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import simulate_data
from stress_analysis import (
    STRESSES,
    apply_stress,
    paired_regret_shift,
    run_stress_experiment,
    stress_estimators,
)


@pytest.mark.parametrize("stress", STRESSES)
def test_stress_preserves_true_test_outcomes_and_is_reproducible(stress: str) -> None:
    original = simulate_data(seed=8, n_train=80, n_test=100)
    first = apply_stress(original, stress, seed=19)
    second = apply_stress(original, stress, seed=19)
    np.testing.assert_array_equal(first.x_train, second.x_train)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.mean_test, original.mean_test)
    np.testing.assert_array_equal(first.y_test, original.y_test)
    assert np.isfinite(first.y_train).all()
    assert np.isfinite(first.y_test).all()
    assert np.isfinite(original.x_train).all()
    if stress == "clean":
        assert first.missing_train_rate == 0
        assert first.contaminated_train_rate == 0
    if stress in ("outliers", "mar_outliers"):
        assert first.contaminated_train_rate == pytest.approx(0.1)
    else:
        assert first.contaminated_train_rate == 0


def test_mar_missingness_depends_on_observed_anchor_but_mcar_does_not() -> None:
    data = simulate_data(seed=21, n_train=4000, n_test=100)
    mcar = apply_stress(data, "mcar", seed=30)
    mar = apply_stress(data, "mar", seed=30)
    anchor = data.x_train[:, 2]
    high = anchor > 0.9
    low = anchor < -0.9
    mar_gap = np.isnan(mar.x_train[high, 0]).mean() - np.isnan(
        mar.x_train[low, 0]
    ).mean()
    mcar_gap = np.isnan(mcar.x_train[high, 0]).mean() - np.isnan(
        mcar.x_train[low, 0]
    ).mean()
    assert mar_gap > 0.15
    assert abs(mcar_gap) < 0.08
    assert not np.isnan(mar.x_train[:, 2]).any()
    assert not np.isnan(mcar.x_train[:, 2]).any()


def test_training_only_outliers_and_predictor_imputation() -> None:
    data = simulate_data(seed=12, n_train=100, n_test=35)
    corrupted = apply_stress(data, "mar_outliers", seed=10)
    assert np.isnan(corrupted.x_train).any()
    assert np.isnan(corrupted.x_test).any()
    assert np.sum(corrupted.y_train != data.y_train) == 10
    for model in stress_estimators().values():
        model.fit(corrupted.x_train, corrupted.y_train)
        predicted = np.asarray(model.predict(corrupted.x_test)).reshape(-1)
        assert predicted.shape == (35,)
        assert np.isfinite(predicted).all()
        assert model.steps[0][0] == "simpleimputer"


def test_smoke_benchmark_is_paired_and_regrets_are_nonnegative() -> None:
    results = run_stress_experiment(
        repeats=1,
        n_train_values=(80,),
        n_test=60,
        signal_scenarios=("low_variance_signal",),
        stresses=("clean", "mcar", "outliers"),
        seed=91,
    )
    assert set(results["stress"]) == {"clean", "mcar", "outliers"}
    assert len(results) == 3 * 8
    assert set(results.loc[results["stress"] == "mcar", "model"]) == set(
        results.loc[results["stress"] == "clean", "model"]
    )
    assert "Validation utility" in set(results["model"])
    assert "Validation RMSE" in set(results["model"])
    assert (results["regret"] >= -1e-9).all()
    assert np.isfinite(results[["rmse", "regret", "expected_profit"]]).all().all()
    assert results.loc[results["stress"] == "mcar", "missing_train_rate"].gt(0).all()
    assert results.loc[results["stress"] == "clean", "missing_train_rate"].eq(0).all()


def test_paired_differences_use_matching_replicates() -> None:
    frame = pd.DataFrame(
        {
            "signal": ["low"] * 4,
            "n_train": [80] * 4,
            "replicate": [0, 1, 0, 1],
            "model": ["OLS"] * 4,
            "stress": ["clean", "clean", "mar", "mar"],
            "regret": [0.1, 0.2, 0.3, 0.5],
        }
    )
    report = paired_regret_shift(frame, bootstrap_draws=100, seed=42)
    assert len(report) == 1
    assert report.iloc[0]["paired_mean_delta_regret"] == pytest.approx(0.25)
    assert 0.2 - 1e-12 <= report.iloc[0]["bootstrap_lower_95"] <= 0.3
    assert 0.2 <= report.iloc[0]["bootstrap_upper_95"] <= 0.3 + 1e-12


def test_reject_missing_pair_and_invalid_sample_sizes() -> None:
    data = simulate_data(seed=1, n_train=80, n_test=30)
    with pytest.raises(ValueError, match="Unknown stress"):
        apply_stress(data, "not_a_stress", seed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="training sample sizes"):
        run_stress_experiment(repeats=1, n_train_values=(20,), n_test=10)
    incomplete = pd.DataFrame(
        {
            "signal": ["low"],
            "n_train": [80],
            "replicate": [0],
            "stress": ["mar"],
            "model": ["OLS"],
            "regret": [0.2],
        }
    )
    with pytest.raises(ValueError, match="clean-data"):
        paired_regret_shift(incomplete)
