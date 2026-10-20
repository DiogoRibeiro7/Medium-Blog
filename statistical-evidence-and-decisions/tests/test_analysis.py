"""Scientific invariants for the synthetic-evidence notebook."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import (
    expected_profit,
    policy_regret,
    run_experiment,
    simulate_data,
)


def test_simulation_is_reproducible_and_split_is_independent() -> None:
    first = simulate_data(seed=2026, n_train=80, n_test=40, n_pairs=4)
    second = simulate_data(seed=2026, n_train=80, n_test=40, n_pairs=4)
    assert first.x_train.shape == (80, 8)
    assert first.x_test.shape == (40, 8)
    np.testing.assert_array_equal(first.x_train, second.x_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)
    assert not np.array_equal(first.x_train[:40], first.x_test)


def test_full_rank_pca_changes_coordinates_not_ols_predictions() -> None:
    data = simulate_data(seed=17, n_train=100, n_test=50, n_pairs=4)
    raw = make_pipeline(StandardScaler(), LinearRegression())
    transformed = make_pipeline(
        StandardScaler(), PCA(n_components=8), LinearRegression()
    )
    raw.fit(data.x_train, data.y_train)
    transformed.fit(data.x_train, data.y_train)
    np.testing.assert_allclose(
        raw.predict(data.x_test),
        transformed.predict(data.x_test),
        rtol=1e-8,
        atol=1e-8,
    )


def test_policy_oracle_has_zero_regret_and_profit_identity() -> None:
    means = np.array([-2.0, 0.1, 0.5, 1.4], dtype=np.float64)
    cost = 0.4
    assert policy_regret(means, means, cost) == pytest.approx(0.0)
    never_act = np.full(means.shape, -100.0, dtype=np.float64)
    assert expected_profit(means, never_act, cost) == pytest.approx(0.0)
    assert policy_regret(means, never_act, cost) >= 0.0


def test_reject_invalid_generation_parameters() -> None:
    with pytest.raises(ValueError):
        simulate_data(seed=0, n_train=5)
    with pytest.raises(TypeError):
        simulate_data(seed=True)
    with pytest.raises(ValueError):
        simulate_data(seed=1, measurement_sd=0)


def test_small_experiment_produces_finite_benchmark_metrics() -> None:
    result = run_experiment(repeats=1, n_train=80, n_test=50, seed=4)
    assert set(result["scenario"]) == {
        "low_variance_signal",
        "high_variance_signal",
    }
    assert "Decision-guided validation" in set(result["model"])
    assert len(result) == 2 * 7
    assert np.isfinite(result[["rmse", "regret", "expected_profit"]]).all().all()
    assert (result["regret"] >= -1e-10).all()
