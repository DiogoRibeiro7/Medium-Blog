"""Controlled experiments on structural, inferential and decision evidence.

Synthetic outcomes are interpreted as the benefit of an intervention *before*
paying its cost. The known conditional mean is used only to evaluate simulated
test-set decisions; it is never supplied to a fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]
Scenario = Literal["low_variance_signal", "high_variance_signal"]


@dataclass(frozen=True)
class SimulatedData:
    """Independent training/test observations from a known data-generating process."""

    x_train: FloatArray
    y_train: FloatArray
    mean_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    mean_test: FloatArray


def simulate_data(
    seed: int,
    scenario: Scenario = "low_variance_signal",
    n_train: int = 180,
    n_test: int = 600,
    n_pairs: int = 8,
    measurement_sd: float = 0.12,
    outcome_sd: float = 0.65,
) -> SimulatedData:
    """Generate correlated predictor pairs with signal in different PCA directions.

    Each pair consists of two noisy measurements of one latent variable.
    In the low-variance case, the response depends on their difference;
    in the high-variance case, it depends on their sum.
    """
    for label, count, minimum in (
        ("seed", seed, 0),
        ("n_train", n_train, 2 * n_pairs + 6),
        ("n_test", n_test, 1),
        ("n_pairs", n_pairs, 2),
    ):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{label} must be an integer")
        if count < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError(f"Unknown scenario: {scenario}")
    if not (np.isfinite(measurement_sd) and measurement_sd > 0):
        raise ValueError("measurement_sd must be positive and finite")
    if not (np.isfinite(outcome_sd) and outcome_sd > 0):
        raise ValueError("outcome_sd must be positive and finite")

    rng = np.random.default_rng(seed)
    count = n_train + n_test
    latent = rng.normal(size=(count, n_pairs))
    # Collinear blocks: common factor plus small independent measurement errors.
    x = np.repeat(latent, 2, axis=1) + rng.normal(
        scale=measurement_sd, size=(count, 2 * n_pairs)
    )
    if scenario == "low_variance_signal":
        conditional_mean = 10.0 * (x[:, 0] - x[:, 1])
    else:
        conditional_mean = 1.2 * (x[:, 0] + x[:, 1])
    y = conditional_mean + rng.normal(scale=outcome_sd, size=count)
    return SimulatedData(
        x_train=np.asarray(x[:n_train], dtype=np.float64),
        y_train=np.asarray(y[:n_train], dtype=np.float64),
        mean_train=np.asarray(conditional_mean[:n_train], dtype=np.float64),
        x_test=np.asarray(x[n_train:], dtype=np.float64),
        y_test=np.asarray(y[n_train:], dtype=np.float64),
        mean_test=np.asarray(conditional_mean[n_train:], dtype=np.float64),
    )


def candidate_estimators() -> dict[str, object]:
    """Return fresh estimators; transforms and tuning remain inside training folds."""
    return {
        "OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 15), cv=3),
        ),
        "Elastic net": make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.25, 0.5, 0.8, 1.0],
                alphas=np.logspace(-3, 0.5, 15),
                cv=3,
                max_iter=5000,
                random_state=0,
            ),
        ),
        "PCR (95% X variance)": make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            LinearRegression(),
        ),
        "PLS (2 components)": make_pipeline(
            StandardScaler(),
            PLSRegression(n_components=2, scale=False),
        ),
    }


def _predict(estimator: object, x: FloatArray) -> FloatArray:
    """Normalise one- and two-dimensional scikit-learn regression outputs."""
    if not hasattr(estimator, "predict"):
        raise TypeError("The estimator must support predict")
    prediction = estimator.predict(x)  # type: ignore[attr-defined]
    output = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if output.shape[0] != x.shape[0] or not np.all(np.isfinite(output)):
        raise ValueError("The estimator produced invalid predictions")
    return output


def pvalue_screened_prediction(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    alpha: float = 0.05,
) -> FloatArray:
    """Fit OLS after exploratory training-only full-model p-value screening.

    These post-selection coefficients/p-values must NOT be used for confirmatory
    inference. This intentionally naive baseline is included for comparison.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    full_fit = sm.OLS(y_train, sm.add_constant(x_train, has_constant="add")).fit()
    pvalues = np.asarray(full_fit.pvalues[1:], dtype=np.float64)
    included = np.isfinite(pvalues) & (pvalues < alpha)
    if np.any(included):
        model = LinearRegression().fit(x_train[:, included], y_train)
        return _predict(model, x_test[:, included])
    model = DummyRegressor(strategy="mean").fit(x_train, y_train)
    return _predict(model, x_test)


def expected_profit(
    conditional_mean: FloatArray, prediction: FloatArray, cost: float
) -> float:
    """Expected benefit minus intervention cost under a threshold policy."""
    if conditional_mean.shape != prediction.shape:
        raise ValueError("conditional_mean and prediction must have equal shape")
    if not (np.isfinite(cost) and cost >= 0):
        raise ValueError("cost must be finite and non-negative")
    if not (np.all(np.isfinite(conditional_mean)) and np.all(np.isfinite(prediction))):
        raise ValueError("Scores and conditional means must be finite")
    take_action = prediction > cost
    return float(np.mean((conditional_mean - cost) * take_action))


def policy_regret(
    conditional_mean: FloatArray, prediction: FloatArray, cost: float
) -> float:
    """Excess expected opportunity loss relative to the known simulation oracle."""
    oracle_profit = float(np.mean(np.maximum(conditional_mean - cost, 0)))
    return oracle_profit - expected_profit(conditional_mean, prediction, cost)


def validation_guided_prediction(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    cost: float,
    seed: int,
) -> tuple[FloatArray, str]:
    """Select a model on an internal validation set using observed net utility.

    The held-out test set and the simulated conditional means never enter model
    selection. Refitting the selected candidate uses the full training set.
    """
    train_idx, valid_idx = train_test_split(
        np.arange(y_train.size), test_size=0.30, random_state=seed
    )
    candidate_scores: dict[str, float] = {}
    for name, estimator in candidate_estimators().items():
        estimator.fit(x_train[train_idx], y_train[train_idx])  # type: ignore[attr-defined]
        prediction = _predict(estimator, x_train[valid_idx])
        # Observed outcomes give an unbiased, noisy validation-payoff estimate.
        candidate_scores[name] = float(
            np.mean((y_train[valid_idx] - cost) * (prediction > cost))
        )
    winner = max(candidate_scores, key=candidate_scores.__getitem__)
    refitted = candidate_estimators()[winner]
    refitted.fit(x_train, y_train)  # type: ignore[attr-defined]
    return _predict(refitted, x_test), winner


def run_experiment(
    repeats: int = 12,
    n_train: int = 180,
    n_test: int = 600,
    cost: float = 0.6,
    seed: int = 2026,
) -> pd.DataFrame:
    """Compare regression, dimension reduction and utility-based selection.

    Each replicate generates fresh independent training/test data. Only test-set
    outcomes and oracle means are used for final evaluation, not model choice.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError("repeats must be an integer")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not (np.isfinite(cost) and cost >= 0):
        raise ValueError("cost must be finite and non-negative")

    records: list[dict[str, str | int | float]] = []
    scenarios: tuple[Scenario, ...] = (
        "low_variance_signal",
        "high_variance_signal",
    )
    for scenario_index, scenario in enumerate(scenarios):
        for replicate in range(repeats):
            local_seed = seed + scenario_index * 100_000 + replicate
            data = simulate_data(
                seed=local_seed, scenario=scenario,
                n_train=n_train, n_test=n_test,
            )
            predictions: dict[str, FloatArray] = {}
            for name, estimator in candidate_estimators().items():
                estimator.fit(data.x_train, data.y_train)  # type: ignore[attr-defined]
                predictions[name] = _predict(estimator, data.x_test)
            predictions["p-value screening (exploratory)"] = pvalue_screened_prediction(
                data.x_train, data.y_train, data.x_test
            )
            guided_prediction, selected = validation_guided_prediction(
                data.x_train, data.y_train, data.x_test, cost, local_seed
            )
            predictions["Decision-guided validation"] = guided_prediction
            for name, prediction in predictions.items():
                records.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "model": name,
                        "selected_candidate": (
                            selected if name == "Decision-guided validation" else ""
                        ),
                        "rmse": float(np.sqrt(np.mean((data.y_test - prediction) ** 2))),
                        "expected_profit": expected_profit(
                            data.mean_test, prediction, cost
                        ),
                        "regret": policy_regret(data.mean_test, prediction, cost),
                        "oracle_profit": float(
                            np.mean(np.maximum(data.mean_test - cost, 0))
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)
