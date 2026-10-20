"""Paired robustness experiments for statistical and decision evidence.

Missing predictors are generated under MCAR or MAR using only a fully observed
anchor variable. Extreme outcome errors affect training labels only. In every
case the original test outcomes and test conditional means remain unchanged.

The full-information conditional mean is reserved for external test evaluation;
it is not available to the trained estimators or validation selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNetCV,
    HuberRegressor,
    LinearRegression,
    RidgeCV,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import (
    FloatArray,
    Scenario,
    SimulatedData,
    expected_profit,
    policy_regret,
    simulate_data,
)

Stress = Literal["clean", "mcar", "mar", "outliers", "mar_outliers"]
STRESSES: tuple[Stress, ...] = (
    "clean",
    "mcar",
    "mar",
    "outliers",
    "mar_outliers",
)
SIGNALS: tuple[Scenario, ...] = (
    "low_variance_signal",
    "high_variance_signal",
)


@dataclass(frozen=True)
class StressedData:
    """Observed train/test data, and full-information test evaluation targets."""

    x_train: FloatArray
    y_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    mean_test: FloatArray
    missing_train_rate: float
    missing_test_rate: float
    contaminated_train_rate: float


def _mask_predictors(
    x: FloatArray,
    mechanism: Literal["mcar", "mar"],
    rng: np.random.Generator,
) -> FloatArray:
    """Mask 15/16 columns, keeping the MAR conditioning variable observed.

    The MCAR probability is 0.20. The MAR probability for all other columns
    depends solely on observed X3: 0.20 + 0.15*tanh(X3), in [0.05, 0.35].
    The full-information X3 is never masked under either mechanism.
    """
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("x must have at least three predictor columns")
    if not np.isfinite(x).all():
        raise ValueError("Source predictors must be complete and finite")

    if mechanism == "mcar":
        probability = np.full((x.shape[0], 1), 0.20, dtype=np.float64)
    elif mechanism == "mar":
        probability = (0.20 + 0.15 * np.tanh(x[:, 2]))[:, None]
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")

    missing = rng.random(size=x.shape) < probability
    missing[:, 2] = False  # Always-observed anchor: missingness is MAR, not MNAR.
    observed = np.asarray(x.copy(), dtype=np.float64)
    observed[missing] = np.nan
    return observed


def apply_stress(
    data: SimulatedData,
    stress: Stress,
    seed: int,
    outlier_fraction: float = 0.10,
    outlier_magnitude: float = 12.0,
) -> StressedData:
    """Perturb observations without modifying the underlying true outcomes.

    "mcar" and "mar" affect both training and test predictor availability.
    "outliers" corrupts a fraction of training Y with symmetric gross errors.
    Combined stress applies MAR predictor missingness and training-Y errors.
    """
    if stress not in STRESSES:
        raise ValueError(f"Unknown stress: {stress}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not np.isfinite(outlier_fraction) or not 0 <= outlier_fraction <= 1:
        raise ValueError("outlier_fraction must lie in [0, 1]")
    if not np.isfinite(outlier_magnitude) or outlier_magnitude < 0:
        raise ValueError("outlier_magnitude must be finite and non-negative")

    rng = np.random.default_rng(int(seed))
    x_train = np.asarray(data.x_train.copy(), dtype=np.float64)
    x_test = np.asarray(data.x_test.copy(), dtype=np.float64)
    y_train = np.asarray(data.y_train.copy(), dtype=np.float64)

    if stress in ("mcar", "mar", "mar_outliers"):
        mechanism: Literal["mcar", "mar"] = (
            "mcar" if stress == "mcar" else "mar"
        )
        x_train = _mask_predictors(x_train, mechanism, rng)
        x_test = _mask_predictors(x_test, mechanism, rng)

    outlier_count = 0
    if stress in ("outliers", "mar_outliers"):
        outlier_count = int(np.floor(outlier_fraction * y_train.size))
        if outlier_count > 0:
            affected = rng.choice(y_train.size, size=outlier_count, replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), size=outlier_count)
            y_train[affected] += signs * outlier_magnitude

    return StressedData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=np.asarray(data.y_test.copy(), dtype=np.float64),
        mean_test=np.asarray(data.mean_test.copy(), dtype=np.float64),
        missing_train_rate=float(np.isnan(x_train).mean()),
        missing_test_rate=float(np.isnan(x_test).mean()),
        contaminated_train_rate=float(outlier_count / y_train.size),
    )


def stress_estimators() -> dict[str, Pipeline]:
    """Fresh comparable estimators, each with training-fold-only imputation.

    Missingness indicators are included, but median imputation remains a
    pragmatic baseline rather than a valid universal solution under MAR.
    """
    def pipeline(regressor: object) -> Pipeline:
        return make_pipeline(
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
            StandardScaler(),
            regressor,
        )

    return {
        "OLS": pipeline(LinearRegression()),
        "Ridge": pipeline(RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3)),
        "Elastic net": pipeline(
            ElasticNetCV(
                l1_ratio=[0.25, 0.5, 0.8, 1.0],
                alphas=np.logspace(-3, 0.5, 12),
                cv=3,
                max_iter=7000,
                random_state=0,
            )
        ),
        "Huber": pipeline(
            HuberRegressor(alpha=0.1, epsilon=1.35, max_iter=700)
        ),
        "PCR (95% X variance)": make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            StandardScaler(),
            PCA(n_components=0.95),
            LinearRegression(),
        ),
        "PLS (2 components)": pipeline(
            PLSRegression(n_components=2, scale=False)
        ),
    }


def _predict(model: Pipeline, x: FloatArray) -> FloatArray:
    """Return a finite one-dimensional numeric prediction vector."""
    predicted = np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
    if predicted.shape != (x.shape[0],) or not np.isfinite(predicted).all():
        raise ValueError("Estimator returned invalid predictions")
    return predicted


def choose_by_validation(
    data: StressedData,
    cost: float,
    seed: int,
) -> tuple[dict[str, FloatArray], dict[str, str]]:
    """Select models on an inner split using observed RMSE and net payoff.

    All preprocessing, including imputation and indicator construction,
    happens inside the fit on the inner training split. Labels used for
    selection may be contaminated, which is itself a study limitation.
    """
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("cost must be finite and non-negative")
    idx_train, idx_valid = train_test_split(
        np.arange(data.y_train.size),
        test_size=0.30,
        random_state=seed,
    )
    utilities: dict[str, float] = {}
    errors: dict[str, float] = {}
    for name, model in stress_estimators().items():
        model.fit(data.x_train[idx_train], data.y_train[idx_train])
        predicted = _predict(model, data.x_train[idx_valid])
        observed = data.y_train[idx_valid]
        utilities[name] = float(np.mean((observed - cost) * (predicted > cost)))
        errors[name] = float(np.mean((observed - predicted) ** 2))

    selected = {
        "Validation utility": max(utilities, key=utilities.__getitem__),
        "Validation RMSE": min(errors, key=errors.__getitem__),
    }
    predictions: dict[str, FloatArray] = {}
    trained: dict[str, FloatArray] = {}
    for label, winner in selected.items():
        if winner not in trained:
            model = stress_estimators()[winner]
            model.fit(data.x_train, data.y_train)
            trained[winner] = _predict(model, data.x_test)
        predictions[label] = trained[winner]
    return predictions, selected


def run_stress_experiment(
    repeats: int = 6,
    n_train_values: tuple[int, ...] = (80, 180, 360),
    n_test: int = 500,
    cost: float = 0.6,
    seed: int = 2026,
    signal_scenarios: tuple[Scenario, ...] = SIGNALS,
    stresses: tuple[Stress, ...] = STRESSES,
) -> pd.DataFrame:
    """Evaluate all stress conditions on paired, clean underlying samples.

    Within one (signal, sample size, replicate), all stress conditions share
    the same generated clean data. A new test set is drawn per replicate.
    Selection never sees test outcomes or the full-information conditional mean.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(n_test, int) or isinstance(n_test, bool) or n_test < 10:
        raise ValueError("n_test must be at least 10")
    if not n_train_values or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 64
        for n in n_train_values
    ):
        raise ValueError("training sample sizes must all be integers >= 64")
    if len(set(n_train_values)) != len(n_train_values):
        raise ValueError("training sample sizes must be distinct")
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("cost must be finite and non-negative")
    if not signal_scenarios or any(s not in SIGNALS for s in signal_scenarios):
        raise ValueError("Unknown or missing signal scenarios")
    if not stresses or any(s not in STRESSES for s in stresses):
        raise ValueError("Unknown or missing stresses")

    records: list[dict[str, str | int | float]] = []
    for signal_index, signal in enumerate(signal_scenarios):
        for n_train in n_train_values:
            for replicate in range(repeats):
                data_seed = seed + signal_index * 10_000_000 + n_train * 100 + replicate
                clean_data = simulate_data(
                    seed=data_seed,
                    scenario=signal,
                    n_train=n_train,
                    n_test=n_test,
                )
                for stress_index, stress in enumerate(stresses):
                    observed = apply_stress(
                        clean_data, stress, seed=data_seed + 100_000 * (stress_index + 1)
                    )
                    predictions: dict[str, FloatArray] = {}
                    for name, model in stress_estimators().items():
                        model.fit(observed.x_train, observed.y_train)
                        predictions[name] = _predict(model, observed.x_test)
                    chosen_predictions, chosen_names = choose_by_validation(
                        observed, cost=cost, seed=data_seed
                    )
                    predictions.update(chosen_predictions)

                    oracle_profit = float(
                        np.mean(np.maximum(observed.mean_test - cost, 0.0))
                    )
                    for model_name, predicted in predictions.items():
                        records.append(
                            {
                                "signal": signal,
                                "n_train": n_train,
                                "replicate": replicate,
                                "stress": stress,
                                "model": model_name,
                                "selected_candidate": chosen_names.get(model_name, ""),
                                "rmse": float(
                                    np.sqrt(np.mean((observed.y_test - predicted) ** 2))
                                ),
                                "regret": policy_regret(
                                    observed.mean_test, predicted, cost
                                ),
                                "expected_profit": expected_profit(
                                    observed.mean_test, predicted, cost
                                ),
                                "oracle_profit": oracle_profit,
                                "missing_train_rate": observed.missing_train_rate,
                                "missing_test_rate": observed.missing_test_rate,
                                "outlier_train_rate": observed.contaminated_train_rate,
                            }
                        )
    return pd.DataFrame.from_records(records)


def paired_regret_shift(
    experiment: pd.DataFrame,
    bootstrap_draws: int = 500,
    seed: int = 7,
) -> pd.DataFrame:
    """Bootstrap paired replicate-level regret changes from clean data.

    Positive values indicate larger decision regret under a perturbation.
    Bootstrap units are independent simulated datasets, *not individual rows*.
    The intervals describe finite Monte Carlo variability, not external validity.
    """
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws < 10:
        raise ValueError("Require at least 10 bootstrap draws")
    required = {"signal", "n_train", "replicate", "stress", "model", "regret"}
    if not required.issubset(experiment.columns):
        raise ValueError(f"Missing experiment columns: {required - set(experiment.columns)}")
    keys = ["signal", "n_train", "replicate", "model"]
    clean = experiment.loc[experiment["stress"] == "clean", keys + ["regret"]]
    if clean.empty:
        raise ValueError("Paired comparison requires the clean-data baseline")
    clean = clean.rename(columns={"regret": "clean_regret"})
    affected = experiment.loc[experiment["stress"] != "clean"].copy()
    paired = affected.merge(clean, on=keys, how="left", validate="many_to_one")
    if paired["clean_regret"].isna().any():
        raise ValueError("Every stressed replicate must have a clean counterpart")
    paired["delta_regret"] = paired["regret"] - paired["clean_regret"]

    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int | float]] = []
    for (signal, n_train, stress, model_name), group in paired.groupby(
        ["signal", "n_train", "stress", "model"], sort=True
    ):
        values = group["delta_regret"].to_numpy(dtype=np.float64)
        indices = rng.integers(
            0, values.size, size=(bootstrap_draws, values.size)
        )
        boot_mean = values[indices].mean(axis=1)
        rows.append(
            {
                "signal": str(signal),
                "n_train": int(n_train),
                "stress": str(stress),
                "model": str(model_name),
                "paired_mean_delta_regret": float(np.mean(values)),
                "bootstrap_lower_95": float(np.quantile(boot_mean, 0.025)),
                "bootstrap_upper_95": float(np.quantile(boot_mean, 0.975)),
                "replicates": int(values.size),
            }
        )
    return pd.DataFrame(rows)
