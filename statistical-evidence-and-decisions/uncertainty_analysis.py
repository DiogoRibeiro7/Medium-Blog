"""Bootstrap uncertainty of estimation, model selection and economic decisions.

The purpose is a *descriptive*, frequentist sensitivity analysis. Bootstrap
replicates do not constitute a Bayesian posterior, and selection frequencies
are not confidence probabilities or evidence of causal effects.

A bootstrap in-bag sample is fit against its strictly out-of-bag (OOB)
observations. This avoids validating on duplicated records that are also in
the training sample. Candidate hyperparameters are fixed before resampling,
so there is no duplicate-contaminated inner cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import FloatArray, Scenario, expected_profit, policy_regret, simulate_data
from cost_selection import CostGrid, DEFAULT_COSTS, score_validation, validate_design
from stress_analysis import Stress, apply_stress

PolicyName = Literal["Fixed OLS", "Fixed Ridge", "Selected RMSE", "Selected utility"]
DecisionRule = Literal["Ensemble mean", "Majority vote", "Lower 10% bound"]
POLICIES: tuple[PolicyName, ...] = (
    "Fixed OLS",
    "Fixed Ridge",
    "Selected RMSE",
    "Selected utility",
)
SELECTION_POLICIES: tuple[PolicyName, ...] = (
    "Selected RMSE", "Selected utility"
)
DECISION_RULES: tuple[DecisionRule, ...] = (
    "Ensemble mean",
    "Majority vote",
    "Lower 10% bound",
)


@dataclass(frozen=True)
class BootstrapDraws:
    """Predicted *conditional benefits*, never posterior samples.

    predictions[label] has shape (bootstrap_replicates, heldout_observations).
    selected_models contains the algorithm picked in each OOB draw.
    """

    predictions: dict[str, FloatArray]
    selected_models: dict[str, tuple[str, ...]]
    oob_sizes: NDArray[np.int64]


@dataclass(frozen=True)
class UncertaintyStudy:
    """Outputs for one independent simulated training/test dataset."""

    draws: BootstrapDraws
    coefficient_table: pd.DataFrame | None
    decision_table: pd.DataFrame
    variance_table: pd.DataFrame
    coverage_table: pd.DataFrame


def fixed_candidates() -> dict[str, Pipeline]:
    """Create fresh, pre-specified pipelines, with no post-bootstrap CV.

    Preprocessing is estimated exclusively using in-bag observations.
    Ridge alpha=5 and Huber parameters are prespecified for this experiment,
    not optimised using the held-out test observations.
    """

    def base(estimator: object) -> Pipeline:
        return make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            StandardScaler(),
            estimator,
        )

    return {
        "OLS": base(LinearRegression()),
        "Ridge": base(Ridge(alpha=5.0)),
        "Huber": base(HuberRegressor(alpha=0.1, epsilon=1.35, max_iter=600)),
        "PCR (95%)": make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            StandardScaler(),
            PCA(n_components=0.95),
            LinearRegression(),
        ),
    }


def bootstrap_oob_indices(inbag: NDArray[np.int64], n_rows: int) -> NDArray[np.int64]:
    """Return indices not sampled by a bootstrap draw, without overlap."""
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 2:
        raise ValueError("n_rows must be an integer >= 2")
    index = np.asarray(inbag)
    if index.ndim != 1 or index.size != n_rows:
        raise ValueError("inbag must have exactly n_rows elements")
    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError("inbag must contain integer indices")
    if (index < 0).any() or (index >= n_rows).any():
        raise ValueError("inbag indices are outside the data range")
    occupied = np.zeros(n_rows, dtype=bool)
    occupied[index] = True
    oob = np.flatnonzero(~occupied).astype(np.int64)
    if np.intersect1d(index, oob).size != 0:
        raise AssertionError("In-bag and OOB observations overlap")
    return oob


def _validate_xy(
    x_train: FloatArray, y_train: FloatArray, x_test: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Accept incomplete predictors, never infinite values or missing labels."""
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    test = np.asarray(x_test, dtype=np.float64)
    if x.ndim != 2 or test.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be matrices and y must be a vector")
    if x.shape[0] < 64 or x.shape[1] < 2 or test.shape[1] != x.shape[1]:
        raise ValueError("Need >=64 training rows, >=2 predictors, matching columns")
    if y.shape != (x.shape[0],) or test.shape[0] < 1:
        raise ValueError("Training X/y and test X are misaligned")
    if np.isinf(x).any() or np.isinf(test).any() or not np.isfinite(y).all():
        raise ValueError("Non-finite labels and infinite predictors are invalid")
    return x, y, test


def bootstrap_predictive_draws(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    bootstrap_replicates: int = 60,
    costs: CostGrid = DEFAULT_COSTS,
    nominal_cost: float = 0.6,
    seed: int = 2026,
) -> BootstrapDraws:
    """Bootstrap training observations, compare candidates using OOB data.

    Each replicate fits fixed-hyperparameter models on a bootstrap sample
    and evaluates the same *original*, unsampled observations out of bag.
    Selected models are not refitted on the full sample: each final prediction
    captures uncertainty from resampling the training data.

    The scoring criteria are out-of-bag MSE and cost-grid observed net utility.
    Neither test responses nor test conditional means enter this interface.
    """
    validate_design(costs, nominal_cost, split_count=2)
    if isinstance(bootstrap_replicates, bool) or not isinstance(
        bootstrap_replicates, int
    ) or bootstrap_replicates < 6:
        raise ValueError("bootstrap_replicates must be an integer >= 6")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    x, y, xt = _validate_xy(x_train, y_train, x_test)
    rng = np.random.default_rng(seed)
    n_train = y.size
    min_oob = max(8, n_train // 10)
    predictions: dict[str, FloatArray] = {
        name: np.empty((bootstrap_replicates, len(xt)), dtype=np.float64)
        for name in POLICIES
    }
    selected: dict[str, list[str]] = {
        policy: [] for policy in SELECTION_POLICIES
    }
    oob_sizes: list[int] = []
    for draw in range(bootstrap_replicates):
        # Re-sample until we have a usable OOB set (rarely needed at n>=64).
        for _ in range(30):
            inbag = rng.integers(0, n_train, size=n_train, dtype=np.int64)
            oob = bootstrap_oob_indices(inbag, n_train)
            if len(oob) >= min_oob:
                break
        else:
            raise RuntimeError("Unable to obtain a sufficiently large OOB set")

        oob_sizes.append(len(oob))
        candidate_predictions: dict[str, FloatArray] = {}
        candidate_scores: dict[str, tuple[float, float]] = {}
        for name, model in fixed_candidates().items():
            model.fit(x[inbag], y[inbag])
            oob_prediction = np.asarray(
                model.predict(x[oob]), dtype=np.float64
            ).reshape(-1)
            test_prediction = np.asarray(
                model.predict(xt), dtype=np.float64
            ).reshape(-1)
            if (
                oob_prediction.shape != (len(oob),)
                or test_prediction.shape != (len(xt),)
                or not np.isfinite(oob_prediction).all()
                or not np.isfinite(test_prediction).all()
            ):
                raise ValueError("Candidate generated non-finite predictions")
            mse, _, grid_utility = score_validation(
                y[oob], oob_prediction, costs, nominal_cost
            )
            candidate_scores[name] = (mse, grid_utility)
            candidate_predictions[name] = test_prediction
        # Stable lexical ordering resolves exact ties without using test data.
        mse_winner = min(
            sorted(candidate_scores),
            key=lambda name: candidate_scores[name][0],
        )
        utility_winner = max(
            sorted(candidate_scores),
            key=lambda name: candidate_scores[name][1],
        )
        selected["Selected RMSE"].append(mse_winner)
        selected["Selected utility"].append(utility_winner)
        predictions["Fixed OLS"][draw] = candidate_predictions["OLS"]
        predictions["Fixed Ridge"][draw] = candidate_predictions["Ridge"]
        predictions["Selected RMSE"][draw] = candidate_predictions[mse_winner]
        predictions["Selected utility"][draw] = candidate_predictions[utility_winner]

    return BootstrapDraws(
        predictions=predictions,
        selected_models={name: tuple(value) for name, value in selected.items()},
        oob_sizes=np.asarray(oob_sizes, dtype=np.int64),
    )


def coefficient_uncertainty(
    x_train: FloatArray,
    y_train: FloatArray,
    bootstrap_replicates: int = 200,
    seed: int = 2026,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Compare analytic OLS intervals with percentile bootstrap intervals.

    Only complete matrices are admitted. Both intervals refer to *prespecified
    full-model OLS coefficients*; they are not valid post-selection inference.
    The analytic intervals assume the usual homoskedastic Gaussian linear model.
    """
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("Aligned matrix X and vector y are required")
    if x.shape[0] <= x.shape[1] + 2:
        raise ValueError("Need more observations than full-model parameters")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Coefficient audit requires complete finite inputs")
    if isinstance(bootstrap_replicates, bool) or not isinstance(
        bootstrap_replicates, int
    ) or bootstrap_replicates < 6:
        raise ValueError("bootstrap_replicates must be an integer >= 6")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    design = sm.add_constant(x, has_constant="add")
    fitted = sm.OLS(y, design).fit()
    analytic = np.asarray(fitted.conf_int(alpha=alpha), dtype=np.float64)
    rng = np.random.default_rng(seed)
    coefficients = np.empty((bootstrap_replicates, x.shape[1] + 1))
    for draw in range(bootstrap_replicates):
        selected = rng.integers(0, len(y), size=len(y))
        coefficients[draw] = np.linalg.lstsq(
            design[selected], y[selected], rcond=None
        )[0]
    bootstrap_bounds = np.quantile(
        coefficients, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0
    )
    return pd.DataFrame(
        {
            "term": ["intercept"] + [
                f"x{col + 1:02d}" for col in range(x.shape[1])
            ],
            "estimate": np.asarray(fitted.params, dtype=np.float64),
            "p_value": np.asarray(fitted.pvalues, dtype=np.float64),
            "analytic_lower": analytic[:, 0],
            "analytic_upper": analytic[:, 1],
            "bootstrap_lower": bootstrap_bounds[0],
            "bootstrap_upper": bootstrap_bounds[1],
            "bootstrap_sd": np.std(coefficients, axis=0, ddof=1),
            "fraction_positive": np.mean(coefficients > 0, axis=0),
        }
    )


def variance_decomposition(
    predicted: FloatArray, model_labels: tuple[str, ...]
) -> pd.DataFrame:
    """Describe selection-conditioned variation using total variance identity.

    Var(P) = E[Var(P | selected-model label)]
             + Var(E[P | selected-model label]).

    This is a *descriptive partition* of a joint resampling distribution,
    not a causal attribution of variance to model selection. Group membership
    itself is selected using the bootstrap data.
    """
    samples = np.asarray(predicted, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("Predictions must be a 2-D bootstrap-by-test array")
    if len(model_labels) != len(samples) or not model_labels:
        raise ValueError("Every bootstrap row needs a model-selection label")
    if not np.isfinite(samples).all():
        raise ValueError("Predictions must be finite")
    overall_mean = samples.mean(axis=0)
    total = samples.var(axis=0, ddof=0)
    within = np.zeros(samples.shape[1], dtype=np.float64)
    between = np.zeros(samples.shape[1], dtype=np.float64)
    for label in sorted(set(model_labels)):
        block = samples[np.asarray(model_labels) == label]
        weight = len(block) / len(samples)
        block_mean = block.mean(axis=0)
        within += weight * block.var(axis=0, ddof=0)
        between += weight * (block_mean - overall_mean) ** 2
    if not np.allclose(total, within + between, rtol=1e-9, atol=1e-11):
        raise ArithmeticError("Finite-mixture variance identity failed")
    return pd.DataFrame(
        {
            "test_case": np.arange(samples.shape[1]),
            "total_variance": total,
            "within_model_label": within,
            "between_model_labels": between,
        }
    )


def decision_policy_audit(
    predictions: dict[str, FloatArray],
    conditional_mean: FloatArray,
    costs: CostGrid = DEFAULT_COSTS,
) -> pd.DataFrame:
    """Score deterministic ensemble actions against the simulation oracle.

    These ensemble decision rules use only bootstrap predictions and a
    cost disclosed before action. The true conditional mean appears solely
    in evaluation of expected regret and unprofitable decisions.
    """
    validate_design(costs, costs[0], split_count=2)
    truth = np.asarray(conditional_mean, dtype=np.float64)
    if truth.ndim != 1 or truth.size < 1 or not np.isfinite(truth).all():
        raise ValueError("conditional_mean must be a finite 1-D vector")
    output: list[dict[str, str | float]] = []
    for name, matrix in predictions.items():
        draws = np.asarray(matrix, dtype=np.float64)
        if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] != truth.size:
            raise ValueError("Unexpected bootstrap prediction shape")
        if not np.isfinite(draws).all():
            raise ValueError("Bootstrap predictions must be finite")
        prediction_average = draws.mean(axis=0)
        prediction_lower = np.quantile(draws, 0.10, axis=0)
        for cost in costs:
            probability_act = np.mean(draws > cost, axis=0)
            actions = {
                "Ensemble mean": prediction_average > cost,
                "Majority vote": probability_act > 0.50,
                "Lower 10% bound": prediction_lower > cost,
            }
            ambiguity = float(
                np.mean((probability_act >= 0.25) & (probability_act <= 0.75))
            )
            for rule, choice in actions.items():
                # Passing 1/0 as a prediction ensures that the shared
                # economic metric is applied to exactly the specified action.
                net = float(np.mean((truth - cost) * choice))
                oracle = float(np.mean(np.maximum(truth - cost, 0.0)))
                output.append(
                    {
                        "estimator": name,
                        "rule": rule,
                        "cost": float(cost),
                        "action_rate": float(np.mean(choice)),
                        "ambiguous_case_rate": ambiguity,
                        "unprofitable_action_rate": float(
                            np.mean(choice & (truth <= cost))
                        ),
                        "missed_profitable_action_rate": float(
                            np.mean(~choice & (truth > cost))
                        ),
                        "expected_profit": net,
                        "regret": oracle - net,
                        "oracle_profit": oracle,
                    }
                )
    return pd.DataFrame(output)


def interval_audit(
    predictions: dict[str, FloatArray],
    conditional_mean: FloatArray,
    observed_y: FloatArray,
    noise_sd: float = 0.65,
    seed: int = 37,
    noise_replicates: int = 4,
) -> pd.DataFrame:
    """Inspect 90% bootstrap-mean and Gaussian-mixture predictive intervals.

    The observation-noise SD is specified by the synthetic data-generating
    mechanism. It is NOT inferred from test outcomes. The predictive interval
    draws new independent N(0, noise_sd**2) errors for each fitted predictor.

    Empirical coverage is measured on independent test observations. Neither
    interval has a finite-sample calibration guarantee.
    """
    truth = np.asarray(conditional_mean, dtype=np.float64)
    observed = np.asarray(observed_y, dtype=np.float64)
    if truth.ndim != 1 or observed.shape != truth.shape or truth.size < 1:
        raise ValueError("Independent test outcome and mean arrays must align")
    if not np.isfinite(truth).all() or not np.isfinite(observed).all():
        raise ValueError("Test evaluation outcomes must be finite")
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError("noise_sd must be positive and finite")
    if isinstance(noise_replicates, bool) or not isinstance(
        noise_replicates, int
    ) or noise_replicates < 2:
        raise ValueError("noise_replicates must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    output: list[dict[str, str | float]] = []
    for name, draw_matrix in predictions.items():
        draws = np.asarray(draw_matrix, dtype=np.float64)
        if draws.ndim != 2 or draws.shape[0] < 2 or draws.shape[1] != len(truth):
            raise ValueError("Predictor bootstrap draws have invalid shape")
        if not np.isfinite(draws).all():
            raise ValueError("Predictions must be finite")
        low_mu, high_mu = np.quantile(draws, [0.05, 0.95], axis=0)
        new_noise = rng.normal(
            loc=0.0, scale=noise_sd,
            size=(noise_replicates, draws.shape[0], draws.shape[1]),
        )
        prediction_samples = (draws[None, :, :] + new_noise).reshape(
            -1, len(truth)
        )
        low_y, high_y = np.quantile(
            prediction_samples, [0.05, 0.95], axis=0
        )
        output.append(
            {
                "estimator": name,
                "conditional_mean_90pct_coverage": float(
                    np.mean((truth >= low_mu) & (truth <= high_mu))
                ),
                "observation_90pct_coverage": float(
                    np.mean((observed >= low_y) & (observed <= high_y))
                ),
                "mean_interval_width": float(np.mean(high_mu - low_mu)),
                "predictive_interval_width": float(np.mean(high_y - low_y)),
            }
        )
    return pd.DataFrame(output)


def run_uncertainty_study(
    scenario: Scenario = "low_variance_signal",
    stress: Stress = "clean",
    n_train: int = 180,
    n_test: int = 300,
    bootstrap_replicates: int = 60,
    costs: CostGrid = DEFAULT_COSTS,
    seed: int = 2026,
) -> UncertaintyStudy:
    """Run one independent simulation and descriptive uncertainty analysis.

    Only the synthetic evaluator sees the independent test conditional mean.
    The coefficient analysis is restricted to clean, unselected OLS.
    """
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError("Invalid response signal mechanism")
    if stress not in ("clean", "mcar", "mar", "outliers", "mar_outliers"):
        raise ValueError("Invalid perturbation")
    validate_design(costs, nominal_cost=0.6, split_count=2)
    data = simulate_data(seed=seed, scenario=scenario, n_train=n_train, n_test=n_test)
    affected = apply_stress(data, stress=stress, seed=seed + 100_000)
    draws = bootstrap_predictive_draws(
        affected.x_train,
        affected.y_train,
        affected.x_test,
        bootstrap_replicates=bootstrap_replicates,
        costs=costs,
        nominal_cost=0.6,
        seed=seed + 1,
    )
    coefficient_table = (
        coefficient_uncertainty(
            affected.x_train, affected.y_train,
            bootstrap_replicates=max(bootstrap_replicates, 30),
            seed=seed + 2,
        )
        if stress == "clean" else None
    )
    policies = decision_policy_audit(draws.predictions, affected.mean_test, costs)
    variance_records: list[dict[str, str | float]] = []
    for policy in SELECTION_POLICIES:
        component = variance_decomposition(
            draws.predictions[policy], draws.selected_models[policy]
        )
        total = float(component["total_variance"].mean())
        within = float(component["within_model_label"].mean())
        between = float(component["between_model_labels"].mean())
        variance_records.append(
            {
                "selection_policy": policy,
                "mean_total_variance": total,
                "mean_within_label_variance": within,
                "mean_between_label_variance": between,
                "between_fraction": between / total if total > 0 else 0.0,
                "unique_selected_models": float(
                    len(set(draws.selected_models[policy]))
                ),
            }
        )
    coverage = interval_audit(
        draws.predictions,
        affected.mean_test,
        affected.y_test,
        noise_sd=0.65,
        seed=seed + 3,
    )
    return UncertaintyStudy(
        draws=draws,
        coefficient_table=coefficient_table,
        decision_table=policies,
        variance_table=pd.DataFrame(variance_records),
        coverage_table=coverage,
    )
