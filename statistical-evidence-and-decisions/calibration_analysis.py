"""Independent calibration, coverage diagnostics and cost-sensitive decisions.

A split-conformal prediction interval constructed from a calibration set
separate from all model fitting and model selection has finite-sample
* marginal * coverage at least (1 - alpha) under exchangeability, apart from
any unavoidable discreteness. The guarantee targets a future observed Y,
not a conditional mean, an individual subgroup, or an intervention effect.

This study deliberately breaks exchangeability by doubling test observation
noise while holding the underlying conditional mean fixed. Test outcomes and
known synthetic conditional means are used only for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analysis import FloatArray, Scenario, simulate_data
from cost_selection import CostGrid, DEFAULT_COSTS, validate_design
from stress_analysis import Stress, apply_stress
from uncertainty_analysis import fixed_candidates

CalibrationMethod = Literal["Gaussian plug-in", "Split conformal"]
Distribution = Literal["IID", "Noise inflation"]
ModelName = Literal["OLS", "Ridge", "Fit-only selected"]
METHODS: tuple[CalibrationMethod, ...] = (
    "Gaussian plug-in",
    "Split conformal",
)
MODELS: tuple[ModelName, ...] = ("OLS", "Ridge", "Fit-only selected")
DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)
DECISION_ALPHA = 0.10


@dataclass(frozen=True)
class CalibrationSplit:
    """Disjoint data used for fitting, calibration, and external evaluation."""

    x_fit: FloatArray
    y_fit: FloatArray
    x_cal: FloatArray
    y_cal: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    mean_test: FloatArray


def make_calibration_split(
    seed: int,
    scenario: Scenario,
    stress: Stress = "clean",
    n_fit: int = 180,
    n_cal: int = 120,
    n_test: int = 300,
) -> CalibrationSplit:
    """Draw independent fit, calibration and test samples from one DGP.

    MCAR/MAR are legal possible stresses in the simulation, but this study
    deliberately restricts itself to clean and MAR data. Outcome contamination
    is excluded because it would make the calibration/test label mechanisms
    different, invalidating their exchangeability.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for label, count, minimum in (
        ("n_fit", n_fit, 64),
        ("n_cal", n_cal, 20),
        ("n_test", n_test, 20),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError("Unknown data-generating response scenario")
    if stress not in ("clean", "mar"):
        raise ValueError("This calibration study supports only clean or MAR data")
    data = simulate_data(
        seed=seed, scenario=scenario, n_train=n_fit + n_cal, n_test=n_test
    )
    observed = apply_stress(data, stress=stress, seed=seed + 100_001)
    return CalibrationSplit(
        x_fit=observed.x_train[:n_fit].copy(),
        y_fit=observed.y_train[:n_fit].copy(),
        x_cal=observed.x_train[n_fit:].copy(),
        y_cal=observed.y_train[n_fit:].copy(),
        x_test=observed.x_test.copy(),
        y_test=observed.y_test.copy(),
        mean_test=observed.mean_test.copy(),
    )


def _check_alpha(alpha: float) -> float:
    """Validate finite one-sided tail probability for symmetric intervals."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be a real scalar")
    alpha_float = float(alpha)
    if not np.isfinite(alpha_float) or not 0 < alpha_float < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return alpha_float


def split_conformal_radius(
    observed_cal: FloatArray,
    predicted_cal: FloatArray,
    alpha: float,
) -> float:
    """Finite-sample corrected quantile of absolute calibration residuals.

    With n calibration examples, use the k-th smallest absolute residual,
    k = ceil((n+1)(1-alpha)). If k > n, the correct conservative threshold
    is +infinity. Ties may cause overcoverage.

    Exchangeability must hold for the fitted model's calibration/test scores.
    """
    level = _check_alpha(alpha)
    observed = np.asarray(observed_cal, dtype=np.float64)
    predicted = np.asarray(predicted_cal, dtype=np.float64)
    if observed.ndim != 1 or observed.size < 1 or predicted.shape != observed.shape:
        raise ValueError("Calibration arrays must be aligned, nonempty vectors")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Calibration outcomes and predictions must be finite")
    residuals = np.abs(observed - predicted)
    index = int(np.ceil((observed.size + 1) * (1.0 - level)))
    if index > observed.size:
        return float("inf")
    return float(np.partition(residuals, index - 1)[index - 1])


def gaussian_plugin_radius(
    observed_fit: FloatArray,
    predicted_fit: FloatArray,
    alpha: float,
) -> float:
    """Gaussian interval radius from in-sample residual standard deviation.

    This is an intentionally optimistic *plug-in* comparison baseline. It
    omits parameter uncertainty and in-sample degrees-of-freedom correction;
    its nominal level does not provide a finite-sample coverage guarantee.
    """
    level = _check_alpha(alpha)
    observed = np.asarray(observed_fit, dtype=np.float64)
    predicted = np.asarray(predicted_fit, dtype=np.float64)
    if observed.ndim != 1 or observed.size < 2 or predicted.shape != observed.shape:
        raise ValueError("Need at least two aligned fitted observations")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Fitted outcomes and predictions must be finite")
    standard_deviation = float(np.std(observed - predicted, ddof=1))
    return float(norm.ppf(1.0 - level / 2.0) * standard_deviation)


def _prediction(model: Pipeline, x: FloatArray) -> FloatArray:
    """Convert regression prediction to a finite vector."""
    pred = np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
    if pred.shape != (len(x),) or not np.isfinite(pred).all():
        raise ValueError("Model produced invalid predictions")
    return pred


def fit_calibration_models(
    split: CalibrationSplit,
    seed: int,
) -> tuple[dict[ModelName, Pipeline], str]:
    """Fit candidates without touching calibration or independent test data.

    The selected candidate is chosen using a holdout taken *only* from the
    fitting set, and is then refitted using all fitting observations. Model
    selection and all preprocessing must remain upstream of calibration.
    """
    n_fit = len(split.y_fit)
    if n_fit < 64:
        raise ValueError("Need at least 64 fitting observations")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    idx_train, idx_valid = train_test_split(
        np.arange(n_fit), test_size=0.25, random_state=seed
    )
    loss: dict[str, float] = {}
    for name, model in fixed_candidates().items():
        model.fit(split.x_fit[idx_train], split.y_fit[idx_train])
        prediction = _prediction(model, split.x_fit[idx_valid])
        loss[name] = float(np.mean((split.y_fit[idx_valid] - prediction) ** 2))
    selected = min(sorted(loss), key=loss.__getitem__)

    selected_models: dict[ModelName, Pipeline] = {}
    candidate_names: dict[ModelName, str] = {
        "OLS": "OLS",
        "Ridge": "Ridge",
        "Fit-only selected": selected,
    }
    for label, name in candidate_names.items():
        model = fixed_candidates()[name]
        model.fit(split.x_fit, split.y_fit)
        selected_models[label] = model
    return selected_models, selected


def _case_coverage(
    observed_test: FloatArray,
    prediction: FloatArray,
    radius: float,
    anchor: FloatArray,
) -> tuple[float, float, float]:
    """Return marginal and pre-specified anchor-subgroup empirical coverages."""
    actual = np.asarray(observed_test, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    x_anchor = np.asarray(anchor, dtype=np.float64)
    if (
        actual.ndim != 1
        or actual.shape != forecast.shape
        or actual.shape != x_anchor.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(forecast).all()
        or not np.isfinite(x_anchor).all()
    ):
        raise ValueError("Test response, prediction and anchor must align")
    if np.isnan(radius) or radius < 0:
        raise ValueError("Interval radius must be non-negative")
    included = np.abs(actual - forecast) <= radius
    positive = x_anchor > 0
    nonpositive = ~positive
    return (
        float(included.mean()),
        float(included[positive].mean()) if positive.any() else float("nan"),
        float(included[nonpositive].mean()) if nonpositive.any() else float("nan"),
    )


def decision_evaluation(
    mean_test: FloatArray,
    prediction: FloatArray,
    radii: dict[CalibrationMethod, float],
    costs: CostGrid = DEFAULT_COSTS,
    review_cost: float = 0.15,
) -> pd.DataFrame:
    """Score point actions, conservative actions, and idealised review.

    The interval is a *prediction interval for noisy Y*, not a confidence
    interval for the expected treatment benefit. Therefore applying it as
    a decision gate is a heuristic, NOT a calibrated error-risk guarantee.

    For ideal review, an external perfect reviewer learns the *true*
    full-information conditional mean on deferred cases, and each review
    incurs cost. This is an optimistic benchmark, not a deployable algorithm.
    """
    grid = validate_design(costs, nominal_cost=0.6, split_count=2)
    true_mean = np.asarray(mean_test, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if (
        true_mean.ndim != 1
        or true_mean.size < 1
        or true_mean.shape != pred.shape
        or not np.isfinite(true_mean).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError("True conditional means and predictions must align")
    if (
        isinstance(review_cost, bool)
        or not isinstance(review_cost, (int, float, np.number))
        or not np.isfinite(review_cost)
        or review_cost < 0
    ):
        raise ValueError("Review cost must be finite and non-negative")
    if set(radii) != set(METHODS):
        raise ValueError("Radii must include Gaussian plug-in and split conformal")
    for radius in radii.values():
        if np.isnan(radius) or radius < 0:
            raise ValueError("Interval radii must be non-negative")

    records: list[dict[str, str | float]] = []
    for cost in grid:
        oracle_profit = float(np.mean(np.maximum(true_mean - cost, 0.0)))

        def record(
            rule: str,
            method: str,
            act_direct: NDArray[np.bool_],
            reviewed: NDArray[np.bool_],
            ambiguous: NDArray[np.bool_],
        ) -> None:
            """Score decisions: deferred cases are perfectly reviewed, at a fee."""
            if np.any(act_direct & reviewed):
                raise AssertionError("Cannot act directly and review the same case")
            contribution = (
                (true_mean - cost) * act_direct
                + (np.maximum(true_mean - cost, 0) - float(review_cost)) * reviewed
            )
            profit = float(np.mean(contribution))
            regret = float(oracle_profit - profit)
            records.append(
                {
                    "method": method,
                    "rule": rule,
                    "cost": float(cost),
                    "action_rate": float(np.mean(act_direct)),
                    "review_rate": float(np.mean(reviewed)),
                    "ambiguous_case_rate": float(np.mean(ambiguous)),
                    "unprofitable_action_rate": float(
                        np.mean(act_direct & (true_mean <= cost))
                    ),
                    "missed_profitable_action_rate": float(
                        np.mean(~(act_direct | reviewed) & (true_mean > cost))
                    ),
                    "expected_profit": profit,
                    "oracle_profit": oracle_profit,
                    "regret": regret,
                }
            )

        empty = np.zeros(len(pred), dtype=bool)
        record("Point threshold", "Point estimate", pred > cost, empty, empty)
        for method, radius in radii.items():
            lower = pred - radius
            upper = pred + radius
            ambiguous = (lower <= cost) & (upper >= cost)
            direct_act = lower > cost
            record("Conservative", method, direct_act, empty, ambiguous)
            record("Ideal review", method, direct_act, ambiguous, ambiguous)
    return pd.DataFrame.from_records(records)


def run_calibration_experiment(
    repeats: int = 12,
    n_fit: int = 180,
    n_cal_values: tuple[int, ...] = (60, 140),
    n_test: int = 250,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    costs: CostGrid = DEFAULT_COSTS,
    review_cost: float = 0.15,
    noise_inflation: float = 2.0,
    seed: int = 2026,
    scenarios: tuple[Scenario, ...] = (
        "low_variance_signal", "high_variance_signal"
    ),
    stresses: tuple[Stress, ...] = ("clean", "mar"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired independent simulation experiments and return two data tables.

    Models are fit on n_fit observations and conformal scores are calibrated
    on n_cal *different* observations. IID and noise-inflated test outcomes
    are paired on the same independent covariates and underlying noise draw.

    Interval rows contain test Y coverage, including subgroup diagnostics.
    Decision rows use the simulation's known conditional mean and are
    unaffected by outcome noise inflation because decisions depend on X,
    expected benefit, and cost—not the realised observation noise.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, number, lower in (("n_fit", n_fit, 64), ("n_test", n_test, 20)):
        if isinstance(number, bool) or not isinstance(number, int) or number < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if not n_cal_values or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 20
        for n in n_cal_values
    ) or len(set(n_cal_values)) != len(n_cal_values):
        raise ValueError("Calibration sizes must be distinct integers >= 20")
    levels = tuple(_check_alpha(alpha) for alpha in alphas)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("alpha levels must be nonempty and distinct")
    if DECISION_ALPHA not in levels:
        raise ValueError("alpha levels must include 0.10 for decision evaluation")
    validate_design(costs, nominal_cost=0.6, split_count=2)
    if not np.isfinite(noise_inflation) or noise_inflation <= 0:
        raise ValueError("noise_inflation must be positive and finite")
    if not np.isfinite(review_cost) or review_cost < 0:
        raise ValueError("review_cost must be non-negative and finite")
    if (
        not scenarios
        or any(s not in ("low_variance_signal", "high_variance_signal") for s in scenarios)
        or len(set(scenarios)) != len(scenarios)
        or not stresses
        or any(s not in ("clean", "mar") for s in stresses)
        or len(set(stresses)) != len(stresses)
    ):
        raise ValueError("Only distinct, supported response/stress scenarios are valid")

    interval_rows: list[dict[str, str | int | float]] = []
    decision_rows: list[dict[str, str | int | float]] = []
    for scenario_idx, scenario in enumerate(scenarios):
        for stress_idx, stress in enumerate(stresses):
            for n_cal in n_cal_values:
                for replicate in range(repeats):
                    local_seed = (
                        seed
                        + scenario_idx * 100_000_000
                        + stress_idx * 10_000_000
                        + n_cal * 10_000
                        + replicate
                    )
                    split = make_calibration_split(
                        seed=local_seed,
                        scenario=scenario,
                        stress=stress,
                        n_fit=n_fit,
                        n_cal=n_cal,
                        n_test=n_test,
                    )
                    models, chosen = fit_calibration_models(
                        split=split, seed=local_seed + 17
                    )
                    y_iid = split.y_test
                    y_inflated = (
                        split.mean_test
                        + float(noise_inflation) * (
                            split.y_test - split.mean_test
                        )
                    )
                    populations: dict[Distribution, FloatArray] = {
                        "IID": y_iid,
                        "Noise inflation": y_inflated,
                    }
                    for label, fitted in models.items():
                        fit_pred = _prediction(fitted, split.x_fit)
                        cal_pred = _prediction(fitted, split.x_cal)
                        test_pred = _prediction(fitted, split.x_test)
                        radii: dict[CalibrationMethod, float] = {}
                        for alpha in levels:
                            widths: dict[CalibrationMethod, float] = {
                                "Gaussian plug-in": gaussian_plugin_radius(
                                    split.y_fit, fit_pred, alpha
                                ),
                                "Split conformal": split_conformal_radius(
                                    split.y_cal, cal_pred, alpha
                                ),
                            }
                            if alpha == DECISION_ALPHA:
                                radii = widths
                            for method, width in widths.items():
                                for population, test_y in populations.items():
                                    cov, plus, minus = _case_coverage(
                                        test_y, test_pred, width, split.x_test[:, 2]
                                    )
                                    interval_rows.append(
                                        {
                                            "scenario": scenario,
                                            "stress": stress,
                                            "n_fit": n_fit,
                                            "n_cal": n_cal,
                                            "replicate": replicate,
                                            "model": label,
                                            "selected_candidate": (
                                                chosen if label == "Fit-only selected"
                                                else label
                                            ),
                                            "method": method,
                                            "distribution": population,
                                            "alpha": alpha,
                                            "target_coverage": 1.0 - alpha,
                                            "coverage": cov,
                                            "anchor_positive_coverage": plus,
                                            "anchor_nonpositive_coverage": minus,
                                            "half_width": width,
                                            "interval_width": 2.0 * width,
                                            "test_rmse": float(
                                                np.sqrt(np.mean((test_y - test_pred) ** 2))
                                            ),
                                        }
                                    )
                        if not radii:
                            raise RuntimeError("Decision calibration level was not computed")
                        decisions = decision_evaluation(
                            split.mean_test,
                            test_pred,
                            radii,
                            costs=costs,
                            review_cost=review_cost,
                        )
                        metadata: dict[str, str | int] = {
                            "scenario": scenario,
                            "stress": stress,
                            "n_fit": n_fit,
                            "n_cal": n_cal,
                            "replicate": replicate,
                            "model": label,
                            "selected_candidate": (
                                chosen if label == "Fit-only selected" else label
                            ),
                        }
                        for entry in decisions.to_dict("records"):
                            decision_rows.append({**metadata, **entry})
    return pd.DataFrame.from_records(interval_rows), pd.DataFrame.from_records(
        decision_rows
    )


def aggregate_coverage(
    intervals: pd.DataFrame,
    bootstrap_draws: int = 400,
    seed: int = 37,
) -> pd.DataFrame:
    """Bootstrap across independent generated datasets, not individual rows.

    Each replicate contains many test observations sharing one fitted model
    and one calibration radius. A binomial CI over *all pooled test rows*
    would incorrectly treat their fitted/calibrated intervals as independent.
    """
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws < 10:
        raise ValueError("bootstrap_draws must be >= 10")
    required = {
        "scenario", "stress", "n_fit", "n_cal", "replicate", "model",
        "method", "distribution", "alpha", "target_coverage",
        "coverage", "interval_width", "anchor_positive_coverage",
        "anchor_nonpositive_coverage",
    }
    if not required.issubset(intervals.columns):
        raise ValueError(f"Missing required columns: {required - set(intervals.columns)}")
    keys = [
        "scenario", "stress", "n_fit", "n_cal", "model", "method",
        "distribution", "alpha",
    ]
    if intervals.duplicated(keys + ["replicate"]).any():
        raise ValueError("Duplicate dataset replicate in one comparison group")
    rng = np.random.default_rng(seed)
    output: list[dict[str, str | int | float]] = []
    for group_key, frame in intervals.groupby(keys, sort=True, dropna=False):
        values = frame["coverage"].to_numpy(dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Coverage proportions must be finite")
        target = float(frame["target_coverage"].iloc[0])
        if not np.allclose(frame["target_coverage"], target):
            raise ValueError("Target coverage varies within comparison group")
        if values.size >= 3:
            sampled = rng.integers(
                0, values.size, size=(bootstrap_draws, values.size)
            )
            lower, upper = np.quantile(
                values[sampled].mean(axis=1), [0.025, 0.975]
            )
        else:
            lower, upper = float("nan"), float("nan")
        rec: dict[str, str | int | float] = dict(zip(keys, group_key))
        rec.update(
            {
                "replicates": int(values.size),
                "target_coverage": target,
                "mean_coverage": float(values.mean()),
                "coverage_error": float(values.mean() - target),
                "bootstrap_lower_95": float(lower),
                "bootstrap_upper_95": float(upper),
                "mean_interval_width": float(frame["interval_width"].mean()),
                "anchor_positive_mean_coverage": float(
                    frame["anchor_positive_coverage"].mean()
                ),
                "anchor_nonpositive_mean_coverage": float(
                    frame["anchor_nonpositive_coverage"].mean()
                ),
            }
        )
        output.append(rec)
    return pd.DataFrame.from_records(output)
