"""Chronological prequential calibration and decisions under changing processes.

The stable stream has independent innovations and IID examples. Other streams
have a deterministic distributional change at a declared deployment breakpoint.
No serial autocorrelation is simulated here; temporal *order* matters through
the change point and the learner's availability of past labelled outcomes.

Prediction and interval construction always precede observing current Y.
A fixed conformal interval calibrated on an independent predeployment
calibration set has a marginal coverage guarantee under exchangeability, but
not after distribution shift. Rolling radii and periodic refitting are
heuristics and carry no finite-sample coverage guarantee under drift.

The known synthetic conditional mean is used only for decision evaluation.
It is never supplied to forecasting, calibration, or model refitting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from analysis import FloatArray, Scenario
from calibration_analysis import split_conformal_radius
from cost_selection import CostGrid, DEFAULT_COSTS, validate_design
from uncertainty_analysis import fixed_candidates

Regime = Literal[
    "stable", "covariate_shift", "noise_shift", "concept_shift"
]
Policy = Literal["Frozen/static", "Frozen/rolling", "Refit/rolling"]
ActionRule = Literal["Point threshold", "Lower-bound threshold"]

REGIMES: tuple[Regime, ...] = (
    "stable", "covariate_shift", "noise_shift", "concept_shift"
)
POLICIES: tuple[Policy, ...] = (
    "Frozen/static", "Frozen/rolling", "Refit/rolling"
)
RULES: tuple[ActionRule, ...] = (
    "Point threshold", "Lower-bound threshold"
)


@dataclass(frozen=True)
class StreamData:
    """Chronological fit/calibration/deployment partitions of one synthetic DGP."""

    x_fit: FloatArray
    y_fit: FloatArray
    x_cal: FloatArray
    y_cal: FloatArray
    x_deploy: FloatArray
    y_deploy: FloatArray
    mean_deploy: FloatArray
    breakpoint: int


def _positive_integer(value: int, label: str, lower: int) -> None:
    """Reject bool and non-integral scalar parameters explicitly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < lower:
        raise ValueError(f"{label} must be at least {lower}")


def generate_stream(
    seed: int,
    regime: Regime = "stable",
    scenario: Scenario = "low_variance_signal",
    n_fit: int = 160,
    n_cal: int = 80,
    n_deploy: int = 120,
    measurement_sd: float = 0.12,
    noise_sd: float = 0.65,
) -> StreamData:
    """Generate paired chronological regimes with an abrupt deployment change.

    All regimes given the same seed use identical latent innovations, predictor
    measurement errors and observation noise. Regime changes occur only after
    the deployment midpoint.

    - Covariate shift translates the first two predictors equally by +1.8;
      their conditional response formula does not change.
    - Noise shift multiplies deployment observation noise by 2.4.
    - Concept shift reverses the regression signal's sign.
    - Stable makes no change; its midpoint is only a diagnostic partition.

    The conditional-mean truth is computed *after* the predictor translation
    and before adding outcome noise.
    """
    for label, value, minimum in (
        ("seed", seed, 0), ("n_fit", n_fit, 64),
        ("n_cal", n_cal, 20), ("n_deploy", n_deploy, 30),
    ):
        _positive_integer(value, label, minimum)
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError(f"Unknown response scenario: {scenario}")
    if not np.isfinite(measurement_sd) or measurement_sd <= 0:
        raise ValueError("measurement_sd must be positive and finite")
    if not np.isfinite(noise_sd) or noise_sd <= 0:
        raise ValueError("noise_sd must be positive and finite")

    n_total = n_fit + n_cal + n_deploy
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_total, 8))
    predictors = np.repeat(latent, 2, axis=1) + rng.normal(
        scale=measurement_sd, size=(n_total, 16)
    )
    noise = rng.normal(scale=noise_sd, size=n_total)
    breakpoint = n_deploy // 2
    after = np.arange(n_total) >= (n_fit + n_cal + breakpoint)

    if regime == "covariate_shift":
        predictors[after, 0:2] += 1.8

    if scenario == "low_variance_signal":
        expected_benefit = 10.0 * (predictors[:, 0] - predictors[:, 1])
    else:
        expected_benefit = 1.2 * (predictors[:, 0] + predictors[:, 1])

    if regime == "concept_shift":
        expected_benefit[after] *= -1.0
    if regime == "noise_shift":
        noise[after] *= 2.4

    observed = expected_benefit + noise
    deploy_start = n_fit + n_cal
    return StreamData(
        x_fit=np.asarray(predictors[:n_fit], dtype=np.float64).copy(),
        y_fit=np.asarray(observed[:n_fit], dtype=np.float64).copy(),
        x_cal=np.asarray(predictors[n_fit:deploy_start], dtype=np.float64).copy(),
        y_cal=np.asarray(observed[n_fit:deploy_start], dtype=np.float64).copy(),
        x_deploy=np.asarray(predictors[deploy_start:], dtype=np.float64).copy(),
        y_deploy=np.asarray(observed[deploy_start:], dtype=np.float64).copy(),
        mean_deploy=np.asarray(expected_benefit[deploy_start:], dtype=np.float64).copy(),
        breakpoint=breakpoint,
    )


def _fit_ols(x: FloatArray, y: FloatArray) -> Pipeline:
    """Fit the same prespecified OLS pipeline on only available observations."""
    model = fixed_candidates()["OLS"]
    model.fit(x, y)
    return model


def _predict_one(model: Pipeline, x: FloatArray) -> float:
    """Predict one row and enforce a finite scalar response."""
    raw = np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
    if raw.shape != (1,) or not np.isfinite(raw[0]):
        raise ValueError("Model must return exactly one finite prediction")
    return float(raw[0])


def _rolling_quantile(scores: deque[float], alpha: float) -> float:
    """Apply the finite-sample corrected conformal quantile to past residuals."""
    residuals = np.asarray(scores, dtype=np.float64)
    return split_conformal_radius(
        observed_cal=residuals,
        predicted_cal=np.zeros_like(residuals),
        alpha=alpha,
    )


def evaluate_stream(
    stream: StreamData,
    alpha: float = 0.10,
    costs: CostGrid = DEFAULT_COSTS,
    rolling_window: int = 60,
    refit_every: int = 20,
    refit_window: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate three interval policies using predict-then-update chronology.

    At deployment step t, predictions, intervals, and economic actions are
    determined from fit/calibration data plus labelled deployment rows < t.
    Only then is Y_t observed and its residual appended to the rolling queue.

    Refit/rolling may use old calibration observations as labelled history
    *after deployment begins*, so it makes no independent calibration claim.
    Its rolling buffer contains historical prequential forecast errors, which
    may arise from different fitted models.
    """
    validate_design(costs, nominal_cost=0.6, split_count=2)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be a real scalar")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    for label, value, minimum in (
        ("rolling_window", rolling_window, 20),
        ("refit_every", refit_every, 1),
        ("refit_window", refit_window, 64),
    ):
        _positive_integer(value, label, minimum)

    if (
        stream.x_fit.ndim != 2
        or stream.x_cal.ndim != 2
        or stream.x_deploy.ndim != 2
        or stream.x_fit.shape[1] != stream.x_cal.shape[1]
        or stream.x_fit.shape[1] != stream.x_deploy.shape[1]
    ):
        raise ValueError("Training, calibration, and deployment feature shapes must agree")
    if (
        stream.y_fit.shape != (len(stream.x_fit),)
        or stream.y_cal.shape != (len(stream.x_cal),)
        or stream.y_deploy.shape != (len(stream.x_deploy),)
        or stream.mean_deploy.shape != (len(stream.x_deploy),)
        or not 0 < stream.breakpoint < len(stream.x_deploy)
    ):
        raise ValueError("Outcome partition shapes or breakpoint are invalid")
    if any(
        not np.isfinite(array).all()
        for array in (
            stream.x_fit, stream.x_cal, stream.x_deploy,
            stream.y_fit, stream.y_cal,
            stream.y_deploy, stream.mean_deploy,
        )
    ):
        raise ValueError("Chronological experiment requires complete finite data")

    fixed_model = _fit_ols(stream.x_fit, stream.y_fit)
    refitted_model = _fit_ols(stream.x_fit, stream.y_fit)
    initial_cal_prediction = np.asarray(
        fixed_model.predict(stream.x_cal), dtype=np.float64
    ).reshape(-1)
    initial_scores = np.abs(stream.y_cal - initial_cal_prediction)
    static_radius = split_conformal_radius(
        stream.y_cal, initial_cal_prediction, alpha
    )
    history_frozen: deque[float] = deque(
        (float(v) for v in initial_scores[-rolling_window:]),
        maxlen=rolling_window,
    )
    history_refitted: deque[float] = deque(
        (float(v) for v in initial_scores[-rolling_window:]),
        maxlen=rolling_window,
    )

    interval_rows: list[dict[str, str | int | float]] = []
    decision_rows: list[dict[str, str | int | float]] = []
    for t in range(len(stream.y_deploy)):
        if t > 0 and t % refit_every == 0:
            # Only labels from prior deployment steps are included.  The
            # initial calibration partition has become historical data;
            # rolling recalibration no longer carries a formal guarantee.
            available_x = np.concatenate(
                (stream.x_fit, stream.x_cal, stream.x_deploy[:t]), axis=0
            )
            available_y = np.concatenate(
                (stream.y_fit, stream.y_cal, stream.y_deploy[:t]), axis=0
            )
            refitted_model = _fit_ols(
                available_x[-refit_window:],
                available_y[-refit_window:],
            )

        one_x = stream.x_deploy[t : t + 1]
        observed_y = float(stream.y_deploy[t])
        true_mean = float(stream.mean_deploy[t])
        frozen_prediction = _predict_one(fixed_model, one_x)
        refitted_prediction = _predict_one(refitted_model, one_x)
        frozen_radius = _rolling_quantile(history_frozen, alpha)
        refitted_radius = _rolling_quantile(history_refitted, alpha)
        policies: dict[Policy, tuple[float, float]] = {
            "Frozen/static": (frozen_prediction, static_radius),
            "Frozen/rolling": (frozen_prediction, frozen_radius),
            "Refit/rolling": (refitted_prediction, refitted_radius),
        }
        phase = "before_shift" if t < stream.breakpoint else "after_shift"
        for policy, (prediction, radius) in policies.items():
            included = float(abs(observed_y - prediction) <= radius)
            interval_rows.append(
                {
                    "time": t,
                    "phase": phase,
                    "policy": policy,
                    "prediction": prediction,
                    "radius": float(radius),
                    "interval_width": float(2.0 * radius),
                    "observed_y": observed_y,
                    "true_mean": true_mean,
                    "covered": included,
                    "absolute_error": abs(observed_y - prediction),
                }
            )
            for cost in costs:
                oracle_profit = max(true_mean - cost, 0.0)
                for action_rule, action in (
                    ("Point threshold", prediction > cost),
                    ("Lower-bound threshold", prediction - radius > cost),
                ):
                    profit = (true_mean - cost) * float(action)
                    decision_rows.append(
                        {
                            "time": t,
                            "phase": phase,
                            "policy": policy,
                            "rule": action_rule,
                            "cost": float(cost),
                            "action": float(action),
                            "expected_profit": float(profit),
                            "oracle_profit": float(oracle_profit),
                            "regret": float(oracle_profit - profit),
                            "unprofitable_action": float(
                                action and true_mean <= cost
                            ),
                            "missed_profitable_action": float(
                                not action and true_mean > cost
                            ),
                        }
                    )

        # CURRENT observation is only exposed after all current predictions,
        # radius calculations, and decisions are recorded.
        history_frozen.append(abs(observed_y - frozen_prediction))
        history_refitted.append(abs(observed_y - refitted_prediction))

    return pd.DataFrame.from_records(interval_rows), pd.DataFrame.from_records(
        decision_rows
    )


def run_temporal_experiment(
    repeats: int = 8,
    n_fit: int = 160,
    n_cal: int = 80,
    n_deploy: int = 120,
    costs: CostGrid = DEFAULT_COSTS,
    alpha: float = 0.10,
    rolling_window: int = 60,
    refit_every: int = 20,
    refit_window: int = 100,
    seed: int = 2026,
    scenarios: tuple[Scenario, ...] = (
        "low_variance_signal", "high_variance_signal"
    ),
    regimes: tuple[Regime, ...] = REGIMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair temporal perturbations on identical seeded underlying innovations.

    Every replicate is a distinct simulated stream. Regimes for the same
    scenario and replicate share pre-breakpoint data and random innovations.
    Overlapping time steps *within* a stream are not independent replicates.
    """
    _positive_integer(repeats, "repeats", 1)
    _positive_integer(seed, "seed", 0)
    if (
        not scenarios
        or any(s not in ("low_variance_signal", "high_variance_signal") for s in scenarios)
        or len(set(scenarios)) != len(scenarios)
    ):
        raise ValueError("Unknown, duplicate, or missing response scenarios")
    if (
        not regimes or any(r not in REGIMES for r in regimes)
        or len(set(regimes)) != len(regimes)
    ):
        raise ValueError("Unknown, duplicate, or missing drift regimes")

    all_intervals: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    for scenario_idx, scenario in enumerate(scenarios):
        for replicate in range(repeats):
            shared_seed = seed + scenario_idx * 1_000_000 + replicate
            for regime in regimes:
                stream = generate_stream(
                    seed=shared_seed,
                    scenario=scenario,
                    regime=regime,
                    n_fit=n_fit,
                    n_cal=n_cal,
                    n_deploy=n_deploy,
                )
                intervals, decisions = evaluate_stream(
                    stream,
                    alpha=alpha,
                    costs=costs,
                    rolling_window=rolling_window,
                    refit_every=refit_every,
                    refit_window=refit_window,
                )
                for frame in (intervals, decisions):
                    frame.insert(0, "replicate", replicate)
                    frame.insert(0, "regime", regime)
                    frame.insert(0, "scenario", scenario)
                all_intervals.append(intervals)
                all_decisions.append(decisions)
    return (
        pd.concat(all_intervals, ignore_index=True),
        pd.concat(all_decisions, ignore_index=True),
    )


def aggregate_temporal_coverage(
    intervals: pd.DataFrame,
    bootstrap_draws: int = 500,
    seed: int = 17,
) -> pd.DataFrame:
    """Bootstrap independent streams, not correlated steps within a stream.

    The empirical coverage interval quantifies simulation uncertainty across
    independent generated datasets. It does not restore a conformal guarantee
    under distribution shift, and few replicates give weak precision.
    """
    _positive_integer(bootstrap_draws, "bootstrap_draws", 10)
    _positive_integer(seed, "seed", 0)
    keys = ["scenario", "regime", "phase", "policy", "replicate"]
    required = set(keys) | {"covered", "interval_width"}
    if not required.issubset(intervals.columns):
        raise ValueError(f"Missing columns: {required - set(intervals.columns)}")
    if intervals.empty or not intervals["covered"].between(0, 1).all():
        raise ValueError("Coverage records must be present and within [0, 1]")
    if not np.isfinite(intervals["interval_width"]).all():
        raise ValueError("Interval widths must be finite")

    per_stream = intervals.groupby(keys, as_index=False).agg(
        coverage=("covered", "mean"),
        mean_width=("interval_width", "mean"),
    )
    rng = np.random.default_rng(seed)
    groups = ["scenario", "regime", "phase", "policy"]
    records: list[dict[str, str | int | float]] = []
    for identifiers, part in per_stream.groupby(groups, sort=True):
        vals = part["coverage"].to_numpy(dtype=np.float64)
        if len(vals) >= 3:
            indices = rng.integers(0, len(vals), size=(bootstrap_draws, len(vals)))
            bootstrap_coverage = np.mean(vals[indices], axis=1)
            lower, upper = np.quantile(bootstrap_coverage, [0.025, 0.975])
        else:
            lower, upper = float("nan"), float("nan")
        record: dict[str, str | int | float] = dict(zip(groups, identifiers))
        record.update(
            {
                "replicates": int(len(vals)),
                "mean_coverage": float(vals.mean()),
                "mean_width": float(part["mean_width"].mean()),
                "bootstrap_lower_95": float(lower),
                "bootstrap_upper_95": float(upper),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_temporal_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stream economic outcomes before averaging streams."""
    keys = [
        "scenario", "regime", "phase", "policy",
        "cost", "rule", "replicate",
    ]
    required = set(keys) | {
        "regret", "expected_profit", "action",
        "unprofitable_action", "missed_profitable_action"
    }
    if not required.issubset(decisions.columns):
        raise ValueError(f"Missing columns: {required - set(decisions.columns)}")
    if decisions.empty or not np.isfinite(
        decisions[list(required - set(keys))]
    ).all().all():
        raise ValueError("Decision metrics must be finite and nonempty")
    series = decisions.groupby(keys, as_index=False).agg(
        mean_regret=("regret", "mean"),
        mean_profit=("expected_profit", "mean"),
        action_rate=("action", "mean"),
        unprofitable_rate=("unprofitable_action", "mean"),
        missed_profitable_rate=("missed_profitable_action", "mean"),
    )
    group = [column for column in keys if column != "replicate"]
    return series.groupby(group, as_index=False).agg(
        replicates=("replicate", "nunique"),
        mean_regret=("mean_regret", "mean"),
        mean_profit=("mean_profit", "mean"),
        mean_action_rate=("action_rate", "mean"),
        mean_unprofitable_rate=("unprofitable_rate", "mean"),
        mean_missed_profitable_rate=("missed_profitable_rate", "mean"),
    )
