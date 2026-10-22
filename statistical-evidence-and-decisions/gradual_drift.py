"""Prequential prediction with gradual concept drift and delayed feedback.

Three policies are evaluated on a fixed generated deployment stream. At step t
they can use only the initial labelled fit/calibration partitions and deployed
labels j satisfying j + feedback_delay + 1 <= t.

The immediate-feedback convention is deliberately explicit: a label observed
at the end of step j first becomes usable at the beginning of step j + 1.
With feedback delay d, that first usable step becomes j + d + 1.

The AR(1) scenarios violate the exchangeability assumptions needed for
ordinary split-conformal marginal coverage. Rolling residual recalibration
and refitting are empirical heuristics without coverage guarantees.
Conditional means are reserved for evaluation of synthetic decision regret.
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

Regime = Literal["stable", "gradual_drift", "ar1", "gradual_ar1"]
Policy = Literal["Frozen/static", "Frozen/rolling", "Refit/rolling"]
Phase = Literal["early", "transition", "late"]
REGIMES: tuple[Regime, ...] = (
    "stable", "gradual_drift", "ar1", "gradual_ar1"
)
POLICIES: tuple[Policy, ...] = (
    "Frozen/static", "Frozen/rolling", "Refit/rolling"
)
DEFAULT_DELAYS: tuple[int, ...] = (0, 12)


@dataclass(frozen=True)
class DelayedStream:
    """Known-fit, known-calibration, and sequentially revealed deployment rows."""

    x_fit: FloatArray
    y_fit: FloatArray
    x_cal: FloatArray
    y_cal: FloatArray
    x_live: FloatArray
    y_live: FloatArray
    mean_live: FloatArray
    drift_start: int


def _check_integer(value: int, name: str, minimum: int) -> None:
    """Reject bool values, negative delays and non-integral configurations."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _ar1_process(innovations: FloatArray, rho: float) -> FloatArray:
    """Transform IID standard-normal innovations to stationary Gaussian AR(1).

    Uses X0 = epsilon0 and X_t = rho X_{t-1} + sqrt(1-rho²) epsilon_t.
    The stationary marginal variance is one. The series may be multivariate,
    with independent innovation columns and a common temporal correlation.
    """
    signal = np.asarray(innovations, dtype=np.float64)
    if signal.ndim not in (1, 2) or signal.shape[0] < 1:
        raise ValueError("innovations must be a nonempty vector or matrix")
    if not np.isfinite(signal).all():
        raise ValueError("innovations must be finite")
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must lie in [0, 1)")
    result = np.empty_like(signal)
    result[0] = signal[0]
    new_weight = np.sqrt(1.0 - rho**2)
    for step in range(1, len(signal)):
        result[step] = rho * result[step - 1] + new_weight * signal[step]
    return result


def generate_delayed_stream(
    seed: int,
    regime: Regime = "stable",
    scenario: Scenario = "low_variance_signal",
    n_fit: int = 160,
    n_cal: int = 80,
    n_live: int = 120,
    measurement_sd: float = 0.12,
    outcome_sd: float = 0.65,
    rho: float = 0.75,
) -> DelayedStream:
    """Generate matched stable/gradual and IID/autocorrelated synthetic streams.

    Regimes share innovation arrays at the same seed, but applying AR(1)
    changes the realised paths, including predeployment observations.
    A gradual drift reverses the conditional-mean coefficient continuously
    over the final two thirds of deployment; earlier rows remain unchanged.

    Observation noise and latent factors have AR(1) correlation only in the
    ar1 and gradual_ar1 regimes. Measurement errors remain IID.
    """
    for label, value, limit in (
        ("seed", seed, 0), ("n_fit", n_fit, 64),
        ("n_cal", n_cal, 30), ("n_live", n_live, 45),
    ):
        _check_integer(value, label, limit)
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError(f"Unknown signal mechanism: {scenario}")
    for name, value in (
        ("measurement_sd", measurement_sd),
        ("outcome_sd", outcome_sd),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must lie in [0, 1)")

    rng = np.random.default_rng(seed)
    n_total = n_fit + n_cal + n_live
    z = rng.normal(size=(n_total, 8))
    measurement_noise = rng.normal(scale=measurement_sd, size=(n_total, 16))
    outcome_innovations = rng.normal(size=n_total)
    correlated = regime in ("ar1", "gradual_ar1")
    if correlated:
        z = _ar1_process(z, rho)
        outcome_innovations = _ar1_process(outcome_innovations, rho)

    features = np.repeat(z, 2, axis=1) + measurement_noise
    if scenario == "low_variance_signal":
        mean = 10.0 * (features[:, 0] - features[:, 1])
    else:
        mean = 1.2 * (features[:, 0] + features[:, 1])

    drift_start = n_live // 3
    if regime in ("gradual_drift", "gradual_ar1"):
        location = np.arange(n_live)
        progress = np.clip(
            (location - drift_start) / (n_live - drift_start - 1),
            0.0, 1.0,
        )
        # Initial effect +1 continuously moves to -1 by the final row.
        mean[n_fit + n_cal :] *= 1.0 - 2.0 * progress

    observed = mean + outcome_sd * outcome_innovations
    split = n_fit + n_cal
    return DelayedStream(
        x_fit=features[:n_fit].copy(),
        y_fit=observed[:n_fit].copy(),
        x_cal=features[n_fit:split].copy(),
        y_cal=observed[n_fit:split].copy(),
        x_live=features[split:].copy(),
        y_live=observed[split:].copy(),
        mean_live=mean[split:].copy(),
        drift_start=drift_start,
    )


def available_label_count(time: int, feedback_delay: int) -> int:
    """Number of deployment labels available before predicting at time t.

    Historical index j is eligible only if j + feedback_delay + 1 <= time.
    For example, at t=5 with delay=2, only j=0,1,2 are available.
    """
    _check_integer(time, "time", 0)
    _check_integer(feedback_delay, "feedback_delay", 0)
    return max(0, time - feedback_delay)


def phase_for_step(step: int, n_live: int) -> Phase:
    """Divide one deployment horizon into fixed early/transition/late thirds."""
    _check_integer(step, "step", 0)
    _check_integer(n_live, "n_live", 3)
    if step >= n_live:
        raise ValueError("step must lie within deployment horizon")
    if step < n_live // 3:
        return "early"
    if step < (2 * n_live) // 3:
        return "transition"
    return "late"


def _fresh_ols(x: FloatArray, y: FloatArray) -> Pipeline:
    """Fit the same prespecified pipeline to the currently eligible history."""
    model = fixed_candidates()["OLS"]
    model.fit(x, y)
    return model


def _forecast(model: Pipeline, row: FloatArray) -> float:
    """Return a validated single-row regression prediction."""
    prediction = np.asarray(model.predict(row), dtype=np.float64).reshape(-1)
    if prediction.shape != (1,) or not np.isfinite(prediction[0]):
        raise ValueError("Expected one finite model prediction")
    return float(prediction[0])


def _score_quantile(history: deque[float], alpha: float) -> float:
    """Quantile of already observed absolute prequential forecast errors."""
    scores = np.asarray(history, dtype=np.float64)
    return split_conformal_radius(scores, np.zeros_like(scores), alpha)


def evaluate_delayed_stream(
    stream: DelayedStream,
    feedback_delay: int = 0,
    alpha: float = 0.10,
    costs: CostGrid = DEFAULT_COSTS,
    rolling_window: int = 60,
    refit_every: int = 15,
    refit_window: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict then reveal matured labels, never consulting unreleased outcomes.

    At the beginning of step t:
      1. Release only deployed labels j with j + delay + 1 <= t.
      2. Update two residual queues using the original forecasts stored at j.
      3. Refit the adaptive model periodically on the eligible labelled past.
      4. Produce current predictions, intervals, and decisions.
      5. Store forecasts, without using Y_t until it becomes eligible later.

    A fixed initial conformal interval has a marginal guarantee only when
    calibration and future observations are exchangeable. Rolling intervals
    are unguaranteed under drift, dependence, and refitting.
    """
    _check_integer(feedback_delay, "feedback_delay", 0)
    for label, value, minimum in (
        ("rolling_window", rolling_window, 20),
        ("refit_every", refit_every, 1),
        ("refit_window", refit_window, 64),
    ):
        _check_integer(value, label, minimum)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    checked_costs = validate_design(costs, 0.6, split_count=2)

    if (
        stream.x_fit.ndim != 2 or stream.x_cal.ndim != 2
        or stream.x_live.ndim != 2
        or stream.x_fit.shape[1] != stream.x_cal.shape[1]
        or stream.x_fit.shape[1] != stream.x_live.shape[1]
    ):
        raise ValueError("Predictor matrices must have aligned feature dimensions")
    if (
        stream.y_fit.shape != (len(stream.x_fit),)
        or stream.y_cal.shape != (len(stream.x_cal),)
        or stream.y_live.shape != (len(stream.x_live),)
        or stream.mean_live.shape != stream.y_live.shape
        or not 0 < stream.drift_start < len(stream.y_live)
    ):
        raise ValueError("Response arrays or drift boundary are inconsistent")
    if any(not np.isfinite(x).all() for x in (
        stream.x_fit, stream.y_fit, stream.x_cal, stream.y_cal,
        stream.x_live, stream.y_live, stream.mean_live,
    )):
        raise ValueError("This experiment requires complete finite arrays")

    frozen = _fresh_ols(stream.x_fit, stream.y_fit)
    adaptive = _fresh_ols(stream.x_fit, stream.y_fit)
    cal_pred = np.asarray(frozen.predict(stream.x_cal)).reshape(-1)
    initial_residuals = np.abs(stream.y_cal - cal_pred)
    static_radius = split_conformal_radius(
        stream.y_cal, cal_pred, alpha
    )
    frozen_history: deque[float] = deque(
        map(float, initial_residuals[-rolling_window:]),
        maxlen=rolling_window,
    )
    adaptive_history: deque[float] = deque(
        map(float, initial_residuals[-rolling_window:]),
        maxlen=rolling_window,
    )
    frozen_forecasts: list[float] = []
    adaptive_forecasts: list[float] = []
    released = 0

    interval_rows: list[dict[str, str | int | float]] = []
    decision_rows: list[dict[str, str | int | float]] = []
    for t in range(len(stream.y_live)):
        matured = available_label_count(t, feedback_delay)
        # The only place that previously deployed labels may enter the learner.
        for old in range(released, matured):
            observed = float(stream.y_live[old])
            frozen_history.append(abs(observed - frozen_forecasts[old]))
            adaptive_history.append(abs(observed - adaptive_forecasts[old]))
        released = matured

        if t > 0 and t % refit_every == 0:
            # Calibration is historical, already labelled; deployment labels
            # are restricted to those whose release time has passed.
            historical_x = np.concatenate(
                (stream.x_fit, stream.x_cal, stream.x_live[:matured]), axis=0
            )
            historical_y = np.concatenate(
                (stream.y_fit, stream.y_cal, stream.y_live[:matured]), axis=0
            )
            adaptive = _fresh_ols(
                historical_x[-refit_window:], historical_y[-refit_window:]
            )

        row = stream.x_live[t : t + 1]
        frozen_pred = _forecast(frozen, row)
        adaptive_pred = _forecast(adaptive, row)
        frozen_forecasts.append(frozen_pred)
        adaptive_forecasts.append(adaptive_pred)
        rolling_radius = _score_quantile(frozen_history, alpha)
        adaptive_radius = _score_quantile(adaptive_history, alpha)
        phases: dict[Policy, tuple[float, float]] = {
            "Frozen/static": (frozen_pred, static_radius),
            "Frozen/rolling": (frozen_pred, rolling_radius),
            "Refit/rolling": (adaptive_pred, adaptive_radius),
        }
        phase = phase_for_step(t, len(stream.y_live))
        for policy, (prediction, radius) in phases.items():
            observed = float(stream.y_live[t])
            mean_true = float(stream.mean_live[t])
            interval_rows.append(
                {
                    "step": t,
                    "phase": phase,
                    "delay": feedback_delay,
                    "policy": policy,
                    "available_labels": matured,
                    "prediction": prediction,
                    "radius": float(radius),
                    "width": float(2.0 * radius),
                    "absolute_error": abs(observed - prediction),
                    "covered": float(abs(observed - prediction) <= radius),
                }
            )
            # The synthetic conditional mean is referenced only by the
            # evaluator, after the model has chosen its predictive score.
            for cost in checked_costs:
                oracle = max(mean_true - cost, 0.0)
                for rule, acting in (
                    ("Point", prediction > cost),
                    ("Lower bound", prediction - radius > cost),
                ):
                    expected_profit = (mean_true - cost) * float(acting)
                    decision_rows.append(
                        {
                            "step": t,
                            "phase": phase,
                            "delay": feedback_delay,
                            "policy": policy,
                            "rule": rule,
                            "cost": float(cost),
                            "available_labels": matured,
                            "acted": float(acting),
                            "oracle_profit": float(oracle),
                            "expected_profit": float(expected_profit),
                            "regret": float(oracle - expected_profit),
                        }
                    )
    return pd.DataFrame.from_records(interval_rows), pd.DataFrame.from_records(
        decision_rows
    )


def run_gradual_drift_study(
    repeats: int = 6,
    delays: tuple[int, ...] = DEFAULT_DELAYS,
    regimes: tuple[Regime, ...] = REGIMES,
    scenarios: tuple[Scenario, ...] = (
        "low_variance_signal", "high_variance_signal"
    ),
    n_fit: int = 160,
    n_cal: int = 80,
    n_live: int = 120,
    rolling_window: int = 60,
    refit_every: int = 15,
    refit_window: int = 120,
    costs: CostGrid = DEFAULT_COSTS,
    alpha: float = 0.10,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate delays on *identical streams* within each independent repeat."""
    _check_integer(repeats, "repeats", 1)
    _check_integer(seed, "seed", 0)
    if (
        not delays or 0 not in delays or len(set(delays)) != len(delays)
        or any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in delays)
    ):
        raise ValueError("Delays must be distinct non-negative ints and include 0")
    if (
        not regimes or len(set(regimes)) != len(regimes)
        or any(r not in REGIMES for r in regimes)
    ):
        raise ValueError("Regimes must be distinct recognised labels")
    allowed_scenarios = ("low_variance_signal", "high_variance_signal")
    if (
        not scenarios or len(set(scenarios)) != len(scenarios)
        or any(s not in allowed_scenarios for s in scenarios)
    ):
        raise ValueError("Scenarios must be distinct recognised labels")
    all_intervals: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    for scenario_idx, scenario in enumerate(scenarios):
        for replicate in range(repeats):
            sim_seed = seed + scenario_idx * 1_000_000 + replicate
            for regime in regimes:
                stream = generate_delayed_stream(
                    seed=sim_seed, regime=regime, scenario=scenario,
                    n_fit=n_fit, n_cal=n_cal, n_live=n_live,
                )
                for delay in delays:
                    intervals, decisions = evaluate_delayed_stream(
                        stream, feedback_delay=delay, alpha=alpha,
                        costs=costs, rolling_window=rolling_window,
                        refit_every=refit_every, refit_window=refit_window,
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


def aggregate_study(
    intervals: pd.DataFrame,
    decisions: pd.DataFrame,
    bootstrap_draws: int = 300,
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate by independent generated stream, never individual time rows.

    Bootstrapped coverage intervals reflect only Monte Carlo uncertainty in
    mean *stream-level* coverage. There is no conformal guarantee under
    autocorrelation, gradual drift, rolling quantiles or refitting.
    """
    _check_integer(bootstrap_draws, "bootstrap_draws", 10)
    _check_integer(seed, "seed", 0)
    interval_keys = [
        "scenario", "regime", "replicate", "delay", "phase", "policy"
    ]
    decision_keys = interval_keys + ["rule", "cost"]
    if not set(interval_keys + ["covered", "width"]).issubset(intervals.columns):
        raise ValueError("Interval data missing required columns")
    if not set(decision_keys + ["regret", "acted"]).issubset(decisions.columns):
        raise ValueError("Decision data missing required columns")
    if intervals.empty or decisions.empty:
        raise ValueError("Cannot summarise an empty experiment")
    by_stream = intervals.groupby(interval_keys, as_index=False).agg(
        coverage=("covered", "mean"),
        width=("width", "mean"),
    )
    by_decision = decisions.groupby(decision_keys, as_index=False).agg(
        regret=("regret", "mean"), action_rate=("acted", "mean"),
    )
    group_keys = [x for x in interval_keys if x != "replicate"]
    rng = np.random.default_rng(seed)
    coverage_rows: list[dict[str, str | int | float]] = []
    for group, df in by_stream.groupby(group_keys, sort=True):
        values = df["coverage"].to_numpy(dtype=np.float64)
        if len(values) >= 3:
            bootstrap_index = rng.integers(
                0, len(values), size=(bootstrap_draws, len(values))
            )
            lower, upper = np.quantile(
                values[bootstrap_index].mean(axis=1), [0.025, 0.975]
            )
        else:
            lower, upper = float("nan"), float("nan")
        coverage_rows.append(
            {
                **dict(zip(group_keys, group)),
                "n_streams": int(len(values)),
                "coverage": float(np.mean(values)),
                "width": float(df["width"].mean()),
                "mc_lower": float(lower),
                "mc_upper": float(upper),
            }
        )
    decision_groups = [x for x in decision_keys if x != "replicate"]
    economic = by_decision.groupby(decision_groups, as_index=False).agg(
        n_streams=("replicate", "nunique"),
        mean_regret=("regret", "mean"),
        mean_action_rate=("action_rate", "mean"),
    )
    return pd.DataFrame(coverage_rows), economic


def paired_delay_penalty(
    decisions: pd.DataFrame,
    comparison_delay: int,
    baseline_delay: int = 0,
    bootstrap_draws: int = 300,
    seed: int = 19,
) -> pd.DataFrame:
    """Compare delayed-minus-immediate regret in matched generated streams.

    Each bootstrap unit is a separately regenerated stream/replicate.
    Variation among steps and cost thresholds within a stream does NOT
    constitute independent evidence.
    """
    _check_integer(comparison_delay, "comparison_delay", 1)
    _check_integer(baseline_delay, "baseline_delay", 0)
    _check_integer(bootstrap_draws, "bootstrap_draws", 10)
    _check_integer(seed, "seed", 0)
    if comparison_delay == baseline_delay:
        raise ValueError("Comparison and baseline delays must differ")
    keys = [
        "scenario", "regime", "replicate", "phase", "policy", "rule", "cost"
    ]
    if not set(keys + ["delay", "regret"]).issubset(decisions.columns):
        raise ValueError("Missing decision columns for paired comparison")
    reduced = decisions.groupby(keys + ["delay"], as_index=False)["regret"].mean()
    base = reduced.loc[reduced["delay"] == baseline_delay, keys + ["regret"]]
    delayed = reduced.loc[reduced["delay"] == comparison_delay, keys + ["regret"]]
    if base.empty or delayed.empty:
        raise ValueError("Comparison requires both requested feedback delays")
    joined = base.merge(
        delayed, on=keys, how="outer", validate="one_to_one",
        suffixes=("_baseline", "_delayed"),
    )
    if joined[["regret_baseline", "regret_delayed"]].isna().any().any():
        raise ValueError("Matched stream observations missing for a delay")
    joined["delta_regret"] = joined["regret_delayed"] - joined["regret_baseline"]
    group_keys = [key for key in keys if key != "replicate"]
    rng = np.random.default_rng(seed)
    output: list[dict[str, str | int | float]] = []
    for group, frame in joined.groupby(group_keys, sort=True):
        shifts = frame["delta_regret"].to_numpy(dtype=np.float64)
        if len(shifts) >= 3:
            selection = rng.integers(
                0, len(shifts), size=(bootstrap_draws, len(shifts))
            )
            ci = np.quantile(shifts[selection].mean(axis=1), [0.025, 0.975])
        else:
            ci = [float("nan"), float("nan")]
        output.append(
            {
                **dict(zip(group_keys, group)),
                "baseline_delay": baseline_delay,
                "comparison_delay": comparison_delay,
                "n_streams": len(shifts),
                "mean_delay_minus_baseline_regret": float(np.mean(shifts)),
                "mc_lower": float(ci[0]),
                "mc_upper": float(ci[1]),
            }
        )
    return pd.DataFrame(output)
