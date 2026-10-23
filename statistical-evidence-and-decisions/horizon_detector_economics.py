"""Horizon-calibrated CUSUM and the economic cost of model maintenance.

The threshold is calibrated from *independent simulated null streams*. The
corrected rank of the maximum no-reset CUSUM statistic controls the probability
of one or more alerts over this fixed monitoring horizon, marginally across
exchangeable null streams of the same data-generating design. It is NOT an
anytime-valid sequential test or a guarantee under a different data regime.

Policies predict and act using only labels that have matured by the current
step. The synthetic conditional mean is used solely to evaluate economics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from analysis import FloatArray, Scenario
from change_point_detection import residual_disagreement
from gradual_drift import (
    DelayedStream,
    Regime,
    available_label_count,
    generate_delayed_stream,
)
from uncertainty_analysis import fixed_candidates

NullRegime = Literal["stable", "ar1"]
Policy = Literal["Frozen", "Scheduled", "CUSUM IID", "CUSUM matched"]
POLICIES: tuple[Policy, ...] = (
    "Frozen", "Scheduled", "CUSUM IID", "CUSUM matched"
)
DEFAULT_COSTS = (0.2, 0.6, 1.0, 1.4)
DEFAULT_REFIT_COSTS = (0.0, 0.05, 0.25)


@dataclass(frozen=True)
class MaintenanceDesign:
    """Predeclared simulation and policy parameters."""

    n_fit: int = 160
    n_cal: int = 80
    n_live: int = 120
    calibration_streams: int = 39
    false_alarm_target: float = 0.10
    delay_values: tuple[int, ...] = (0, 12)
    refit_every: int = 15
    refit_window: int = 100
    cooldown: int = 15
    allowance: float = 0.2
    action_costs: tuple[float, ...] = DEFAULT_COSTS
    refit_costs: tuple[float, ...] = DEFAULT_REFIT_COSTS

    def __post_init__(self) -> None:
        """Reject invalid experiments before calibrating any threshold."""
        for name, minimum in (
            ("n_fit", 64), ("n_cal", 30), ("n_live", 45),
            ("calibration_streams", 2), ("refit_every", 1),
            ("refit_window", 64), ("cooldown", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if not np.isfinite(self.false_alarm_target) or not (
            0 < self.false_alarm_target < 1
        ):
            raise ValueError("false_alarm_target must lie in (0, 1)")
        if not np.isfinite(self.allowance) or self.allowance < 0:
            raise ValueError("allowance must be finite and nonnegative")
        if (
            not self.delay_values
            or len(set(self.delay_values)) != len(self.delay_values)
            or any(
                isinstance(delay, bool)
                or not isinstance(delay, int)
                or delay < 0
                for delay in self.delay_values
            )
        ):
            raise ValueError("delay_values must contain distinct nonnegative integers")
        for name in ("action_costs", "refit_costs"):
            values = getattr(self, name)
            if not values or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not np.isfinite(v) or v < 0
                for v in values
            ):
                raise ValueError(f"{name} must contain finite nonnegative numbers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class HorizonThreshold:
    """One threshold calibrated on a disjoint null-stream population."""

    null_regime: NullRegime
    scenario: Scenario
    delay: int
    threshold: float
    target: float
    calibration_streams: int
    quantile_rank: int
    null_maxima: tuple[float, ...]


def _fit(x: FloatArray, y: FloatArray) -> Pipeline:
    """Fit a prespecified model using historical labels only."""
    model = fixed_candidates()["OLS"]
    model.fit(x, y)
    return model


def reference_increments(
    stream: DelayedStream,
    delay: int,
    allowance: float = 0.2,
) -> FloatArray:
    """Construct standardised CUSUM increments at legal label-release times.

    The training/calibration partitions set the reference predictor, score
    location and score scale. For live label j, the score is processed only
    at time j + delay + 1, never on or before the observation's own step.
    Missing releases are encoded as NaN; a NaN causes no CUSUM update.
    """
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError("delay must be a nonnegative integer")
    if not np.isfinite(allowance) or allowance < 0:
        raise ValueError("allowance must be finite and nonnegative")

    reference = _fit(stream.x_fit, stream.y_fit)
    fitted = np.asarray(reference.predict(stream.x_fit), dtype=np.float64).reshape(-1)
    noise_sd = max(float(np.std(stream.y_fit - fitted, ddof=1)), 1e-8)
    predicted_cal = np.asarray(
        reference.predict(stream.x_cal), dtype=np.float64
    ).reshape(-1)
    calibration_scores = residual_disagreement(
        stream.y_cal, predicted_cal, noise_sd
    )
    center = float(calibration_scores.mean())
    spread = max(float(np.std(calibration_scores, ddof=1)), 1e-8)
    predicted_live = np.asarray(
        reference.predict(stream.x_live), dtype=np.float64
    ).reshape(-1)
    output = np.full(len(stream.y_live), np.nan, dtype=np.float64)
    for time in range(len(stream.y_live)):
        matured_index = time - delay - 1
        if matured_index < 0:
            continue
        observed_score = float(
            residual_disagreement(
                np.asarray([stream.y_live[matured_index]], dtype=np.float64),
                np.asarray([predicted_live[matured_index]], dtype=np.float64),
                noise_sd,
            )[0]
        )
        output[time] = (observed_score - center) / spread - allowance
    return output


def horizon_maximum(increments: FloatArray) -> float:
    """Find the maximum un-reset, nonnegative CUSUM on one fixed horizon."""
    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or np.isinf(values).any():
        raise ValueError("Increments must be a nonempty vector without infinity")
    running = maximum = 0.0
    for value in values:
        if np.isnan(value):
            continue
        running = max(0.0, running + float(value))
        maximum = max(maximum, running)
    return maximum


def corrected_horizon_quantile(
    null_maxima: FloatArray,
    false_alarm_target: float,
) -> tuple[float, int]:
    """Conservative order-statistic threshold for P(max > threshold) <= alpha.

    With N exchangeable null maxima and one new null maximum, choose the
    k-th smallest calibration maximum with k=ceil((N+1)(1-alpha)).
    If k>N, the threshold is +infinity. Ties can make the result conservative.
    This is a fixed-horizon, marginal *simulation-stream* statement only.
    """
    scores = np.asarray(null_maxima, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 2 or not np.isfinite(scores).all():
        raise ValueError("Need at least two finite calibration maxima")
    if (scores < 0).any():
        raise ValueError("CUSUM maxima must be nonnegative")
    if not np.isfinite(false_alarm_target) or not 0 < false_alarm_target < 1:
        raise ValueError("false_alarm_target must be in (0, 1)")
    rank = int(np.ceil((scores.size + 1) * (1.0 - false_alarm_target)))
    if rank > scores.size:
        return float("inf"), rank
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


def calibrate_null_horizon(
    design: MaintenanceDesign,
    scenario: Scenario,
    delay: int,
    null_regime: NullRegime,
    seed: int,
) -> HorizonThreshold:
    """Generate null streams not reused in economic policy evaluation."""
    if null_regime not in ("stable", "ar1"):
        raise ValueError("null_regime must be stable or ar1")
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError("Unknown response scenario")
    if delay not in design.delay_values:
        raise ValueError("Requested delay is outside the study design")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    maxima: list[float] = []
    for index in range(design.calibration_streams):
        stream = generate_delayed_stream(
            seed=seed + index, regime=null_regime, scenario=scenario,
            n_fit=design.n_fit, n_cal=design.n_cal, n_live=design.n_live,
        )
        increments = reference_increments(stream, delay, design.allowance)
        maxima.append(horizon_maximum(increments))
    threshold, rank = corrected_horizon_quantile(
        np.asarray(maxima, dtype=np.float64), design.false_alarm_target
    )
    return HorizonThreshold(
        null_regime=null_regime,
        scenario=scenario,
        delay=delay,
        threshold=threshold,
        target=design.false_alarm_target,
        calibration_streams=len(maxima),
        quantile_rank=rank,
        null_maxima=tuple(maxima),
    )


def _null_regime(regime: Regime) -> NullRegime:
    """Map gradual regimes to a no-drift noise-dependence reference."""
    return "ar1" if regime in ("ar1", "gradual_ar1") else "stable"


def evaluate_maintenance(
    stream: DelayedStream,
    design: MaintenanceDesign,
    delay: int,
    iid_threshold: float,
    matched_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate four policies on one shared prequential deployment stream.

    A detector consumes only an eligible label from the *frozen reference*
    and can retrain immediately before the current prediction. A subsequent
    alarm is blocked during cooldown. Any alarm is a refit event.

    Economic regret includes a charge *only at retraining steps*. The oracle
    is allowed to know the true conditional mean but is not charged for
    model maintenance; this is an explicit full-information benchmark.
    """
    if delay not in design.delay_values:
        raise ValueError("Invalid delay")
    if any(np.isnan(t) or t < 0 for t in (iid_threshold, matched_threshold)):
        raise ValueError("Thresholds must be nonnegative (infinity permitted)")
    increments = reference_increments(stream, delay, design.allowance)
    models: dict[str, Pipeline] = {
        policy: _fit(stream.x_fit, stream.y_fit) for policy in POLICIES
    }
    stats = {"CUSUM IID": 0.0, "CUSUM matched": 0.0}
    blocked_until = {"CUSUM IID": 0, "CUSUM matched": 0}
    threshold_by_policy = {
        "CUSUM IID": iid_threshold, "CUSUM matched": matched_threshold
    }
    cumulative_refits: dict[str, int] = {p: 0 for p in POLICIES}
    model_rows: list[dict[str, str | int | float]] = []
    economic_rows: list[dict[str, str | int | float]] = []
    for time in range(len(stream.y_live)):
        matured = available_label_count(time, delay)
        events = {p: False for p in POLICIES}
        if time > 0 and matured > 0 and time % design.refit_every == 0:
            events["Scheduled"] = True
        for label in ("CUSUM IID", "CUSUM matched"):
            value = increments[time]
            if time < blocked_until[label]:
                stats[label] = 0.0
            elif np.isfinite(value):
                stats[label] = max(0.0, stats[label] + float(value))
                if stats[label] > threshold_by_policy[label] and matured > 0:
                    events[label] = True
                    stats[label] = 0.0
                    blocked_until[label] = time + design.cooldown + 1

        if any(events.values()):
            past_x = np.concatenate(
                (stream.x_fit, stream.x_cal, stream.x_live[:matured]), axis=0
            )
            past_y = np.concatenate(
                (stream.y_fit, stream.y_cal, stream.y_live[:matured]), axis=0
            )
            for label, event in events.items():
                if event:
                    models[label] = _fit(
                        past_x[-design.refit_window:],
                        past_y[-design.refit_window:],
                    )
                    cumulative_refits[label] += 1

        for label, model in models.items():
            prediction = float(
                np.asarray(
                    model.predict(stream.x_live[time : time + 1]),
                    dtype=np.float64,
                ).reshape(-1)[0]
            )
            if not np.isfinite(prediction):
                raise ValueError("Model returned a non-finite forecast")
            observed = float(stream.y_live[time])
            true_mean = float(stream.mean_live[time])
            model_rows.append(
                {
                    "step": time,
                    "policy": label,
                    "delay": delay,
                    "available_labels": matured,
                    "prediction": prediction,
                    "squared_error": (observed - prediction) ** 2,
                    "alarm": int(events[label] and label.startswith("CUSUM")),
                    "refit_event": int(events[label]),
                    "cumulative_refits": cumulative_refits[label],
                }
            )
            for action_cost in design.action_costs:
                margin = true_mean - action_cost
                action = prediction > action_cost
                premaintenance_profit = margin * float(action)
                oracle_profit = max(margin, 0.0)
                for refit_cost in design.refit_costs:
                    maintenance_charge = refit_cost * int(events[label])
                    profit = premaintenance_profit - maintenance_charge
                    economic_rows.append(
                        {
                            "step": time,
                            "policy": label,
                            "delay": delay,
                            "action_cost": float(action_cost),
                            "refit_cost": float(refit_cost),
                            "expected_net_profit": float(profit),
                            "oracle_profit": float(oracle_profit),
                            "net_regret": float(oracle_profit - profit),
                        }
                    )
    return pd.DataFrame.from_records(model_rows), pd.DataFrame.from_records(
        economic_rows
    )


def run_maintenance_study(
    design: MaintenanceDesign,
    evaluation_repeats: int = 8,
    scenarios: tuple[Scenario, ...] = (
        "low_variance_signal", "high_variance_signal"
    ),
    regimes: tuple[Regime, ...] = (
        "stable", "gradual_drift", "ar1", "gradual_ar1"
    ),
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate on disjoint null streams; evaluate on new paired streams.

    Null calibration uses seeds in a separate, nonoverlapping range. Within
    evaluation, all drift/no-drift regimes for a given replicate use the same
    underlying random seed. The IID threshold is also evaluated on AR(1)
    streams as an intentionally misspecified calibration comparison.
    """
    for label, value in (("evaluation_repeats", evaluation_repeats), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < (1 if label == "evaluation_repeats" else 0):
            raise ValueError(f"{label} is too small")
    if (
        not scenarios or len(set(scenarios)) != len(scenarios)
        or any(
            s not in ("low_variance_signal", "high_variance_signal")
            for s in scenarios
        )
    ):
        raise ValueError("Invalid response scenario list")
    if (
        not regimes or len(set(regimes)) != len(regimes)
        or any(r not in ("stable", "gradual_drift", "ar1", "gradual_ar1")
               for r in regimes)
    ):
        raise ValueError("Invalid process regime list")

    needed_nulls = {"stable"} | {_null_regime(regime) for regime in regimes}
    thresholds: dict[tuple[str, int, str], HorizonThreshold] = {}
    calibration_records: list[dict[str, str | int | float]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for delay in design.delay_values:
            for regime_idx, null_regime in enumerate(sorted(needed_nulls)):
                # A disjoint seed range prevents accidental use of the
                # evaluation streams in the threshold calibration set.
                cal_seed = (
                    seed + 10_000_000 + scenario_index * 1_000_000
                    + delay * 10000 + regime_idx * 100_000
                )
                result = calibrate_null_horizon(
                    design, scenario, delay, null_regime, seed=cal_seed
                )
                thresholds[(scenario, delay, null_regime)] = result
                calibration_records.append(
                    {
                        "scenario": scenario,
                        "delay": delay,
                        "null_regime": null_regime,
                        "target_false_alarm_rate": result.target,
                        "threshold": result.threshold,
                        "quantile_rank": result.quantile_rank,
                        "calibration_streams": result.calibration_streams,
                        "mean_null_maximum": float(np.mean(result.null_maxima)),
                    }
                )

    all_models: list[pd.DataFrame] = []
    all_economics: list[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenarios):
        for replicate in range(evaluation_repeats):
            evaluation_seed = seed + scenario_index * 1_000_000 + replicate
            for regime in regimes:
                stream = generate_delayed_stream(
                    evaluation_seed, regime=regime, scenario=scenario,
                    n_fit=design.n_fit, n_cal=design.n_cal,
                    n_live=design.n_live,
                )
                for delay in design.delay_values:
                    iid = thresholds[(scenario, delay, "stable")].threshold
                    matched = thresholds[
                        (scenario, delay, _null_regime(regime))
                    ].threshold
                    models, economics = evaluate_maintenance(
                        stream, design, delay,
                        iid_threshold=iid, matched_threshold=matched
                    )
                    for frame in (models, economics):
                        frame.insert(0, "replicate", replicate)
                        frame.insert(0, "regime", regime)
                        frame.insert(0, "scenario", scenario)
                    all_models.append(models)
                    all_economics.append(economics)
    return (
        pd.DataFrame.from_records(calibration_records),
        pd.concat(all_models, ignore_index=True),
        pd.concat(all_economics, ignore_index=True),
    )


def summarize_maintenance(
    model_records: pd.DataFrame,
    economics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise independent streams before pooling performance statistics.

    Alarm probability is the fraction of independent generated evaluation
    streams with at least one detector alarm; on null regimes, this is the
    fixed-horizon empirical false-alarm rate. Gradual changes have no
    uniquely defined onset time, so the detection delay is descriptive.
    """
    key = ["scenario", "regime", "delay", "replicate", "policy"]
    if not set(key + ["squared_error", "alarm", "refit_event", "step"]).issubset(
        model_records.columns
    ):
        raise ValueError("Missing model records")
    if not set(key + ["action_cost", "refit_cost", "net_regret"]).issubset(
        economics.columns
    ):
        raise ValueError("Missing economic records")
    if model_records.empty or economics.empty:
        raise ValueError("No observations to summarise")

    per_stream = model_records.groupby(key, as_index=False).agg(
        rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
        alarms=("alarm", "sum"),
        refits=("refit_event", "sum"),
    )
    per_stream["any_alarm"] = (per_stream["alarms"] > 0).astype(float)
    group = ["scenario", "regime", "delay", "policy"]
    operational = per_stream.groupby(group, as_index=False).agg(
        independent_streams=("replicate", "nunique"),
        mean_rmse=("rmse", "mean"),
        probability_any_alarm=("any_alarm", "mean"),
        mean_alarms=("alarms", "mean"),
        mean_refits=("refits", "mean"),
    )
    economic_key = key + ["action_cost", "refit_cost"]
    per_stream_loss = economics.groupby(
        economic_key, as_index=False
    ).agg(mean_net_regret=("net_regret", "mean"))
    loss_group = group + ["action_cost", "refit_cost"]
    decision_summary = per_stream_loss.groupby(
        loss_group, as_index=False
    ).agg(
        independent_streams=("replicate", "nunique"),
        mean_net_regret=("mean_net_regret", "mean"),
    )
    return operational, decision_summary
