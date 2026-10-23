"""Compare CUSUM, Gaussian GLR and Bayesian online change-point detection.

Detection is performed on a *predeclared*, univariate score measuring
disagreement between observed outcomes and a frozen regression reference.
All three methods consume the same delayed sequence of standardised scores.

- CUSUM: one-sided positive mean-shift accumulation.
- GLR: maximum Gaussian log-likelihood ratio for an unknown positive mean
  shift over a fixed collection of recent run lengths.
- BOCPD: posterior mass on short run lengths under a constant-hazard,
  normal-mean conjugate model with known unit observation variance.

These are different models of the same univariate score, not a comparison
of all possible multivariate or regression-specific change detectors.

Thresholds are calibrated on *independent* no-change simulated streams using
the corrected order statistic of the maximum detector score over a fixed
horizon. Under exchangeability of entire null-stream maxima, this controls
the marginal probability of at least one alarm on a NEW null stream. It is
not an anytime-valid guarantee, nor a power or decision-value guarantee.

Each label from live step j is unavailable until step j + delay + 1.
The true conditional benefit is available only to the evaluation of regret.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logsumexp
from sklearn.pipeline import Pipeline

from analysis import FloatArray, Scenario
from gradual_drift import (
    DelayedStream,
    Regime,
    available_label_count,
    generate_delayed_stream,
)
from horizon_detector_economics import (
    corrected_horizon_quantile,
    reference_increments,
)
from uncertainty_analysis import fixed_candidates

DetectorName = Literal["CUSUM", "GLR", "BOCPD"]
PolicyName = Literal["Frozen", "Scheduled", "CUSUM", "GLR", "BOCPD"]
NullRegime = Literal["stable", "ar1"]
DETECTORS: tuple[DetectorName, ...] = ("CUSUM", "GLR", "BOCPD")
POLICIES: tuple[PolicyName, ...] = (
    "Frozen", "Scheduled", "CUSUM", "GLR", "BOCPD"
)


@dataclass(frozen=True)
class ComparisonDesign:
    """Predeclared parameters for calibration, alarms and model maintenance."""

    n_fit: int = 160
    n_cal: int = 80
    n_live: int = 120
    null_streams: int = 49
    alpha: float = 0.10
    delays: tuple[int, ...] = (0, 12)
    scheduled_every: int = 15
    refit_window: int = 100
    cooldown: int = 15
    cusum_allowance: float = 0.2
    glr_min_window: int = 4
    glr_max_window: int = 32
    bocpd_hazard: float = 0.03
    bocpd_prior_precision: float = 0.25
    bocpd_short_window: int = 5
    bocpd_warmup: int = 15
    action_costs: tuple[float, ...] = (0.2, 0.6, 1.0, 1.4)
    refit_costs: tuple[float, ...] = (0.0, 0.05, 0.25)

    def __post_init__(self) -> None:
        """Fail early on invalid or misleading experimental configurations."""
        for name, minimum in (
            ("n_fit", 64), ("n_cal", 30), ("n_live", 45),
            ("null_streams", 2), ("scheduled_every", 1),
            ("refit_window", 64), ("cooldown", 0),
            ("glr_min_window", 2), ("glr_max_window", 2),
            ("bocpd_short_window", 1), ("bocpd_warmup", 2),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if (
            self.glr_max_window < self.glr_min_window
            or self.bocpd_warmup <= self.bocpd_short_window
        ):
            raise ValueError("Invalid detector window configuration")
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if not np.isfinite(self.cusum_allowance) or self.cusum_allowance < 0:
            raise ValueError("CUSUM allowance must be finite and nonnegative")
        if not np.isfinite(self.bocpd_hazard) or not 0 < self.bocpd_hazard < 1:
            raise ValueError("Bayesian hazard must lie in (0, 1)")
        if (
            not np.isfinite(self.bocpd_prior_precision)
            or self.bocpd_prior_precision <= 0
        ):
            raise ValueError("Bayesian prior precision must be positive")
        for name in ("delays", "action_costs", "refit_costs"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and distinct")
        if any(isinstance(d, bool) or not isinstance(d, int) or d < 0
               for d in self.delays):
            raise ValueError("Delays must be nonnegative integers")
        for name in ("action_costs", "refit_costs"):
            if any(
                isinstance(v, bool)
                or not isinstance(v, (float, int))
                or not np.isfinite(v)
                or v < 0
                for v in getattr(self, name)
            ):
                raise ValueError(f"{name} must contain finite nonnegative costs")


class CUSUMTracker:
    """One-sided CUSUM for a standardised positive mean shift."""

    def __init__(self, allowance: float) -> None:
        if not np.isfinite(allowance) or allowance < 0:
            raise ValueError("allowance must be finite and nonnegative")
        self.allowance = float(allowance)
        self.value = 0.0

    def update(self, observation: float) -> float:
        """Process a single matured standardised score."""
        if not np.isfinite(observation):
            raise ValueError("CUSUM observations must be finite")
        self.value = max(0.0, self.value + observation - self.allowance)
        return self.value

    def reset(self) -> None:
        """Start another detection episode after an alarm."""
        self.value = 0.0


class GLRTracker:
    """One-sided Gaussian GLR for an unknown positive shift in unit-variance Z."""

    def __init__(self, minimum: int = 4, maximum: int = 32) -> None:
        if (
            isinstance(minimum, bool) or isinstance(maximum, bool)
            or not isinstance(minimum, int) or not isinstance(maximum, int)
            or minimum < 2 or maximum < minimum
        ):
            raise ValueError("GLR scan lengths must satisfy 2 <= minimum <= maximum")
        self.minimum = minimum
        self.maximum = maximum
        self.history: deque[float] = deque(maxlen=maximum)

    def update(self, observation: float) -> float:
        """Maximise the log-likelihood ratio over the scanned recent suffixes.

        Under Z ~ N(mu, 1), testing mu=0 against an unknown mu>0 gives
        Lambda = (max(0, sum Z)^2) / (2 * segment_length).
        """
        if not np.isfinite(observation):
            raise ValueError("GLR observations must be finite")
        self.history.append(float(observation))
        if len(self.history) < self.minimum:
            return 0.0
        suffix_sums = np.cumsum(
            np.asarray(self.history, dtype=np.float64)[::-1]
        )
        lengths = np.arange(self.minimum, len(self.history) + 1)
        candidate = np.maximum(suffix_sums[self.minimum - 1 :], 0.0)
        statistic = candidate * candidate / (2.0 * lengths)
        return float(np.max(statistic))

    def reset(self) -> None:
        """Discard the old scanning window after a triggered alarm."""
        self.history.clear()


class BOCPDTracker:
    """Conjugate normal-mean BOCPD with constant hazard and known noise variance.

    For each possible run length, we retain the posterior run-length mass and
    the normal posterior mean/precision for the current segment mean.
    A change resets the segment mean to its N(0, 1/prior_precision) prior.

    The returned statistic is P(run_length < short_window | observed scores).
    A short initial warmup suppresses the trivial probability one at t=1.

    These are Bayesian probabilities *conditional on this score likelihood,
    prior and hazard*, not probabilities that the physical system changed.
    """

    def __init__(
        self,
        hazard: float = 0.03,
        prior_precision: float = 0.25,
        short_window: int = 5,
        warmup: int = 15,
    ) -> None:
        if not np.isfinite(hazard) or not 0 < hazard < 1:
            raise ValueError("hazard must be in (0, 1)")
        if not np.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("prior_precision must be finite and positive")
        if (
            isinstance(short_window, bool) or isinstance(warmup, bool)
            or not isinstance(short_window, int) or not isinstance(warmup, int)
            or short_window < 1 or warmup <= short_window
        ):
            raise ValueError("Require 1 <= short_window < warmup")
        self.hazard = float(hazard)
        self.prior_precision = float(prior_precision)
        self.short_window = short_window
        self.warmup = warmup
        self.reset()

    def reset(self) -> None:
        """Restart the run-length recursion after a retraining event."""
        self.posterior: FloatArray = np.asarray([1.0], dtype=np.float64)
        self.means: FloatArray = np.asarray([0.0], dtype=np.float64)
        self.precisions: FloatArray = np.asarray(
            [self.prior_precision], dtype=np.float64
        )
        self.observation_count = 0

    @staticmethod
    def _normal_logpdf(
        observation: float, means: FloatArray, variances: FloatArray
    ) -> FloatArray:
        """Normal log density for unit-noise observations with mean uncertainty."""
        return -0.5 * (
            np.log(2.0 * np.pi * variances)
            + (observation - means) ** 2 / variances
        )

    def update(self, observation: float) -> float:
        """Apply BOCPD growth/change update and return short-run posterior mass."""
        if not np.isfinite(observation):
            raise ValueError("BOCPD observations must be finite")
        z = float(observation)
        first_precision = self.prior_precision + 1.0
        first_mean = z / first_precision
        if self.observation_count == 0:
            # At t=1 the segment necessarily consists of the first observation.
            self.posterior = np.asarray([1.0], dtype=np.float64)
            self.means = np.asarray([first_mean], dtype=np.float64)
            self.precisions = np.asarray([first_precision], dtype=np.float64)
        else:
            predictive_variances = 1.0 + 1.0 / self.precisions
            log_growth = (
                np.log(np.maximum(self.posterior, np.finfo(float).tiny))
                + np.log1p(-self.hazard)
                + self._normal_logpdf(z, self.means, predictive_variances)
            )
            prior_log_likelihood = self._normal_logpdf(
                z,
                np.asarray([0.0]),
                np.asarray([1.0 + 1.0 / self.prior_precision]),
            )[0]
            log_change = np.log(self.hazard) + prior_log_likelihood
            all_log_weights = np.concatenate(
                (np.asarray([log_change]), log_growth)
            )
            self.posterior = np.exp(
                all_log_weights - logsumexp(all_log_weights)
            )
            updated_precision = self.precisions + 1.0
            updated_mean = (
                self.precisions * self.means + z
            ) / updated_precision
            self.means = np.concatenate(
                (np.asarray([first_mean]), updated_mean)
            )
            self.precisions = np.concatenate(
                (np.asarray([first_precision]), updated_precision)
            )
        self.observation_count += 1
        if self.observation_count < self.warmup:
            return 0.0
        return float(np.sum(self.posterior[: self.short_window]))


def make_trackers(design: ComparisonDesign) -> dict[DetectorName, object]:
    """Create new, independent detector state for one score sequence."""
    return {
        "CUSUM": CUSUMTracker(design.cusum_allowance),
        "GLR": GLRTracker(design.glr_min_window, design.glr_max_window),
        "BOCPD": BOCPDTracker(
            design.bocpd_hazard,
            design.bocpd_prior_precision,
            design.bocpd_short_window,
            design.bocpd_warmup,
        ),
    }


def score_maxima(
    released_scores: FloatArray,
    design: ComparisonDesign,
) -> dict[DetectorName, float]:
    """Maximum unreset detector score across an entire fixed deployment horizon.

    NaN means no label has yet matured. No detector observes those steps.
    There is no cooldown in null calibration: first-alarm probability is
    defined by whether the unreset maximum exceeds a threshold.
    """
    scores = np.asarray(released_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1 or np.isinf(scores).any():
        raise ValueError("Scores must be a nonempty vector without infinities")
    maximum: dict[DetectorName, float] = {
        name: 0.0 for name in DETECTORS
    }
    trackers = make_trackers(design)
    for value in scores:
        if np.isnan(value):
            continue
        for name in DETECTORS:
            score = float(trackers[name].update(float(value)))
            maximum[name] = max(maximum[name], score)
    return maximum


def null_family(regime: Regime) -> NullRegime:
    """Match autocorrelation while removing gradual coefficient changes."""
    return "ar1" if regime in ("ar1", "gradual_ar1") else "stable"


def calibrate_detectors(
    design: ComparisonDesign,
    scenario: Scenario,
    delay: int,
    regime: NullRegime,
    seed: int,
) -> pd.DataFrame:
    """Calibrate detector thresholds from disjoint simulated null streams.

    Each row represents one detector calibrated independently using the same
    set of generated null streams. Do NOT tune parameters or choose among
    detectors using the independent evaluation streams.
    """
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError("Unknown response scenario")
    if regime not in ("stable", "ar1"):
        raise ValueError("Null regime must be stable or ar1")
    if delay not in design.delays:
        raise ValueError("Delay must belong to declared design")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    collected: dict[DetectorName, list[float]] = {
        name: [] for name in DETECTORS
    }
    for replicate in range(design.null_streams):
        stream = generate_delayed_stream(
            seed=seed + replicate,
            regime=regime,
            scenario=scenario,
            n_fit=design.n_fit,
            n_cal=design.n_cal,
            n_live=design.n_live,
        )
        # allowance zero preserves the standardised Z required by GLR/BOCPD.
        scores = reference_increments(stream, delay, allowance=0.0)
        maxima = score_maxima(scores, design)
        for detector in DETECTORS:
            collected[detector].append(maxima[detector])
    records: list[dict[str, str | int | float]] = []
    for detector in DETECTORS:
        values = np.asarray(collected[detector], dtype=np.float64)
        threshold, rank = corrected_horizon_quantile(
            values, design.alpha
        )
        records.append(
            {
                "scenario": scenario,
                "delay": delay,
                "null_family": regime,
                "detector": detector,
                "threshold": float(threshold),
                "rank": int(rank),
                "calibration_streams": design.null_streams,
                "target_false_alarm_rate": design.alpha,
                "mean_null_maximum": float(np.mean(values)),
            }
        )
    return pd.DataFrame.from_records(records)


def _fit_model(x: FloatArray, y: FloatArray) -> Pipeline:
    """Use the same prespecified linear regression across all policies."""
    model = fixed_candidates()["OLS"]
    model.fit(x, y)
    return model


def evaluate_policies(
    stream: DelayedStream,
    design: ComparisonDesign,
    delay: int,
    thresholds: dict[DetectorName, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare frozen, scheduled and detector-triggered model maintenance.

    For each step, only matured *past* labels may update detector states
    or fitting samples. Models are evaluated on the *same* deployment row.
    The initial score/reference transformation is trained on historical data.

    After an alarm, a detector is reset and ignores scores during its
    cooldown. First-alarm calibration still matches the unreset maximum
    used by the threshold calibration. Later alarms have no such guarantee.
    """
    if delay not in design.delays:
        raise ValueError("Delay is not configured")
    if set(thresholds) != set(DETECTORS):
        raise ValueError("Thresholds must contain all detector names")
    if any(np.isnan(v) or v < 0 for v in thresholds.values()):
        raise ValueError("Thresholds must be nonnegative (infinity permitted)")

    released = reference_increments(stream, delay, allowance=0.0)
    trackers = make_trackers(design)
    frozen = _fit_model(stream.x_fit, stream.y_fit)
    models: dict[str, Pipeline] = {
        name: _fit_model(stream.x_fit, stream.y_fit)
        for name in POLICIES
    }
    models["Frozen"] = frozen
    blocked_until = {name: 0 for name in DETECTORS}
    refits = {name: 0 for name in POLICIES}
    model_rows: list[dict[str, str | int | float]] = []
    economic_rows: list[dict[str, str | int | float]] = []

    for t in range(len(stream.y_live)):
        matured_count = available_label_count(t, delay)
        event = {name: False for name in POLICIES}
        detector_score = {name: 0.0 for name in DETECTORS}
        if t > 0 and matured_count > 0 and t % design.scheduled_every == 0:
            event["Scheduled"] = True

        for name in DETECTORS:
            # No future outcome is used: released[t] is constructed from a
            # label that was already observed at t - delay - 1.
            if np.isfinite(released[t]) and t >= blocked_until[name]:
                statistic = float(trackers[name].update(float(released[t])))
                detector_score[name] = statistic
                if statistic > thresholds[name] and matured_count > 0:
                    event[name] = True
                    trackers[name].reset()
                    blocked_until[name] = t + design.cooldown + 1

        if any(event.values()):
            historical_x = np.concatenate(
                (stream.x_fit, stream.x_cal, stream.x_live[:matured_count])
            )
            historical_y = np.concatenate(
                (stream.y_fit, stream.y_cal, stream.y_live[:matured_count])
            )
            for name in POLICIES:
                if event[name]:
                    models[name] = _fit_model(
                        historical_x[-design.refit_window:],
                        historical_y[-design.refit_window:],
                    )
                    refits[name] += 1

        for name in POLICIES:
            predicted = float(
                np.asarray(
                    models[name].predict(stream.x_live[t : t + 1]),
                    dtype=np.float64,
                ).reshape(-1)[0]
            )
            if not np.isfinite(predicted):
                raise ValueError("Non-finite model prediction")
            observed = float(stream.y_live[t])
            true_mean = float(stream.mean_live[t])
            model_rows.append(
                {
                    "step": t,
                    "policy": name,
                    "delay": delay,
                    "available_labels": matured_count,
                    "prediction": predicted,
                    "squared_error": (observed - predicted) ** 2,
                    "alarm": float(event[name] and name in DETECTORS),
                    "refit_event": float(event[name]),
                    "cumulative_refits": refits[name],
                    "detector_statistic": detector_score.get(name, 0.0),
                    "change_onset": stream.drift_start,
                }
            )
            for action_cost in design.action_costs:
                margin = true_mean - action_cost
                action = predicted > action_cost
                oracle = max(margin, 0.0)
                for maintenance_cost in design.refit_costs:
                    charged = maintenance_cost * float(event[name])
                    profit = margin * float(action) - charged
                    economic_rows.append(
                        {
                            "step": t,
                            "policy": name,
                            "delay": delay,
                            "action_cost": float(action_cost),
                            "refit_cost": float(maintenance_cost),
                            "expected_net_profit": float(profit),
                            "oracle_profit": float(oracle),
                            "net_regret": float(oracle - profit),
                        }
                    )
    return pd.DataFrame.from_records(model_rows), pd.DataFrame.from_records(
        economic_rows
    )


def run_comparison(
    design: ComparisonDesign,
    evaluation_repeats: int = 8,
    scenarios: tuple[Scenario, ...] = (
        "low_variance_signal", "high_variance_signal"
    ),
    regimes: tuple[Regime, ...] = (
        "stable", "gradual_drift", "ar1", "gradual_ar1"
    ),
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use disjoint null calibration and paired independent evaluation streams."""
    if (
        isinstance(evaluation_repeats, bool)
        or not isinstance(evaluation_repeats, int)
        or evaluation_repeats < 1
    ):
        raise ValueError("evaluation_repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    allowed_signals = ("low_variance_signal", "high_variance_signal")
    allowed_regimes = ("stable", "gradual_drift", "ar1", "gradual_ar1")
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(
        s not in allowed_signals for s in scenarios
    ):
        raise ValueError("Invalid response scenarios")
    if not regimes or len(set(regimes)) != len(regimes) or any(
        r not in allowed_regimes for r in regimes
    ):
        raise ValueError("Invalid regimes")

    needed_nulls: set[NullRegime] = {
        null_family(regime) for regime in regimes
    }
    thresholds: dict[tuple[str, int, str, str], float] = {}
    calibration_results: list[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenarios):
        for delay in design.delays:
            for null_index, family in enumerate(sorted(needed_nulls)):
                calibration_seed = (
                    seed + 50_000_000 + scenario_index * 5_000_000
                    + delay * 100_000 + null_index * 10_000
                )
                table = calibrate_detectors(
                    design, scenario, delay, family, calibration_seed
                )
                calibration_results.append(table)
                for row in table.to_dict("records"):
                    thresholds[(
                        scenario, delay, family, str(row["detector"])
                    )] = float(row["threshold"])

    model_tables: list[pd.DataFrame] = []
    decision_tables: list[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenarios):
        for replicate in range(evaluation_repeats):
            evaluation_seed = seed + scenario_index * 1_000_000 + replicate
            for regime in regimes:
                stream = generate_delayed_stream(
                    seed=evaluation_seed,
                    scenario=scenario,
                    regime=regime,
                    n_fit=design.n_fit,
                    n_cal=design.n_cal,
                    n_live=design.n_live,
                )
                for delay in design.delays:
                    family = null_family(regime)
                    chosen_thresholds: dict[DetectorName, float] = {
                        name: thresholds[(scenario, delay, family, name)]
                        for name in DETECTORS
                    }
                    models, economics = evaluate_policies(
                        stream, design, delay, chosen_thresholds
                    )
                    for frame in (models, economics):
                        frame.insert(0, "scenario", scenario)
                        frame.insert(1, "regime", regime)
                        frame.insert(2, "replicate", replicate)
                    model_tables.append(models)
                    decision_tables.append(economics)
    return (
        pd.concat(calibration_results, ignore_index=True),
        pd.concat(model_tables, ignore_index=True),
        pd.concat(decision_tables, ignore_index=True),
    )


def summarize_comparison(
    models: pd.DataFrame,
    economics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise operating characteristics by independent simulated stream.

    For stable and AR(1) null regimes, any alarm is a false alarm. In drifting
    regimes, detection latency is measured from the defined *start* of gradual
    drift; it is descriptive because there is no unique abrupt change time.

    Time steps, policies and economic-cost thresholds from the same stream
    are correlated and must not be counted as independent Monte Carlo trials.
    """
    keys = ["scenario", "regime", "delay", "replicate", "policy"]
    model_fields = set(keys + [
        "step", "squared_error", "alarm", "refit_event", "change_onset"
    ])
    economic_fields = set(keys + [
        "action_cost", "refit_cost", "net_regret"
    ])
    if not model_fields.issubset(models.columns):
        raise ValueError(f"Missing model fields: {model_fields - set(models.columns)}")
    if not economic_fields.issubset(economics.columns):
        raise ValueError("Missing economic fields")
    if models.empty or economics.empty:
        raise ValueError("Cannot summarise empty experiment")

    streams: list[dict[str, str | int | float]] = []
    for identity, frame in models.groupby(keys, sort=True):
        scenario, regime, delay, replicate, policy = identity
        alarms = frame.loc[frame["alarm"] > 0, "step"].to_numpy(dtype=int)
        onset = int(frame["change_onset"].iloc[0])
        has_drift = regime in ("gradual_drift", "gradual_ar1")
        after_change = alarms[alarms >= onset] if has_drift else np.array([])
        latency = (
            float(after_change[0] - onset)
            if len(after_change) > 0 else float("nan")
        )
        streams.append(
            {
                "scenario": scenario, "regime": regime,
                "delay": int(delay), "replicate": int(replicate),
                "policy": policy,
                "rmse": float(np.sqrt(frame["squared_error"].mean())),
                "alarms": float(len(alarms)),
                "any_alarm": float(len(alarms) > 0),
                "premature_alarms": float(np.sum(alarms < onset)),
                "refits": float(frame["refit_event"].sum()),
                "detected_after_onset": float(len(after_change) > 0),
                "detection_latency": latency,
            }
        )

    stream_frame = pd.DataFrame.from_records(streams)
    groups = ["scenario", "regime", "delay", "policy"]
    operational = stream_frame.groupby(groups, as_index=False).agg(
        independent_streams=("replicate", "nunique"),
        mean_rmse=("rmse", "mean"),
        probability_any_alarm=("any_alarm", "mean"),
        mean_alarms=("alarms", "mean"),
        mean_refits=("refits", "mean"),
        mean_premature_alarms=("premature_alarms", "mean"),
        detection_fraction=("detected_after_onset", "mean"),
        mean_detection_latency=("detection_latency", "mean"),
    )
    operational["empirical_false_alarm_probability"] = np.where(
        operational["regime"].isin(["stable", "ar1"]),
        operational["probability_any_alarm"],
        np.nan,
    )

    cost_keys = keys + ["action_cost", "refit_cost"]
    per_stream = economics.groupby(cost_keys, as_index=False).agg(
        stream_net_regret=("net_regret", "mean")
    )
    summary_group = [
        "scenario", "regime", "delay", "policy", "action_cost", "refit_cost"
    ]
    decision = per_stream.groupby(summary_group, as_index=False).agg(
        independent_streams=("replicate", "nunique"),
        mean_net_regret=("stream_net_regret", "mean"),
    )
    return operational, decision
