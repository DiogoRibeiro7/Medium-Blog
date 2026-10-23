"""Monte Carlo validation of detector alarms and economic decisions.

Independent generated *streams* are the resampling units. Time observations,
maintenance costs and detectors within a stream are not independent data.

Finite-horizon thresholds based on exchangeable whole-null-stream maxima
are not anytime-valid. The Bonferroni correction protects the declared
family of three detectors, not a post-hoc choice among sensitivity profiles.
"""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from analysis import FloatArray, Scenario
from comparative_change_detection import (
    ComparisonDesign,
    DETECTORS,
    DetectorName,
    make_trackers,
    null_family,
    score_maxima,
)
from gradual_drift import Regime, generate_delayed_stream
from horizon_detector_economics import (
    corrected_horizon_quantile,
    reference_increments,
)

CalibrationRule = Literal["Individual", "Bonferroni"]
ProfileName = Literal["Baseline", "Short memory", "Long memory"]
RULES: tuple[CalibrationRule, ...] = ("Individual", "Bonferroni")
PROFILES: tuple[ProfileName, ...] = (
    "Baseline", "Short memory", "Long memory"
)


def _integer(value: int, name: str, minimum: int) -> None:
    """Validate scalar replication, sample-size or seed parameters."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson confidence interval for a whole-stream Bernoulli rate.

    Intervals condition on the *realised* threshold calibration dataset;
    they do not account for sampling variation of that threshold.
    """
    _integer(successes, "successes", 0)
    _integer(total, "total", 1)
    if successes > total:
        raise ValueError("successes cannot exceed total")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    p = successes / total
    z = float(norm.ppf((1 + confidence) / 2))
    z2 = z * z
    denom = 1 + z2 / total
    mid = (p + z2 / (2 * total)) / denom
    radius = z / denom * sqrt(
        p * (1 - p) / total + z2 / (4 * total * total)
    )
    return max(0.0, mid - radius), min(1.0, mid + radius)


def paired_regret(
    economic_rows: pd.DataFrame,
    reference: str = "CUSUM",
    candidates: tuple[str, ...] = ("Frozen", "Scheduled", "GLR", "BOCPD"),
    bootstrap_draws: int = 600,
    seed: int = 37,
) -> pd.DataFrame:
    """Compare maintenance strategies on matched independent simulated streams.

    Delta = candidate net regret minus reference regret; negative is better.
    A percentile bootstrap resamples entire independent replicate datasets,
    not time steps. With fewer than three replicates, return NaN intervals.
    """
    _integer(bootstrap_draws, "bootstrap_draws", 10)
    _integer(seed, "seed", 0)
    if not reference or not candidates or reference in candidates:
        raise ValueError("Reference must differ from nonempty candidate list")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidates must be unique")
    key = [
        "scenario", "regime", "delay", "replicate",
        "action_cost", "refit_cost",
    ]
    required = set(key) | {"policy", "net_regret"}
    if not required.issubset(economic_rows.columns):
        raise ValueError(f"Missing economic columns: {required - set(economic_rows.columns)}")
    if economic_rows.empty or not np.isfinite(economic_rows["net_regret"]).all():
        raise ValueError("Economic rows must be finite and nonempty")
    per_stream = economic_rows.groupby(key + ["policy"], as_index=False).agg(
        regret=("net_regret", "mean")
    )
    baseline = per_stream.loc[
        per_stream["policy"] == reference, key + ["regret"]
    ].rename(columns={"regret": "baseline_regret"})
    if baseline.empty:
        raise ValueError("Missing reference policy")

    grouping = [name for name in key if name != "replicate"]
    rng = np.random.default_rng(seed)
    out: list[dict[str, str | int | float]] = []
    for candidate in candidates:
        alternative = per_stream.loc[
            per_stream["policy"] == candidate, key + ["regret"]
        ].rename(columns={"regret": "candidate_regret"})
        if alternative.empty:
            raise ValueError(f"Missing candidate policy: {candidate}")
        joined = baseline.merge(
            alternative, on=key, how="outer", validate="one_to_one"
        )
        if joined[["baseline_regret", "candidate_regret"]].isna().any().any():
            raise ValueError("Candidate and reference streams must pair")
        joined["delta"] = joined["candidate_regret"] - joined["baseline_regret"]
        for identity, subset in joined.groupby(grouping, sort=True):
            values = subset["delta"].to_numpy(dtype=np.float64)
            count = len(values)
            if count >= 3:
                indexes = rng.integers(
                    0, count, size=(bootstrap_draws, count)
                )
                lower, upper = np.quantile(
                    values[indexes].mean(axis=1), [0.025, 0.975]
                )
                mc_se = float(np.std(values, ddof=1) / sqrt(count))
            else:
                lower = upper = mc_se = float("nan")
            out.append(
                {
                    **dict(zip(grouping, identity)),
                    "reference": reference,
                    "candidate": candidate,
                    "independent_streams": count,
                    "delta_regret": float(values.mean()),
                    "mc_standard_error": mc_se,
                    "bootstrap_lower_95": float(lower),
                    "bootstrap_upper_95": float(upper),
                }
            )
    return pd.DataFrame.from_records(out)


def empirical_null_alarms(model_rows: pd.DataFrame) -> pd.DataFrame:
    """Estimate individual and three-detector family null false-alarm rates.

    This audits the *uncorrected*, individually calibrated detector comparison.
    Probability of at least one alarm in the detector family may exceed the
    single-detector target. It is measured per independent evaluation stream.
    """
    required = {
        "scenario", "regime", "delay", "replicate", "policy", "alarm"
    }
    if not required.issubset(model_rows.columns):
        raise ValueError("Missing detector model columns")
    if model_rows.empty or not np.isfinite(model_rows["alarm"]).all():
        raise ValueError("Model rows must be nonempty and finite")
    observed = model_rows.loc[
        model_rows["policy"].isin(DETECTORS)
        & model_rows["regime"].isin(["stable", "ar1"])
    ]
    if observed.empty:
        raise ValueError("No stable null detection records")
    key = ["scenario", "regime", "delay", "replicate"]
    flagged = observed.groupby(key + ["policy"], as_index=False).agg(
        alarm=("alarm", lambda x: int((x > 0).any()))
    )
    if flagged.groupby(key)["policy"].nunique().ne(len(DETECTORS)).any():
        raise ValueError("Each null stream must contain all detectors")
    family = flagged.groupby(key, as_index=False).agg(
        alarm=("alarm", "max")
    )
    family["policy"] = "Any detector"
    pooled = pd.concat([flagged, family], ignore_index=True)
    out: list[dict[str, str | int | float]] = []
    for group, frame in pooled.groupby(
        ["scenario", "regime", "delay", "policy"], sort=True
    ):
        count = len(frame)
        alarms = int(frame["alarm"].sum())
        low, high = wilson_interval(alarms, count)
        out.append(
            {
                **dict(zip(["scenario", "regime", "delay", "policy"], group)),
                "independent_streams": count,
                "alarm_count": alarms,
                "alarm_probability": alarms / count,
                "wilson_lower_95": low if count >= 3 else float("nan"),
                "wilson_upper_95": high if count >= 3 else float("nan"),
            }
        )
    return pd.DataFrame.from_records(out)


def detector_profiles(
    original: ComparisonDesign,
) -> dict[ProfileName, ComparisonDesign]:
    """Predeclare parameter sensitivity without evaluation-based tuning."""
    return {
        "Baseline": original,
        "Short memory": replace(
            original, cusum_allowance=0.10,
            glr_min_window=2, glr_max_window=12,
            bocpd_hazard=0.10,
        ),
        "Long memory": replace(
            original, cusum_allowance=0.50,
            glr_min_window=6, glr_max_window=40,
            bocpd_hazard=0.01,
        ),
    }


def detector_traces(
    released_scores: FloatArray,
    design: ComparisonDesign,
) -> dict[DetectorName, FloatArray]:
    """Return unreset online score traces with NaN at unreleased steps."""
    values = np.asarray(released_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.isinf(values).any():
        raise ValueError("Released scores must be a vector without infinities")
    trackers = make_trackers(design)
    traces: dict[DetectorName, FloatArray] = {
        name: np.full(values.size, np.nan) for name in DETECTORS
    }
    for time, value in enumerate(values):
        if np.isnan(value):
            continue
        for detector in DETECTORS:
            traces[detector][time] = float(
                trackers[detector].update(float(value))
            )
    return traces


def first_alarm_step(trace: FloatArray, threshold: float) -> int | None:
    """First threshold crossing, censored to None if the alarm never occurs."""
    scores = np.asarray(trace, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or np.isinf(scores).any():
        raise ValueError("Invalid score trace")
    if np.isnan(threshold) or threshold < 0:
        raise ValueError("Threshold cannot be negative or NaN")
    crossing = np.flatnonzero(scores > threshold)
    return int(crossing[0]) if crossing.size else None


def run_sensitivity(
    design: ComparisonDesign,
    evaluation_repeats: int = 30,
    profiles: tuple[ProfileName, ...] = PROFILES,
    regimes: tuple[Regime, ...] = (
        "stable", "gradual_drift", "ar1", "gradual_ar1"
    ),
    scenario: Scenario = "low_variance_signal",
    delay: int = 0,
    seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate individual and Bonferroni thresholds on disjoint null streams.

    Each profile shares the *same null score series* and evaluation streams.
    The familywise rule uses per-detector alpha / 3; it applies only to the
    predeclared three-detector family, not the family of profiles.

    Sensitivity evaluates first alarms *without* refitting or cooldown.
    The maintenance-policy experiment is a separate evaluation.
    """
    _integer(evaluation_repeats, "evaluation_repeats", 1)
    _integer(seed, "seed", 0)
    profile_map = detector_profiles(design)
    if not profiles or len(set(profiles)) != len(profiles) or any(
        profile not in profile_map for profile in profiles
    ):
        raise ValueError("Profiles must be distinct and recognised")
    allowed = ("stable", "gradual_drift", "ar1", "gradual_ar1")
    if not regimes or len(set(regimes)) != len(regimes) or any(
        regime not in allowed for regime in regimes
    ):
        raise ValueError("Regimes must be distinct and recognised")
    if scenario not in ("low_variance_signal", "high_variance_signal"):
        raise ValueError("Unsupported response scenario")
    if delay not in design.delays:
        raise ValueError("Requested delay must belong to design")

    families = sorted(set(null_family(regime) for regime in regimes))
    null_samples: dict[str, list[FloatArray]] = {}
    for index, family in enumerate(families):
        samples: list[FloatArray] = []
        for draw in range(design.null_streams):
            # Null-calibration seeds are separated by 20 million from
            # evaluation seeds; profiles share null trajectories.
            stream = generate_delayed_stream(
                seed + 20_000_000 + index * 1_000_000 + draw,
                regime=family, scenario=scenario,
                n_fit=design.n_fit, n_cal=design.n_cal,
                n_live=design.n_live,
            )
            samples.append(reference_increments(
                stream, delay, allowance=0.0
            ))
        null_samples[family] = samples

    thresholds: dict[tuple[str, str, str, str], float] = {}
    calibration_rows: list[dict[str, str | int | float]] = []
    for profile in profiles:
        profile_design = profile_map[profile]
        for family in families:
            maxima_by_stream = [
                score_maxima(sample, profile_design)
                for sample in null_samples[family]
            ]
            for detector in DETECTORS:
                maxima = np.asarray(
                    [item[detector] for item in maxima_by_stream], dtype=np.float64
                )
                for rule in RULES:
                    target = design.alpha / (
                        len(DETECTORS) if rule == "Bonferroni" else 1
                    )
                    threshold, rank = corrected_horizon_quantile(
                        maxima, target
                    )
                    thresholds[(profile, family, rule, detector)] = threshold
                    calibration_rows.append(
                        {
                            "profile": profile, "null_family": family,
                            "rule": rule, "detector": detector,
                            "per_detector_target": float(target),
                            "threshold": float(threshold),
                            "rank": int(rank),
                            "null_streams": design.null_streams,
                        }
                    )

    evaluation_rows: list[dict[str, str | int | float]] = []
    for regime_index, regime in enumerate(regimes):
        family = null_family(regime)
        has_drift = regime in ("gradual_drift", "gradual_ar1")
        for replicate in range(evaluation_repeats):
            stream = generate_delayed_stream(
                seed + regime_index * 100_000 + replicate,
                regime=regime, scenario=scenario,
                n_fit=design.n_fit, n_cal=design.n_cal,
                n_live=design.n_live,
            )
            released = reference_increments(
                stream, delay, allowance=0.0
            )
            for profile in profiles:
                traces = detector_traces(released, profile_map[profile])
                for rule in RULES:
                    for detector in DETECTORS:
                        first = first_alarm_step(
                            traces[detector],
                            thresholds[(profile, family, rule, detector)],
                        )
                        post = (
                            has_drift
                            and first is not None
                            and first >= stream.drift_start
                        )
                        evaluation_rows.append(
                            {
                                "scenario": scenario, "regime": regime,
                                "delay": delay, "replicate": replicate,
                                "profile": profile, "rule": rule,
                                "detector": detector,
                                "first_alarm": (
                                    float(first) if first is not None
                                    else float("nan")
                                ),
                                "any_alarm": int(first is not None),
                                "early_alarm": int(
                                    has_drift and first is not None
                                    and first < stream.drift_start
                                ),
                                "post_onset_first_alarm": int(post),
                                "latency_if_detected": (
                                    float(first - stream.drift_start)
                                    if post and first is not None
                                    else float("nan")
                                ),
                            }
                        )
    return (
        pd.DataFrame.from_records(calibration_rows),
        pd.DataFrame.from_records(evaluation_rows),
    )


def summarize_sensitivity(records: pd.DataFrame) -> pd.DataFrame:
    """Estimate detector-specific and combined fixed-horizon alarm rates.

    In changed regimes, first-alarm-after-onset excludes streams with
    premature alarms. No claim about repeated alarms is made.
    """
    key = [
        "scenario", "regime", "delay", "replicate", "profile", "rule"
    ]
    required = set(key) | {
        "detector", "any_alarm", "early_alarm",
        "post_onset_first_alarm", "latency_if_detected"
    }
    if not required.issubset(records.columns):
        raise ValueError("Missing sensitivity data columns")
    if records.empty or records.duplicated(key + ["detector"]).any():
        raise ValueError("Duplicate or empty sensitivity observations")
    if records.groupby(key)["detector"].nunique().ne(len(DETECTORS)).any():
        raise ValueError("Every stream must have three detector scores")

    family = records.groupby(key, as_index=False).agg(
        any_alarm=("any_alarm", "max"),
        early_alarm=("early_alarm", "max"),
        post_onset_first_alarm=("post_onset_first_alarm", "max"),
    )
    family["detector"] = "Any detector"
    family["latency_if_detected"] = float("nan")
    combined = pd.concat([records, family], ignore_index=True)
    grouping = [
        "scenario", "regime", "delay", "profile", "rule", "detector"
    ]
    result: list[dict[str, str | int | float]] = []
    for labels, frame in combined.groupby(grouping, sort=True):
        n = len(frame)
        alarms = int(frame["any_alarm"].sum())
        low, high = wilson_interval(alarms, n)
        latency = frame["latency_if_detected"].dropna()
        result.append(
            {
                **dict(zip(grouping, labels)),
                "independent_streams": n,
                "alarm_probability": alarms / n,
                "wilson_lower_95": low if n >= 3 else float("nan"),
                "wilson_upper_95": high if n >= 3 else float("nan"),
                "early_alarm_probability": float(frame["early_alarm"].mean()),
                "post_onset_first_alarm_rate": float(
                    frame["post_onset_first_alarm"].mean()
                ),
                "latency_if_detected": (
                    float(latency.mean()) if len(latency) else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(result)
