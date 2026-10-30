"""Reproducible horizon-calibrated online change-detection experiment.

This study distinguishes signal time from the true change time, and matches
detectors by their chance of a false alarm within a *finite* 200-event horizon.
It does not claim equal average run lengths or indefinite false-alarm control.
All streams are synthetic and generated from independent seed namespaces.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Method: TypeAlias = Literal["shewhart", "cusum_mean", "ewma_mean", "cusum_variance"]
Row: TypeAlias = dict[str, str | float | int]

SEED = 20261030
HORIZON = 200
CHANGE_INDEX = 80
DETECTION_WINDOW = 50
EARLY_WINDOW = 5
NULL_FALSE_ALARM_TARGET = 0.10
EWMA_WEIGHT = 0.20
MEAN_CUSUM_REFERENCE = 0.50
SCALE_ALTERNATIVE = 1.70
AR_COEFFICIENT = 0.60
METHODS: tuple[Method, ...] = (
    "shewhart",
    "cusum_mean",
    "ewma_mean",
    "cusum_variance",
)
LABELS: dict[Method, str] = {
    "shewhart": "Shewhart (upward)",
    "cusum_mean": "CUSUM (upward mean)",
    "ewma_mean": "EWMA (upward mean)",
    "cusum_variance": "CUSUM (variance)",
}
SCENARIOS = ("mean_0_5", "mean_1_0", "mean_2_0", "single_spike", "variance_1_7")


@dataclass(frozen=True)
class Study:
    """Small, versionable metrics plus arrays required for eight figures."""

    thresholds: dict[Method, float]
    calibration_maxima: dict[Method, FloatArray]
    example_paths: dict[str, FloatArray]
    example_statistics: dict[Method, FloatArray]
    calibration_rows: list[Row]
    scenario_rows: list[Row]
    dependence_rows: list[Row]
    delay_rows: list[Row]
    sample_path_rows: list[Row]
    summary_rows: list[Row]


def _generator(namespace: int) -> np.random.Generator:
    """Construct independent random streams from a stable master seed."""
    if not isinstance(namespace, int) or isinstance(namespace, bool) or namespace < 0:
        raise ValueError("namespace must be a nonnegative integer")
    return np.random.default_rng(np.random.SeedSequence((SEED, namespace)))


def _validate_streams(streams: FloatArray) -> FloatArray:
    """Require a rectangular, finite, two-dimensional set of observations."""
    array = np.asarray(streams, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("streams must have shape (positive_paths, at_least_two_times)")
    if not np.isfinite(array).all():
        raise ValueError("all observations must be finite")
    return array


def detector_statistics(streams: FloatArray) -> dict[Method, FloatArray]:
    """Calculate four *causal* statistics, retaining every time step.

    The one-sided mean CUSUM has reference k=0.5, suitable for a nominal
    N(0,1) to N(1,1) mean shift. The variance CUSUM uses the Gaussian
    N(0,1) versus N(0,1.7^2) log-likelihood increment. Both are recursive
    and clipped at zero. EWMA starts at the known in-control mean zero.
    """
    observations = _validate_streams(streams)
    n_paths, horizon = observations.shape
    shewhart = observations.copy()
    cusum = np.empty((n_paths, horizon), dtype=np.float64)
    ewma = np.empty((n_paths, horizon), dtype=np.float64)
    variance_cusum = np.empty((n_paths, horizon), dtype=np.float64)
    state_cusum = np.zeros(n_paths, dtype=np.float64)
    state_ewma = np.zeros(n_paths, dtype=np.float64)
    state_variance = np.zeros(n_paths, dtype=np.float64)
    variance_coefficient = 0.5 * (1.0 - SCALE_ALTERNATIVE ** -2)
    for time in range(horizon):
        value = observations[:, time]
        state_cusum = np.maximum(0.0, state_cusum + value - MEAN_CUSUM_REFERENCE)
        state_ewma = EWMA_WEIGHT * value + (1.0 - EWMA_WEIGHT) * state_ewma
        log_likelihood_ratio = (
            -np.log(SCALE_ALTERNATIVE) + variance_coefficient * value**2
        )
        state_variance = np.maximum(0.0, state_variance + log_likelihood_ratio)
        cusum[:, time] = state_cusum
        ewma[:, time] = state_ewma
        variance_cusum[:, time] = state_variance
    return {
        "shewhart": shewhart,
        "cusum_mean": cusum,
        "ewma_mean": ewma,
        "cusum_variance": variance_cusum,
    }


def _sample_streams(
    rng: np.random.Generator,
    n_paths: int,
    scenario: str,
) -> FloatArray:
    """Generate a stationary null or a controlled post-index change."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    if scenario not in {"null_iid", "null_ar1", *SCENARIOS}:
        raise ValueError(f"unsupported scenario: {scenario}")
    if scenario == "null_ar1":
        innovations = rng.standard_normal((n_paths, HORIZON))
        series = np.empty_like(innovations)
        series[:, 0] = innovations[:, 0]
        coefficient = np.sqrt(1.0 - AR_COEFFICIENT**2)
        for time in range(1, HORIZON):
            series[:, time] = (
                AR_COEFFICIENT * series[:, time - 1]
                + coefficient * innovations[:, time]
            )
        return series

    series = np.asarray(rng.standard_normal((n_paths, HORIZON)), dtype=np.float64)
    if scenario == "mean_0_5":
        series[:, CHANGE_INDEX:] += 0.5
    elif scenario == "mean_1_0":
        series[:, CHANGE_INDEX:] += 1.0
    elif scenario == "mean_2_0":
        series[:, CHANGE_INDEX:] += 2.0
    elif scenario == "single_spike":
        series[:, CHANGE_INDEX] += 5.0
    elif scenario == "variance_1_7":
        series[:, CHANGE_INDEX:] *= SCALE_ALTERNATIVE
    return series


def _first_alarm(statistics: FloatArray, threshold: float) -> NDArray[np.int64]:
    """Return first zero-based crossing index, or horizon if never signaled."""
    series = _validate_streams(statistics)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    crossings = series > threshold
    has_crossing = np.any(crossings, axis=1)
    indices = np.argmax(crossings, axis=1)
    return np.asarray(
        np.where(has_crossing, indices, series.shape[1]),
        dtype=np.int64,
    )


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for a binomial event proportion."""
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("counts must be integers")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    z = 1.959963984540054
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return float(center - half), float(center + half)


def _scenario_metrics(
    scenario: str,
    paths: FloatArray,
    thresholds: dict[Method, float],
) -> tuple[list[Row], list[Row]]:
    """Measure conditional detection and delay, censoring later signals."""
    statistics = detector_statistics(paths)
    metrics: list[Row] = []
    delay_rows: list[Row] = []
    for method in METHODS:
        first = _first_alarm(statistics[method], thresholds[method])
        survivors = first >= CHANGE_INDEX
        n_survive = int(survivors.sum())
        if n_survive < 1:
            raise RuntimeError("no stream survived before the known change point")
        alarm_after_change = first - CHANGE_INDEX
        signaled_50 = survivors & (alarm_after_change < DETECTION_WINDOW)
        signaled_5 = survivors & (alarm_after_change < EARLY_WINDOW)
        n_50 = int(signaled_50.sum())
        n_5 = int(signaled_5.sum())
        low_50, high_50 = wilson_interval(n_50, n_survive)
        low_5, high_5 = wilson_interval(n_5, n_survive)
        delays = np.asarray(alarm_after_change[signaled_50], dtype=np.int64)
        median_delay = float(np.median(delays)) if delays.size else float("nan")
        metrics.append(
            {
                "scenario": scenario,
                "method": method,
                "streams": paths.shape[0],
                "prechange_survivors": n_survive,
                "prechange_alarm_fraction": float(1.0 - n_survive / paths.shape[0]),
                "detected_within_50": n_50,
                "conditional_detection_50": n_50 / n_survive,
                "ci95_50_low": low_50,
                "ci95_50_high": high_50,
                "detected_within_5": n_5,
                "conditional_detection_5": n_5 / n_survive,
                "ci95_5_low": low_5,
                "ci95_5_high": high_5,
                "median_delay_among_detected_50": median_delay,
            }
        )
        if scenario == "mean_1_0":
            for delay in range(DETECTION_WINDOW):
                delay_rows.append(
                    {
                        "method": method,
                        "delay": delay,
                        "survivors": n_survive,
                        "detected_within_50": n_50,
                        "fraction_of_detected_by_delay": float(
                            np.mean(delays <= delay) if delays.size else float("nan")
                        ),
                        "fraction_of_survivors_detected_by_delay": float(
                            np.sum(delays <= delay) / n_survive
                        ),
                    }
                )
    return metrics, delay_rows


def run_study(quick: bool = False) -> Study:
    """Calibrate a fixed-horizon alarm rule and assess independent scenarios."""
    if not isinstance(quick, bool):
        raise TypeError("quick must be bool")
    calibration_count = 1_000 if quick else 5_000
    validation_count = 1_500 if quick else 6_000
    scenario_count = 1_500 if quick else 5_000

    # Calibrate to P0(any signal by event 200) = 10%, not a target ARL0.
    calibration = _sample_streams(_generator(10), calibration_count, "null_iid")
    null_statistics = detector_statistics(calibration)
    null_maxima = {
        method: values.max(axis=1)
        for method, values in null_statistics.items()
    }
    thresholds: dict[Method, float] = {
        method: float(
            np.quantile(null_maxima[method], 1.0 - NULL_FALSE_ALARM_TARGET)
        )
        for method in METHODS
    }

    validation = _sample_streams(_generator(11), validation_count, "null_iid")
    validation_statistics = detector_statistics(validation)
    calibration_rows: list[Row] = []
    dependence_rows: list[Row] = []
    ar_null = _sample_streams(_generator(12), validation_count, "null_ar1")
    ar_stats = detector_statistics(ar_null)
    for method in METHODS:
        validation_alarm_count = int(
            (_first_alarm(validation_statistics[method], thresholds[method]) < HORIZON).sum()
        )
        ar_alarm_count = int((_first_alarm(ar_stats[method], thresholds[method]) < HORIZON).sum())
        iid_lower, iid_upper = wilson_interval(validation_alarm_count, validation_count)
        ar_lower, ar_upper = wilson_interval(ar_alarm_count, validation_count)
        calibration_rows.append(
            {
                "method": method,
                "threshold": thresholds[method],
                "calibration_streams": calibration_count,
                "validation_streams": validation_count,
                "target_null_probability_by_200": NULL_FALSE_ALARM_TARGET,
                "validated_iid_alarm_count": validation_alarm_count,
                "validated_iid_false_alarm_probability": validation_alarm_count / validation_count,
                "iid_ci95_low": iid_lower,
                "iid_ci95_high": iid_upper,
            }
        )
        for process, count, low, high in (
            ("iid_gaussian", validation_alarm_count, iid_lower, iid_upper),
            ("ar1_gaussian_phi_0_6", ar_alarm_count, ar_lower, ar_upper),
        ):
            dependence_rows.append(
                {
                    "method": method,
                    "null_process": process,
                    "streams": validation_count,
                    "alarm_count_by_200": count,
                    "false_alarm_probability_by_200": count / validation_count,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    scenario_rows: list[Row] = []
    delay_rows: list[Row] = []
    for namespace, scenario in enumerate(SCENARIOS, start=20):
        paths = _sample_streams(_generator(namespace), scenario_count, scenario)
        metrics, delays = _scenario_metrics(scenario, paths, thresholds)
        scenario_rows.extend(metrics)
        delay_rows.extend(delays)

    example_paths: dict[str, FloatArray] = {}
    for namespace, scenario in (
        (45, "mean_1_0"), (41, "single_spike"), (42, "variance_1_7")
    ):
        example_paths[scenario] = _sample_streams(_generator(namespace), 1, scenario)[0]
    example_stat = detector_statistics(example_paths["mean_1_0"][None, :])
    example_statistics: dict[Method, FloatArray] = {
        method: series[0] for method, series in example_stat.items()
    }
    sample_path_rows: list[Row] = []
    for t in range(HORIZON):
        sample_path_rows.append(
            {
                "time_index_0_based": t,
                "known_change_index": CHANGE_INDEX,
                "mean_shift_observation": float(example_paths["mean_1_0"][t]),
                "pulse_observation": float(example_paths["single_spike"][t]),
                "variance_shift_observation": float(example_paths["variance_1_7"][t]),
                **{
                    f"{method}_statistic": float(example_statistics[method][t])
                    for method in METHODS
                },
            }
        )
    summary_rows: list[Row] = []
    for method in METHODS:
        reference = next(row for row in calibration_rows if row["method"] == method)
        mean_one = next(
            row for row in scenario_rows
            if row["method"] == method and row["scenario"] == "mean_1_0"
        )
        variance = next(
            row for row in scenario_rows
            if row["method"] == method and row["scenario"] == "variance_1_7"
        )
        summary_rows.append(
            {
                "seed": SEED,
                "horizon": HORIZON,
                "change_index": CHANGE_INDEX,
                "detection_window": DETECTION_WINDOW,
                "method": method,
                "threshold": thresholds[method],
                "iid_null_false_alarm_by_200": float(reference["validated_iid_false_alarm_probability"]),
                "mean_shift_1_detection_given_survival_by_50": float(
                    mean_one["conditional_detection_50"]
                ),
                "variance_shift_detection_given_survival_by_50": float(
                    variance["conditional_detection_50"]
                ),
                "mean_shift_1_median_delay_among_detected": float(
                    mean_one["median_delay_among_detected_50"]
                ),
            }
        )
    return Study(
        thresholds=thresholds,
        calibration_maxima=null_maxima,
        example_paths=example_paths,
        example_statistics=example_statistics,
        calibration_rows=calibration_rows,
        scenario_rows=scenario_rows,
        dependence_rows=dependence_rows,
        delay_rows=delay_rows,
        sample_path_rows=sample_path_rows,
        summary_rows=summary_rows,
    )


def write_csv(path: Path, rows: list[Row]) -> None:
    """Persist deterministic UTF-8 tables with stable insertion-order columns."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not rows:
        raise ValueError("at least one table row is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scenario_row(study: Study, scenario: str, method: Method) -> Row:
    """Fetch one uniquely defined method/scenario comparison row."""
    return next(
        row for row in study.scenario_rows
        if row["scenario"] == scenario and row["method"] == method
    )


def make_figures(study: Study, output: Path, *, suffix: str = "") -> None:
    """Generate eight publication figures from seeded simulation results."""
    if not isinstance(study, Study) or not isinstance(output, Path):
        raise TypeError("expected Study and Path arguments")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False}
    )
    navy, teal, orange, red = "#385779", "#4a9290", "#d08e47", "#b64d4a"
    colors = (navy, teal, orange, red)
    time = np.arange(HORIZON)

    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8), sharex=True, constrained_layout=True)
    for ax, scenario, caption in zip(
        axes,
        ("mean_1_0", "single_spike", "variance_1_7"),
        ("Persistent mean shift: +1", "One-time spike: +5", "Persistent scale shift: ×1.7"),
    ):
        ax.plot(time, study.example_paths[scenario], color=navy, lw=1.3)
        ax.axvline(CHANGE_INDEX, ls="--", color=red, label="Known intervention")
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.4)
        ax.set_ylabel("Observed value")
        ax.set_title(caption)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Event index (zero-based)")
    fig.suptitle("Three different changes can look similar at a glance")
    fig.savefig(output / f"fig1_three_types_of_change{suffix}.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7), constrained_layout=True)
    for ax, method, color in zip(axes.flat, METHODS, colors):
        ax.plot(time, study.example_statistics[method], color=color, lw=1.4)
        ax.axhline(study.thresholds[method], color=red, ls="--", label="Calibrated limit")
        ax.axvline(CHANGE_INDEX, color="black", ls=":", label="True change")
        ax.set_title(LABELS[method])
        ax.set_xlabel("Event index")
        ax.set_ylabel("Sequential statistic")
        ax.grid(alpha=0.23)
        ax.legend(fontsize=8)
    fig.suptitle("A change time is not the same as the first alarm time")
    fig.savefig(output / f"fig2_statistic_alarm_paths{suffix}.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7), constrained_layout=True)
    for ax, method, color in zip(axes.flat, METHODS, colors):
        values = study.calibration_maxima[method]
        ax.hist(values, bins=30, color=color, alpha=0.8)
        ax.axvline(study.thresholds[method], color=red, ls="--", label="Empirical 90th percentile")
        ax.set_title(LABELS[method])
        ax.set_xlabel("Largest statistic over 200 null events")
        ax.set_ylabel("Calibration paths")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Match a 10% false-alarm probability over a fixed 200-event horizon")
    fig.savefig(output / f"fig3_horizon_calibration{suffix}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    shifts = (("mean_0_5", 0.5), ("mean_1_0", 1.0), ("mean_2_0", 2.0))
    for method, color in zip(METHODS, colors):
        rows = [_scenario_row(study, scenario, method) for scenario, _ in shifts]
        probs = np.array([float(row["conditional_detection_50"]) for row in rows])
        lo = np.array([float(row["ci95_50_low"]) for row in rows])
        hi = np.array([float(row["ci95_50_high"]) for row in rows])
        ax.errorbar(
            [amount for _, amount in shifts],
            probs,
            # Guard numerical roundoff in Wilson endpoints at proportions 0/1.
            yerr=[np.maximum(0.0, probs - lo), np.maximum(0.0, hi - probs)],
            marker="o",
            capsize=4,
            color=color,
            label=LABELS[method],
        )
    ax.set_ylim(0, 1.05)
    ax.set_xticks((0.5, 1.0, 2.0))
    ax.set_xlabel("Persistent upward change in mean (standard deviations)")
    ax.set_ylabel("P(alarm within 50 | survived before change)")
    ax.set_title("Sensitivity to persistent mean shifts at matched null risk")
    ax.grid(alpha=0.23)
    ax.legend()
    fig.savefig(output / f"fig4_mean_shift_detection{suffix}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    for method, color in zip(METHODS, colors):
        rows = [row for row in study.delay_rows if row["method"] == method]
        n_detected = int(rows[0]["detected_within_50"])
        n_survived = int(rows[0]["survivors"])
        ax.plot(
            [int(row["delay"]) for row in rows],
            [float(row["fraction_of_survivors_detected_by_delay"]) for row in rows],
            color=color,
            label=f"{LABELS[method]}: {n_detected}/{n_survived}",
        )
    ax.set_xlim(0, DETECTION_WINDOW - 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Events elapsed since true mean +1 shift")
    ax.set_ylabel("Fraction of prechange survivors signaled by delay")
    ax.set_title("Cumulative detection by time: undetected runs remain in denominator")
    ax.legend()
    ax.grid(alpha=0.23)
    fig.savefig(output / f"fig5_detection_by_delay{suffix}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 5), constrained_layout=True)
    positions = np.arange(len(METHODS))
    width = 0.36
    for scenario, caption, offset, color in (
        ("single_spike", "Single +5 spike", -width / 2, orange),
        ("mean_1_0", "Persistent +1 mean", width / 2, teal),
    ):
        rates = [
            float(_scenario_row(study, scenario, method)["conditional_detection_5"])
            for method in METHODS
        ]
        ax.bar(positions + offset, rates, width=width, color=color, label=caption)
    ax.set_xticks(positions, [LABELS[method].replace(" (", "\n(") for method in METHODS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(alarm within 5 | survived before change)")
    ax.set_title("A spike and a persistent shift ask different questions")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / f"fig6_spike_vs_persistence{suffix}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5), constrained_layout=True)
    for process, caption, offset, color in (
        ("iid_gaussian", "iid N(0,1)", -width / 2, navy),
        ("ar1_gaussian_phi_0_6", "AR(1), φ = 0.6", width / 2, red),
    ):
        rows = [
            next(
                row for row in study.dependence_rows
                if row["method"] == method and row["null_process"] == process
            )
            for method in METHODS
        ]
        rates = [float(row["false_alarm_probability_by_200"]) for row in rows]
        ax.bar(positions + offset, rates, width=width, color=color, label=caption)
    ax.axhline(NULL_FALSE_ALARM_TARGET, ls="--", color="black", label="Target under iid")
    ax.set_xticks(positions, [LABELS[method].replace(" (", "\n(") for method in METHODS])
    ax.set_ylabel("P(false alarm by event 200)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Calibration changes when the null observations are autocorrelated")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / f"fig7_autocorrelation_miscalibration{suffix}.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5), constrained_layout=True)
    for scenario, caption, offset, color in (
        ("mean_1_0", "Mean shift: +1", -width / 2, teal),
        ("variance_1_7", "Scale shift: ×1.7", width / 2, orange),
    ):
        rates = [
            float(_scenario_row(study, scenario, method)["conditional_detection_50"])
            for method in METHODS
        ]
        bars = ax.bar(positions + offset, rates, width=width, label=caption, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set_xticks(positions, [LABELS[method].replace(" (", "\n(") for method in METHODS])
    ax.set_ylabel("P(alarm within 50 | survived before change)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Detecting a variance increase requires monitoring more than the mean")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / f"fig8_variance_change_detection{suffix}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    """Persist full evidence or isolated suffixed smoke-test artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced independent smoke run")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not isinstance(args.output_root, Path):
        raise TypeError("output root must be a pathlib.Path")
    study = run_study(quick=bool(args.quick))
    suffix = "_quick" if args.quick else ""
    outputs = (
        ("summary", study.summary_rows),
        ("calibration", study.calibration_rows),
        ("scenario_metrics", study.scenario_rows),
        ("dependence_null", study.dependence_rows),
        ("detection_by_delay", study.delay_rows),
        ("sample_path", study.sample_path_rows),
    )
    for name, rows in outputs:
        write_csv(args.output_root / "results" / f"{name}{suffix}.csv", rows)
    make_figures(study, args.output_root / "figures", suffix=suffix)
    for row in study.summary_rows:
        print(
            f"{row['method']:16}  h={float(row['threshold']):6.3f}  "
            f"false0={float(row['iid_null_false_alarm_by_200']):.3f}  "
            f"Pdetect50(mean+1)={float(row['mean_shift_1_detection_given_survival_by_50']):.3f}  "
            f"Pdetect50(scale×1.7)={float(row['variance_shift_detection_given_survival_by_50']):.3f}"
        )


if __name__ == "__main__":
    main()
