"""Reproduce the public statistical analysis for the blood-pressure tracker.

The repository intentionally contains only a day-indexed aggregate snapshot, not
raw personal health records. The script validates that snapshot, computes the
reported statistics, fits exploratory models, and regenerates the figures.

This is a statistical case study, not a clinical interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class DailyRecord:
    """One calendar day in the public day-indexed analysis snapshot."""

    day_index: int
    observed: bool
    n_readings: int
    n_sessions: int
    mean_systolic_mmHg: Optional[float]
    mean_diastolic_mmHg: Optional[float]
    mean_pulse_pressure_mmHg: Optional[float]
    mean_bpm: Optional[float]


METRIC_NAMES: Tuple[str, ...] = (
    "mean_systolic_mmHg",
    "mean_diastolic_mmHg",
    "mean_pulse_pressure_mmHg",
    "mean_bpm",
)


def _optional_float(value: str) -> Optional[float]:
    """Convert an optional numeric CSV field to ``float`` or ``None``."""

    text = value.strip()
    if text == "":
        return None
    return float(text)


def load_snapshot(path: Path) -> List[DailyRecord]:
    """Load and validate the day-indexed public snapshot.

    Args:
        path: CSV file containing one row per calendar day.

    Returns:
        Validated records in increasing ``day_index`` order.

    Raises:
        ValueError: If the snapshot is malformed or internally inconsistent.
    """

    records: List[DailyRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "day_index",
            "observed",
            "n_readings",
            "n_sessions",
            *METRIC_NAMES,
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Snapshot columns do not match the expected schema.")

        for row in reader:
            record = DailyRecord(
                day_index=int(row["day_index"]),
                observed=bool(int(row["observed"])),
                n_readings=int(row["n_readings"]),
                n_sessions=int(row["n_sessions"]),
                mean_systolic_mmHg=_optional_float(row["mean_systolic_mmHg"]),
                mean_diastolic_mmHg=_optional_float(row["mean_diastolic_mmHg"]),
                mean_pulse_pressure_mmHg=_optional_float(
                    row["mean_pulse_pressure_mmHg"]
                ),
                mean_bpm=_optional_float(row["mean_bpm"]),
            )
            records.append(record)

    if not records:
        raise ValueError("Snapshot is empty.")

    expected_indices = list(range(len(records)))
    actual_indices = [record.day_index for record in records]
    if actual_indices != expected_indices:
        raise ValueError("day_index must be contiguous and start at zero.")

    for record in records:
        metric_values = [getattr(record, name) for name in METRIC_NAMES]
        if record.observed:
            if record.n_readings <= 0:
                raise ValueError("Observed days must contain at least one reading.")
            if record.n_sessions <= 0 or record.n_sessions > record.n_readings:
                raise ValueError("Observed-day session counts are inconsistent.")
            if any(value is None for value in metric_values):
                raise ValueError("Observed days must contain all daily means.")
        else:
            if record.n_readings != 0 or record.n_sessions != 0:
                raise ValueError("Unobserved days cannot contain readings or sessions.")
            if any(value is not None for value in metric_values):
                raise ValueError("Unobserved days cannot contain daily means.")

    return records


def load_audit(path: Path) -> Dict[str, object]:
    """Load the aggregate source-audit metadata."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Source audit must be a JSON object.")
    return payload


def observed_records(records: Sequence[DailyRecord]) -> List[DailyRecord]:
    """Return only days on which at least one valid measurement was recorded."""

    return [record for record in records if record.observed]


def longest_missing_run(records: Sequence[DailyRecord]) -> int:
    """Return the maximum number of consecutive unobserved calendar days."""

    longest = 0
    current = 0
    for record in records:
        if record.observed:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _metric_array(
    records: Sequence[DailyRecord], metric_name: str
) -> np.ndarray:
    """Extract one observed daily metric as a NumPy vector."""

    if metric_name not in METRIC_NAMES:
        raise ValueError(f"Unsupported metric: {metric_name}")
    values: List[float] = []
    for record in observed_records(records):
        value = getattr(record, metric_name)
        if value is None:
            raise ValueError("Observed metric unexpectedly contains a missing value.")
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _day_index_array(records: Sequence[DailyRecord]) -> np.ndarray:
    """Return observed day indices as floating-point values."""

    return np.asarray(
        [record.day_index for record in observed_records(records)], dtype=float
    )


def _reading_count_array(records: Sequence[DailyRecord]) -> np.ndarray:
    """Return observed per-day reading counts as floating-point values."""

    return np.asarray(
        [record.n_readings for record in observed_records(records)], dtype=float
    )


def linear_trend(
    records: Sequence[DailyRecord], metric_name: str
) -> Dict[str, float]:
    """Fit an exploratory linear trend with HC3 heteroskedasticity-robust SEs."""

    x = _day_index_array(records)
    y = _metric_array(records, metric_name)
    design = sm.add_constant(x)
    fitted = sm.OLS(y, design).fit(cov_type="HC3")
    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)[1]

    return {
        "slope_per_day": float(fitted.params[1]),
        "slope_per_30_days": float(fitted.params[1] * 30.0),
        "ci95_low_per_30_days": float(interval[0] * 30.0),
        "ci95_high_per_30_days": float(interval[1] * 30.0),
        "p_value": float(fitted.pvalues[1]),
        "r_squared": float(fitted.rsquared),
    }


def sampling_association(
    records: Sequence[DailyRecord], metric_name: str
) -> Dict[str, float]:
    """Measure association between sampling intensity and the observed daily mean."""

    counts = _reading_count_array(records)
    values = _metric_array(records, metric_name)
    pearson = stats.pearsonr(counts, values)
    spearman = stats.spearmanr(counts, values)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
    }


def weighted_and_equal_day_means(
    records: Sequence[DailyRecord], metric_name: str
) -> Dict[str, float]:
    """Compare reading-weighted and equal-calendar-day averages."""

    values = _metric_array(records, metric_name)
    counts = _reading_count_array(records)
    return {
        "reading_weighted_mean": float(np.average(values, weights=counts)),
        "equal_observed_day_mean": float(np.mean(values)),
        "difference_weighted_minus_equal_day": float(
            np.average(values, weights=counts) - np.mean(values)
        ),
    }


def state_space_level(
    records: Sequence[DailyRecord], metric_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a local-linear-trend latent level over the full calendar grid.

    Missing calendar days are supplied as ``NaN`` and handled directly by the
    Kalman filter/smoother. Returned intervals describe uncertainty in the
    latent level. They are not replacement observations.
    """

    series = np.full(len(records), np.nan, dtype=float)
    for record in records:
        if record.observed:
            value = getattr(record, metric_name)
            if value is None:
                raise ValueError(
                    "Observed metric unexpectedly contains a missing value."
                )
            series[record.day_index] = float(value)

    model = sm.tsa.UnobservedComponents(series, level="local linear trend")
    fitted = model.fit(disp=False)
    level = np.asarray(fitted.smoothed_state[0], dtype=float)
    variance = np.asarray(fitted.smoothed_state_cov[0, 0, :], dtype=float)
    std = np.sqrt(np.maximum(variance, 0.0))
    lower = level - 1.96 * std
    upper = level + 1.96 * std
    return level, lower, upper


def build_summary(
    records: Sequence[DailyRecord], audit: Dict[str, object]
) -> Dict[str, object]:
    """Build the machine-readable statistical results used in the article."""

    observed = observed_records(records)
    missing_days = len(records) - len(observed)
    counts = _reading_count_array(records)
    longest_run = longest_missing_run(records)

    metrics: Dict[str, object] = {}
    for metric_name in METRIC_NAMES:
        metrics[metric_name] = {
            "means": weighted_and_equal_day_means(records, metric_name),
            "sampling_association": sampling_association(records, metric_name),
            "linear_trend": linear_trend(records, metric_name),
        }

    return {
        "source_audit": audit,
        "calendar": {
            "n_calendar_days": len(records),
            "n_observed_days": len(observed),
            "n_missing_days": missing_days,
            "coverage_rate": len(observed) / len(records),
            "missing_rate": missing_days / len(records),
            "longest_missing_run_days": longest_run,
            "share_of_missing_days_in_longest_run": (
                longest_run / missing_days if missing_days else 0.0
            ),
        },
        "sampling": {
            "total_valid_readings": int(np.sum(counts)),
            "total_sessions": int(sum(record.n_sessions for record in observed)),
            "readings_per_observed_day": {
                "mean": float(np.mean(counts)),
                "median": float(np.median(counts)),
                "q1": float(np.quantile(counts, 0.25)),
                "q3": float(np.quantile(counts, 0.75)),
                "min": float(np.min(counts)),
                "max": float(np.max(counts)),
            },
        },
        "metrics": metrics,
    }


def plot_reading_counts(records: Sequence[DailyRecord], output_path: Path) -> None:
    """Plot the number of valid readings on every calendar day."""

    x = np.asarray([record.day_index for record in records], dtype=int)
    y = np.asarray([record.n_readings for record in records], dtype=int)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, y)
    ax.set_xlabel("Day index from first observation")
    ax.set_ylabel("Valid readings")
    ax.set_title("Sampling intensity is highly uneven across calendar days")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_daily_metric_with_trend(
    records: Sequence[DailyRecord],
    metric_name: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot observed daily means with the HC3 linear-trend point estimate."""

    x = _day_index_array(records)
    y = _metric_array(records, metric_name)
    design = sm.add_constant(x)
    fitted = sm.OLS(y, design).fit(cov_type="HC3")
    grid = np.linspace(0.0, float(len(records) - 1), 200)
    trend = fitted.params[0] + fitted.params[1] * grid

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(x, y, label="Observed daily mean")
    ax.plot(grid, trend, label="Exploratory linear trend")
    ax.set_xlabel("Day index from first observation")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sampling_association(
    records: Sequence[DailyRecord], output_path: Path
) -> None:
    """Plot reading count against observed daily systolic mean."""

    x = _reading_count_array(records)
    y = _metric_array(records, "mean_systolic_mmHg")
    slope, intercept = np.polyfit(x, y, deg=1)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(x, y)
    ax.plot(grid, intercept + slope * grid)
    ax.set_xlabel("Readings on observed day")
    ax.set_ylabel("Daily mean systolic pressure (mmHg)")
    ax.set_title("Sampling intensity is associated with the observed daily mean")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_state_space(
    records: Sequence[DailyRecord],
    metric_name: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot observed values and the smoothed local-linear-trend latent level."""

    level, lower, upper = state_space_level(records, metric_name)
    full_x = np.arange(len(records), dtype=float)
    obs_x = _day_index_array(records)
    obs_y = _metric_array(records, metric_name)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(obs_x, obs_y, label="Observed daily mean")
    ax.plot(full_x, level, label="Smoothed latent level")
    ax.fill_between(full_x, lower, upper, alpha=0.2, label="95% state interval")
    ax.set_xlabel("Day index from first observation")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/analysis_snapshot.csv"),
        help="Day-indexed aggregate snapshot CSV.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/source_audit.json"),
        help="Aggregate source-audit metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated figures and results.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate inputs, reproduce statistics, and regenerate article figures."""

    args = parse_args()
    records = load_snapshot(args.data)
    audit = load_audit(args.audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(records, audit)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_reading_counts(records, args.output_dir / "reading_count_by_day.svg")
    plot_daily_metric_with_trend(
        records,
        "mean_systolic_mmHg",
        "Daily mean systolic pressure (mmHg)",
        "Observed daily systolic means: equal weight per observed day",
        args.output_dir / "daily_systolic_mean.svg",
    )
    plot_sampling_association(
        records, args.output_dir / "sampling_intensity_vs_systolic.svg"
    )
    plot_state_space(
        records,
        "mean_systolic_mmHg",
        "Daily mean systolic pressure (mmHg)",
        "Local-linear-trend smoothing: uncertainty spans unobserved days",
        args.output_dir / "state_space_systolic.svg",
    )
    plot_state_space(
        records,
        "mean_diastolic_mmHg",
        "Daily mean diastolic pressure (mmHg)",
        "Local-linear-trend smoothing for diastolic pressure",
        args.output_dir / "state_space_diastolic.svg",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
