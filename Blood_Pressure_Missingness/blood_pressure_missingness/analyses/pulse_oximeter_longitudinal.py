"""Coverage-aware longitudinal diagnostics for pulse-oximeter aggregates.

The pulse-oximeter device was added after the blood-pressure tracker was already
running, so a time association must be interpreted together with the observation
pattern. This module therefore reports coverage first and estimates descriptive
HC3 time associations only when enough pulse-oximeter days are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import statsmodels.api as sm

DAYS_PER_REPORTING_PERIOD: Final[float] = 30.0
MIN_TREND_DAYS: Final[int] = 4


@dataclass(frozen=True)
class PulseOximeterDay:
    """One privacy-safe daily pulse-oximeter aggregate row."""

    day_index: int
    n_spo2_readings: int
    mean_spo2_percent: float | None
    n_bpm_spo2_readings: int
    mean_bpm_spo2: float | None
    n_paired_bpm_readings: int
    mean_bpm_difference: float | None


@dataclass(frozen=True)
class BloodPressureCalendarDay:
    """One day from the established privacy-safe blood-pressure calendar grid."""

    day_index: int
    observed: bool


@dataclass(frozen=True)
class TrendEstimate:
    """One descriptive 30-day trend estimate with an HC3 interval."""

    slope_per_30_days: float
    ci95_low_per_30_days: float
    ci95_high_per_30_days: float
    p_value: float


def _optional_float(value: str, field: str) -> float | None:
    """Parse an optional finite float from a public aggregate CSV cell."""

    text = value.strip()
    if not text:
        return None
    parsed = float(text)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite when present.")
    return parsed


def _validate_count_and_mean(
    *,
    count: int,
    mean: float | None,
    count_field: str,
    mean_field: str,
) -> None:
    """Require aggregate count/mean cells to be internally consistent."""

    if count < 0:
        raise ValueError(f"{count_field} must be non-negative.")
    if count == 0 and mean is not None:
        raise ValueError(f"{mean_field} must be blank when {count_field} is zero.")
    if count > 0 and mean is None:
        raise ValueError(f"{mean_field} is required when {count_field} is positive.")


def load_pulse_oximeter_days(path: Path) -> list[PulseOximeterDay]:
    """Load and validate the privacy-safe daily pulse-oximeter snapshot."""

    required = {
        "day_index",
        "n_spo2_readings",
        "mean_spo2_percent",
        "n_bpm_spo2_readings",
        "mean_bpm_spo2",
        "n_paired_bpm_readings",
        "mean_bpm_difference",
    }
    rows: list[PulseOximeterDay] = []
    seen_day_indices: set[int] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError("Pulse-oximeter daily snapshot schema is invalid.")

        for raw in reader:
            day_index = int(raw["day_index"])
            if day_index < 0:
                raise ValueError("day_index must be non-negative.")
            if day_index in seen_day_indices:
                raise ValueError("Pulse-oximeter day_index values must be unique.")
            seen_day_indices.add(day_index)

            n_spo2 = int(raw["n_spo2_readings"])
            mean_spo2 = _optional_float(raw["mean_spo2_percent"], "mean_spo2_percent")
            n_bpm_spo2 = int(raw["n_bpm_spo2_readings"])
            mean_bpm_spo2 = _optional_float(raw["mean_bpm_spo2"], "mean_bpm_spo2")
            n_paired = int(raw["n_paired_bpm_readings"])
            mean_difference = _optional_float(
                raw["mean_bpm_difference"],
                "mean_bpm_difference",
            )

            _validate_count_and_mean(
                count=n_spo2,
                mean=mean_spo2,
                count_field="n_spo2_readings",
                mean_field="mean_spo2_percent",
            )
            _validate_count_and_mean(
                count=n_bpm_spo2,
                mean=mean_bpm_spo2,
                count_field="n_bpm_spo2_readings",
                mean_field="mean_bpm_spo2",
            )
            _validate_count_and_mean(
                count=n_paired,
                mean=mean_difference,
                count_field="n_paired_bpm_readings",
                mean_field="mean_bpm_difference",
            )

            if mean_spo2 is not None and not 0.0 <= mean_spo2 <= 100.0:
                raise ValueError("mean_spo2_percent must be between 0 and 100.")
            if mean_bpm_spo2 is not None and mean_bpm_spo2 <= 0.0:
                raise ValueError("mean_bpm_spo2 must be positive.")
            if n_paired > n_bpm_spo2:
                raise ValueError(
                    "n_paired_bpm_readings cannot exceed n_bpm_spo2_readings."
                )
            if n_spo2 == 0 and n_bpm_spo2 == 0:
                raise ValueError(
                    "Every pulse-oximeter snapshot row must contain at least one device value."
                )

            rows.append(
                PulseOximeterDay(
                    day_index=day_index,
                    n_spo2_readings=n_spo2,
                    mean_spo2_percent=mean_spo2,
                    n_bpm_spo2_readings=n_bpm_spo2,
                    mean_bpm_spo2=mean_bpm_spo2,
                    n_paired_bpm_readings=n_paired,
                    mean_bpm_difference=mean_difference,
                )
            )

    return sorted(rows, key=lambda item: item.day_index)


def load_blood_pressure_calendar(path: Path) -> list[BloodPressureCalendarDay]:
    """Load the established blood-pressure calendar grid for coverage denominators."""

    rows: list[BloodPressureCalendarDay] = []
    seen_day_indices: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"day_index", "observed"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Blood-pressure snapshot is missing calendar columns.")

        for raw in reader:
            day_index = int(raw["day_index"])
            if day_index < 0:
                raise ValueError("Blood-pressure day_index must be non-negative.")
            if day_index in seen_day_indices:
                raise ValueError("Blood-pressure day_index values must be unique.")
            seen_day_indices.add(day_index)

            observed_raw = int(raw["observed"])
            if observed_raw not in (0, 1):
                raise ValueError("Blood-pressure observed must be 0 or 1.")
            rows.append(
                BloodPressureCalendarDay(
                    day_index=day_index,
                    observed=bool(observed_raw),
                )
            )

    if not rows:
        raise ValueError("Blood-pressure calendar snapshot is empty.")

    rows.sort(key=lambda item: item.day_index)
    expected = list(range(rows[-1].day_index + 1))
    actual = [item.day_index for item in rows]
    if actual != expected:
        raise ValueError("Blood-pressure calendar grid must be contiguous from day zero.")
    return rows


def _trend_estimate(fitted: object) -> TrendEstimate:
    """Convert a robust statsmodels fit to the common 30-day representation."""

    params = np.asarray(getattr(fitted, "params"), dtype=float)
    pvalues = np.asarray(getattr(fitted, "pvalues"), dtype=float)
    interval = np.asarray(getattr(fitted, "conf_int")(alpha=0.05), dtype=float)
    return TrendEstimate(
        slope_per_30_days=float(params[1] * DAYS_PER_REPORTING_PERIOD),
        ci95_low_per_30_days=float(interval[1, 0] * DAYS_PER_REPORTING_PERIOD),
        ci95_high_per_30_days=float(interval[1, 1] * DAYS_PER_REPORTING_PERIOD),
        p_value=float(pvalues[1]),
    )


def _fit_time_association(
    *,
    day_indices: Sequence[int],
    values: Sequence[float],
    weights: Sequence[int],
) -> dict[str, Any]:
    """Fit equal-day and count-weighted HC3 trends when support is sufficient."""

    n_days = len(values)
    if not (len(day_indices) == n_days == len(weights)):
        raise ValueError("Trend inputs must have the same length.")
    if n_days < MIN_TREND_DAYS:
        return {
            "estimable": False,
            "n_days": n_days,
            "minimum_days_required": MIN_TREND_DAYS,
            "reason": "insufficient_observed_days",
            "equal_day": None,
            "reading_count_weighted": None,
        }
    if len(set(day_indices)) < 2:
        return {
            "estimable": False,
            "n_days": n_days,
            "minimum_days_required": MIN_TREND_DAYS,
            "reason": "no_time_variation",
            "equal_day": None,
            "reading_count_weighted": None,
        }
    if any(weight <= 0 for weight in weights):
        raise ValueError("Trend weights must be positive.")

    t = np.asarray(day_indices, dtype=float)
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    design = sm.add_constant(t)

    equal_day = sm.OLS(y, design).fit(cov_type="HC3")
    weighted = sm.WLS(y, design, weights=w).fit(cov_type="HC3")

    return {
        "estimable": True,
        "n_days": n_days,
        "minimum_days_required": MIN_TREND_DAYS,
        "reason": None,
        "equal_day": _trend_estimate(equal_day).__dict__,
        "reading_count_weighted": _trend_estimate(weighted).__dict__,
    }


def build_longitudinal_diagnostics(
    pulse_days: Sequence[PulseOximeterDay],
    bp_calendar: Sequence[BloodPressureCalendarDay],
) -> dict[str, Any]:
    """Build coverage-aware descriptive longitudinal pulse-oximeter diagnostics."""

    if not bp_calendar:
        raise ValueError("Blood-pressure calendar cannot be empty.")

    bp_day_indices = {item.day_index for item in bp_calendar}
    pulse_day_indices = [item.day_index for item in pulse_days]
    if any(day_index not in bp_day_indices for day_index in pulse_day_indices):
        raise ValueError("Pulse-oximeter days must lie inside the blood-pressure calendar.")

    n_calendar_days = len(bp_calendar)
    n_bp_observed_days = sum(item.observed for item in bp_calendar)
    n_pulse_days = len(pulse_days)

    if pulse_days:
        first_pulse_day = pulse_days[0].day_index
        last_pulse_day = pulse_days[-1].day_index
        pulse_span_days = last_pulse_day - first_pulse_day + 1
        pulse_window_coverage = n_pulse_days / pulse_span_days
    else:
        first_pulse_day = None
        last_pulse_day = None
        pulse_span_days = 0
        pulse_window_coverage = 0.0

    coverage = {
        "blood_pressure_calendar_days": n_calendar_days,
        "blood_pressure_observed_days": n_bp_observed_days,
        "pulse_oximeter_observed_days": n_pulse_days,
        "first_pulse_oximeter_day_index": first_pulse_day,
        "last_pulse_oximeter_day_index": last_pulse_day,
        "pulse_oximeter_span_days": pulse_span_days,
        "pulse_oximeter_coverage_of_full_calendar": round(
            n_pulse_days / n_calendar_days,
            8,
        ),
        "pulse_oximeter_coverage_of_bp_observed_days": (
            round(n_pulse_days / n_bp_observed_days, 8)
            if n_bp_observed_days
            else 0.0
        ),
        "pulse_oximeter_coverage_within_its_observed_span": round(
            pulse_window_coverage,
            8,
        ),
        "total_spo2_readings": sum(item.n_spo2_readings for item in pulse_days),
        "total_bpm_spo2_readings": sum(
            item.n_bpm_spo2_readings for item in pulse_days
        ),
        "total_paired_bpm_readings": sum(
            item.n_paired_bpm_readings for item in pulse_days
        ),
    }

    spo2_days = [item for item in pulse_days if item.mean_spo2_percent is not None]
    spo2_association = _fit_time_association(
        day_indices=[item.day_index for item in spo2_days],
        values=[float(item.mean_spo2_percent) for item in spo2_days],
        weights=[item.n_spo2_readings for item in spo2_days],
    )

    paired_days = [
        item for item in pulse_days if item.mean_bpm_difference is not None
    ]
    bpm_difference_association = _fit_time_association(
        day_indices=[item.day_index for item in paired_days],
        values=[float(item.mean_bpm_difference) for item in paired_days],
        weights=[item.n_paired_bpm_readings for item in paired_days],
    )

    return {
        "coverage": coverage,
        "spo2_time_association": spo2_association,
        "bpm_device_difference_time_association": bpm_difference_association,
        "interpretation": {
            "scope": "descriptive_longitudinal_diagnostics",
            "coverage_must_be_considered_with_time_associations": True,
            "missingness_mechanism_identified": False,
            "clinical_thresholds_applied": False,
            "causal_interpretation": False,
        },
    }


def write_results(path: Path, results: dict[str, Any]) -> None:
    """Write aggregate longitudinal diagnostics as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pulse-data",
        type=Path,
        default=Path("data/pulse_oximeter_daily_snapshot.csv"),
        help="Privacy-safe daily pulse-oximeter aggregate snapshot.",
    )
    parser.add_argument(
        "--bp-data",
        type=Path,
        default=Path("data/analysis_snapshot.csv"),
        help="Established blood-pressure calendar snapshot used for coverage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pulse_oximeter_longitudinal_diagnostics.json"),
        help="Path receiving aggregate longitudinal diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    """Load public aggregate inputs and write coverage-aware diagnostics."""

    args = parse_args()
    pulse_days = load_pulse_oximeter_days(args.pulse_data)
    bp_calendar = load_blood_pressure_calendar(args.bp_data)
    results = build_longitudinal_diagnostics(pulse_days, bp_calendar)
    write_results(args.output, results)
    coverage = results["coverage"]
    print(
        "Wrote pulse-oximeter longitudinal diagnostics: "
        f"{coverage['pulse_oximeter_observed_days']} pulse-oximeter days across "
        f"{coverage['blood_pressure_calendar_days']} calendar days."
    )


if __name__ == "__main__":
    main()
