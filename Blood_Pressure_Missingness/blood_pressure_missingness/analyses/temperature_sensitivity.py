"""Sensitivity analysis for the time-matched Fiães temperature covariate.

The primary temperature model uses nearest-hour matching, equal weighting across
observed days, and a linear temperature term. This module perturbs those choices
without persisting the private date-temperature join.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import statsmodels.api as sm

from blood_pressure_missingness.analyses import temperature as primary
from blood_pressure_missingness.data_sources import google_sheets as source


@dataclass(frozen=True)
class SensitivityDailyRecord:
    """One private daily record used only while fitting sensitivity models."""

    day_index: int
    n_readings: int
    mean_temperature_2m_c: float
    mean_systolic_mmHg: float
    mean_diastolic_mmHg: float
    mean_pulse_pressure_mmHg: float
    mean_bpm: float


def interpolate_temperature(
    series: primary.WeatherSeries,
    local_timestamp: datetime,
) -> float:
    """Linearly interpolate temperature between surrounding hourly timestamps."""

    if local_timestamp.tzinfo is None:
        target = local_timestamp.replace(tzinfo=primary.FIANES_TIMEZONE)
    else:
        target = local_timestamp.astimezone(primary.FIANES_TIMEZONE)

    insertion = bisect.bisect_left(series.timestamps, target)
    if insertion < len(series.timestamps) and series.timestamps[insertion] == target:
        return series.temperature_2m_c[insertion]
    if insertion == 0 or insertion >= len(series.timestamps):
        raise ValueError("Interpolation requires weather observations on both sides.")

    left_index = insertion - 1
    right_index = insertion
    left_time = series.timestamps[left_index]
    right_time = series.timestamps[right_index]
    total_seconds = (right_time - left_time).total_seconds()
    if total_seconds <= 0.0:
        raise ValueError("Weather timestamps must be strictly increasing.")

    left_gap = (target - left_time).total_seconds()
    right_gap = (right_time - target).total_seconds()
    if max(left_gap, right_gap) > primary.MAX_MATCH_OFFSET_MINUTES * 60.0:
        raise ValueError("Interpolation neighbours are too far from the measurement.")

    weight = left_gap / total_seconds
    left_temperature = series.temperature_2m_c[left_index]
    right_temperature = series.temperature_2m_c[right_index]
    return float(left_temperature + weight * (right_temperature - left_temperature))


def build_interpolated_daily_records(
    measurements: Sequence[source.Measurement],
    weather: primary.WeatherSeries,
) -> list[SensitivityDailyRecord]:
    """Build daily records using linearly interpolated reading-time temperatures."""

    if not measurements:
        raise ValueError("At least one blood-pressure measurement is required.")

    grouped: defaultdict[date, list[tuple[source.Measurement, float]]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.day].append(
            (measurement, interpolate_temperature(weather, measurement.timestamp))
        )

    first_day = min(grouped)
    records: list[SensitivityDailyRecord] = []
    for day_value in sorted(grouped):
        rows = grouped[day_value]
        readings = [measurement for measurement, _ in rows]
        temperatures = [temperature for _, temperature in rows]
        records.append(
            SensitivityDailyRecord(
                day_index=(day_value - first_day).days,
                n_readings=len(rows),
                mean_temperature_2m_c=statistics.fmean(temperatures),
                mean_systolic_mmHg=statistics.fmean(row.systolic for row in readings),
                mean_diastolic_mmHg=statistics.fmean(row.diastolic for row in readings),
                mean_pulse_pressure_mmHg=statistics.fmean(
                    row.pulse_pressure for row in readings
                ),
                mean_bpm=statistics.fmean(row.bpm for row in readings),
            )
        )
    return records


def _arrays(
    records: Sequence[primary.DailyTemperatureRecord | SensitivityDailyRecord],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return day, centered temperature, outcome, and reading-count arrays."""

    if metric_name not in primary.METRIC_NAMES:
        raise ValueError(f"Unsupported metric: {metric_name}")
    if len(records) < 4:
        raise ValueError("At least four observed days are required.")

    day = np.asarray([record.day_index for record in records], dtype=float)
    temperature = np.asarray(
        [record.mean_temperature_2m_c for record in records], dtype=float
    )
    outcome = np.asarray([getattr(record, metric_name) for record in records], dtype=float)
    counts = np.asarray([record.n_readings for record in records], dtype=float)
    if not all(np.all(np.isfinite(values)) for values in (day, temperature, outcome, counts)):
        raise ValueError("Sensitivity model inputs must be finite.")
    if np.any(counts <= 0.0):
        raise ValueError("Reading counts must be positive.")
    return day, temperature - float(np.mean(temperature)), outcome, counts


def _fit_linear(
    records: Sequence[primary.DailyTemperatureRecord | SensitivityDailyRecord],
    metric_name: str,
    *,
    weighted: bool,
) -> dict[str, float]:
    """Fit a linear temperature model with optional reading-count weights."""

    day, centered_temperature, outcome, counts = _arrays(records, metric_name)
    design = np.column_stack(
        [np.ones(len(records), dtype=float), day, centered_temperature]
    )
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Sensitivity design matrix is rank deficient.")

    model = sm.WLS(outcome, design, weights=counts) if weighted else sm.OLS(outcome, design)
    fitted = model.fit(cov_type="HC3")
    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
    return {
        "day_slope_per_30_days": float(fitted.params[1] * 30.0),
        "day_ci95_low_per_30_days": float(interval[1, 0] * 30.0),
        "day_ci95_high_per_30_days": float(interval[1, 1] * 30.0),
        "temperature_coefficient_per_c": float(fitted.params[2]),
        "temperature_ci95_low_per_c": float(interval[2, 0]),
        "temperature_ci95_high_per_c": float(interval[2, 1]),
        "r_squared": float(fitted.rsquared),
    }


def _fit_quadratic(
    records: Sequence[primary.DailyTemperatureRecord],
    metric_name: str,
) -> dict[str, float]:
    """Fit a quadratic temperature sensitivity model with equal day weights."""

    day, centered_temperature, outcome, _ = _arrays(records, metric_name)
    squared_temperature = centered_temperature**2
    design = np.column_stack(
        [
            np.ones(len(records), dtype=float),
            day,
            centered_temperature,
            squared_temperature,
        ]
    )
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Quadratic sensitivity design matrix is rank deficient.")

    fitted = sm.OLS(outcome, design).fit(cov_type="HC3")
    interval = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
    return {
        "day_slope_per_30_days": float(fitted.params[1] * 30.0),
        "linear_temperature_coefficient_per_c_at_mean": float(fitted.params[2]),
        "linear_temperature_ci95_low_per_c_at_mean": float(interval[2, 0]),
        "linear_temperature_ci95_high_per_c_at_mean": float(interval[2, 1]),
        "quadratic_temperature_coefficient_per_c2": float(fitted.params[3]),
        "quadratic_temperature_ci95_low_per_c2": float(interval[3, 0]),
        "quadratic_temperature_ci95_high_per_c2": float(interval[3, 1]),
        "r_squared": float(fitted.rsquared),
    }


def build_sensitivity_summary(
    measurements: Sequence[source.Measurement],
    weather: primary.WeatherSeries,
) -> dict[str, object]:
    """Return aggregate sensitivity results without exposing calendar alignment."""

    nearest_records, _ = primary.build_daily_records(measurements, weather)
    interpolated_records = build_interpolated_daily_records(measurements, weather)

    metric_results: dict[str, object] = {}
    for metric_name in primary.METRIC_NAMES:
        primary_fit = primary.fit_temperature_adjusted_model(nearest_records, metric_name)
        interpolated_fit = _fit_linear(
            interpolated_records, metric_name, weighted=False
        )
        reading_weighted_fit = _fit_linear(
            nearest_records, metric_name, weighted=True
        )
        quadratic_fit = _fit_quadratic(nearest_records, metric_name)

        temperature_signs = [
            math.copysign(1.0, coefficient)
            for coefficient in (
                primary_fit["temperature_coefficient_per_c"],
                interpolated_fit["temperature_coefficient_per_c"],
                reading_weighted_fit["temperature_coefficient_per_c"],
                quadratic_fit["linear_temperature_coefficient_per_c_at_mean"],
            )
            if coefficient != 0.0
        ]
        day_signs = [
            math.copysign(1.0, coefficient)
            for coefficient in (
                primary_fit["day_slope_per_30_days"],
                interpolated_fit["day_slope_per_30_days"],
                reading_weighted_fit["day_slope_per_30_days"],
                quadratic_fit["day_slope_per_30_days"],
            )
            if coefficient != 0.0
        ]
        metric_results[metric_name] = {
            "primary_nearest_hour_equal_day_linear": primary_fit,
            "linear_interpolation_equal_day_linear": interpolated_fit,
            "nearest_hour_reading_weighted_linear": reading_weighted_fit,
            "nearest_hour_equal_day_quadratic": quadratic_fit,
            "robustness": {
                "temperature_direction_consistent_across_specs": (
                    len(set(temperature_signs)) <= 1
                ),
                "day_trend_direction_consistent_across_specs": len(set(day_signs)) <= 1,
            },
        }

    return {
        "purpose": (
            "Assess sensitivity of the time and temperature associations to "
            "weather matching, day weighting, and temperature functional form."
        ),
        "specifications": {
            "primary": "nearest-hour match; equal observed-day weight; linear temperature",
            "interpolation": (
                "linear interpolation between surrounding weather hours; "
                "equal observed-day weight; linear temperature"
            ),
            "reading_weighted": (
                "nearest-hour match; observed days weighted by number of readings; "
                "linear temperature"
            ),
            "quadratic": (
                "nearest-hour match; equal observed-day weight; linear and squared "
                "centered temperature"
            ),
            "covariance": "HC3 heteroskedasticity-robust for every specification",
        },
        "metrics": metric_results,
        "privacy": {
            "calendar_dates_written": False,
            "measurement_times_written": False,
            "matched_temperature_series_written": False,
            "interpolated_temperature_series_written": False,
            "note": (
                "All alternative temperature joins are computed only while private "
                "measurement timestamps are in memory."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/temperature_covariate_sensitivity.json"),
        help="Path receiving privacy-safe aggregate sensitivity diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    """Read private measurements, fetch weather once, and write sensitivities."""

    args = parse_args()
    values = source.fetch_sheet_values()
    measurements, _ = source.parse_measurements(values)
    start_day = min(measurement.day for measurement in measurements)
    end_day = max(measurement.day for measurement in measurements)
    weather = primary.fetch_hourly_temperature(start_day, end_day)
    summary = build_sensitivity_summary(measurements, weather)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Generated privacy-safe temperature-covariate sensitivity summary.")


if __name__ == "__main__":
    main()