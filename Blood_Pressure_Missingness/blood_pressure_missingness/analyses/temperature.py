"""Fit privacy-preserving, time-matched temperature covariate models.

This module reads the private blood-pressure Google Sheet in memory, retrieves
hourly 2 m air temperature for Fiães, Santa Maria da Feira, Portugal, and fits
exploratory day-level models adjusted for measurement-time temperature.

Only aggregate model diagnostics are written. Row-level timestamps, calendar
dates, and the matched temperature series are deliberately not persisted.
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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import statsmodels.api as sm

import refresh_from_google_sheets as source

FIANES_LATITUDE = 40.994459
FIANES_LONGITUDE = -8.525370
FIANES_LABEL = "Fiães, Santa Maria da Feira, Portugal"
FIANES_TIMEZONE_NAME = "Europe/Lisbon"
FIANES_TIMEZONE = ZoneInfo(FIANES_TIMEZONE_NAME)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_MODEL = "ecmwf_ifs"
OPEN_METEO_VARIABLE = "temperature_2m"
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_MATCH_OFFSET_MINUTES = 60.0

METRIC_NAMES: tuple[str, ...] = (
    "mean_systolic_mmHg",
    "mean_diastolic_mmHg",
    "mean_pulse_pressure_mmHg",
    "mean_bpm",
)


@dataclass(frozen=True)
class WeatherSeries:
    """Validated hourly temperature series with timezone-aware timestamps."""

    timestamps: tuple[datetime, ...]
    temperature_2m_c: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate ordering and one-to-one timestamp/value alignment."""

        if not self.timestamps:
            raise ValueError("Weather series is empty.")
        if len(self.timestamps) != len(self.temperature_2m_c):
            raise ValueError(
                "Weather timestamps and temperatures have different lengths."
            )
        if any(
            current <= previous
            for previous, current in zip(self.timestamps, self.timestamps[1:])
        ):
            raise ValueError("Weather timestamps must be strictly increasing.")
        if any(timestamp.tzinfo is None for timestamp in self.timestamps):
            raise ValueError("Weather timestamps must be timezone-aware.")
        if any(not math.isfinite(value) for value in self.temperature_2m_c):
            raise ValueError("Weather temperatures must be finite.")


@dataclass(frozen=True)
class TemperatureMatch:
    """One nearest-hour temperature match for a private measurement."""

    temperature_c: float
    offset_minutes: float


@dataclass(frozen=True)
class DailyTemperatureRecord:
    """One observed day used only in memory for covariate modelling."""

    day_index: int
    n_readings: int
    mean_temperature_2m_c: float
    mean_systolic_mmHg: float
    mean_diastolic_mmHg: float
    mean_pulse_pressure_mmHg: float
    mean_bpm: float


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    """Return a JSON object after a runtime type check."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _parse_open_meteo_payload(payload: object) -> WeatherSeries:
    """Validate an Open-Meteo response and return an hourly weather series."""

    root = _as_mapping(payload, "Open-Meteo response")
    if root.get("error") is True:
        reason = root.get("reason", "unspecified Open-Meteo error")
        raise RuntimeError(f"Open-Meteo returned an error: {reason}")

    hourly = _as_mapping(root.get("hourly"), "Open-Meteo hourly payload")
    raw_times = hourly.get("time")
    raw_temperatures = hourly.get(OPEN_METEO_VARIABLE)
    if not isinstance(raw_times, list) or not isinstance(raw_temperatures, list):
        raise ValueError("Open-Meteo hourly arrays are missing or malformed.")
    if len(raw_times) != len(raw_temperatures):
        raise ValueError("Open-Meteo hourly arrays have different lengths.")

    timestamps: list[datetime] = []
    temperatures: list[float] = []
    for index, (raw_time, raw_temperature) in enumerate(
        zip(raw_times, raw_temperatures, strict=True)
    ):
        if isinstance(raw_time, bool) or not isinstance(raw_time, (int, float)):
            raise ValueError(f"Open-Meteo time[{index}] is not a Unix timestamp.")
        if isinstance(raw_temperature, bool) or not isinstance(
            raw_temperature, (int, float)
        ):
            raise ValueError(f"Open-Meteo temperature[{index}] is not numeric.")

        temperature_c = float(raw_temperature)
        if not math.isfinite(temperature_c):
            raise ValueError(f"Open-Meteo temperature[{index}] must be finite.")

        timestamp_utc = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
        timestamps.append(timestamp_utc.astimezone(FIANES_TIMEZONE))
        temperatures.append(temperature_c)

    return WeatherSeries(tuple(timestamps), tuple(temperatures))


def _read_json_url(url: str) -> Mapping[str, Any]:
    """Fetch one JSON object over HTTPS with an explicit timeout."""

    request = Request(
        url,
        headers={"User-Agent": "Medium-Blog/blood-pressure-temperature-covariate"},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Could not retrieve historical temperature data.") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Open-Meteo returned invalid JSON.") from exc
    return _as_mapping(payload, "Open-Meteo response")


def fetch_hourly_temperature(start_day: date, end_day: date) -> WeatherSeries:
    """Fetch hourly 2 m temperature for Fiães over an inclusive date interval."""

    if end_day < start_day:
        raise ValueError("end_day must not precede start_day.")

    query = urlencode(
        {
            "latitude": f"{FIANES_LATITUDE:.6f}",
            "longitude": f"{FIANES_LONGITUDE:.6f}",
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "hourly": OPEN_METEO_VARIABLE,
            "temperature_unit": "celsius",
            "timeformat": "unixtime",
            "timezone": FIANES_TIMEZONE_NAME,
            "models": OPEN_METEO_MODEL,
            "cell_selection": "land",
        }
    )
    url = f"{OPEN_METEO_ARCHIVE_URL}?{query}"
    return _parse_open_meteo_payload(_read_json_url(url))


def match_temperature(
    series: WeatherSeries, local_timestamp: datetime
) -> TemperatureMatch:
    """Match a local measurement timestamp to the nearest available hourly value."""

    if local_timestamp.tzinfo is None:
        target = local_timestamp.replace(tzinfo=FIANES_TIMEZONE)
    else:
        target = local_timestamp.astimezone(FIANES_TIMEZONE)

    insertion = bisect.bisect_left(series.timestamps, target)
    candidate_indices = {
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(series.timestamps)
    }
    if not candidate_indices:
        raise ValueError("No weather timestamp can be matched to the measurement.")

    best_index = min(
        candidate_indices,
        key=lambda index: abs((series.timestamps[index] - target).total_seconds()),
    )
    offset_minutes = abs(
        (series.timestamps[best_index] - target).total_seconds()
    ) / 60.0
    if offset_minutes > MAX_MATCH_OFFSET_MINUTES:
        raise ValueError(
            "Nearest weather observation is more than "
            f"{MAX_MATCH_OFFSET_MINUTES:g} minutes from a measurement."
        )

    return TemperatureMatch(
        temperature_c=series.temperature_2m_c[best_index],
        offset_minutes=offset_minutes,
    )


def build_daily_records(
    measurements: Sequence[source.Measurement],
    weather: WeatherSeries,
) -> tuple[list[DailyTemperatureRecord], list[float]]:
    """Build private daily analysis records from time-matched temperatures."""

    if not measurements:
        raise ValueError("At least one blood-pressure measurement is required.")

    by_day: defaultdict[date, list[tuple[source.Measurement, TemperatureMatch]]] = (
        defaultdict(list)
    )
    offsets: list[float] = []
    for measurement in measurements:
        matched = match_temperature(weather, measurement.timestamp)
        by_day[measurement.day].append((measurement, matched))
        offsets.append(matched.offset_minutes)

    first_day = min(by_day)
    records: list[DailyTemperatureRecord] = []
    for day_value in sorted(by_day):
        rows = by_day[day_value]
        readings = [measurement for measurement, _ in rows]
        matches = [match for _, match in rows]
        records.append(
            DailyTemperatureRecord(
                day_index=(day_value - first_day).days,
                n_readings=len(rows),
                mean_temperature_2m_c=statistics.fmean(
                    match.temperature_c for match in matches
                ),
                mean_systolic_mmHg=statistics.fmean(
                    measurement.systolic for measurement in readings
                ),
                mean_diastolic_mmHg=statistics.fmean(
                    measurement.diastolic for measurement in readings
                ),
                mean_pulse_pressure_mmHg=statistics.fmean(
                    measurement.pulse_pressure for measurement in readings
                ),
                mean_bpm=statistics.fmean(measurement.bpm for measurement in readings),
            )
        )

    return records, offsets


def _metric_array(
    records: Sequence[DailyTemperatureRecord],
    metric_name: str,
) -> np.ndarray:
    """Return one validated daily metric vector."""

    if metric_name not in METRIC_NAMES:
        raise ValueError(f"Unsupported metric: {metric_name}")
    values = np.asarray(
        [getattr(record, metric_name) for record in records], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{metric_name} contains non-finite values.")
    return values


def fit_temperature_adjusted_model(
    records: Sequence[DailyTemperatureRecord],
    metric_name: str,
) -> dict[str, float]:
    """Fit ``metric ~ day_index + centered temperature`` using HC3 standard errors."""

    if len(records) < 4:
        raise ValueError(
            "At least four observed days are required for covariate modelling."
        )

    day_index = np.asarray([record.day_index for record in records], dtype=float)
    temperature = np.asarray(
        [record.mean_temperature_2m_c for record in records], dtype=float
    )
    outcome = _metric_array(records, metric_name)
    centered_temperature = temperature - float(np.mean(temperature))

    design = np.column_stack(
        [
            np.ones(len(records), dtype=float),
            day_index,
            centered_temperature,
        ]
    )
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Temperature-adjusted design matrix is rank deficient.")

    fitted = sm.OLS(outcome, design).fit(cov_type="HC3")
    confidence = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)

    return {
        "day_slope_per_30_days": float(fitted.params[1] * 30.0),
        "day_ci95_low_per_30_days": float(confidence[1, 0] * 30.0),
        "day_ci95_high_per_30_days": float(confidence[1, 1] * 30.0),
        "day_p_value": float(fitted.pvalues[1]),
        "temperature_coefficient_per_c": float(fitted.params[2]),
        "temperature_ci95_low_per_c": float(confidence[2, 0]),
        "temperature_ci95_high_per_c": float(confidence[2, 1]),
        "temperature_p_value": float(fitted.pvalues[2]),
        "r_squared": float(fitted.rsquared),
    }


def build_summary(
    measurements: Sequence[source.Measurement],
    weather: WeatherSeries,
) -> dict[str, object]:
    """Build the aggregate, privacy-safe temperature-covariate result."""

    records, offsets = build_daily_records(measurements, weather)
    temperatures = np.asarray(
        [record.mean_temperature_2m_c for record in records], dtype=float
    )
    day_index = np.asarray([record.day_index for record in records], dtype=float)

    if len(records) < 2 or np.std(temperatures) == 0.0:
        day_temperature_correlation: float | None = None
    else:
        day_temperature_correlation = float(
            np.corrcoef(day_index, temperatures)[0, 1]
        )

    return {
        "weather_source": {
            "provider": "Open-Meteo Historical Weather API",
            "model": OPEN_METEO_MODEL,
            "variable": OPEN_METEO_VARIABLE,
            "unit": "degC",
            "location": FIANES_LABEL,
            "latitude": FIANES_LATITUDE,
            "longitude": FIANES_LONGITUDE,
            "timezone": FIANES_TIMEZONE_NAME,
            "spatial_interpretation": (
                "gridded model/reanalysis estimate; not an on-site thermometer"
            ),
        },
        "matching": {
            "policy": "nearest available hourly temperature to each measurement time",
            "max_allowed_offset_minutes": MAX_MATCH_OFFSET_MINUTES,
            "n_measurements_matched": len(offsets),
            "n_observed_days": len(records),
            "median_match_offset_minutes": float(np.median(offsets)),
            "max_match_offset_minutes": float(np.max(offsets)),
        },
        "temperature_summary": {
            "mean_of_observed_day_matched_temperature_c": float(np.mean(temperatures)),
            "std_of_observed_day_matched_temperature_c": float(
                np.std(temperatures, ddof=1)
            )
            if len(temperatures) > 1
            else 0.0,
            "min_of_observed_day_matched_temperature_c": float(np.min(temperatures)),
            "max_of_observed_day_matched_temperature_c": float(np.max(temperatures)),
            "correlation_with_day_index": day_temperature_correlation,
        },
        "model": {
            "formula": (
                "daily_metric ~ day_index + "
                "(mean_time_matched_temperature_2m_c - overall_mean_temperature)"
            ),
            "day_weighting": "equal weight per observed calendar day",
            "within_day_temperature": (
                "mean of temperatures matched to the actual measurement times"
            ),
            "covariance": "HC3 heteroskedasticity-robust",
            "metrics": {
                metric_name: fit_temperature_adjusted_model(records, metric_name)
                for metric_name in METRIC_NAMES
            },
        },
        "privacy": {
            "row_level_dates_written": False,
            "row_level_times_written": False,
            "row_level_temperatures_written": False,
            "day_indexed_temperature_series_written": False,
            "note": (
                "Temperature is joined to private timestamps only in memory. "
                "Only aggregate diagnostics and coefficients are persisted."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/temperature_covariate.json"),
        help="Path receiving aggregate temperature-covariate diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    """Read the private source, match weather, fit models, and write aggregates."""

    args = parse_args()
    values = source.fetch_sheet_values()
    measurements, _ = source.parse_measurements(values)

    start_day = min(measurement.day for measurement in measurements)
    end_day = max(measurement.day for measurement in measurements)
    weather = fetch_hourly_temperature(start_day, end_day)
    summary = build_summary(measurements, weather)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    matching = summary["matching"]
    if not isinstance(matching, Mapping):
        raise TypeError("matching summary must be a mapping.")
    print(
        "Generated privacy-safe temperature covariate summary: "
        f"{matching['n_measurements_matched']} matched readings across "
        f"{matching['n_observed_days']} observed days."
    )


if __name__ == "__main__":
    main()
