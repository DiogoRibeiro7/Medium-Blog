"""Regression tests for temperature-covariate sensitivity analysis."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import temperature_covariate as primary
import temperature_covariate_sensitivity as sensitivity
from refresh_from_google_sheets import Measurement


class TemperatureCovariateSensitivityTests(unittest.TestCase):
    """Protect interpolation, weighting, functional form, and privacy behavior."""

    def _measurement(
        self,
        *,
        day: date,
        hour: int,
        minute: int,
        systolic: float,
        diastolic: float,
        bpm: float = 70.0,
    ) -> Measurement:
        return Measurement(
            day=day,
            timestamp=datetime(day.year, day.month, day.day, hour, minute),
            systolic=systolic,
            diastolic=diastolic,
            pulse_pressure=systolic - diastolic,
            bpm=bpm,
            pill=None,
            home=None,
            sleep=None,
            meal=None,
            symptoms=None,
        )

    def test_interpolation_uses_surrounding_hour_values(self) -> None:
        """Half-hour timestamps should interpolate halfway between hourly values."""

        tz = primary.FIANES_TIMEZONE
        series = primary.WeatherSeries(
            timestamps=(
                datetime(2026, 7, 4, 11, 0, tzinfo=tz),
                datetime(2026, 7, 4, 12, 0, tzinfo=tz),
            ),
            temperature_2m_c=(18.0, 22.0),
        )
        value = sensitivity.interpolate_temperature(
            series, datetime(2026, 7, 4, 11, 30)
        )
        self.assertAlmostEqual(value, 20.0)

    def test_interpolation_rejects_missing_bracket(self) -> None:
        """Interpolation must not extrapolate beyond the weather series."""

        tz = primary.FIANES_TIMEZONE
        series = primary.WeatherSeries(
            timestamps=(
                datetime(2026, 7, 4, 11, 0, tzinfo=tz),
                datetime(2026, 7, 4, 12, 0, tzinfo=tz),
            ),
            temperature_2m_c=(18.0, 22.0),
        )
        with self.assertRaisesRegex(ValueError, "both sides"):
            sensitivity.interpolate_temperature(
                series, datetime(2026, 7, 4, 10, 30)
            )

    def test_reading_weighted_model_changes_when_counts_are_informative(self) -> None:
        """Reading-count weighting must actually alter the fitted estimating equation."""

        records: list[primary.DailyTemperatureRecord] = []
        temperatures = [16.0, 18.0, 17.0, 21.0, 19.0, 23.0]
        counts = [1, 1, 1, 1, 1, 20]
        for day_index, (temp_c, count) in enumerate(
            zip(temperatures, counts, strict=True)
        ):
            systolic = 110.0 + 0.3 * day_index + 1.2 * (temp_c - 19.0)
            if day_index == 5:
                systolic += 8.0
            records.append(
                primary.DailyTemperatureRecord(
                    day_index=day_index,
                    n_readings=count,
                    mean_temperature_2m_c=temp_c,
                    mean_systolic_mmHg=systolic,
                    mean_diastolic_mmHg=70.0 + 0.1 * day_index,
                    mean_pulse_pressure_mmHg=systolic - (70.0 + 0.1 * day_index),
                    mean_bpm=65.0 + 0.05 * day_index,
                )
            )

        equal_fit = sensitivity._fit_linear(
            records, "mean_systolic_mmHg", weighted=False
        )
        weighted_fit = sensitivity._fit_linear(
            records, "mean_systolic_mmHg", weighted=True
        )
        self.assertNotAlmostEqual(
            equal_fit["temperature_coefficient_per_c"],
            weighted_fit["temperature_coefficient_per_c"],
        )

    def test_quadratic_model_recovers_curvature(self) -> None:
        """Quadratic sensitivity fit should recover an exact synthetic curvature."""

        records: list[primary.DailyTemperatureRecord] = []
        temperatures = [15.0, 19.0, 17.0, 22.0, 18.0, 24.0, 16.0, 21.0]
        mean_temperature = sum(temperatures) / len(temperatures)
        for day_index, temp_c in enumerate(temperatures):
            centered = temp_c - mean_temperature
            systolic = 120.0 + 0.2 * day_index + 0.7 * centered + 0.4 * centered**2
            records.append(
                primary.DailyTemperatureRecord(
                    day_index=day_index,
                    n_readings=1,
                    mean_temperature_2m_c=temp_c,
                    mean_systolic_mmHg=systolic,
                    mean_diastolic_mmHg=75.0 + 0.05 * day_index,
                    mean_pulse_pressure_mmHg=systolic - (75.0 + 0.05 * day_index),
                    mean_bpm=68.0 + 0.03 * day_index,
                )
            )

        fitted = sensitivity._fit_quadratic(records, "mean_systolic_mmHg")
        self.assertAlmostEqual(
            fitted["linear_temperature_coefficient_per_c_at_mean"], 0.7, places=9
        )
        self.assertAlmostEqual(
            fitted["quadratic_temperature_coefficient_per_c2"], 0.4, places=9
        )
        self.assertAlmostEqual(fitted["day_slope_per_30_days"], 6.0, places=9)

    def test_summary_does_not_expose_calendar_or_temperature_sequence(self) -> None:
        """Sensitivity output must remain aggregate and non-date-indexed."""

        start = date(2026, 7, 4)
        measurements: list[Measurement] = []
        weather_times: list[datetime] = []
        weather_temperatures: list[float] = []
        day_temperatures = [18.0, 20.0, 17.0, 23.0, 19.0, 22.0]
        tz = primary.FIANES_TIMEZONE

        for index, base_temp in enumerate(day_temperatures):
            day_value = start + timedelta(days=index)
            measurements.append(
                self._measurement(
                    day=day_value,
                    hour=11,
                    minute=20,
                    systolic=120.0 + index,
                    diastolic=80.0 + 0.2 * index,
                    bpm=70.0 + 0.1 * index,
                )
            )
            for hour, temp in ((11, base_temp), (12, base_temp + 2.0)):
                weather_times.append(
                    datetime(
                        day_value.year,
                        day_value.month,
                        day_value.day,
                        hour,
                        0,
                        tzinfo=tz,
                    )
                )
                weather_temperatures.append(temp)

        weather = primary.WeatherSeries(
            timestamps=tuple(weather_times),
            temperature_2m_c=tuple(weather_temperatures),
        )
        summary = sensitivity.build_sensitivity_summary(measurements, weather)
        serialized = json.dumps(summary)

        self.assertNotIn("2026-07-", serialized)
        privacy = summary["privacy"]
        self.assertIsInstance(privacy, dict)
        self.assertFalse(privacy["calendar_dates_written"])
        self.assertFalse(privacy["measurement_times_written"])
        self.assertFalse(privacy["matched_temperature_series_written"])
        self.assertFalse(privacy["interpolated_temperature_series_written"])


if __name__ == "__main__":
    unittest.main()
