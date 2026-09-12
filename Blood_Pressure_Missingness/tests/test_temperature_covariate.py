"""Regression tests for the private temperature-covariate analysis."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone

from blood_pressure_missingness.analyses import temperature
from blood_pressure_missingness.data_sources.google_sheets import Measurement


class TemperatureCovariateTests(unittest.TestCase):
    """Protect weather parsing, time matching, modelling, and privacy invariants."""

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
        """Build one valid synthetic blood-pressure measurement."""

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

    def test_open_meteo_payload_parses_unix_time_in_lisbon_timezone(self) -> None:
        """Unix weather times must become timezone-aware local timestamps."""

        base = datetime(2026, 7, 4, 10, tzinfo=timezone.utc)
        payload = {
            "hourly": {
                "time": [
                    int(base.timestamp()),
                    int((base + timedelta(hours=1)).timestamp()),
                ],
                "temperature_2m": [20.0, 21.5],
            }
        }

        series = temperature._parse_open_meteo_payload(payload)

        self.assertEqual(len(series.timestamps), 2)
        self.assertIsNotNone(series.timestamps[0].tzinfo)
        self.assertEqual(series.timestamps[0].hour, 11)
        self.assertEqual(series.temperature_2m_c, (20.0, 21.5))

    def test_open_meteo_payload_rejects_misaligned_arrays(self) -> None:
        """Weather timestamps and temperatures must be one-to-one."""

        payload = {
            "hourly": {
                "time": [1_784_000_000, 1_784_003_600],
                "temperature_2m": [20.0],
            }
        }
        with self.assertRaisesRegex(ValueError, "different lengths"):
            temperature._parse_open_meteo_payload(payload)

    def test_nearest_hour_matching_uses_measurement_time(self) -> None:
        """The covariate must reflect clock time, not a daily temperature mean."""

        local_tz = temperature.FIANES_TIMEZONE
        series = temperature.WeatherSeries(
            timestamps=(
                datetime(2026, 7, 4, 11, 0, tzinfo=local_tz),
                datetime(2026, 7, 4, 12, 0, tzinfo=local_tz),
            ),
            temperature_2m_c=(19.0, 23.0),
        )

        before_half_hour = temperature.match_temperature(
            series, datetime(2026, 7, 4, 11, 20)
        )
        after_half_hour = temperature.match_temperature(
            series, datetime(2026, 7, 4, 11, 40)
        )

        self.assertEqual(before_half_hour.temperature_c, 19.0)
        self.assertAlmostEqual(before_half_hour.offset_minutes, 20.0)
        self.assertEqual(after_half_hour.temperature_c, 23.0)
        self.assertAlmostEqual(after_half_hour.offset_minutes, 20.0)

    def test_adjusted_model_recovers_linear_temperature_effect(self) -> None:
        """The HC3 model must use day and temperature as separate regressors."""

        records: list[temperature.DailyTemperatureRecord] = []
        for day_index in range(8):
            temp_c = 15.0 + (day_index % 3) * 2.0
            systolic = 110.0 + 0.2 * day_index + 1.5 * (temp_c - 17.0)
            records.append(
                temperature.DailyTemperatureRecord(
                    day_index=day_index,
                    n_readings=1,
                    mean_temperature_2m_c=temp_c,
                    mean_systolic_mmHg=systolic,
                    mean_diastolic_mmHg=70.0 + 0.1 * day_index,
                    mean_pulse_pressure_mmHg=systolic - (70.0 + 0.1 * day_index),
                    mean_bpm=65.0 + 0.05 * day_index,
                )
            )

        fitted = temperature.fit_temperature_adjusted_model(
            records, "mean_systolic_mmHg"
        )

        self.assertAlmostEqual(fitted["temperature_coefficient_per_c"], 1.5, places=10)
        self.assertAlmostEqual(fitted["day_slope_per_30_days"], 6.0, places=10)

    def test_public_summary_does_not_write_dates_or_temperature_series(self) -> None:
        """Persisted output must not expose the private date-temperature join."""

        start = date(2026, 7, 4)
        measurements = [
            self._measurement(
                day=start + timedelta(days=index),
                hour=11,
                minute=10,
                systolic=120.0 + index,
                diastolic=80.0 + 0.2 * index,
                bpm=70.0 + 0.5 * index,
            )
            for index in range(5)
        ]
        local_tz = temperature.FIANES_TIMEZONE
        weather = temperature.WeatherSeries(
            timestamps=tuple(
                datetime(
                    (start + timedelta(days=index)).year,
                    (start + timedelta(days=index)).month,
                    (start + timedelta(days=index)).day,
                    11,
                    0,
                    tzinfo=local_tz,
                )
                for index in range(5)
            ),
            temperature_2m_c=(18.0, 20.0, 19.0, 22.0, 21.0),
        )

        summary = temperature.build_summary(measurements, weather)
        serialized = __import__("json").dumps(summary)

        self.assertNotIn("2026-07-", serialized)
        privacy = summary["privacy"]
        self.assertIsInstance(privacy, dict)
        self.assertFalse(privacy["row_level_dates_written"])
        self.assertFalse(privacy["row_level_times_written"])
        self.assertFalse(privacy["row_level_temperatures_written"])
        self.assertFalse(privacy["day_indexed_temperature_series_written"])


if __name__ == "__main__":
    unittest.main()
