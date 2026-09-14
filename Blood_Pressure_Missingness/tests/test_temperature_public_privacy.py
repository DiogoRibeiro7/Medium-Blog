"""Focused privacy regression for the persisted temperature summary."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta

from blood_pressure_missingness.analyses import temperature
from blood_pressure_missingness.data_sources.google_sheets import Measurement


class TemperaturePublicPrivacyTests(unittest.TestCase):
    """Keep precise weather-query coordinates out of public diagnostics."""

    def test_public_weather_source_omits_coordinates(self) -> None:
        start = date(2026, 7, 4)
        measurements = [
            Measurement(
                day=start + timedelta(days=index),
                timestamp=datetime.combine(
                    start + timedelta(days=index), datetime.min.time()
                ).replace(hour=11, minute=10),
                systolic=120.0 + index,
                diastolic=80.0,
                pulse_pressure=40.0 + index,
                bpm=70.0,
                pill=None,
                home=None,
                sleep=None,
                meal=None,
                symptoms=None,
            )
            for index in range(5)
        ]
        weather = temperature.WeatherSeries(
            timestamps=tuple(
                datetime.combine(
                    start + timedelta(days=index), datetime.min.time()
                ).replace(hour=11, tzinfo=temperature.FIANES_TIMEZONE)
                for index in range(5)
            ),
            temperature_2m_c=(18.0, 20.0, 19.0, 22.0, 21.0),
        )

        summary = temperature.build_summary(measurements, weather)
        weather_source = summary["weather_source"]

        self.assertIsInstance(weather_source, dict)
        self.assertEqual(weather_source["location"], temperature.FIANES_LABEL)
        self.assertNotIn("latitude", weather_source)
        self.assertNotIn("longitude", weather_source)


if __name__ == "__main__":
    unittest.main()
