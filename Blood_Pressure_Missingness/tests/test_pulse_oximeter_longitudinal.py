"""Regression tests for coverage-aware pulse-oximeter longitudinal analysis."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from blood_pressure_missingness.analyses import pulse_oximeter_longitudinal as analysis


class PulseOximeterLongitudinalTests(unittest.TestCase):
    """Protect coverage accounting and conservative trend estimation."""

    def test_sparse_support_is_reported_not_fitted(self) -> None:
        """Too few pulse-oximeter days must not produce a nominal trend."""

        pulse_days = [
            analysis.PulseOximeterDay(10, 2, 98.0, 2, 70.0, 2, 1.0),
            analysis.PulseOximeterDay(12, 1, 97.0, 1, 72.0, 1, -1.0),
        ]
        calendar = [
            analysis.BloodPressureCalendarDay(day_index=index, observed=True)
            for index in range(15)
        ]

        results = analysis.build_longitudinal_diagnostics(pulse_days, calendar)

        self.assertFalse(results["spo2_time_association"]["estimable"])
        self.assertEqual(
            results["spo2_time_association"]["reason"],
            "insufficient_observed_days",
        )
        self.assertFalse(
            results["bpm_device_difference_time_association"]["estimable"]
        )
        self.assertEqual(results["coverage"]["pulse_oximeter_observed_days"], 2)
        self.assertEqual(results["coverage"]["first_pulse_oximeter_day_index"], 10)
        self.assertEqual(results["coverage"]["last_pulse_oximeter_day_index"], 12)
        self.assertEqual(results["coverage"]["pulse_oximeter_span_days"], 3)

    def test_four_days_produce_equal_day_and_weighted_trends(self) -> None:
        """Supported time series should expose both descriptive weighting choices."""

        pulse_days = [
            analysis.PulseOximeterDay(0, 1, 96.0, 1, 70.0, 1, -1.0),
            analysis.PulseOximeterDay(1, 2, 97.0, 2, 71.0, 2, 0.0),
            analysis.PulseOximeterDay(2, 3, 98.0, 3, 72.0, 3, 1.0),
            analysis.PulseOximeterDay(3, 4, 99.0, 4, 73.0, 4, 2.0),
        ]
        calendar = [
            analysis.BloodPressureCalendarDay(day_index=index, observed=True)
            for index in range(4)
        ]

        results = analysis.build_longitudinal_diagnostics(pulse_days, calendar)

        spo2 = results["spo2_time_association"]
        bpm_difference = results["bpm_device_difference_time_association"]
        self.assertTrue(spo2["estimable"])
        self.assertTrue(bpm_difference["estimable"])
        self.assertAlmostEqual(spo2["equal_day"]["slope_per_30_days"], 30.0)
        self.assertAlmostEqual(
            bpm_difference["equal_day"]["slope_per_30_days"],
            30.0,
        )

    def test_pulse_days_must_lie_inside_bp_calendar(self) -> None:
        """Relative-day alignment cannot silently extrapolate beyond the BP grid."""

        pulse_days = [analysis.PulseOximeterDay(4, 1, 98.0, 1, 70.0, 1, 0.0)]
        calendar = [
            analysis.BloodPressureCalendarDay(day_index=index, observed=True)
            for index in range(4)
        ]

        with self.assertRaisesRegex(ValueError, "inside the blood-pressure calendar"):
            analysis.build_longitudinal_diagnostics(pulse_days, calendar)

    def test_loader_rejects_inconsistent_count_mean_pair(self) -> None:
        """A positive count with a blank daily mean is malformed aggregate data."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pulse.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "day_index",
                        "n_spo2_readings",
                        "mean_spo2_percent",
                        "n_bpm_spo2_readings",
                        "mean_bpm_spo2",
                        "n_paired_bpm_readings",
                        "mean_bpm_difference",
                    ]
                )
                writer.writerow([0, 1, "", 1, 70.0, 1, 0.0])

            with self.assertRaisesRegex(ValueError, "mean_spo2_percent is required"):
                analysis.load_pulse_oximeter_days(path)


if __name__ == "__main__":
    unittest.main()
