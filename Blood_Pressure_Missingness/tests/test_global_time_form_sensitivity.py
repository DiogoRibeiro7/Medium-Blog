"""Regression tests for global systolic time-form sensitivity."""

from __future__ import annotations

import unittest

from blood_pressure_missingness import public_analysis as primary
from blood_pressure_missingness.analyses import global_time_form as analysis


class GlobalTimeFormSensitivityTests(unittest.TestCase):
    """Protect distinct global time-association estimands."""

    def _record(self, day: int, systolic: float) -> primary.DailyRecord:
        return primary.DailyRecord(
            day_index=day,
            observed=True,
            n_readings=1,
            n_sessions=1,
            mean_systolic_mmHg=systolic,
            mean_diastolic_mmHg=70.0,
            mean_pulse_pressure_mmHg=systolic - 70.0,
            mean_bpm=70.0,
        )

    def test_non_linear_series_separates_sensitivity_estimands(self) -> None:
        records = [
            self._record(0, 100.0),
            self._record(1, 100.0),
            self._record(2, 100.0),
            self._record(3, 110.0),
            self._record(4, 110.0),
        ]

        results = analysis.analyze_global_time_form_sensitivity(records)
        estimates = results["estimates"]

        self.assertEqual(results["n_observed_days"], 5)
        self.assertFalse(results["interpretation"]["estimands_are_interchangeable"])
        self.assertTrue(
            results["interpretation"]["quadratic_summary_is_endpoint_average_change"]
        )
        self.assertTrue(
            results["interpretation"]["theil_sen_is_median_pairwise_slope"]
        )

        linear = estimates["linear_equal_day_hc3"]["estimate_per_30_days"]
        quadratic = estimates["quadratic_end_to_end_average_hc3"][
            "estimate_per_30_days"
        ]
        theil_sen = estimates["theil_sen_median_pairwise_slope"][
            "estimate_per_30_days"
        ]
        self.assertNotAlmostEqual(linear, quadratic, places=8)
        self.assertNotAlmostEqual(linear, theil_sen, places=8)

    def test_exact_linear_series_agrees_on_point_slope(self) -> None:
        records = [
            self._record(day, 120.0 - 0.5 * day)
            for day in (0, 1, 2, 4, 7)
        ]

        results = analysis.analyze_global_time_form_sensitivity(records)
        estimates = results["estimates"]
        expected = -15.0
        for item in estimates.values():
            self.assertAlmostEqual(item["estimate_per_30_days"], expected, places=8)


if __name__ == "__main__":
    unittest.main()
