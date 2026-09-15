"""Regression tests for pulse longitudinal weighting semantics."""

from __future__ import annotations

import unittest

from blood_pressure_missingness.analyses import pulse_oximeter_longitudinal as analysis


class PulseOximeterWeightingSemanticsTests(unittest.TestCase):
    """Protect the distinction between equal-day and reading-count estimands."""

    def test_unequal_counts_change_the_weighted_time_association(self) -> None:
        """A non-linear series must react to strongly unequal reading counts."""

        results = analysis._fit_time_association(
            day_indices=[0, 1, 2, 3],
            values=[0.0, 0.0, 0.0, 10.0],
            weights=[1, 1, 1, 10],
        )

        self.assertTrue(results["estimable"])
        equal_day = results["equal_day"]
        weighted = results["reading_count_weighted"]
        assert equal_day is not None
        assert weighted is not None

        self.assertAlmostEqual(equal_day["slope_per_30_days"], 90.0, places=8)
        self.assertAlmostEqual(
            weighted["slope_per_30_days"],
            123.2876712328767,
            places=8,
        )
        self.assertNotAlmostEqual(
            equal_day["slope_per_30_days"],
            weighted["slope_per_30_days"],
            places=8,
        )

    def test_uniform_counts_recover_the_equal_day_estimand(self) -> None:
        """Equal weights must collapse WLS back to the equal-day OLS slope."""

        results = analysis._fit_time_association(
            day_indices=[0, 1, 2, 3],
            values=[0.0, 0.0, 0.0, 10.0],
            weights=[1, 1, 1, 1],
        )

        equal_day = results["equal_day"]
        weighted = results["reading_count_weighted"]
        assert equal_day is not None
        assert weighted is not None
        self.assertAlmostEqual(
            equal_day["slope_per_30_days"],
            weighted["slope_per_30_days"],
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
