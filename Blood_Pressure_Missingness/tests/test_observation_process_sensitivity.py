"""Regression tests for observation-process sensitivity analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import observation_process_sensitivity as sensitivity


class ObservationProcessSensitivityTests(unittest.TestCase):
    """Protect the interpretation of unequal measurement intensity."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the committed privacy-safe snapshot once for all tests."""

        cls.days = sensitivity.load_observed_days(
            PROJECT_DIR / "data" / "analysis_snapshot.csv"
        )
        cls.results = sensitivity.fit_sensitivity_models(cls.days)

    def _estimate(self, name: str) -> dict[str, float | str]:
        """Return one named sensitivity estimate from the machine results."""

        estimates = self.results["trend_estimates"]
        self.assertIsInstance(estimates, list)
        for item in estimates:
            self.assertIsInstance(item, dict)
            if item.get("name") == name:
                return item
        self.fail(f"Sensitivity estimate {name!r} was not found.")

    def test_committed_snapshot_contains_25_observed_days(self) -> None:
        """Sensitivity analysis must use the same 25 observed calendar days."""

        self.assertEqual(len(self.days), 25)
        self.assertEqual(self.results["n_observed_days"], 25)

    def test_equal_day_trend_matches_primary_analysis(self) -> None:
        """Baseline sensitivity specification must reproduce the published trend."""

        estimate = self._estimate("equal_day")
        self.assertAlmostEqual(float(estimate["slope_per_30_days"]), -4.26751331, 6)
        self.assertLess(float(estimate["ci95_high_per_30_days"]), 0.0)

    def test_sampling_adjustment_weakens_but_preserves_negative_trend(self) -> None:
        """Adjusting for observation intensity should not be presented as neutral."""

        baseline = self._estimate("equal_day")
        adjusted = self._estimate("sampling_adjusted")
        self.assertLess(float(adjusted["slope_per_30_days"]), 0.0)
        self.assertGreater(
            float(adjusted["slope_per_30_days"]),
            float(baseline["slope_per_30_days"]),
        )
        self.assertLess(float(adjusted["ci95_high_per_30_days"]), 0.0)

    def test_inverse_intensity_stress_test_crosses_zero(self) -> None:
        """The deliberate sparse-day stress test should expose trend fragility."""

        estimate = self._estimate("inverse_intensity_stress")
        self.assertLess(float(estimate["ci95_low_per_30_days"]), 0.0)
        self.assertGreater(float(estimate["ci95_high_per_30_days"]), 0.0)

    def test_inverse_intensity_is_not_labeled_as_ipw(self) -> None:
        """No inverse-probability interpretation is allowed without probabilities."""

        interpretation = self.results["interpretation"]
        self.assertIsInstance(interpretation, dict)
        self.assertFalse(interpretation["identified_missingness_mechanism"])
        self.assertTrue(interpretation["inverse_intensity_is_stress_test_not_ipw"])


if __name__ == "__main__":
    unittest.main()
