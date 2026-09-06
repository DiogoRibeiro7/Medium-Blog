"""Regression tests for episode time-form sensitivity analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import analysis as primary
import episode_time_form_sensitivity as time_form
import gap_aware_trend_decomposition as gap_aware


class EpisodeTimeFormSensitivityTests(unittest.TestCase):
    """Protect cross-analysis and structural time-form invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = primary.load_snapshot(
            PROJECT_DIR / "data" / "analysis_snapshot.csv"
        )
        cls.results = time_form.fit_episode_time_form_sensitivity(cls.records)

    def _estimate(self, name: str) -> dict[str, object]:
        estimates = self.results["estimates"]
        self.assertIsInstance(estimates, list)
        for item in estimates:
            self.assertIsInstance(item, dict)
            if item.get("name") == name:
                return item
        self.fail(f"Estimate {name!r} was not found.")

    def test_common_linear_matches_gap_aware_episode_contrast(self) -> None:
        """The established gap-aware model must be reproduced exactly."""

        gap = gap_aware.dominant_internal_gap(self.records)
        before, after = gap_aware.split_observation_episodes(self.records, gap)
        established = gap_aware._episode_centered_model(before, after)
        common = self._estimate("common_linear")
        self.assertAlmostEqual(
            float(common["episode_difference_mmHg"]),
            float(established["post_minus_pre_mean_difference_mmHg"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(common["episode_ci95_low_mmHg"]),
            float(established["post_minus_pre_ci95_low_mmHg"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(common["episode_ci95_high_mmHg"]),
            float(established["post_minus_pre_ci95_high_mmHg"]),
            places=10,
        )

    def test_specifications_have_increasing_or_equal_complexity(self) -> None:
        """The stress model must be the most parameter-rich specification."""

        estimates = self.results["estimates"]
        counts = {str(item["name"]): int(item["n_parameters"]) for item in estimates}
        self.assertEqual(counts["episode_only"], 2)
        self.assertEqual(counts["common_linear"], 3)
        self.assertEqual(counts["separate_linear_slopes"], 4)
        self.assertEqual(counts["common_quadratic"], 4)
        self.assertEqual(
            counts["separate_linear_plus_common_quadratic_stress"], 5
        )

    def test_episode_split_matches_observed_sample(self) -> None:
        """All observed days must belong to exactly one frozen episode."""

        sizes = self.results["episode_sizes"]
        self.assertIsInstance(sizes, dict)
        total = int(sizes["pre_gap_observed_days"]) + int(
            sizes["post_gap_observed_days"]
        )
        self.assertEqual(total, len(primary.observed_records(self.records)))
        self.assertEqual(total, int(self.results["n_observed_days"]))

    def test_stress_model_is_not_declared_preferred(self) -> None:
        """The flexible model must remain a stress specification only."""

        interpretation = self.results["interpretation"]
        self.assertIsInstance(interpretation, dict)
        self.assertFalse(interpretation["preferred_model_selected_by_this_analysis"])
        self.assertFalse(interpretation["stress_model_is_claimed_true_trajectory"])
        self.assertTrue(
            interpretation["common_linear_matches_gap_aware_primary_estimand"]
        )


if __name__ == "__main__":
    unittest.main()
