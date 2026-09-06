"""Regression tests for episode observation-intensity sensitivity analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import analysis as primary
import episode_observation_sensitivity as episode_sensitivity
import gap_aware_trend_decomposition as gap_aware


class EpisodeObservationSensitivityTests(unittest.TestCase):
    """Protect gap-aware episode sensitivity invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = primary.load_snapshot(
            PROJECT_DIR / "data" / "analysis_snapshot.csv"
        )
        cls.results = episode_sensitivity.fit_episode_observation_sensitivity(
            cls.records
        )

    def _estimate(self, name: str) -> dict[str, float | str]:
        raw = self.results["estimates"]
        self.assertIsInstance(raw, list)
        for item in raw:
            self.assertIsInstance(item, dict)
            if item.get("name") == name:
                return item
        self.fail(f"Estimate {name!r} was not found.")

    def test_equal_day_matches_gap_aware_episode_model(self) -> None:
        """The baseline contrast must reuse the established gap-aware estimand."""

        gap = gap_aware.dominant_internal_gap(self.records)
        before, after = gap_aware.split_observation_episodes(self.records, gap)
        baseline = gap_aware._episode_centered_model(before, after)
        estimate = self._estimate("equal_day")

        self.assertAlmostEqual(
            float(estimate["episode_difference_mmHg"]),
            float(baseline["post_minus_pre_mean_difference_mmHg"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(estimate["within_episode_slope_per_30_days"]),
            float(baseline["common_within_episode_slope_per_30_days"]),
            places=10,
        )

    def test_all_expected_specifications_are_present_once(self) -> None:
        """The sensitivity grid must remain complete and non-duplicated."""

        raw = self.results["estimates"]
        self.assertIsInstance(raw, list)
        names = [str(item["name"]) for item in raw]
        self.assertEqual(
            sorted(names),
            sorted(episode_sensitivity.SPECIFICATION_NAMES),
        )
        self.assertEqual(len(names), len(set(names)))

    def test_inverse_intensity_is_not_labeled_as_ipw(self) -> None:
        """No probability-weighting interpretation is allowed without probabilities."""

        interpretation = self.results["interpretation"]
        self.assertIsInstance(interpretation, dict)
        self.assertFalse(interpretation["identified_missingness_mechanism"])
        self.assertTrue(interpretation["inverse_intensity_is_stress_test_not_ipw"])
        self.assertTrue(
            interpretation["episode_split_is_gap_defined_not_estimated_from_outcomes"]
        )

    def test_episode_sizes_cover_all_observed_days(self) -> None:
        """The two gap-defined episodes must partition every observed day."""

        sizes = self.results["episode_sizes"]
        self.assertIsInstance(sizes, dict)
        total = int(sizes["pre_gap_observed_days"]) + int(
            sizes["post_gap_observed_days"]
        )
        self.assertEqual(total, len(primary.observed_records(self.records)))


if __name__ == "__main__":
    unittest.main()
