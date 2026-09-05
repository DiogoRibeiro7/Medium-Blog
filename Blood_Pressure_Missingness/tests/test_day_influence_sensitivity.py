"""Regression tests for leave-one-observed-day-out influence analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import analysis as primary
import day_influence_sensitivity as influence
import gap_aware_trend_decomposition as gap_aware


class DayInfluenceSensitivityTests(unittest.TestCase):
    """Protect small-sample influence-analysis invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = primary.load_snapshot(
            PROJECT_DIR / "data" / "analysis_snapshot.csv"
        )
        cls.results = influence.summarize_influence(cls.records)

    def test_one_deletion_per_observed_day(self) -> None:
        """Every observed calendar day must be removed exactly once."""

        observed = primary.observed_records(self.records)
        deletions = self.results["deletions"]
        self.assertIsInstance(deletions, list)
        removed = [int(item["removed_day_index"]) for item in deletions]
        expected = [record.day_index for record in observed]
        self.assertEqual(sorted(removed), sorted(expected))
        self.assertEqual(len(removed), len(set(removed)))

    def test_baseline_matches_primary_and_gap_aware_analyses(self) -> None:
        """Influence baseline must reuse the established estimands exactly."""

        baseline = self.results["baseline"]
        self.assertIsInstance(baseline, dict)

        primary_fit = primary.linear_trend(
            self.records, "mean_systolic_mmHg"
        )
        gap = gap_aware.dominant_internal_gap(self.records)
        before, after = gap_aware.split_observation_episodes(self.records, gap)
        episode = gap_aware._episode_centered_model(before, after)

        self.assertAlmostEqual(
            float(baseline["global_slope_per_30_days"]),
            float(primary_fit["slope_per_30_days"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(baseline["episode_level_difference_mmHg"]),
            float(episode["post_minus_pre_mean_difference_mmHg"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(baseline["within_episode_slope_per_30_days"]),
            float(episode["common_within_episode_slope_per_30_days"]),
            places=10,
        )

    def test_current_global_and_episode_conclusions_survive_every_deletion(self) -> None:
        """Current main claims should not hinge on one observed day."""

        summary = self.results["leave_one_day_out_summary"]
        self.assertIsInstance(summary, dict)
        self.assertTrue(
            summary["global_interval_excludes_zero_for_all_deletions"]
        )
        self.assertTrue(
            summary["episode_level_interval_excludes_zero_for_all_deletions"]
        )
        self.assertLess(float(summary["global_slope_max_per_30_days"]), 0.0)
        self.assertLess(float(summary["episode_level_difference_max_mmHg"]), 0.0)

    def test_within_episode_significance_is_not_stable(self) -> None:
        """The within-episode significance conclusion is deletion-sensitive."""

        summary = self.results["leave_one_day_out_summary"]
        self.assertIsInstance(summary, dict)
        significant = summary["within_episode_significant_negative_deletions"]
        self.assertIsInstance(significant, list)
        self.assertGreater(len(significant), 0)
        self.assertLess(len(significant), self.results["n_observed_days"])

    def test_influence_diagnostics_reference_observed_days(self) -> None:
        """Cook's-distance and DFBETA maxima must point to real observed days."""

        observed_days = {
            record.day_index for record in primary.observed_records(self.records)
        }
        cook = self.results["most_influential_by_cooks_distance"]
        dfbeta = self.results["largest_absolute_slope_dfbeta"]
        self.assertIn(int(cook["day_index"]), observed_days)
        self.assertIn(int(dfbeta["day_index"]), observed_days)


if __name__ == "__main__":
    unittest.main()
