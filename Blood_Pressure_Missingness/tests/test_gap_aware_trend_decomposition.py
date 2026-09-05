"""Regression tests for the gap-aware blood-pressure trend decomposition."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import analysis as primary
import gap_aware_trend_decomposition as gap_analysis


class GapAwareTrendDecompositionTests(unittest.TestCase):
    """Protect gap detection, decomposition identities, and interpretation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the current privacy-safe snapshot once for all tests."""

        cls.snapshot_path = PROJECT_DIR / "data" / "analysis_snapshot.csv"
        cls.records = primary.load_snapshot(cls.snapshot_path)
        cls.results = gap_analysis.analyze_gap_aware_trend(cls.records)

    def test_dominant_gap_matches_primary_longest_missing_run(self) -> None:
        """The episode split must use the primary analysis's longest gap."""

        gap = self.results["dominant_internal_gap"]
        self.assertIsInstance(gap, dict)
        self.assertEqual(
            int(gap["length_days"]),
            primary.longest_missing_run(self.records),
        )

    def test_episode_split_preserves_all_observed_days(self) -> None:
        """Every observed day must belong to exactly one episode."""

        pre = self.results["pre_gap_episode"]
        post = self.results["post_gap_episode"]
        self.assertIsInstance(pre, dict)
        self.assertIsInstance(post, dict)
        expected = len(primary.observed_records(self.records))
        self.assertEqual(int(pre["n_days"]) + int(post["n_days"]), expected)
        self.assertEqual(int(self.results["n_observed_days"]), expected)

    def test_global_slope_matches_primary_hc3_point_estimate(self) -> None:
        """The decomposition must reproduce the primary global OLS slope."""

        primary_trend = primary.linear_trend(
            self.records,
            "mean_systolic_mmHg",
        )
        decomposition = self.results["exact_global_slope_decomposition"]
        self.assertIsInstance(decomposition, dict)
        self.assertAlmostEqual(
            float(decomposition["global_ols_slope_per_30_days"]),
            float(primary_trend["slope_per_30_days"]),
            places=10,
        )

    def test_within_and_between_contributions_add_to_global_slope(self) -> None:
        """The covariance decomposition must satisfy its exact additive identity."""

        decomposition = self.results["exact_global_slope_decomposition"]
        self.assertIsInstance(decomposition, dict)
        self.assertAlmostEqual(
            float(decomposition["within_episode_contribution_per_30_days"])
            + float(decomposition["between_episode_contribution_per_30_days"]),
            float(decomposition["global_ols_slope_per_30_days"]),
            places=10,
        )

    def test_interpretation_does_not_claim_a_change_point(self) -> None:
        """A long unobserved interval cannot identify a transition time."""

        interpretation = self.results["interpretation"]
        self.assertIsInstance(interpretation, dict)
        self.assertFalse(interpretation["is_change_point_test"])
        self.assertFalse(interpretation["transition_inside_gap_is_observed"])
        self.assertFalse(interpretation["can_identify_when_level_difference_arose"])

    def test_tied_longest_internal_gaps_are_rejected(self) -> None:
        """An ambiguous dominant gap must not trigger an arbitrary episode split."""

        records = [
            primary.DailyRecord(0, True, 1, 1, 120.0, 80.0, 40.0, 70.0),
            primary.DailyRecord(1, False, 0, 0, None, None, None, None),
            primary.DailyRecord(2, True, 1, 1, 119.0, 79.0, 40.0, 71.0),
            primary.DailyRecord(3, True, 1, 1, 118.0, 78.0, 40.0, 72.0),
            primary.DailyRecord(4, False, 0, 0, None, None, None, None),
            primary.DailyRecord(5, True, 1, 1, 117.0, 77.0, 40.0, 73.0),
        ]
        with self.assertRaisesRegex(ValueError, "not unique"):
            gap_analysis.dominant_internal_gap(records)

    def test_plot_writes_only_derived_figure(self) -> None:
        """Plotting the public snapshot must not require or emit row-level data."""

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gap.svg"
            gap_analysis.plot_gap_aware_trend(
                self.records,
                self.results,
                output,
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
