"""Regression tests for leave-one-observed-day-out influence analysis."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import analysis as primary
import day_influence_sensitivity as influence
import gap_aware_trend_decomposition as gap_aware


class DayInfluenceSensitivityTests(unittest.TestCase):
    """Protect structural invariants of the influence analysis."""

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

    def test_episode_definition_is_frozen_from_full_dataset(self) -> None:
        """Jackknife deletions must not redefine the dominant-gap split."""

        frozen = self.results["frozen_episode_definition"]
        self.assertIsInstance(frozen, dict)
        gap = gap_aware.dominant_internal_gap(self.records)
        self.assertEqual(int(frozen["start_day_index"]), gap.start_day_index)
        self.assertEqual(int(frozen["end_day_index"]), gap.end_day_index)
        interpretation = self.results["interpretation"]
        self.assertIsInstance(interpretation, dict)
        self.assertFalse(
            interpretation["episode_definition_recomputed_after_each_deletion"]
        )

    def test_jackknife_requires_five_days_per_episode(self) -> None:
        """An episode with four days is valid for baseline fit but not jackknife."""

        rows = [
            "day_index,observed,n_readings,n_sessions,mean_systolic_mmHg,mean_diastolic_mmHg,mean_pulse_pressure_mmHg,mean_bpm",
            "0,1,1,1,120,80,40,70",
            "1,1,1,1,119,79,40,70",
            "2,1,1,1,118,78,40,70",
            "3,1,1,1,117,77,40,70",
            "4,0,0,0,,,,",
            "5,0,0,0,,,,",
            "6,0,0,0,,,,",
            "7,0,0,0,,,,",
            "8,0,0,0,,,,",
            "9,1,1,1,116,76,40,70",
            "10,1,1,1,115,75,40,70",
            "11,1,1,1,114,74,40,70",
            "12,1,1,1,113,73,40,70",
            "13,1,1,1,112,72,40,70",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            records = primary.load_snapshot(path)
        with self.assertRaisesRegex(ValueError, "at least 5 observed days"):
            influence.summarize_influence(records)

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
