"""Scientific tests for paired historical forecast-error sensitivity."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from comparison_uncertainty import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CANDIDATES, SOURCE_NAMES,
    compute_paired_evidence, save_paired_evidence,
)
from model_extensions import compute_model_evidence


class PairedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = compute_model_evidence()
        cls.evidence = compute_paired_evidence(cls.models)

    def test_same_ten_chronological_elections_in_both_sources(self) -> None:
        self.assertEqual(len(self.evidence.losses), 20)
        self.assertEqual(len(self.evidence.comparisons), 6)
        self.assertEqual(len(self.evidence.omissions), 60)
        for source in SOURCE_NAMES:
            years = [row.year for row in self.evidence.losses if row.source == source]
            self.assertEqual(years, [1982, 1990, 1994, 1998, 2002, 2006,
                                     2010, 2014, 2018, 2022])
        self.assertEqual(set(row.candidate for row in self.evidence.comparisons),
                         set(CANDIDATES))

    def test_loss_differences_are_candidate_minus_ols_on_same_year(self) -> None:
        for source in SOURCE_NAMES:
            folds = self.models.folds if source == "primary" else self.models.alternative_folds
            rows = [row for row in self.evidence.losses if row.source == source]
            for fold, loss in zip(folds, rows, strict=True):
                expected_ols = (fold.ols - fold.actual) ** 2
                self.assertAlmostEqual(loss.ols_squared_error, expected_ols)
                for model in CANDIDATES:
                    expected = (getattr(fold, model) - fold.actual) ** 2 - expected_ols
                    self.assertAlmostEqual(getattr(loss, model + "_minus_ols"), expected)

    def test_delta_mse_and_rmse_match_independent_manual_calculation(self) -> None:
        for summary in self.evidence.comparisons:
            folds = self.models.folds if summary.source == "primary" else self.models.alternative_folds
            actual = np.array([fold.actual for fold in folds], dtype=float)
            ols = np.array([fold.ols for fold in folds], dtype=float)
            competitor = np.array([getattr(fold, summary.candidate) for fold in folds])
            delta_mse = np.mean((competitor - actual) ** 2 - (ols - actual) ** 2)
            delta_rmse = np.sqrt(np.mean((competitor - actual) ** 2)) - np.sqrt(
                np.mean((ols - actual) ** 2))
            self.assertAlmostEqual(summary.delta_mse, delta_mse)
            self.assertAlmostEqual(summary.delta_rmse, delta_rmse)
            self.assertEqual(np.sign(summary.delta_mse), np.sign(summary.delta_rmse))

    def test_bootstrap_uses_one_common_paired_index_matrix(self) -> None:
        seed = 42
        draws = 700
        evidence = compute_paired_evidence(self.models, draws=draws, seed=seed)
        folds = self.models.folds
        actual = np.array([fold.actual for fold in folds])
        ols_sq = (np.array([fold.ols for fold in folds]) - actual) ** 2
        t_sq = (np.array([fold.student_t for fold in folds]) - actual) ** 2
        indices = np.random.default_rng(seed).integers(0, len(folds), size=(draws, len(folds)))
        manual = (np.sqrt(np.mean(t_sq[indices], axis=1))
                  - np.sqrt(np.mean(ols_sq[indices], axis=1)))
        reported = next(row for row in evidence.comparisons
                        if row.source == "primary" and row.candidate == "student_t")
        self.assertAlmostEqual(reported.bootstrap_q025, np.quantile(manual, 0.025))
        self.assertAlmostEqual(reported.bootstrap_q975, np.quantile(manual, 0.975))
        self.assertAlmostEqual(reported.bootstrap_fraction_better, np.mean(manual < 0))

    def test_seed_and_draw_count_are_reproducible(self) -> None:
        first = compute_paired_evidence(self.models, draws=750, seed=2026)
        second = compute_paired_evidence(self.models, draws=750, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(self.evidence.summary["bootstrap_draws"], BOOTSTRAP_DRAWS)
        self.assertEqual(self.evidence.summary["bootstrap_seed"], BOOTSTRAP_SEED)

    def test_student_t_slight_advantage_is_source_and_year_sensitive(self) -> None:
        lookup = {(row.source, row.candidate): row for row in self.evidence.comparisons}
        primary = lookup[("primary", "student_t")]
        alternate = lookup[("alternative_gallup", "student_t")]
        self.assertAlmostEqual(primary.delta_rmse, -0.2143995, delta=0.02)
        self.assertAlmostEqual(alternate.delta_rmse, +0.0484976, delta=0.02)
        self.assertEqual(primary.lower_absolute_error_count, 5)
        self.assertEqual(primary.largest_loss_difference_year, 2010)
        self.assertAlmostEqual(primary.largest_loss_difference_share, 0.5139, delta=0.02)
        self.assertEqual(primary.omitted_year_reversal_count, 1)
        self.assertEqual(alternate.omitted_year_reversal_count, 3)
        for row in (primary, alternate):
            self.assertLess(row.bootstrap_q025, 0)
            self.assertGreater(row.bootstrap_q975, 0)
            self.assertGreaterEqual(row.bootstrap_fraction_better, 0)
            self.assertLessEqual(row.bootstrap_fraction_better, 1)

    def test_one_year_omission_matches_direct_recomputed_rmse(self) -> None:
        omissions = [row for row in self.evidence.omissions
                     if row.source == "primary" and row.candidate == "student_t"]
        original = self.models.folds
        for row in omissions:
            remainder = [f for f in original if f.year != row.omitted_year]
            actual = np.array([f.actual for f in remainder])
            t_rmse = np.sqrt(np.mean((np.array([f.student_t for f in remainder]) - actual) ** 2))
            o_rmse = np.sqrt(np.mean((np.array([f.ols for f in remainder]) - actual) ** 2))
            self.assertAlmostEqual(row.delta_rmse, t_rmse - o_rmse)

    def test_misaligned_sources_and_future_outcomes_are_rejected(self) -> None:
        primary = list(self.models.folds)
        primary[-1] = replace(primary[-1], year=2026)
        with self.assertRaisesRegex(ValueError, "2026"):
            compute_paired_evidence(replace(self.models, folds=tuple(primary)), draws=100)
        alternative = list(self.models.alternative_folds)
        alternative[-1] = replace(alternative[-1], actual=500)
        with self.assertRaisesRegex(ValueError, "identical"):
            compute_paired_evidence(replace(self.models, alternative_folds=tuple(alternative)), draws=100)

    def test_invalid_draws_and_seeds_are_rejected(self) -> None:
        for draws, seed in ((0, 1), (99, 1), (True, 1), (100, -1), (100, True)):
            with self.subTest(draws=draws, seed=seed):
                with self.assertRaises(ValueError):
                    compute_paired_evidence(self.models, draws=draws, seed=seed)

    def test_all_committed_artifact_schemas_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save_paired_evidence(self.evidence, folder)
            for filename, expected in (
                ("paired_loss_differences.csv", 20),
                ("paired_model_comparisons.csv", 6),
                ("paired_leave_one_year_out.csv", 60),
            ):
                with (folder / filename).open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), expected)
            meta = json.loads((folder / "paired_uncertainty_summary.json").read_text())
            self.assertEqual(meta["forward_count"], 10)
            self.assertEqual(len(meta["model_comparisons"]), 6)
            self.assertEqual(meta["as_of"], "2026-10-31")
            self.assertTrue(all(math.isfinite(row["delta_rmse"])
                                for row in meta["model_comparisons"]))


if __name__ == "__main__":
    unittest.main()
