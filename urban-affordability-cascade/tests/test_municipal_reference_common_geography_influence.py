"""Tests for leave-one-municipality-out matched geographic rank contrasts."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import MUNICIPALITIES
from municipal_reference_common_geography import (
    matched_common_controls,
    rank_residual_correlation,
)
from municipal_reference_common_geography_influence import (
    FIGURES,
    PRIMARY_MODE,
    RESULTS,
    deletion_grid,
    heatmap_svg,
    pair_summary,
    run,
    stability_svg,
    window_summary,
)
from municipal_reference_geography_controls import distance_pairs
from municipal_reference_window_sensitivity import window_pairs
from porto_network_gradient import MODES, WINDOW_STARTS, average_ranks


class ReferenceInfluenceTests(unittest.TestCase):
    """Check complete grid, proper reranking, and published outcomes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.windows = window_pairs()
        cls.distances = distance_pairs()
        cls.matched = matched_common_controls(cls.windows, cls.distances)
        cls.deletions = deletion_grid(cls.windows, cls.distances, cls.matched)
        cls.pairs = pair_summary(cls.deletions)
        cls.summaries = window_summary(cls.pairs)

    def test_full_grid_has_2880_unique_deletions(self) -> None:
        self.assertEqual(len(self.deletions), 4 * 16 * 3 * 15)
        keys = {
            (int(r["window_start_year"]), str(r["alternative_reference"]),
             str(r["exposure_mode"]), str(r["omitted_municipality"]))
            for r in self.deletions
        }
        self.assertEqual(len(keys), 2880)
        self.assertTrue(all(r["remaining_comparators"] == 14 for r in self.deletions))
        self.assertTrue(all(r["shared_control_set"] == "both_distances" for r in self.deletions))
        self.assertTrue(all(
            r["omitted_municipality"] not in ("Porto", r["alternative_reference"])
            for r in self.deletions
        ))

    def test_every_alternative_and_mode_has_fifteen_deletions(self) -> None:
        self.assertEqual(len(self.pairs), 192)
        self.assertEqual(
            {int(r["deletion_samples"]) for r in self.pairs}, {15}
        )
        self.assertEqual(len(self.summaries), 12)
        for summary in self.summaries:
            self.assertEqual(int(summary["alternatives"]), 16)
            self.assertEqual(int(summary["deletion_cases"]), 240)

    def test_primary_mode_reversal_and_margin_counts_are_frozen_oracles(self) -> None:
        expected = {
            2020: (4, 4),
            2021: (5, 6),
            2022: (1, 1),
            2023: (4, 4),
        }
        for summary in self.summaries:
            if summary["exposure_mode"] == PRIMARY_MODE:
                year = int(summary["window_start_year"])
                self.assertEqual(
                    (int(summary["alternatives_with_any_sign_reversal"]),
                     int(summary["alternatives_with_any_margin_crossing"])),
                    expected[year],
                )

    def test_each_14_municipality_projection_retains_rank_variance(self) -> None:
        for row in self.deletions:
            for key in (
                "porto_exposure_remaining_rank_variance_fraction",
                "alternative_exposure_remaining_rank_variance_fraction",
                "outcome_remaining_rank_variance_fraction",
            ):
                self.assertGreater(float(row[key]), 0)
                self.assertLessEqual(float(row[key]), 1 + 1e-10)
            self.assertTrue(math.isfinite(float(
                row["deleted_shared_alternative_minus_porto_rho"]
            )))

    def test_deletion_reranks_instead_of_subsetting_original_ranks(self) -> None:
        # Removing one of a tied pair changes average ranks. Reranking the
        # reduced sample must agree with the 14-observation ranks.
        x = np.array([1., 2., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.])
        keep = np.arange(x.size) != 1
        actual = average_ranks(x[keep])
        precomputed_subsample = average_ranks(x)[keep]
        self.assertFalse(np.array_equal(actual, precomputed_subsample))
        np.testing.assert_array_equal(actual, np.arange(1, 15, dtype=float))
        y = np.array([1, 5, 3, 7, 9, 6, 11, 8, 13, 10, 12, 14, 4, 15, 2.], dtype=float)
        z = np.column_stack((np.arange(15.) ** 2, np.arange(15.)[::-1] ** 2))
        result = rank_residual_correlation(x[keep], y[keep], z[keep, :])
        self.assertTrue(math.isfinite(result[0]))

    def test_source_baseline_mismatch_is_rejected(self) -> None:
        changed = [dict(r) for r in self.matched]
        target = next(r for r in changed if r["shared_control_set"] == "both_distances")
        target["shared_alternative_minus_porto_rho"] = "0.999999999"
        with self.assertRaisesRegex(ValueError, "baseline disagrees"):
            deletion_grid(self.windows, self.distances, changed)

    def test_missing_or_duplicate_parent_grid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "576-row"):
            deletion_grid(self.windows, self.distances, self.matched[:-1])
        with self.assertRaisesRegex(ValueError, "576-row"):
            deletion_grid(self.windows, self.distances,
                          [*self.matched[:-1], self.matched[0]])

    def test_missing_deletion_or_pair_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2,880"):
            pair_summary(self.deletions[:-1])
        with self.assertRaisesRegex(ValueError, "2,880"):
            pair_summary([*self.deletions[:-1], self.deletions[0]])
        with self.assertRaisesRegex(ValueError, "192"):
            window_summary(self.pairs[:-1])

    def test_every_dropped_name_is_in_declared_geography(self) -> None:
        for r in self.deletions:
            self.assertIn(r["alternative_reference"], MUNICIPALITIES)
            self.assertIn(r["omitted_municipality"], MUNICIPALITIES)
        self.assertEqual(len(WINDOW_STARTS), 4)
        self.assertEqual(len(MODES), 3)

    def test_visualisations_are_deterministic_and_complete(self) -> None:
        heat = heatmap_svg(self.pairs)
        bars = stability_svg(self.summaries)
        self.assertEqual(heat, heatmap_svg(self.pairs))
        self.assertEqual(bars, stability_svg(self.summaries))
        self.assertTrue(heat.startswith('<svg xmlns="http://www.w3.org/2000/svg"'))
        self.assertIn("São João da Madeira", heat)
        for year in WINDOW_STARTS:
            self.assertIn(f"{year}–25", heat)
            self.assertIn(f"{year}–25", bars)
        with self.assertRaisesRegex(ValueError, "16 x 4"):
            heatmap_svg([r for r in self.pairs if not (
                r["exposure_mode"] == PRIMARY_MODE
                and int(r["window_start_year"]) == 2020
                and r["alternative_reference"] == "Arouca")])
        with self.assertRaisesRegex(ValueError, "all four"):
            stability_svg(self.summaries[:-1])

    def test_output_provenance_and_tables_regenerate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            report = run(output)
            self.assertEqual(report["municipality_deletion_cases"], 2880)
            self.assertEqual(report["matched_reference_mode_year_cases"], 192)
            self.assertEqual(report["window_mode_summary_rows"], 12)
            self.assertEqual(report["comparators_after_deletion"], 14)
            self.assertEqual(report["shared_control_set"], "both_distances")
            self.assertEqual(report["centroid_crs_epsg"], 3763)
            self.assertEqual(report["commuting_network_year"], 2021)
            for figure in FIGURES:
                self.assertTrue(
                    (output / "figures" / figure).read_text(
                        encoding="utf-8").startswith("<svg")
                )
            for name, expected_rows in zip(
                RESULTS[:3], (self.deletions, self.pairs, self.summaries)
            ):
                with (output / "results" / name).open(
                    encoding="utf-8", newline=""
                ) as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), len(expected_rows))
                self.assertEqual(list(rows[0]), list(expected_rows[0]))
            actual = json.loads(
                (output / "results" / RESULTS[3]).read_text(encoding="utf-8")
            )
            self.assertEqual(actual["housing_panel_sha256"],
                             report["housing_panel_sha256"])
            self.assertEqual(actual["caop2025_derived_csv_sha256"],
                             report["caop2025_derived_csv_sha256"])
            self.assertEqual(actual["commuting_derived_pair_csv_sha256"],
                             report["commuting_derived_pair_csv_sha256"])


if __name__ == "__main__":
    unittest.main()
