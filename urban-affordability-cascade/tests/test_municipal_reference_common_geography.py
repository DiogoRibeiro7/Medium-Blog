"""Regression and falsification tests for matched common-geography controls."""

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
    CONTROLS,
    FIGURES,
    PRIMARY_MODE,
    RESULTS,
    matched_common_controls,
    rank_residual_correlation,
    run,
    summary_grid,
)
from municipal_reference_geography_controls import (
    distance_pairs,
    matched_reference_controls,
)
from municipal_reference_window_sensitivity import (
    MATERIAL_RHO_DIFFERENCE,
    OUTCOMES,
    window_pairs,
)
from porto_distance_sensitivity import partial_rank_correlation
from porto_network_gradient import MODES, WINDOW_STARTS, spearman_rho


class CommonGeographyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.windows = window_pairs()
        cls.distances = distance_pairs()
        cls.old = matched_reference_controls(cls.windows, cls.distances)
        cls.matched = matched_common_controls(cls.windows, cls.distances)
        cls.summary = summary_grid(cls.matched)

    def test_exact_full_design(self) -> None:
        expected = {(year, name, mode, control)
                    for year in WINDOW_STARTS for name in MUNICIPALITIES - {"Porto"}
                    for mode in MODES for control in CONTROLS}
        actual = {(row["window_start_year"], row["alternative_reference"],
                   row["exposure_mode"], row["shared_control_set"])
                  for row in self.matched}
        self.assertEqual(len(self.matched), 576)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(self.matched))

    def test_exact_15_shared_municipalities(self) -> None:
        for row in self.matched:
            self.assertEqual(int(row["shared_comparators"]), 15)
            actual = set(str(row["shared_municipality_names"]).split("|"))
            self.assertEqual(actual, MUNICIPALITIES - {"Porto", row["alternative_reference"]})

    def test_only_invariant_own_rent_growth_is_used(self) -> None:
        self.assertTrue(all(row["outcome"] == OUTCOMES[0] for row in self.matched))
        self.assertTrue(all(row["baseline_reference"] == "Porto" for row in self.matched))
        self.assertEqual(len(OUTCOMES), 2)  # The reference-relative one stays excluded.

    def test_matching_keeps_raw_coefficients_fixed_across_controls(self) -> None:
        for year in WINDOW_STARTS:
            for alternative in MUNICIPALITIES - {"Porto"}:
                for mode in MODES:
                    rows = [r for r in self.matched if r["window_start_year"] == year
                            and r["alternative_reference"] == alternative
                            and r["exposure_mode"] == mode]
                    self.assertEqual(len(rows), 3)
                    for key in ("raw_porto_rho", "raw_alternative_rho",
                                "raw_alternative_minus_porto_rho"):
                        self.assertEqual(len({row[key] for row in rows}), 1)

    def test_shared_outcome_is_residualised_identically(self) -> None:
        for row in self.matched:
            self.assertTrue(0 < float(row["outcome_remaining_rank_variance_fraction"]) <= 1)
            self.assertEqual(int(row["window_end_year"]), 2025)

    def test_one_control_projection_matches_existing_formula(self) -> None:
        x = np.array([2, 3, 1, 7, 8, 6, 9, 5, 4, 10, 11], dtype=float)
        y = np.array([6, 9, 1, 3, 11, 8, 7, 5, 2, 4, 10], dtype=float)
        d = np.array([2, 5, 3, 6, 1, 4, 9, 10, 7, 8, 11], dtype=float)
        rho, _, _ = rank_residual_correlation(x, y, d[:, None])
        self.assertAlmostEqual(rho, partial_rank_correlation(x, y, d), places=12)

    def test_primary_matched_porto_and_alternative_one_control_oracles(self) -> None:
        prior = {(r["window_start_year"], r["alternative_reference"], r["exposure_mode"]): r
                 for r in self.old}
        for row in self.matched:
            original = prior[row["window_start_year"], row["alternative_reference"],
                             row["exposure_mode"]]
            if row["shared_control_set"] == "porto_distance":
                self.assertAlmostEqual(float(row["shared_porto_conditional_rho"]),
                                       float(original["porto_distance_conditional_rho_on_same_15"]), places=8)
            elif row["shared_control_set"] == "alternative_distance":
                self.assertAlmostEqual(float(row["shared_alternative_conditional_rho"]),
                                       float(original["alternative_distance_conditional_rho_on_same_15"]), places=8)

    def test_previous_separate_controls_are_a_benchmark_not_the_new_contrast(self) -> None:
        original = {(r["window_start_year"], r["alternative_reference"], r["exposure_mode"]): r
                    for r in self.old}
        for row in self.matched:
            old = original[row["window_start_year"], row["alternative_reference"], row["exposure_mode"]]
            self.assertAlmostEqual(float(row["prior_different_controls_alternative_minus_porto_rho"]),
                                   float(old["conditional_alternative_minus_porto_rho"]), places=8)
            self.assertAlmostEqual(float(row["shared_minus_different_controls_delta"]),
                                   float(row["shared_alternative_minus_porto_rho"]) -
                                   float(row["prior_different_controls_alternative_minus_porto_rho"]), places=8)

    def test_common_control_changes_reference_contrast(self) -> None:
        chosen = [r for r in self.matched if r["window_start_year"] == 2023
                  and r["exposure_mode"] == PRIMARY_MODE]
        porto = {r["alternative_reference"]: r for r in chosen
                 if r["shared_control_set"] == "porto_distance"}
        alternative = {r["alternative_reference"]: r for r in chosen
                       if r["shared_control_set"] == "alternative_distance"}
        self.assertEqual(set(porto), set(alternative))
        self.assertTrue(any(abs(float(porto[name]["shared_alternative_minus_porto_rho"]) -
                                float(alternative[name]["shared_alternative_minus_porto_rho"])) > 0.05
                            for name in porto))

    def test_all_36_summary_cells_and_counts(self) -> None:
        expected = {(year, mode, control) for year in WINDOW_STARTS
                    for mode in MODES for control in CONTROLS}
        actual = {(r["window_start_year"], r["exposure_mode"], r["shared_control_set"])
                  for r in self.summary}
        self.assertEqual(len(self.summary), 36)
        self.assertEqual(actual, expected)
        self.assertTrue(all(r["alternatives"] == 16 for r in self.summary))

    def test_predeclared_primary_robustness_counts(self) -> None:
        expected = {
            "porto_distance": (1, 6, 0, 1),
            "alternative_distance": (2, 4, 1, 7),
            "both_distances": (2, 4, 0, 4),
        }
        for control, values in expected.items():
            observed = tuple(int(r["alternatives_more_negative_than_porto"]) for r in self.summary
                             if r["exposure_mode"] == PRIMARY_MODE
                             and r["shared_control_set"] == control)
            self.assertEqual(observed, values)

    def test_threshold_is_descriptive_and_preexisting(self) -> None:
        self.assertAlmostEqual(MATERIAL_RHO_DIFFERENCE, 0.05)
        for row in self.summary:
            self.assertLessEqual(int(row["alternatives_at_least_0_05_more_negative"]),
                                 int(row["alternatives_more_negative_than_porto"]))

    def test_remaining_rank_variances_are_finite_and_bounded(self) -> None:
        for row in self.matched:
            for key in ("porto_exposure_remaining_rank_variance_fraction",
                        "alternative_exposure_remaining_rank_variance_fraction",
                        "outcome_remaining_rank_variance_fraction"):
                self.assertTrue(1e-8 <= float(row[key]) <= 1 + 1e-8)
            for key in ("shared_porto_conditional_rho", "shared_alternative_conditional_rho",
                        "shared_alternative_minus_porto_rho"):
                self.assertTrue(math.isfinite(float(row[key])))

    def test_common_geography_is_invariant_to_row_order(self) -> None:
        x = np.array([1, 5, 7, 2, 9, 3, 4, 8, 10, 6, 11], dtype=float)
        y = np.array([8, 2, 3, 9, 1, 10, 7, 5, 11, 6, 4], dtype=float)
        z = np.column_stack((x + y, np.array([4, 9, 7, 1, 3, 6, 2, 11, 5, 10, 8])))
        original = rank_residual_correlation(x, y, z)
        perm = np.array([3, 0, 9, 2, 6, 10, 1, 8, 7, 4, 5])
        reordered = rank_residual_correlation(x[perm], y[perm], z[perm, :])
        self.assertTrue(np.allclose(original, reordered, atol=1e-12))

    def test_two_control_result_is_invariant_to_control_column_order(self) -> None:
        x = np.array([1, 5, 7, 2, 9, 3, 4, 8, 10, 6, 11], dtype=float)
        y = np.array([8, 2, 3, 9, 1, 10, 7, 5, 11, 6, 4], dtype=float)
        z = np.column_stack((x + y, np.array([4, 9, 7, 1, 3, 6, 2, 11, 5, 10, 8])))
        result = rank_residual_correlation(x, y, z)
        reversed_controls = rank_residual_correlation(x, y, z[:, ::-1])
        self.assertTrue(np.allclose(result, reversed_controls, atol=1e-12))

    def test_ties_use_average_ranks(self) -> None:
        x = np.array([1, 2, 2, 4, 5, 5, 7, 8, 9, 10], dtype=float)
        y = np.array([5, 6, 2, 1, 1, 3, 7, 9, 8, 4], dtype=float)
        z = np.array([1, 3, 4, 5, 5, 2, 7, 6, 9, 8], dtype=float)
        rho, ex, oy = rank_residual_correlation(x, y, z[:, None])
        self.assertAlmostEqual(rho, partial_rank_correlation(x, y, z), places=12)
        self.assertGreater(ex, 0)
        self.assertGreater(oy, 0)

    def test_nonfinite_and_wrong_shapes_fail_closed(self) -> None:
        x = np.arange(10, dtype=float)
        y = 1.0 + x ** 2
        z = np.array([9, 4, 1, 2, 3, 6, 5, 10, 8, 7], dtype=float)[:, None]
        with self.assertRaisesRegex(ValueError, "one or two controls"):
            rank_residual_correlation(x, y[:-1], z)
        with self.assertRaisesRegex(ValueError, "one or two controls"):
            rank_residual_correlation(x, y, z[:, 0])
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_residual_correlation(x, y, np.where(z == 10, np.nan, z))

    def test_collinear_or_exhausted_rank_variation_fails_closed(self) -> None:
        x = np.arange(1, 11, dtype=float)
        y = np.array([6, 1, 5, 2, 7, 3, 10, 4, 8, 9], dtype=float)
        with self.assertRaisesRegex(ValueError, "almost all"):
            rank_residual_correlation(x, y, x[:, None])
        z = np.column_stack((x, x * 3))
        with self.assertRaisesRegex(ValueError, "Rank-deficient"):
            rank_residual_correlation(x, y, z)

    def test_missing_or_duplicate_municipality_grid_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
            matched_common_controls(self.windows[:-1], self.distances)
        with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
            matched_common_controls([*self.windows, self.windows[0]], self.distances)

    def test_missing_summary_rows_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "576"):
            summary_grid(self.matched[:-1])
        with self.assertRaisesRegex(ValueError, "576"):
            summary_grid([*self.matched[:-1], self.matched[0]])

    def test_outputs_regenerate_with_verified_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            report = run(destination)
            self.assertEqual(report["matched_rows"], 576)
            self.assertEqual(report["summary_rows"], 36)
            self.assertEqual(report["centroid_crs_epsg"], 3763)
            self.assertEqual(report["commuting_network_year"], 2021)
            self.assertEqual(report["shared_control_sets"], list(CONTROLS))
            self.assertEqual(len(report["primary_mode_summary"]), 12)
            for figure in FIGURES:
                self.assertTrue((destination / "figures" / figure).read_bytes().startswith(b"\x89PNG"))
            for filename, expected_rows in zip(RESULTS[:2], (self.matched, self.summary)):
                with (destination / "results" / filename).open(encoding="utf-8", newline="") as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), len(expected_rows))
                self.assertEqual(list(rows[0]), list(expected_rows[0]))
            saved = json.loads((destination / "results" / RESULTS[2]).read_text(encoding="utf-8"))
            self.assertEqual(saved["housing_panel_sha256"], report["housing_panel_sha256"])
            self.assertEqual(saved["caop2025_derived_csv_sha256"], report["caop2025_derived_csv_sha256"])
            self.assertEqual(saved["commuting_derived_pair_csv_sha256"], report["commuting_derived_pair_csv_sha256"])

    def test_original_geometry_and_network_sources_are_unmodified(self) -> None:
        self.assertEqual(len(self.distances), 272)
        self.assertTrue(all(r["caop_year"] == 2025 for r in self.distances))
        self.assertEqual(len({(r["reference_municipality"], r["comparator_municipality"])
                              for r in self.distances}), 272)
        primary = next(r for r in self.matched if r["window_start_year"] == 2020
                       and r["alternative_reference"] == "Gondomar"
                       and r["exposure_mode"] == PRIMARY_MODE
                       and r["shared_control_set"] == "both_distances")
        self.assertAlmostEqual(float(primary["shared_alternative_minus_porto_rho"]),
                               -0.166462234, places=8)
        self.assertAlmostEqual(spearman_rho(np.arange(10), np.arange(10)), 1.0)


if __name__ == "__main__":
    unittest.main()
