"""Regression checks for official-geometry confounding of reference ranks."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import MUNICIPALITIES
from municipal_reference_geography_controls import (
    FIGURES, HIGH_RANK_DISTANCE_CORRELATION, RESULTS, _distance_index,
    _distances, distance_pairs, full_reference_grid,
    matched_reference_controls, porto_influence, primary_summary, run,
)
from municipal_reference_window_sensitivity import (
    OUTCOMES, matched_window_ranks, window_pairs, window_ranks,
)
from porto_distance_sensitivity import (
    comparison_table, diagnostics, load_centroids, partial_rank_correlation,
)
from porto_network_gradient import MODES, WINDOW_STARTS


class ReferenceGeographyControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows = window_pairs()
        cls.distances = distance_pairs()
        cls.full = full_reference_grid(cls.windows, cls.distances)
        cls.matched = matched_reference_controls(cls.windows, cls.distances)
        cls.influence = porto_influence(cls.windows, cls.distances)
        cls.summary = primary_summary(cls.full, cls.matched, cls.influence)

    def test_official_centroids_are_used_without_a_new_download(self):
        self.assertEqual(len(load_centroids()), 17)
        self.assertEqual(len(self.distances), 17 * 16)
        self.assertTrue(all(r["caop_year"] == 2025 for r in self.distances))

    def test_every_ordered_municipality_pair_is_present(self):
        expected = {(a, b) for a in MUNICIPALITIES for b in MUNICIPALITIES if a != b}
        actual = {(r["reference_municipality"], r["comparator_municipality"])
                  for r in self.distances}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 272)

    def test_derived_distances_are_symmetric_and_positive(self):
        indexed = _distance_index(self.distances)
        self.assertTrue(all(d > 0 and math.isfinite(d) for d in indexed.values()))
        for (a, b), d in indexed.items():
            self.assertAlmostEqual(d, indexed[b, a], places=8)

    def test_porto_distances_agree_with_prior_geography_module(self):
        old = {r["municipality"]: float(r["caop2025_porto_centroid_euclidean_distance_km"])
               for r in comparison_table()}
        indexed = _distance_index(self.distances)
        for name in MUNICIPALITIES - {"Porto"}:
            self.assertAlmostEqual(indexed["Porto", name], old[name], delta=1e-6)

    def test_reject_incomplete_geometry_registry(self):
        centroids = load_centroids()
        centroids.pop("Trofa")
        with self.assertRaisesRegex(ValueError, "17 AMP"):
            distance_pairs(centroids)

    def test_reject_identical_centroids_for_distinct_municipalities(self):
        centroids = load_centroids()
        centroids["Trofa"] = centroids["Porto"]
        with self.assertRaisesRegex(ValueError, "Positive distance"):
            distance_pairs(centroids)

    def test_reject_nonfinite_projected_centroids(self):
        centroids = load_centroids()
        centroids["Porto"] = (float("nan"), centroids["Porto"][1])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            distance_pairs(centroids)

    def test_reject_missing_or_duplicated_distance_pair(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            _distance_index(self.distances[:-1])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            _distance_index([*self.distances, self.distances[0]])

    def test_reject_asymmetric_pair_distance(self):
        bad = [dict(row) for row in self.distances]
        bad[0]["centroid_distance_km"] = "100.000000000"
        with self.assertRaisesRegex(ValueError, "symmetric"):
            _distance_index(bad)

    def test_reject_bad_centroid_year_and_negative_distance(self):
        changed = [dict(row) for row in self.distances]
        changed[0]["caop_year"] = 2024
        with self.assertRaisesRegex(ValueError, "year"):
            _distance_index(changed)
        changed = [dict(row) for row in self.distances]
        changed[0]["centroid_distance_km"] = "-1"
        with self.assertRaisesRegex(ValueError, "non-positive"):
            _distance_index(changed)

    def test_distance_vector_order_tracks_the_comparator_rows(self):
        selected = [r for r in self.windows if r["window_start_year"] == 2020
                    and r["reference_municipality"] == "Porto"]
        indexed = _distance_index(self.distances)
        expected = [indexed["Porto", r["comparator_municipality"]] for r in selected]
        self.assertTrue(np.array_equal(_distances(selected, indexed), expected))

    def test_all_408_full_reference_cells_exist(self):
        expected = {(y, ref, mode, outcome) for y in WINDOW_STARTS
                    for ref in MUNICIPALITIES for mode in MODES for outcome in OUTCOMES}
        actual = {(r["window_start_year"], r["reference_municipality"],
                   r["exposure_mode"], r["outcome"]) for r in self.full}
        self.assertEqual(actual, expected)
        self.assertEqual(len(self.full), 408)
        self.assertTrue(all(r["comparators"] == 16 for r in self.full))

    def test_unadjusted_coefficients_match_previous_all_window_module(self):
        earlier = {(r["window_start_year"], r["reference_municipality"],
                    r["exposure_mode"], r["outcome"]): r for r in window_ranks(self.windows)}
        for row in self.full:
            key = (row["window_start_year"], row["reference_municipality"],
                   row["exposure_mode"], row["outcome"])
            self.assertAlmostEqual(float(row["unadjusted_spearman_rho"]),
                                   float(earlier[key]["spearman_rank_rho"]), places=8)
            self.assertEqual(row["unadjusted_absolute_rank_among_17"],
                             earlier[key]["absolute_rho_rank_among_17"])

    def test_all_partial_coefficients_and_absolute_ranks_are_bounded(self):
        for row in self.full:
            self.assertTrue(-1 <= float(row["conditional_on_centroid_distance_partial_spearman_rho"]) <= 1)
            self.assertTrue(1 <= int(row["conditional_absolute_rank_among_17"]) <= 17)
            self.assertTrue(float(row["rank_distance_vif_diagnostic"]) >= 1)

    def test_porto_primary_own_growth_conditional_coefficients(self):
        rows = [r for r in self.full if r["reference_municipality"] == "Porto"
                and r["exposure_mode"] == "per_1000_residents" and r["outcome"] == OUTCOMES[0]]
        self.assertEqual([r["window_start_year"] for r in rows], list(WINDOW_STARTS))
        self.assertEqual([r["conditional_absolute_rank_among_17"] for r in rows], [3, 14, 7, 6])
        for r, rho in zip(rows, (-.408715708, -.119518581, -.501237054, -.439712626)):
            self.assertAlmostEqual(float(r["conditional_on_centroid_distance_partial_spearman_rho"]),
                                   rho, places=7)

    def test_porto_exposure_geography_collinearity_is_explicit(self):
        rows = [r for r in self.full if r["window_start_year"] == 2020
                and r["exposure_mode"] == "per_1000_residents" and r["outcome"] == OUTCOMES[0]]
        porto = next(r for r in rows if r["reference_municipality"] == "Porto")
        self.assertAlmostEqual(float(porto["exposure_vs_centroid_distance_spearman_rho"]), -.95)
        self.assertEqual(int(porto["strong_exposure_distance_collinearity_flag"]), 1)
        self.assertEqual(sum(int(r["strong_exposure_distance_collinearity_flag"]) for r in rows), 4)
        self.assertEqual(HIGH_RANK_DISTANCE_CORRELATION, .90)

    def test_porto_2023_relative_compression_changes_sign_after_conditioning(self):
        row = next(r for r in self.full if r["window_start_year"] == 2023
                   and r["reference_municipality"] == "Porto"
                   and r["exposure_mode"] == "per_1000_residents"
                   and r["outcome"] == OUTCOMES[1])
        self.assertGreater(float(row["unadjusted_spearman_rho"]), 0)
        self.assertAlmostEqual(float(row["conditional_on_centroid_distance_partial_spearman_rho"]),
                               -.403844231, places=7)

    def test_2020_porto_relative_compression_matches_prior_distance_module(self):
        original, _ = diagnostics(comparison_table())
        original_score = next(float(r["partial_spearman_controlling_centroid_distance"])
                              for r in original if r["start_year"] == 2020
                              and r["exposure"] == "per_1000_residents")
        recent = next(float(r["conditional_on_centroid_distance_partial_spearman_rho"])
                      for r in self.full if r["window_start_year"] == 2020
                      and r["reference_municipality"] == "Porto"
                      and r["outcome"] == OUTCOMES[1]
                      and r["exposure_mode"] == "per_1000_residents")
        self.assertAlmostEqual(recent, original_score, delta=1e-6)

    def test_identical_network_distance_rhos_across_rent_windows(self):
        for reference in MUNICIPALITIES:
            for mode in MODES:
                scores = {r["exposure_vs_centroid_distance_spearman_rho"]
                          for r in self.full if r["reference_municipality"] == reference
                          and r["exposure_mode"] == mode}
                self.assertEqual(len(scores), 1)

    def test_matched_support_has_192_valid_shared_outcome_comparisons(self):
        self.assertEqual(len(self.matched), 192)
        self.assertTrue(all(r["outcome"] == OUTCOMES[0] for r in self.matched))
        for row in self.matched:
            self.assertEqual(int(row["shared_comparators"]), 15)
            self.assertEqual(set(row["shared_municipality_names"].split("|")),
                             MUNICIPALITIES - {"Porto", row["alternative_reference"]})
            self.assertEqual(row["conditioning_distance_differs_by_reference"], 1)

    def test_matched_unadjusted_scores_equal_previous_window_audit(self):
        earlier = {(r["window_start_year"], r["alternative_reference"],
                    r["exposure_mode"]): r for r in matched_window_ranks(self.windows)}
        for row in self.matched:
            key = (row["window_start_year"], row["alternative_reference"], row["exposure_mode"])
            self.assertAlmostEqual(float(row["raw_alternative_minus_porto_rho"]),
                                   float(earlier[key]["alternative_minus_porto_rho"]), places=8)

    def test_matched_conditional_reference_counts_vary_by_window(self):
        expected_strict = (2, 5, 0, 4)
        expected_margin = (1, 4, 0, 4)
        selected = [r for r in self.summary if r["outcome"] == OUTCOMES[0]]
        self.assertEqual([r["matched_alternatives_more_negative_conditional_count"]
                          for r in selected], list(expected_strict))
        self.assertEqual([r["matched_alternatives_at_least_0_05_more_negative_count"]
                          for r in selected], list(expected_margin))

    def test_porto_loo_has_all_64_deletions_and_2021_sign_fragility(self):
        self.assertEqual(len(self.influence), 4 * 16)
        expected = {(y, name) for y in WINDOW_STARTS for name in MUNICIPALITIES - {"Porto"}}
        actual = {(r["window_start_year"], r["removed_comparator"]) for r in self.influence}
        self.assertEqual(actual, expected)
        flips = {y: sum(int(r["conditional_sign_changed"]) for r in self.influence
                        if r["window_start_year"] == y) for y in WINDOW_STARTS}
        self.assertEqual(flips, {2020: 0, 2021: 1, 2022: 0, 2023: 0})

    def test_primary_summary_has_all_eight_outcome_window_rows(self):
        self.assertEqual(len(self.summary), 8)
        self.assertEqual({(r["window_start_year"], r["outcome"]) for r in self.summary},
                         {(y, outcome) for y in WINDOW_STARTS for outcome in OUTCOMES})
        self.assertTrue(all(int(r["high_collinearity_references_out_of_17"]) == 4
                            for r in self.summary))

    def test_reference_grid_rejects_incomplete_window_data(self):
        with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
            full_reference_grid(self.windows[:-1], self.distances)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            matched_reference_controls(self.windows, self.distances[:-1])

    def test_partial_rank_requires_noncollinear_control(self):
        x = np.arange(1, 17, dtype=float)
        with self.assertRaisesRegex(ValueError, "collinearity"):
            partial_rank_correlation(x, x[::-1], x)

    def test_all_outputs_reproduce_offline_in_a_clean_directory(self):
        with tempfile.TemporaryDirectory() as location:
            output = Path(location)
            report = run(output)
            for name in FIGURES:
                path = output / "figures" / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 20_000)
            for name in RESULTS:
                self.assertTrue((output / "results" / name).is_file(), name)
            counts = (272, 408, 192, 64, 8)
            for name, expected in zip(RESULTS[:5], counts):
                with (output / "results" / name).open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(sum(1 for _ in csv.DictReader(handle)), expected)
            saved = json.loads((output / "results" / RESULTS[-1]).read_text(encoding="utf-8"))
            self.assertEqual(report, saved)
            self.assertEqual(saved["centroid_crs_epsg"], 3763)
            self.assertEqual(saved["full_reference_rows"], 408)
            self.assertIn("causal", " ".join(saved["limits"]))


if __name__ == "__main__":
    unittest.main()
