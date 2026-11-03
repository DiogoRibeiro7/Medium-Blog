"""Regression checks for overlapping-window reference-node controls."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from analysis import MUNICIPALITIES, load_panel
from municipal_reference_placebos import pair_table, matched_support
from municipal_reference_window_sensitivity import (
    FIGURES, MATERIAL_RHO_DIFFERENCE, MODE_COLUMNS, OUTCOMES, RESULTS,
    _window_vectors, matched_window_ranks, primary_summary, run,
    window_pairs, window_ranks,
)
from porto_network_gradient import MODES, WINDOW_STARTS


class ReferenceWindowSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = window_pairs()
        cls.ranks = window_ranks(cls.pairs)
        cls.matched = matched_window_ranks(cls.pairs)
        cls.summary = primary_summary(cls.ranks, cls.matched)

    def test_prespecified_baselines_and_declared_outcomes(self):
        self.assertEqual(WINDOW_STARTS, (2020, 2021, 2022, 2023))
        self.assertEqual(len(OUTCOMES), 2)
        self.assertEqual(set(MODE_COLUMNS), set(MODES))

    def test_pair_window_grid_has_all_1088_unique_records(self):
        expected = {(r, c, y) for r in MUNICIPALITIES for c in MUNICIPALITIES
                    for y in WINDOW_STARTS if r != c}
        found = {(r['reference_municipality'], r['comparator_municipality'], r['window_start_year'])
                 for r in self.pairs}
        self.assertEqual(len(self.pairs), 1088)
        self.assertEqual(found, expected)
        self.assertTrue(all(r['window_end_year'] == 2025 for r in self.pairs))

    def test_fixed_2021_interactions_do_not_change_with_rent_baseline(self):
        lookup: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
        for r in self.pairs:
            lookup.setdefault((str(r['reference_municipality']), str(r['comparator_municipality'])), set()).add((
                str(r['interactions_per_1000_comparator_residents_2021']),
                str(r['pair_share_of_comparator_amp_interactions_2021']),
                int(r['undirected_work_study_pair_interactions_2021']),
            ))
        self.assertEqual(len(lookup), 272)
        self.assertTrue(all(len(values) == 1 for values in lookup.values()))

    def test_2020_window_agrees_with_original_reference_module(self):
        original = {(r['reference_municipality'], r['comparator_municipality']): r
                    for r in pair_table()}
        subset = [r for r in self.pairs if r['window_start_year'] == 2020]
        self.assertEqual(len(subset), 272)
        for r in subset:
            old = original[r['reference_municipality'], r['comparator_municipality']]
            self.assertAlmostEqual(float(r[OUTCOMES[0]]),
                                   float(old['own_new_rent_growth_2020_2025_pct']), places=8)
            self.assertAlmostEqual(float(r[OUTCOMES[1]]),
                                   float(old['reference_discount_compression_2020_2025_pp']), places=8)

    def test_each_window_uses_published_own_rent_growth(self):
        panel = load_panel()
        for r in self.pairs:
            comparator = r['comparator_municipality']
            start = int(r['window_start_year'])
            actual = 100 * (panel[comparator, 2025].rent_m2 / panel[comparator, start].rent_m2 - 1)
            self.assertAlmostEqual(float(r[OUTCOMES[0]]), actual, places=8)

    def test_reference_discount_compression_uses_correct_window(self):
        panel = load_panel()
        for r in self.pairs:
            reference, comparator, year = r['reference_municipality'], r['comparator_municipality'], r['window_start_year']
            actual = 100 * (panel[comparator, 2025].rent_m2 / panel[reference, 2025].rent_m2
                            - panel[comparator, year].rent_m2 / panel[reference, year].rent_m2)
            self.assertAlmostEqual(float(r[OUTCOMES[1]]), actual, places=8)

    def test_own_growth_is_reference_invariant_for_each_comparator_year(self):
        for comparator in MUNICIPALITIES:
            for year in WINDOW_STARTS:
                observed = {r[OUTCOMES[0]] for r in self.pairs
                            if r['comparator_municipality'] == comparator and r['window_start_year'] == year}
                self.assertEqual(len(observed), 1)

    def test_full_reference_rank_grid_has_408_rows(self):
        expected = {(year, ref, mode, outcome) for year in WINDOW_STARTS for ref in MUNICIPALITIES
                    for mode in MODES for outcome in OUTCOMES}
        observed = {(r['window_start_year'], r['reference_municipality'], r['exposure_mode'], r['outcome'])
                    for r in self.ranks}
        self.assertEqual(len(self.ranks), 408)
        self.assertEqual(observed, expected)
        self.assertTrue(all(r['comparator_municipalities'] == 16 for r in self.ranks))

    def test_rank_coefficients_and_absolute_ranks_are_bounded(self):
        for r in self.ranks:
            self.assertTrue(-1 <= float(r['spearman_rank_rho']) <= 1)
            self.assertTrue(1 <= int(r['absolute_rho_rank_among_17']) <= 17)

    def test_porto_own_growth_rank_changes_by_window(self):
        selected = [r for r in self.ranks if r['reference_municipality'] == 'Porto'
                    and r['exposure_mode'] == 'per_1000_residents' and r['outcome'] == OUTCOMES[0]]
        self.assertEqual([r['window_start_year'] for r in selected], list(WINDOW_STARTS))
        self.assertEqual([r['absolute_rho_rank_among_17'] for r in selected], [2, 6, 2, 8])
        for actual, expected in zip((float(r['spearman_rank_rho']) for r in selected),
                                    (-.385294117647, -.36470588235, -.56470588235, -.220588235294)):
            self.assertAlmostEqual(actual, expected, delta=.006)

    def test_porto_relative_discount_can_nearly_disappear_in_2023_window(self):
        selected = [r for r in self.ranks if r['reference_municipality'] == 'Porto'
                    and r['exposure_mode'] == 'per_1000_residents' and r['outcome'] == OUTCOMES[1]]
        self.assertEqual([r['absolute_rho_rank_among_17'] for r in selected], [6, 4, 6, 17])
        self.assertAlmostEqual(float(selected[-1]['spearman_rank_rho']), .02352941176, delta=.005)

    def test_matched_comparisons_have_192_exact_cells(self):
        expected = {(y, a, mode) for y in WINDOW_STARTS for a in MUNICIPALITIES - {'Porto'} for mode in MODES}
        observed = {(r['window_start_year'], r['alternative_reference'], r['exposure_mode'])
                    for r in self.matched}
        self.assertEqual(len(self.matched), 192)
        self.assertEqual(observed, expected)

    def test_matched_comparators_exclude_exactly_porto_and_alternative(self):
        for r in self.matched:
            expected = MUNICIPALITIES - {'Porto', r['alternative_reference']}
            self.assertEqual(set(r['shared_municipality_names'].split('|')), expected)
            self.assertEqual(r['shared_comparators'], 15)

    def test_2020_matched_values_agree_with_previous_module(self):
        earlier = {(r['alternative_reference'], r['exposure_mode']): r for r in matched_support(pair_table())
                   if r['outcome'] == 'own_new_rent_growth_2020_2025_pct'}
        for r in self.matched:
            if r['window_start_year'] != 2020:
                continue
            previous = earlier[r['alternative_reference'], r['exposure_mode']]
            self.assertAlmostEqual(float(r['alternative_minus_porto_rho']),
                                   float(previous['alternative_minus_porto_rho']), places=8)

    def test_primary_counts_more_negative_depend_on_window(self):
        selected = [r for r in self.summary if r['outcome'] == OUTCOMES[0]]
        self.assertEqual([r['window_start_year'] for r in selected], list(WINDOW_STARTS))
        self.assertEqual([r['matched_alternatives_more_negative_than_porto_count'] for r in selected], [1, 5, 0, 5])

    def test_material_difference_screen_is_not_statistical_inference(self):
        self.assertEqual(MATERIAL_RHO_DIFFERENCE, .05)
        selected = [r for r in self.summary if r['outcome'] == OUTCOMES[0]]
        self.assertEqual([r['matched_alternatives_at_least_0_05_more_negative_count'] for r in selected],
                         [1, 2, 0, 5])
        self.assertEqual(selected[0]['matched_alternatives_at_least_0_05_more_negative_names'], 'Gondomar')

    def test_primary_summary_has_eight_complete_rows(self):
        self.assertEqual(len(self.summary), 8)
        self.assertEqual({(r['window_start_year'], r['outcome']) for r in self.summary},
                         {(year, outcome) for year in WINDOW_STARTS for outcome in OUTCOMES})
        self.assertTrue(all(r['exposure_mode'] == 'per_1000_residents' for r in self.summary))

    def test_incomplete_or_duplicated_pair_grid_is_rejected(self):
        for malformed in (self.pairs[:-1], self.pairs + [self.pairs[0]]):
            with self.assertRaises(ValueError):
                window_ranks(malformed)
            with self.assertRaises(ValueError):
                matched_window_ranks(malformed)

    def test_unknown_mode_outcome_and_nonfinite_vector_rejected(self):
        subset = self.pairs[:16]
        with self.assertRaises(ValueError):
            _window_vectors(subset, 'invalid', OUTCOMES[0])
        with self.assertRaises(ValueError):
            _window_vectors(subset, 'per_1000_residents', 'invalid')
        with self.assertRaises(ValueError):
            _window_vectors(subset[:4], 'per_1000_residents', OUTCOMES[0])
        with self.assertRaises(ValueError):
            _window_vectors([{**r, OUTCOMES[0]: 'nan'} for r in subset], 'per_1000_residents', OUTCOMES[0])

    def test_primary_summary_rejects_missing_or_duplicate_rows(self):
        with self.assertRaises(ValueError):
            primary_summary(self.ranks[:-1], self.matched)
        with self.assertRaises(ValueError):
            primary_summary(self.ranks, self.matched + [self.matched[0]])

    def test_all_generated_artifacts_and_provenance_are_regenerable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run(root)
            self.assertEqual(report['pair_window_rows'], 1088)
            self.assertEqual(report['reference_rank_rows'], 408)
            self.assertEqual(report['matched_support_rows'], 192)
            self.assertEqual(report['primary_summary_rows'], 8)
            self.assertEqual(report['material_matched_rho_difference_threshold'], .05)
            self.assertEqual(len(report['limits']), 11)
            self.assertIn('not a randomized placebo', ' '.join(report['limits']))
            for fig in FIGURES:
                self.assertGreater((root / 'figures' / fig).stat().st_size, 15000)
            for filename, count in zip(RESULTS[:4], (1088, 408, 192, 8)):
                with (root / 'results' / filename).open(encoding='utf-8', newline='') as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)
            ledger = json.loads((root / 'results' / RESULTS[-1]).read_text(encoding='utf-8'))
            self.assertEqual(ledger, report)


if __name__ == '__main__':
    unittest.main()
