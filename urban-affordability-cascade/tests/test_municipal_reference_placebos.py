"""Regressions for all-reference negative controls and matched support."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from analysis import MUNICIPALITIES, load_panel
from commuting_network import load_weights
from municipal_reference_placebos import (
    FIGURES, MODE_COLUMNS, OUTCOMES, REFERENCE, RESULTS, _vectors,
    leave_one_out, matched_support, pair_table, rank_associations, run,
)
from porto_network_gradient import MODES, exposure_table


class MunicipalReferencePlaceboTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = pair_table()
        cls.ranks = rank_associations(cls.pairs)
        cls.matched = matched_support(cls.pairs)
        cls.deletions = leave_one_out(cls.pairs)

    def test_all_seventeen_references_have_sixteen_distinct_counterparts(self):
        self.assertEqual(len(self.pairs), 272)
        self.assertEqual(Counter(str(r['reference_municipality']) for r in self.pairs),
                         {name: 16 for name in MUNICIPALITIES})
        self.assertEqual(len({(r['reference_municipality'], r['comparator_municipality']) for r in self.pairs}), 272)

    def test_no_reference_pair_is_its_own_comparator(self):
        self.assertTrue(all(r['reference_municipality'] != r['comparator_municipality'] for r in self.pairs))

    def test_original_pair_weights_are_symmetric_across_reference_views(self):
        lookup = {(r['reference_municipality'], r['comparator_municipality']): r for r in self.pairs}
        for (origin, destination), r in lookup.items():
            self.assertEqual(r['undirected_work_study_pair_interactions_2021'],
                             lookup[destination, origin]['undirected_work_study_pair_interactions_2021'])

    def test_pair_counts_match_original_source_matrix(self):
        names, weights = load_weights()
        lookup = {(r['reference_municipality'], r['comparator_municipality']): r for r in self.pairs}
        for i, origin in enumerate(names):
            for j, destination in enumerate(names):
                if i != j:
                    self.assertEqual(lookup[origin, destination]['undirected_work_study_pair_interactions_2021'],
                                     int(weights[i, j]))

    def test_own_growth_is_invariant_across_reference_nodes(self):
        for municipality in MUNICIPALITIES:
            values = {r['own_new_rent_growth_2020_2025_pct'] for r in self.pairs
                      if r['comparator_municipality'] == municipality}
            self.assertEqual(len(values), 1)

    def test_own_growth_uses_published_panel_endpoints(self):
        panel = load_panel()
        for row in self.pairs:
            destination = row['comparator_municipality']
            actual = 100 * (panel[destination, 2025].rent_m2 / panel[destination, 2020].rent_m2 - 1)
            self.assertAlmostEqual(float(row['own_new_rent_growth_2020_2025_pct']), actual, places=8)

    def test_compression_matches_exact_reference_ratio_formula(self):
        for row in self.pairs:
            old = float(row['comparator_2020_new_rent_eur_m2_month']) / float(row['reference_2020_new_rent_eur_m2_month'])
            new = float(row['comparator_2025_new_rent_eur_m2_month']) / float(row['reference_2025_new_rent_eur_m2_month'])
            self.assertAlmostEqual(float(row['reference_discount_compression_2020_2025_pp']),
                                   100 * (new - old), places=8)

    def test_original_porto_exposure_record_agrees_with_prior_module(self):
        earlier = {r['municipality']: r for r in exposure_table()}
        current = {r['comparator_municipality']: r for r in self.pairs
                   if r['reference_municipality'] == REFERENCE}
        self.assertEqual(set(current), set(earlier))
        for name in earlier:
            self.assertEqual(current[name]['undirected_work_study_pair_interactions_2021'],
                             int(earlier[name]['porto_undirected_work_study_interactions_2021']))
            self.assertAlmostEqual(float(current[name]['interactions_per_1000_comparator_residents_2021']),
                                   float(earlier[name]['porto_link_interactions_per_1000_municipal_residents']), places=5)
            self.assertAlmostEqual(float(current[name]['reference_discount_compression_2020_2025_pp']),
                                   float(earlier[name]['discount_compression_2020_2025_percentage_points']), places=5)

    def test_all_modes_and_outcomes_are_declared(self):
        self.assertEqual(set(MODE_COLUMNS), set(MODES))
        self.assertEqual(len(OUTCOMES), 2)
        self.assertIn('own_new_rent_growth_2020_2025_pct', OUTCOMES)

    def test_rank_table_has_complete_102_cell_grid(self):
        expected = {(name, mode, outcome) for name in MUNICIPALITIES for mode in MODES for outcome in OUTCOMES}
        observed = {(r['reference_municipality'], r['exposure_mode'], r['outcome']) for r in self.ranks}
        self.assertEqual(len(self.ranks), 102)
        self.assertEqual(observed, expected)
        self.assertTrue(all(r['comparator_municipalities'] == 16 for r in self.ranks))

    def test_full_reference_correlations_are_bounded(self):
        for row in self.ranks:
            self.assertTrue(-1 <= float(row['spearman_rank_rho']) <= 1)
            self.assertTrue(1 <= int(row['absolute_rho_rank_among_17_references']) <= 17)

    def test_exact_porto_primary_reference_correlations_and_ranks(self):
        lookup = {(r['reference_municipality'], r['exposure_mode'], r['outcome']): r for r in self.ranks}
        own = lookup[REFERENCE, 'per_1000_residents', OUTCOMES[0]]
        relative = lookup[REFERENCE, 'per_1000_residents', OUTCOMES[1]]
        self.assertAlmostEqual(float(own['spearman_rank_rho']), -.385294117647, places=8)
        self.assertAlmostEqual(float(relative['spearman_rank_rho']), -.297058823529, places=8)
        self.assertEqual(own['absolute_rho_rank_among_17_references'], 2)
        self.assertEqual(relative['absolute_rho_rank_among_17_references'], 6)

    def test_gondomar_is_more_negative_than_porto_on_full_16_own_growth(self):
        primary = {(r['reference_municipality'], r['outcome']): float(r['spearman_rank_rho'])
                   for r in self.ranks if r['exposure_mode'] == 'per_1000_residents'}
        self.assertLess(primary['Gondomar', OUTCOMES[0]], primary['Porto', OUTCOMES[0]])

    def test_matched_support_covers_every_alt_reference_and_mode(self):
        self.assertEqual(len(self.matched), 16 * 3 * 2)
        expected = {(alt, mode, outcome) for alt in MUNICIPALITIES - {REFERENCE}
                    for mode in MODES for outcome in OUTCOMES}
        self.assertEqual({(r['alternative_reference'], r['exposure_mode'], r['outcome'])
                          for r in self.matched}, expected)

    def test_matched_support_comparator_lists_are_identical_fifteen(self):
        for r in self.matched:
            alternative = r['alternative_reference']
            actual = set(r['shared_municipality_names'].split('|'))
            self.assertEqual(actual, set(MUNICIPALITIES) - {alternative, REFERENCE})
            self.assertEqual(r['shared_comparators'], 15)

    def test_matched_primary_only_gondomar_is_more_negative(self):
        more_negative = {r['alternative_reference'] for r in self.matched
                         if r['outcome'] == OUTCOMES[0] and r['exposure_mode'] == 'per_1000_residents'
                         and float(r['alternative_minus_porto_rho']) < 0}
        self.assertEqual(more_negative, {'Gondomar'})

    def test_matched_outcome_invariance_is_explicitly_labeled(self):
        for r in self.matched:
            self.assertEqual(r['outcome_reference_invariant'],
                             'true' if r['outcome'] == OUTCOMES[0] else 'false')
            self.assertAlmostEqual(float(r['alternative_reference_rho_on_same_15'])
                                   - float(r['porto_reference_rho_on_same_15']),
                                   float(r['alternative_minus_porto_rho']), places=8)

    def test_all_272_primary_single_comparator_deletions(self):
        self.assertEqual(len(self.deletions), 272)
        self.assertEqual(Counter(r['reference_municipality'] for r in self.deletions),
                         {name: 16 for name in MUNICIPALITIES})
        self.assertTrue(all(r['remaining_comparators'] == 15 for r in self.deletions))
        self.assertEqual(len({(r['reference_municipality'], r['omitted_comparator'])
                              for r in self.deletions}), 272)

    def test_leave_one_out_coefficients_are_finite_and_bounded(self):
        for r in self.deletions:
            value = float(r['leave_one_out_15_rho'])
            self.assertTrue(math.isfinite(value) and -1 <= value <= 1)
            self.assertTrue(-1 <= float(r['full_16_rho']) <= 1)

    def test_incomplete_or_duplicate_pair_grid_is_rejected(self):
        for malformed in (self.pairs[:-1], self.pairs + [self.pairs[0]]):
            with self.assertRaisesRegex(ValueError, '17×16 distinct'):
                rank_associations(malformed)
            with self.assertRaises(ValueError):
                matched_support(malformed)
            with self.assertRaises(ValueError):
                leave_one_out(malformed)

    def test_unknown_mode_outcome_or_bad_vectors_rejected(self):
        with self.assertRaises(ValueError):
            _vectors(self.pairs[:16], 'not_a_mode', OUTCOMES[0])
        with self.assertRaises(ValueError):
            _vectors(self.pairs[:16], 'per_1000_residents', 'not_an_outcome')
        with self.assertRaises(ValueError):
            _vectors(self.pairs[:4], 'per_1000_residents', OUTCOMES[0])
        with self.assertRaises(ValueError):
            _vectors([{**r, MODE_COLUMNS['per_1000_residents']: 'nan'} for r in self.pairs[:16]],
                     'per_1000_residents', OUTCOMES[0])

    def test_all_artifacts_and_provenance_regenerate_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run(root)
            self.assertEqual(report['reference_comparator_rows'], 272)
            self.assertEqual(report['full_reference_rank_rows'], 102)
            self.assertEqual(report['matched_porto_alternative_contrasts'], 96)
            self.assertEqual(report['primary_leave_one_out_rows'], 272)
            self.assertEqual(len(report['limits']), 10)
            self.assertEqual(report['alternative_references_more_negative_than_porto_on_same_15_own_growth'],
                             ['Gondomar'])
            for fig in FIGURES:
                self.assertGreater((root / 'figures' / fig).stat().st_size, 15000)
            for filename, expected in zip(RESULTS[:4], (272, 102, 96, 272)):
                with (root / 'results' / filename).open(encoding='utf-8', newline='') as file:
                    self.assertEqual(len(list(csv.DictReader(file))), expected)
            ledger = json.loads((root / 'results' / RESULTS[-1]).read_text(encoding='utf-8'))
            self.assertEqual(ledger, report)
            self.assertIn('third-party', ledger['commuting_source_status'])


if __name__ == '__main__':
    unittest.main()
