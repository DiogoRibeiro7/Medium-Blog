"""Regression checks for the descriptive Porto-link/rent-gradient audit."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from porto_network_gradient import (
    FIGURES, RESULTS, _compression, _mode_values, average_ranks,
    diagnostic_tables, exposure_table, leave_one_out_rhos,
    random_label_benchmark, run, spearman_rho,
)


class PortoNetworkGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = exposure_table()
        cls.stats, cls.deletions = diagnostic_tables(cls.rows, permutations=99)

    def test_sixteen_unique_municipalities(self):
        self.assertEqual(len(self.rows), 16)
        self.assertEqual(len({r['municipality'] for r in self.rows}), 16)

    def test_average_ranks_with_ties(self):
        np.testing.assert_array_equal(average_ranks(np.array([3, 2, 2, 7])), [3, 1.5, 1.5, 4])

    def test_constant_ranks_rejected(self):
        with self.assertRaises(ValueError):
            average_ranks(np.array([1, 1, 1, 1]))

    def test_spearman_reversal(self):
        self.assertAlmostEqual(spearman_rho(np.arange(5.), np.arange(5.)[::-1]), -1)

    def test_bad_shuffles_rejected(self):
        with self.assertRaises(ValueError):
            random_label_benchmark(np.arange(5.), np.arange(5.), permutations=98)

    def test_shuffles_reproducible_and_bounded(self):
        x, y = np.arange(7.), np.array([5, 2, 6, 3, 1, 4, 0.])
        first = random_label_benchmark(x, y, permutations=99, seed=42)
        self.assertEqual(first, random_label_benchmark(x, y, permutations=99, seed=42))
        self.assertGreaterEqual(first[1], .01)
        self.assertLessEqual(first[1], 1.)

    def test_twelve_declared_windows_and_exposure_modes(self):
        self.assertEqual(len(self.stats), 12)
        self.assertEqual(len({(r['window_start_year'], r['exposure_mode']) for r in self.stats}), 12)

    def test_population_confounds_raw_link_counts(self):
        raw = _mode_values(self.rows, 'raw_interactions')
        pop = np.array([float(r['municipal_population_2021']) for r in self.rows])
        self.assertAlmostEqual(spearman_rho(raw, pop), .9735294117647059)

    def test_primary_normalized_association(self):
        self.assertAlmostEqual(spearman_rho(_mode_values(self.rows, 'per_1000_residents'),
                                            _compression(self.rows, 2020)), -.29705882352941176)

    def test_raw_association(self):
        self.assertAlmostEqual(spearman_rho(_mode_values(self.rows, 'raw_interactions'),
                                            _compression(self.rows, 2020)), -.17941176470588235)

    def test_window_sensitivity(self):
        r = next(r for r in self.stats if r['window_start_year'] == 2023
                 and r['exposure_mode'] == 'raw_interactions')
        self.assertAlmostEqual(float(r['spearman_rho']), .158824, places=6)

    def test_leave_one_out_complete_and_nonpositive(self):
        self.assertEqual(len(self.deletions), 16)
        rho = leave_one_out_rhos(_mode_values(self.rows, 'per_1000_residents'),
                                 _compression(self.rows, 2020))
        self.assertEqual(len(rho), 16)
        self.assertLess(max(rho), 0)

    def test_specific_pair_counts(self):
        counts = {r['municipality']: int(r['porto_undirected_work_study_interactions_2021'])
                  for r in self.rows}
        self.assertEqual(counts['Vila Nova de Gaia'], 39960)
        self.assertEqual(counts['Matosinhos'], 31142)
        self.assertEqual(counts['Arouca'], 512)

    def test_artifacts_and_caveat_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            report = run(base, permutations=99)
            self.assertEqual(report['municipalities'], 16)
            self.assertEqual(len(report['limits']), 10)
            self.assertIn('not inferential p-values', ' '.join(report['limits']))
            for name in FIGURES:
                self.assertGreater((base / 'figures' / name).stat().st_size, 10000)
            for name in RESULTS:
                self.assertTrue((base / 'results' / name).is_file())
            with (base / 'results' / RESULTS[1]).open(newline='', encoding='utf-8') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 12)
            saved = json.loads((base / 'results' / RESULTS[3]).read_text(encoding='utf-8'))
            self.assertEqual(saved['raw_porto_pair_count_vs_2021_population_spearman_rho'], '0.973529')


if __name__ == '__main__':
    unittest.main()
