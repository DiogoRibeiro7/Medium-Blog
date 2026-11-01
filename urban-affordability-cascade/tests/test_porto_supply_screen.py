"""Audit the pinned construction-flow versus new-rent screening diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis import INPUT_PANEL, MUNICIPALITIES, load_panel
from porto_supply_screen import (FIGURES, MODES, OUTCOMES, RESULTS, WINDOW_ENDS,
                                 NORMALIZATIONS, _outcome, _supply, construction_panel,
                                 diagnostics, municipal_exposures, run)


class PortoSupplyScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = construction_panel()
        cls.rows = municipal_exposures()
        cls.stats, cls.deletions = diagnostics(cls.rows)

    def test_complete_seventeen_by_seven_construction_counts(self):
        self.assertEqual(len(self.records), 119)
        self.assertEqual({name for name, _ in self.records}, MUNICIPALITIES)

    def test_2021_2025_preliminary_flags(self):
        prelim = sum('construction_preliminary' in str(row['flags']) for row in self.records.values())
        self.assertEqual(prelim, 85)

    def test_2024_2025_estimated_completions_flags(self):
        estimated = sum('completions_estimated' in str(row['flags']) for row in self.records.values())
        self.assertEqual(estimated, 34)

    def test_sixteen_named_comparators(self):
        self.assertEqual(len(self.rows), 16)
        self.assertEqual({r['municipality'] for r in self.rows}, MUNICIPALITIES - {'Porto'})

    def test_construction_counts_nonnegative_integers(self):
        for record in self.records.values():
            for key in ('completions', 'permits'):
                self.assertIsInstance(record[key], int)
                self.assertGreaterEqual(record[key], 0)

    def test_exact_three_windows(self):
        self.assertEqual(WINDOW_ENDS, (2023, 2024, 2025))
        self.assertEqual(len(self.stats), 36)
        self.assertEqual(len({(r['construction_window_end'], r['construction_exposure'],
                              r['normalization'], r['rent_outcome']) for r in self.stats}), 36)

    def test_primary_completed_dwellings_association(self):
        self.assertAlmostEqual(self._rho(2023, MODES[0], OUTCOMES[0]), .179412, places=6)

    def test_primary_licensed_dwellings_association(self):
        self.assertAlmostEqual(self._rho(2023, MODES[1], OUTCOMES[0]), -.055882, places=6)

    def test_primary_within_municipality_rent_growth_check(self):
        self.assertAlmostEqual(self._rho(2023, MODES[0], OUTCOMES[1]), .161765, places=6)
        self.assertAlmostEqual(self._rho(2023, MODES[1], OUTCOMES[1]), -.085294, places=6)

    def test_2024_2025_window_sensitivity(self):
        self.assertAlmostEqual(self._rho(2024, MODES[0], OUTCOMES[0]), .197059, places=6)
        self.assertAlmostEqual(self._rho(2025, MODES[0], OUTCOMES[0]), .167647, places=6)
        self.assertAlmostEqual(self._rho(2024, MODES[1], OUTCOMES[0]), -.032353, places=6)
        self.assertAlmostEqual(self._rho(2025, MODES[1], OUTCOMES[0]), -.044118, places=6)

    def test_output_is_independent_of_commuting_matrix(self):
        with mock.patch('commuting_network.load_weights', side_effect=AssertionError('commuting source not required')):
            self.assertEqual(len(municipal_exposures()), 16)

    def test_population_denominator_arithmetic(self):
        for row in self.rows:
            raw = float(row['completed_dwellings_2020_2023_raw_count'])
            pop = float(row['municipal_population_2020'])
            scaled = float(row['completed_dwellings_2020_2023_per_1000_population_2020'])
            self.assertAlmostEqual(scaled, 1000 * raw / pop, places=6)

    def test_outcome_matches_raw_published_rents(self):
        observations = load_panel()
        porto_2020, porto_2025 = observations['Porto', 2020].rent_m2, observations['Porto', 2025].rent_m2
        for row in self.rows:
            name = row['municipality']
            rent_2020, rent_2025 = observations[name, 2020].rent_m2, observations[name, 2025].rent_m2
            self.assertAlmostEqual(float(row[OUTCOMES[0]]),
                                   100 * (rent_2025 / porto_2025 - rent_2020 / porto_2020), places=6)

    def test_invalid_exposure_mode_rejected(self):
        with self.assertRaises(ValueError):
            _supply(self.rows, 'other', 2023, NORMALIZATIONS[1])

    def test_invalid_outcome_rejected(self):
        with self.assertRaises(ValueError):
            _outcome(self.rows, 'causal_displacement')

    def test_invalid_comparator_identity_rejected(self):
        modified = [dict(row) for row in self.rows]
        modified[0]['municipality'] = 'Outside the AMP'
        with self.assertRaises(ValueError):
            diagnostics(modified)

    def test_leave_one_out_complete_and_bounded(self):
        self.assertEqual(len(self.deletions), 32)
        for mode in MODES:
            selected = [float(r['leave_one_out_15_spearman_rho']) for r in self.deletions
                        if r['construction_exposure'] == mode]
            self.assertEqual(len(selected), 16)
            self.assertTrue(all(-1 <= x <= 1 for x in selected))

    def test_tampered_source_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            changed = Path(temp) / 'panel.csv'
            changed.write_bytes(INPUT_PANEL.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                construction_panel(changed)

    def test_invalid_preliminary_flag_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            changed = Path(temp) / 'panel.csv'
            content = INPUT_PANEL.read_text(encoding='utf-8')
            content = content.replace('construction_preliminary;population_administrative_basis;completions_estimated',
                                      'population_administrative_basis;completions_estimated', 1)
            changed.write_text(content, encoding='utf-8')
            with mock.patch('porto_supply_screen.SUBSET_SHA256', hashlib.sha256(changed.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'preliminary flag'):
                    construction_panel(changed)

    def test_diagnostic_provenance_and_artifact_reproduction(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            report = run(base)
            self.assertEqual(report['rank_diagnostic_rows'], 36)
            self.assertEqual(report['influence_rows'], 32)
            self.assertEqual(report['construction_preliminary_rows_2021_2025'], 85)
            self.assertEqual(report['estimated_completed_dwellings_rows_2024_2025'], 34)
            self.assertEqual(report['primary_completion_vs_discount_rho'], '0.179412')
            self.assertEqual(report['primary_permit_vs_discount_rho'], '-0.055882')
            self.assertEqual(len(report['limitations']), 9)
            for name in FIGURES:
                self.assertGreater((base / 'figures' / name).stat().st_size, 10_000)
            for name in RESULTS:
                self.assertTrue((base / 'results' / name).is_file())
            with (base / 'results' / RESULTS[1]).open(encoding='utf-8', newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 36)
            loaded = json.loads((base / 'results' / RESULTS[3]).read_text(encoding='utf-8'))
            self.assertEqual(loaded['evidence_class'], report['evidence_class'])

    def _rho(self, end: int, mode: str, outcome: str) -> float:
        row = next(r for r in self.stats if r['construction_window_end'] == end
                   and r['construction_exposure'] == mode
                   and r['normalization'] == 'per_1000_population_2020'
                   and r['rent_outcome'] == outcome)
        return float(row['spearman_rho'])


if __name__ == '__main__':
    unittest.main()
