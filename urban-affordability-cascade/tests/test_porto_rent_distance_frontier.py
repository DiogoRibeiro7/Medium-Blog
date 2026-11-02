"""Regression contracts for the descriptive municipal rent/distance screen."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis import load_panel
from porto_rent_options import COMPARATORS
from porto_rent_distance_frontier import (
    AREA_M2, DISTANCE_MARGINS_KM, FIGURES, RENT_MARGINS_EUR_M2, RESULTS, YEARS,
    MunicipalOption, margin_table, municipal_options, municipal_table,
    pareto_dominators, plot_margins, run, status_changes,
)


class PortoRentDistanceFrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = municipal_options()
        cls.table = municipal_table(cls.options)
        cls.margins = margin_table(cls.options)
        cls.changes = status_changes(cls.table)

    def test_both_years_have_exactly_sixteen_named_comparators(self):
        self.assertEqual(set(self.options), set(YEARS))
        for year in YEARS:
            self.assertEqual({o.municipality for o in self.options[year]}, set(COMPARATORS))
            self.assertEqual(len(self.options[year]), 16)

    def test_municipal_medias_are_exactly_source_linked(self):
        panel = load_panel()
        for year in YEARS:
            for point in self.options[year]:
                self.assertEqual(point.new_contract_rent_eur_m2, panel[point.municipality, year].rent_m2)

    def test_distance_is_fixed_geometry_in_both_rent_years(self):
        old = {o.municipality: o.distance_km for o in self.options[2020]}
        new = {o.municipality: o.distance_km for o in self.options[2025]}
        self.assertEqual(old, new)
        self.assertAlmostEqual(new['Matosinhos'], 7.096492, places=5)
        self.assertAlmostEqual(new['Vale de Cambra'], 43.641129, places=5)

    def test_known_2025_rent_and_hypothetical_monthly_amount(self):
        matosinhos = next(r for r in self.table if r['municipality'] == 'Matosinhos' and r['rent_year'] == 2025)
        self.assertEqual(matosinhos['new_contract_median_rent_eur_m2_month'], '12.120000')
        self.assertEqual(matosinhos['synthetic_70m2_monthly_rent_eur'], f'{12.12 * AREA_M2:.6f}')
        self.assertEqual(matosinhos['synthetic_70m2_rent_saving_vs_porto_eur_month'], '140.700000')

    def test_pareto_dominance_on_synthetic_points(self):
        points = [MunicipalOption('A', 2025, 1, 5), MunicipalOption('B', 2025, 2, 4),
                  MunicipalOption('C', 2025, 3, 5), MunicipalOption('D', 2025, 4, 3)]
        d = pareto_dominators(points)
        self.assertEqual(d['C'], ('A', 'B'))
        self.assertEqual(d['A'], ())
        self.assertEqual(d['B'], ())
        self.assertEqual(d['D'], ())

    def test_equal_two_dimensional_points_do_not_dominate_one_another(self):
        a = MunicipalOption('A', 2025, 3, 5)
        b = MunicipalOption('B', 2025, 3, 5)
        self.assertEqual(pareto_dominators([a, b]), {'A': (), 'B': ()})

    def test_both_material_margins_must_be_met(self):
        points = [MunicipalOption('A', 2025, 1, 5), MunicipalOption('B', 2025, 2, 4),
                  MunicipalOption('C', 2025, 3, 5)]
        self.assertEqual(pareto_dominators(points, 0, 0)['C'], ('A', 'B'))
        self.assertEqual(pareto_dominators(points, 0, 1)['C'], ('B',))
        self.assertEqual(pareto_dominators(points, 2, 1)['C'], ())

    def test_invalid_or_duplicate_inputs_are_rejected(self):
        p = MunicipalOption('A', 2025, 3, 5)
        with self.assertRaises(ValueError):
            pareto_dominators([p])
        with self.assertRaises(ValueError):
            pareto_dominators([p, p])
        with self.assertRaises(ValueError):
            pareto_dominators([p, MunicipalOption('B', 2020, 4, 4)])

    def test_invalid_municipal_point_values_are_rejected(self):
        for distance, rent in ((0, 5), (-1, 5), (float('nan'), 5), (3, 0),
                               (3, float('inf')), (3, True)):
            with self.subTest(distance=distance, rent=rent), self.assertRaises(ValueError):
                MunicipalOption('X', 2025, distance, rent)
        with self.assertRaises(ValueError):
            MunicipalOption('X', True, 3, 5)
        with self.assertRaises(ValueError):
            MunicipalOption('', 2025, 3, 5)

    def test_invalid_margin_values_are_rejected(self):
        for invalid in (-1, float('nan'), float('inf'), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pareto_dominators(self.options[2025], invalid, 0)
            with self.subTest(invalid_rent=invalid), self.assertRaises(ValueError):
                pareto_dominators(self.options[2025], 0, invalid)

    def test_exact_frontier_counts_in_2020_and_2025(self):
        counts = {year: sum(not v for v in pareto_dominators(self.options[year]).values()) for year in YEARS}
        self.assertEqual(counts, {2020: 6, 2025: 7})

    def test_exact_non_dominated_municipalities(self):
        expected = {
            2020: {'Matosinhos', 'Maia', 'Valongo', 'Trofa', 'Paredes', 'Santo Tirso'},
            2025: {'Matosinhos', 'Maia', 'Valongo', 'Trofa', 'Paredes', 'Oliveira de Azeméis', 'Arouca'},
        }
        for year in YEARS:
            observed = {name for name, d in pareto_dominators(self.options[year]).items() if not d}
            self.assertEqual(observed, expected[year])

    def test_every_primary_dominated_summary_has_valid_dominator(self):
        for year in YEARS:
            lookup = {o.municipality: o for o in self.options[year]}
            for name, dominators in pareto_dominators(self.options[year]).items():
                candidate = lookup[name]
                for other_name in dominators:
                    other = lookup[other_name]
                    self.assertNotEqual(name, other_name)
                    self.assertLessEqual(other.distance_km, candidate.distance_km)
                    self.assertLessEqual(other.new_contract_rent_eur_m2, candidate.new_contract_rent_eur_m2)
                    self.assertTrue(other.distance_km < candidate.distance_km
                                    or other.new_contract_rent_eur_m2 < candidate.new_contract_rent_eur_m2)

    def test_vila_nova_de_gaia_is_dominated_by_maia_in_2025_summary(self):
        observed = pareto_dominators(self.options[2025])
        self.assertIn('Maia', observed['Vila Nova de Gaia'])
        self.assertEqual(observed['Arouca'], ())

    def test_dominance_unchanged_by_positive_drawing_floor_area(self):
        baseline = pareto_dominators(self.options[2025])
        for area in (50, 70, 90, 140):
            changed = [MunicipalOption(o.municipality, o.year, o.distance_km,
                                       o.new_contract_rent_eur_m2 * area)
                       for o in self.options[2025]]
            self.assertEqual(pareto_dominators(changed), baseline)

    def test_all_thirty_two_output_rows_and_no_duplicate_pairs(self):
        self.assertEqual(len(self.table), 32)
        self.assertEqual(len({(r['municipality'], r['rent_year']) for r in self.table}), 32)
        self.assertTrue(all(r['geography_reference'].startswith('DGT CAOP2025') for r in self.table))

    def test_municipal_grid_rejects_wrong_rent_year_or_missing_comparators(self):
        malformed = {year: list(points) for year, points in self.options.items()}
        malformed[2020][0] = MunicipalOption(malformed[2020][0].municipality, 2025,
                                           malformed[2020][0].distance_km,
                                           malformed[2020][0].new_contract_rent_eur_m2)
        with self.assertRaises(ValueError):
            municipal_table(malformed)
        with self.assertRaises(ValueError):
            margin_table({2020: self.options[2020][:-1], 2025: self.options[2025]})

    def test_2020_to_2025_exact_frontier_entry_and_exit(self):
        self.assertEqual(len(self.changes), 16)
        self.assertEqual({r['municipality'] for r in self.changes if r['summary_frontier_status_2020_to_2025'] == 'entered'},
                         {'Arouca', 'Oliveira de Azeméis'})
        self.assertEqual({r['municipality'] for r in self.changes if r['summary_frontier_status_2020_to_2025'] == 'left'},
                         {'Santo Tirso'})

    def test_missing_or_duplicate_frontier_pairs_rejected(self):
        with self.assertRaises(ValueError):
            status_changes(self.table[:-1])
        with self.assertRaises(ValueError):
            status_changes(self.table + [self.table[0]])

    def test_margin_grid_is_complete_and_known_counts_match(self):
        self.assertEqual(len(self.margins), len(YEARS) * len(DISTANCE_MARGINS_KM) * len(RENT_MARGINS_EUR_M2))
        lookup = {(r['rent_year'], float(r['minimum_centroid_distance_gain_km']),
                   float(r['minimum_new_rent_gain_eur_m2_month'])): r for r in self.margins}
        self.assertEqual(len(lookup), 18)
        self.assertEqual(lookup[2020, 0, 0]['mathematically_dominated_municipal_summaries'], 10)
        self.assertEqual(lookup[2025, 0, 0]['mathematically_dominated_municipal_summaries'], 9)
        self.assertEqual(lookup[2025, 2, .5]['mathematically_dominated_municipal_summaries'], 5)
        self.assertEqual(lookup[2020, 5, 1]['mathematically_dominated_municipal_summaries'], 1)

    def test_dominance_counts_never_increase_with_stronger_margin(self):
        lookup = {(r['rent_year'], float(r['minimum_centroid_distance_gain_km']),
                   float(r['minimum_new_rent_gain_eur_m2_month'])): r for r in self.margins}
        for year in YEARS:
            for distance in DISTANCE_MARGINS_KM:
                counts = [int(lookup[year, distance, r]['mathematically_dominated_municipal_summaries'])
                          for r in RENT_MARGINS_EUR_M2]
                self.assertEqual(counts, sorted(counts, reverse=True))
            for rent in RENT_MARGINS_EUR_M2:
                counts = [int(lookup[year, d, rent]['mathematically_dominated_municipal_summaries'])
                          for d in DISTANCE_MARGINS_KM]
                self.assertEqual(counts, sorted(counts, reverse=True))

    def test_plot_rejects_incomplete_margin_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Incomplete or duplicated'):
                plot_margins(self.margins[:-1], Path(tmp) / 'invalid.png')

    def test_csv_figures_and_provenance_regenerate_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run(root)
            self.assertEqual(report['municipal_year_rows'], 32)
            self.assertEqual(report['margin_scenarios'], 18)
            self.assertEqual(report['comparison_year_changes'], 16)
            self.assertEqual(len(report['limits']), 10)
            self.assertIn('Arouca', report['nondominated_municipal_summaries_by_year']['2025'])
            for figure in FIGURES:
                self.assertGreater((root / 'figures' / figure).stat().st_size, 15000)
            for filename, count in zip(RESULTS[:3], (32, 18, 16)):
                with (root / 'results' / filename).open(encoding='utf-8', newline='') as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)
            loaded = json.loads((root / 'results' / RESULTS[-1]).read_text(encoding='utf-8'))
            self.assertEqual(loaded, report)
            self.assertEqual(report['geography_epsg'], 3763)


if __name__ == '__main__':
    unittest.main()
