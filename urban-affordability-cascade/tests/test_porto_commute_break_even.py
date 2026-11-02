"""Regression checks for declared household commute-cost counterfactuals."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from analysis import load_panel
from extract_hicp import load_hicp
from porto_rent_options import annual_table
from porto_commute_break_even import (
    AREAS_M2, COMPARATORS, EXTRA_ONE_WAY_MINUTES, FIGURES, RESULTS,
    SCENARIOS, CommuteScenario, additional_time_cost_eur_month,
    break_even_extra_one_way_minutes, duration_thresholds,
    net_monthly_saving_eur, primary_municipality_table,
    rent_gaps_2025, run, scenario_table,
)


class PortoCommuteBreakEvenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gaps = rent_gaps_2025()
        cls.rows = scenario_table()
        cls.summary = duration_thresholds(cls.rows)
        cls.primary = primary_municipality_table(cls.rows)

    def test_2025_comparisons_are_complete_and_positive(self):
        self.assertEqual(len(self.gaps), 16)
        self.assertEqual(set(self.gaps), set(COMPARATORS))
        self.assertTrue(all(v > 0 for v in self.gaps.values()))

    def test_reference_scenario_formula(self):
        s = next(s for s in SCENARIOS if s.name == 'reference')
        self.assertAlmostEqual(additional_time_cost_eur_month(30, s), 200)
        self.assertAlmostEqual(break_even_extra_one_way_minutes(400, s), 54)
        self.assertAlmostEqual(net_monthly_saving_eur(400, 54, s), 0)
        self.assertLess(net_monthly_saving_eur(400, 55, s), 0)

    def test_added_time_is_return_trip_and_per_commuter(self):
        s = next(s for s in SCENARIOS if s.name == 'two_commuter_high_cost')
        self.assertAlmostEqual(additional_time_cost_eur_month(30, s), 660)
        self.assertAlmostEqual(net_monthly_saving_eur(800, 30, s), 60)

    def test_low_friction_one_hour_added_cost(self):
        s = SCENARIOS[0]
        self.assertAlmostEqual(additional_time_cost_eur_month(60, s), 160)

    def test_cash_cost_can_exhaust_before_any_added_time(self):
        s = SCENARIOS[1]
        self.assertEqual(break_even_extra_one_way_minutes(39.99, s), 0)
        self.assertLess(net_monthly_saving_eur(39.99, 0, s), 0)

    def test_negative_rent_gap_permitted_without_fake_positive_threshold(self):
        s = SCENARIOS[1]
        self.assertEqual(break_even_extra_one_way_minutes(-100, s), 0)
        self.assertLess(net_monthly_saving_eur(-100, 0, s), 0)

    def test_reject_invalid_assumed_time_value(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CommuteScenario('invalid', 1, 20, value, 0)

    def test_reject_invalid_commuter_counts_and_days(self):
        for number in (0, -1, 1.2, True):
            with self.subTest(value=number), self.assertRaises(ValueError):
                CommuteScenario('bad', number, 20, 10, 0)
        with self.assertRaises(ValueError):
            CommuteScenario('bad', 1, 0, 10, 0)

    def test_reject_invalid_cash_cost_or_name(self):
        with self.assertRaises(ValueError):
            CommuteScenario('', 1, 20, 10, 0)
        with self.assertRaises(ValueError):
            CommuteScenario('bad', 1, 20, 10, -1)

    def test_reject_nonfinite_or_boolean_rent_gaps(self):
        for value in (float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                break_even_extra_one_way_minutes(value, SCENARIOS[1])
            with self.subTest(net=value), self.assertRaises(ValueError):
                net_monthly_saving_eur(value, 30, SCENARIOS[1])

    def test_reject_negative_extra_commute_or_invalid_scenario(self):
        with self.assertRaises(ValueError):
            additional_time_cost_eur_month(-1, SCENARIOS[1])
        with self.assertRaises(ValueError):
            net_monthly_saving_eur(300, 10, 'not a scenario')

    def test_complete_144_municipality_floor_area_scenario_grid(self):
        self.assertEqual(len(self.rows), 16 * 3 * 3)
        self.assertEqual(len({(r['municipality'], r['assumed_dwelling_area_m2'], r['scenario'])
                              for r in self.rows}), 144)

    def test_thirty_six_duration_summary_rows_and_monotonic_exhaustion(self):
        self.assertEqual(len(self.summary), 36)
        for scenario in SCENARIOS:
            for area in AREAS_M2:
                subset = [r for r in self.summary if r['scenario'] == scenario.name
                          and r['assumed_dwelling_area_m2'] == area]
                self.assertEqual([int(r['extra_one_way_minutes_workday']) for r in subset], list(EXTRA_ONE_WAY_MINUTES))
                counts = [int(r['comparisons_with_no_positive_synthetic_net_saving']) for r in subset]
                self.assertEqual(sorted(counts), counts)
                self.assertTrue(all(0 <= x <= 16 for x in counts))

    def test_exact_reference_70m2_duration_counts(self):
        counts = [int(r['comparisons_with_no_positive_synthetic_net_saving'])
                  for r in self.summary if r['scenario'] == 'reference' and r['assumed_dwelling_area_m2'] == 70]
        self.assertEqual(counts, [0, 1, 4, 8])

    def test_two_commuter_70m2_thirty_minute_exhaustion(self):
        target = next(r for r in self.summary if r['scenario'] == 'two_commuter_high_cost'
                      and r['assumed_dwelling_area_m2'] == 70 and r['extra_one_way_minutes_workday'] == 30)
        self.assertEqual(target['comparisons_with_no_positive_synthetic_net_saving'], 16)

    def test_median_reference_70m2_threshold(self):
        target = next(r for r in self.summary if r['scenario'] == 'reference'
                      and r['assumed_dwelling_area_m2'] == 70)
        self.assertAlmostEqual(float(target['median_break_even_extra_one_way_minutes']), 63.1425)

    def test_all_sixteen_primary_70m2_rows(self):
        self.assertEqual(len(self.primary), 16)
        self.assertEqual({r['municipality'] for r in self.primary}, set(COMPARATORS))
        matosinhos = next(r for r in self.primary if r['municipality'] == 'Matosinhos')
        self.assertEqual(matosinhos['synthetic_70m2_rent_only_saving_2025_eur_month'], '140.700000')
        self.assertEqual(matosinhos['reference_break_even_added_one_way_minutes'], '15.105000')

    def test_larger_assumed_area_weakly_raises_break_even(self):
        for municipality in COMPARATORS:
            for scenario in SCENARIOS:
                t = [float(next(r['break_even_extra_one_way_minutes_workday'] for r in self.rows
                                if r['municipality'] == municipality and r['assumed_dwelling_area_m2'] == area
                                and r['scenario'] == scenario.name)) for area in AREAS_M2]
                self.assertEqual(t, sorted(t))

    def test_2025_gap_matches_previous_rent_option_computation(self):
        source = {(r['alternative_municipality'], r['year']): r
                  for r in annual_table(load_panel(), load_hicp())}
        for municipality in COMPARATORS:
            self.assertAlmostEqual(self.gaps[municipality] * 70,
                                   float(source[municipality, 2025]['synthetic_70m2_nominal_monthly_saving_eur']))

    def test_invalid_or_duplicate_scenario_grid_rejected(self):
        with self.assertRaises(ValueError):
            duration_thresholds(self.rows[:-1])
        with self.assertRaises(ValueError):
            primary_municipality_table(self.rows + [self.rows[0]])

    def test_files_figures_and_provenance_regenerate_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = run(output)
            self.assertEqual(report['municipal_scenario_rows'], 144)
            self.assertEqual(report['duration_summary_rows'], 36)
            self.assertEqual(report['reference_70m2_median_break_even_added_one_way_minutes'], '63.142500')
            self.assertEqual(len(report['limits']), 10)
            self.assertIn('not an Andante price', ' '.join(report['limits']))
            for figure in FIGURES:
                self.assertGreater((output / 'figures' / figure).stat().st_size, 10000)
            for result in RESULTS:
                self.assertTrue((output / 'results' / result).is_file())
            with (output / 'results' / RESULTS[0]).open(encoding='utf-8', newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 144)
            loaded = json.loads((output / 'results' / RESULTS[-1]).read_text(encoding='utf-8'))
            self.assertEqual(loaded['comparator_municipalities'], 16)


if __name__ == '__main__':
    unittest.main()
