"""Numerical contracts, data integrity, and artifact regeneration tests."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    FIGURE_NAMES,
    LinearMarket,
    average_cost,
    compounded_index,
    foreign_born_data,
    generate_all,
    growth_data,
    labor_share_path,
    market_outcome,
    railroad_data,
    wealth_data,
)


class MarketTests(unittest.TestCase):
    def test_linear_monopoly_solution(self) -> None:
        result = market_outcome(LinearMarket(12.0, 0.5, 2.0))
        self.assertAlmostEqual(result.competitive_quantity, 20.0)
        self.assertAlmostEqual(result.monopoly_quantity, 10.0)
        self.assertAlmostEqual(result.competitive_price, 2.0)
        self.assertAlmostEqual(result.monopoly_price, 7.0)
        self.assertAlmostEqual(result.deadweight_loss, 25.0)

    def test_monopoly_quantity_is_half_competitive_quantity(self) -> None:
        for slope in (0.25, 0.5, 0.8, 1.5):
            with self.subTest(slope=slope):
                result = market_outcome(LinearMarket(20.0, slope, 4.0))
                self.assertAlmostEqual(result.monopoly_quantity * 2, result.competitive_quantity)
                self.assertGreater(result.deadweight_loss, 0)

    def test_invalid_market_arguments(self) -> None:
        for market in ((12.0, 0.0, 2.0), (5.0, 1.0, 6.0),
                       (10.0, -1.0, 2.0), (math.inf, 1.0, 0.0)):
            with self.subTest(market=market):
                with self.assertRaises(ValueError):
                    LinearMarket(*market)
        with self.assertRaises(TypeError):
            LinearMarket(True, 2.0, 1.0)

    def test_average_cost_reaches_marginal_cost_at_scale(self) -> None:
        result = average_cost(np.array([2.0, 10.0, 1000.0]), 40.0, 2.0)
        np.testing.assert_allclose(result, np.array([22.0, 6.0, 2.04]))
        self.assertTrue(np.all(np.diff(result) < 0))

    def test_average_cost_rejects_invalid_parameters(self) -> None:
        for quantity, fixed in ((0.0, 40.0), (-1.0, 40.0), (3.0, -2.0)):
            with self.subTest(quantity=quantity, fixed=fixed):
                with self.assertRaises(ValueError):
                    average_cost(quantity, fixed, 2.0)


class HistoricalDataTests(unittest.TestCase):
    def test_railroad_values_are_transcribed_correctly(self) -> None:
        years, values = railroad_data()
        self.assertEqual(years.tolist(), [1870, 1880, 1890, 1900])
        self.assertEqual(values.tolist(), [52922.0, 93267.0, 166703.0, 194262.0])
        self.assertGreater(values[-1] / values[0], 3.6)

    def test_foreign_born_counts_and_shares_are_not_conflated(self) -> None:
        years, population, shares = foreign_born_data()
        self.assertEqual(years[0], 1850)
        self.assertEqual(years[-1], 1910)
        self.assertAlmostEqual(population[years.tolist().index(1890)], 9.2)
        self.assertAlmostEqual(shares[years.tolist().index(1890)], 14.8)

    def test_wealth_series_scope_and_missing_1890(self) -> None:
        years, shares = wealth_data()
        self.assertNotIn(1890, years)
        self.assertTrue(np.all(shares[:, 0] >= shares[:, 1]))
        self.assertTrue(np.all(shares[:, 1] >= shares[:, 2]))
        self.assertAlmostEqual(shares[years.tolist().index(1900), 2], 37.2)

    def test_growth_sources_are_exactly_three_comparable_periods(self) -> None:
        rates = growth_data()
        self.assertEqual(len(rates), 3)
        self.assertEqual([span for _, _, span in rates], [29, 29, 29])
        self.assertEqual([rate for _, rate, _ in rates], [2.5, 1.81, 1.43])


class IllustrationTests(unittest.TestCase):
    def test_compounding_uses_non_interpolated_average(self) -> None:
        self.assertEqual(compounded_index(0.0, 30), 100.0)
        self.assertAlmostEqual(compounded_index(2.5, 29), 100.0 * 1.025**29)
        with self.assertRaises(ValueError):
            compounded_index(-100.0, 30)
        with self.assertRaises(ValueError):
            compounded_index(2.0, -1)

    def test_labor_share_accounting_identity(self) -> None:
        times, wage = labor_share_path(0.025, 0.60, 0.48, 30)
        self.assertEqual(len(times), 31)
        self.assertEqual(wage[0], 100.0)
        self.assertAlmostEqual(wage[-1], 100.0 * 1.025**30 * 0.48 / 0.60)

    def test_constant_share_matches_productivity_growth(self) -> None:
        times, wage = labor_share_path(0.025, 0.60, 0.60, 30)
        np.testing.assert_allclose(wage, 100.0 * 1.025**times)

    def test_labor_share_invalid_cases(self) -> None:
        for args in ((0.025, 0.0, 0.6, 30), (0.025, 0.5, 1.2, 30),
                     (-1.0, 0.5, 0.6, 30), (0.025, 0.5, 0.6, 0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    labor_share_path(*args)

    def test_end_to_end_artifacts_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generate_all(output)
            for name in FIGURE_NAMES:
                image = output / "figures" / name
                self.assertTrue(image.is_file())
                self.assertGreater(image.stat().st_size, 10_000)
            expected = {
                "source_based_summary.csv": 6,
                "hypothetical_monopoly_sensitivity.csv": 6,
                "hypothetical_labor_share_paths.csv": 31,
            }
            for name, count in expected.items():
                with (output / "results" / name).open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), count)


if __name__ == "__main__":
    unittest.main()
