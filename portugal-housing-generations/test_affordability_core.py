"""Tests for the pure affordability functions."""

import math
import unittest

import affordability_core as A


class PeriodTests(unittest.TestCase):
    def test_labels_and_codes(self) -> None:
        self.assertEqual(A.period_year("2021"), (2021, None))
        self.assertEqual(A.period_year("4.º Trimestre de 2019"), (2019, 4))
        self.assertEqual(A.period_year("1.º Trimestre de 2026"), (2026, 1))
        self.assertEqual(A.period_year("S5A20194"), (2019, 4))
        self.assertEqual(A.period_year("S7A2024"), (2024, None))
        for bad in ("2019/2020", "5.º Trimestre de 2019", "S5A20195", "Q4 2019", ""):
            with self.subTest(label=bad), self.assertRaises(ValueError):
                A.period_year(bad)


class RatioTests(unittest.TestCase):
    price = {("R", 2019): 1000.0, ("R", 2020): 1100.0, ("R", 2024): 1500.0, ("S", 2019): 2000.0}
    income = {("R", 2019): 20000.0, ("R", 2020): 20000.0, ("R", 2024): 25000.0, ("T", 2019): 1.0}

    def test_m2_per_income_year_uses_only_shared_cells(self) -> None:
        m2 = A.m2_per_income_year(self.price, self.income)
        self.assertEqual(m2, {("R", 2019): 20.0, ("R", 2020): 20000 / 1100, ("R", 2024): 25000 / 1500})
        with self.assertRaises(ValueError):
            A.m2_per_income_year({("R", 2019): 0.0}, {("R", 2019): 1.0})

    def test_index_is_relative_to_the_base_and_skips_gaps(self) -> None:
        idx = A.price_income_index(self.price, self.income, "R", 2019, [2019, 2020, 2021, 2024])
        self.assertEqual(sorted(idx), [2019, 2020, 2024])
        self.assertAlmostEqual(idx[2019], 100.0)
        self.assertAlmostEqual(idx[2020], 110.0)
        self.assertAlmostEqual(idx[2024], 100 * 1.5 / 1.25)
        with self.assertRaises(ValueError):
            A.price_income_index(self.price, self.income, "S", 2019, [2019])

    def test_index_base_change_rescales_the_path(self) -> None:
        a = A.price_income_index(self.price, self.income, "R", 2019, [2020, 2024])
        b = A.price_income_index(self.price, self.income, "R", 2020, [2020, 2024])
        self.assertAlmostEqual(a[2024] / a[2020], b[2024] / b[2020])

    def test_growth_and_gap(self) -> None:
        self.assertAlmostEqual(A.annual_growth(self.price, "R", 2019, 2024) or 0, 100 * (1.5 ** 0.2 - 1))
        self.assertIsNone(A.annual_growth(self.price, "S", 2019, 2024))
        with self.assertRaises(ValueError):
            A.annual_growth(self.price, "R", 2024, 2019)
        c = A.growth_comparison(self.price, self.income, "R", 2019, 2024)
        assert c is not None
        self.assertAlmostEqual(c["gap_log_points"], 100 * (math.log(1.5) - math.log(1.25)))
        self.assertAlmostEqual(c["ratio_change_pct"], 100 * (1.5 / 1.25 - 1))
        idx = A.price_income_index(self.price, self.income, "R", 2019, [2024])[2024]
        self.assertAlmostEqual(idx, 100 + c["ratio_change_pct"])
        self.assertIsNone(A.growth_comparison(self.price, self.income, "S", 2019, 2024))


class StatisticsTests(unittest.TestCase):
    def test_ranks_and_spearman(self) -> None:
        self.assertEqual(A.ranks({"a": 1, "b": 3, "c": 3, "d": 0}), {"d": 1, "a": 2, "b": 3.5, "c": 3.5})
        self.assertAlmostEqual(A.spearman({"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 300}), 1.0)
        self.assertAlmostEqual(A.spearman({"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 2, "c": 1, "z": 9}), -1.0)
        with self.assertRaises(ValueError):
            A.spearman({"a": 1, "b": 2}, {"a": 1, "b": 2})

    def test_pearson_median_dispersion(self) -> None:
        self.assertAlmostEqual(A.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        with self.assertRaises(ValueError):
            A.pearson([1, 1, 1], [1, 2, 3])
        self.assertEqual(A.median([3, 1, 2]), 2)
        self.assertEqual(A.median([4, 1, 2, 3]), 2.5)
        d = A.dispersion([1.0, 3.0])
        self.assertAlmostEqual(d["cv"], 0.5)
        self.assertEqual((d["max_over_min"], d["n"]), (3.0, 2.0))
        with self.assertRaises(ValueError):
            A.dispersion([1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
