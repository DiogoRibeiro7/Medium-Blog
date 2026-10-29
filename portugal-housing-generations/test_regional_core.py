"""Tests for the regional panel's pure functions (no files, no network)."""

import unittest

from regional_core import (
    coefficient_of_variation,
    compare_totals,
    coverage,
    growth,
    longest_complete_span,
    per_thousand,
    period_mean,
    regroup,
)


class RegroupTests(unittest.TestCase):
    """Municipal values are summed into regions without filling gaps."""

    def setUp(self) -> None:
        self.values = {("A1", 2020): 10.0, ("A2", 2020): 5.0, ("B1", 2020): 7.0,
                       ("A1", 2021): 12.0, ("B1", 2021): 8.0}
        self.members = {"A": ["A1", "A2"], "B": ["B1"]}

    def test_sums_complete_region_years(self) -> None:
        totals, gaps = regroup(self.values, self.members, [2020, 2021])
        self.assertEqual(totals[("A", 2020)], 15.0)
        self.assertEqual(totals[("B", 2021)], 8.0)

    def test_region_with_a_missing_member_is_missing_not_partial(self) -> None:
        totals, gaps = regroup(self.values, self.members, [2020, 2021])
        self.assertNotIn(("A", 2021), totals)
        self.assertEqual(gaps[("A", 2021)], ["A2"])

    def test_empty_region_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            regroup(self.values, {"C": []}, [2020])

    def test_compare_totals_reports_only_differences(self) -> None:
        diffs = compare_totals({("A", 2020): 15.0, ("B", 2020): 7.0}, {("A", 2020): 15.0, ("B", 2020): 6.0})
        self.assertEqual(diffs, [("B", 2020, 7.0, 6.0)])


class RateAndGrowthTests(unittest.TestCase):
    """Rates per 1,000 residents, growth and period means."""

    def test_per_thousand(self) -> None:
        rates = per_thousand({("A", 2020): 30.0, ("A", 2021): 5.0}, {("A", 2020): 10000.0})
        self.assertEqual(rates, {("A", 2020): 3.0})
        with self.assertRaises(ValueError):
            per_thousand({("A", 2020): 1.0}, {("A", 2020): 0.0})

    def test_growth_total_and_annual(self) -> None:
        g = growth({("A", 2010): 100.0, ("A", 2020): 121.0}, "A", 2010, 2020)
        assert g is not None
        self.assertAlmostEqual(g["total_pct"], 21.0)
        self.assertAlmostEqual(g["annual_pct"], 100 * (1.21 ** 0.1 - 1))
        self.assertIsNone(growth({("A", 2010): 100.0}, "A", 2010, 2020))

    def test_period_mean_requires_every_year(self) -> None:
        series = {("A", 2020): 1.0, ("A", 2021): 3.0, ("A", 2023): 5.0}
        self.assertEqual(period_mean(series, "A", 2020, 2021), 2.0)
        self.assertIsNone(period_mean(series, "A", 2020, 2023))

    def test_coefficient_of_variation(self) -> None:
        self.assertAlmostEqual(coefficient_of_variation([1.0, 3.0]), 0.5)
        with self.assertRaises(ValueError):
            coefficient_of_variation([2.0])


class CoverageTests(unittest.TestCase):
    """Missingness diagnostics and the longest common span."""

    def test_coverage_counts_and_complete_years(self) -> None:
        series = {("A", 2020): 1.0, ("B", 2020): 1.0, ("A", 2021): 1.0}
        report = coverage(series, ["A", "B"], [2020, 2021])
        self.assertEqual(report["expected"], 4)
        self.assertEqual(report["missing"], 1)
        self.assertEqual(report["missing_cells"], ["B:2021"])
        self.assertEqual(report["complete_years"], [2020])

    def test_longest_complete_span(self) -> None:
        self.assertEqual(longest_complete_span([2001, 2002, 2005, 2006, 2007, 2009]), (2005, 2007))
        self.assertEqual(longest_complete_span([2011]), (2011, 2011))


if __name__ == "__main__":
    unittest.main()
