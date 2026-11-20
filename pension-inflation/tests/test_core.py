"""The methods on cases with a known answer."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pension_core as pc  # noqa: E402


def constant_table(q):
    return {f"Y{a}": q for a in range(60, 95)} | {"Y_GE95": 1.0}


def steady_prices(inflation, first=1950, last=2030):
    return {y: (1 + inflation) ** (y - first) for y in range(first, last + 1)}


def row(year, when, lower, upper, rate, minimum="", regular=""):
    return {
        "year": str(year),
        "when": when,
        "lower": lower,
        "upper": upper,
        "rate": str(rate),
        "regular_rate": regular,
        "minimum": minimum,
        "source": "test",
    }


class Statistics(unittest.TestCase):
    def test_quantile_matches_linear_interpolation(self):
        v = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(pc.quantile(v, 0.0), 1.0)
        self.assertEqual(pc.quantile(v, 1.0), 4.0)
        self.assertAlmostEqual(pc.quantile(v, 0.5), 2.5)
        self.assertAlmostEqual(pc.quantile(v, 0.25), 1.75)


class Survival(unittest.TestCase):
    def test_constant_mortality_is_geometric(self):
        s = pc.survival(constant_table(0.1), n=5)
        for k, x in enumerate(s):
            self.assertAlmostEqual(x, 0.9**k)

    def test_no_mortality_counts_every_year(self):
        s = pc.survival(constant_table(0.0))
        self.assertEqual(len(s), pc.SPAN)
        self.assertAlmostEqual(math.fsum(s), pc.SPAN)


class Paths(unittest.TestCase):
    def test_fixed_payment_loses_at_the_inflation_rate(self):
        p = steady_prices(0.03)
        path = pc.real_path(p, 1960, 21)
        self.assertAlmostEqual(path[20], 1.03**-20)

    def test_rules_on_steady_inflation(self):
        p = steady_prices(0.04)
        self.assertTrue(all(abs(x - 1) < 1e-12 for x in pc.indexed_path(p, 1960, 30, "full")))
        fixed = pc.indexed_path(p, 1960, 30, "fixed")
        self.assertAlmostEqual(fixed[-1], pc.real_path(p, 1960, 30)[-1])
        cap5 = pc.indexed_path(p, 1960, 30, "cap5")
        self.assertAlmostEqual(cap5[-1], 1.0)  # inflation below the cap: fully protected
        cap3 = pc.indexed_path(p, 1960, 30, "cap3")
        self.assertAlmostEqual(cap3[-1], (1.03 / 1.04) ** 29)
        half = pc.indexed_path(p, 1960, 30, "half")
        self.assertAlmostEqual(half[-1], (1.02 / 1.04) ** 29)

    def test_windows_need_complete_years(self):
        cpi = {"A": {y: 1.0 for y in range(1955, 1965)} | {1966: 1.0}}
        self.assertEqual(pc.windows(cpi, ["A"], 5), [("A", y) for y in range(1955, 1961)])

    def test_weighted_real_is_a_weighted_mean(self):
        self.assertAlmostEqual(pc.weighted_real([1.0, 0.5], [3.0, 1.0]), 0.875)


class Annuities(unittest.TestCase):
    def test_factor_without_mortality_or_interest(self):
        s = pc.survival(constant_table(0.0))
        self.assertAlmostEqual(pc.annuity_factor(s, 0.0), pc.SPAN)

    def test_growth_and_rate_cancel(self):
        s = pc.survival(constant_table(0.05))
        self.assertAlmostEqual(pc.annuity_factor(s, 0.03, 0.03), math.fsum(s))

    def test_level_annuity_is_worth_one_when_inflation_is_as_priced(self):
        s = pc.survival(constant_table(0.04))
        nominal = 0.05
        priced = pc.breakeven(nominal, 0.02)
        p = steady_prices(priced)
        self.assertAlmostEqual(pc.level_value(p, 1960, nominal, s, 0.02), 1.0)

    def test_level_annuity_loses_when_inflation_runs_above_the_priced_rate(self):
        s = pc.survival(constant_table(0.04))
        nominal = 0.05
        p = steady_prices(pc.breakeven(nominal, 0.02) + 0.02)
        self.assertLess(pc.level_value(p, 1960, nominal, s, 0.02), 1.0)


class Portugal(unittest.TestCase):
    def test_minimum_increase_and_interim_rise(self):
        rows = [
            row(2001, "jan", "", "500", 2.0),
            row(2001, "jan", "500", "", 1.0, minimum="10"),
            row(2002, "jan", "", "", 0.0),
            row(2003, "jan", "", "500", 4.0),
            row(2003, "jan", "500", "", 3.0),
            row(2003, "jul", "", "", 2.0),
        ]
        out = pc.pt_pension(rows, 600.0, 2000, 2003)
        self.assertAlmostEqual(out[2001][0], 610.0)  # 1% would be 6, the minimum is 10
        self.assertAlmostEqual(out[2002][0], 610.0)
        self.assertAlmostEqual(out[2003][0], 628.3)
        self.assertAlmostEqual(out[2003][1], 628.3 + 12.2)  # 2% of the December amount, 610
        self.assertAlmostEqual(out[2003][2], (628.3 + 640.5) / 2)

    def test_first_bracket_adds_the_two_rises_of_one_year(self):
        rows = [row(2001, "jan", "", "500", 4.0), row(2001, "jul", "", "500", 3.0, regular="")]
        self.assertAlmostEqual(pc.pt_first_bracket(rows, 2000, 2001)[2001], 1.07)

    def test_transcribed_rates_match_the_official_coefficients(self):
        d = pc.load()
        factors = pc.pt_first_bracket(d.pt_updates, regular=True)
        coef = d.pt_coefficients
        for y in range(2007, 2024):
            self.assertAlmostEqual(coef[y] / coef[y + 1], factors[y + 2], delta=3e-4, msg=str(y + 2))


if __name__ == "__main__":
    unittest.main()
