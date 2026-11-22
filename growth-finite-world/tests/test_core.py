"""Checks of growth_core against answers known in closed form or by brute force."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import growth_core as gc  # noqa: E402


class Valuation(unittest.TestCase):
    def test_flat_perpetuity(self):
        self.assertAlmostEqual(gc.perpetuity_value(3.0, 0.08), 37.5)

    def test_growing_perpetuity_requires_r_above_g(self):
        with self.assertRaises(ValueError):
            gc.perpetuity_value(1.0, 0.05, 0.05)

    def test_implied_growth_inverts_the_value(self):
        g = gc.implied_growth(75.0, 3.0, 0.08)
        self.assertAlmostEqual(g, 0.04)
        self.assertAlmostEqual(gc.perpetuity_value(3.0, 0.08, g), 75.0)

    def test_share_beyond_matches_a_brute_force_sum(self):
        r, g, n = 0.07, 0.03, 40
        t = np.arange(1, 5000)
        pv = (1 + g) ** (t - 1) / (1 + r) ** t
        self.assertAlmostEqual(gc.share_beyond(n, r, g), pv[t > n].sum() / pv.sum(), places=9)

    def test_share_beyond_zero_years_is_everything(self):
        self.assertEqual(gc.share_beyond(0, 0.08, 0.04), 1.0)

    def test_years_to_share_inverts_share_beyond(self):
        y = gc.years_to_share(0.25, 0.08, 0.02)
        self.assertAlmostEqual(gc.share_beyond(y, 0.08, 0.02), 0.25)

    def test_halving_and_doubling(self):
        self.assertAlmostEqual(gc.halving_time(0.02), math.log(2) / math.log(1.02))
        self.assertAlmostEqual(1.03 ** gc.doubling_time(0.03), 2.0)


class Saturation(unittest.TestCase):
    def test_years_to_saturate_matches_simulation(self):
        s0, gf, gm = 0.05, 0.10, 0.03
        y = gc.years_to_saturate(s0, gf, gm)
        share, t = s0, 0
        while share < 1:
            share *= (1 + gf) / (1 + gm)
            t += 1
        self.assertTrue(t - 1 < y <= t)

    def test_no_saturation_when_not_outgrowing(self):
        self.assertEqual(gc.years_to_saturate(0.1, 0.03, 0.03), math.inf)

    def test_logistic_starts_at_share0_and_tends_to_one(self):
        s = gc.logistic_share([0, 400], 0.05, 0.12)
        self.assertAlmostEqual(s[0], 0.05)
        self.assertAlmostEqual(s[1], 1.0, places=9)

    def test_firm_growth_converges_to_market_growth(self):
        g = gc.firm_growth_path(np.arange(0, 301), 0.05, 0.12, 0.03)
        self.assertGreater(g[0], 0.10)
        self.assertAlmostEqual(g[-1], 0.03, places=9)


class GrowthAccounting(unittest.TestCase):
    def test_cagr(self):
        self.assertAlmostEqual(gc.cagr(100, 100 * 1.05**10, 10), 0.05)

    def test_required_productivity_growth_with_flat_input(self):
        self.assertAlmostEqual(gc.required_productivity_growth(0.03, 0.0), 0.03)

    def test_years_to_factor(self):
        self.assertAlmostEqual(1.02 ** gc.years_to_factor(10, 0.02), 10)

    def test_decomposition_adds_up(self):
        g, e, p = gc.decompose_log_growth(100, 180, 50, 70, 20)
        self.assertAlmostEqual(g, e + p)
        self.assertAlmostEqual(g, 100 * math.log(1.8) / 20)

    def test_classification(self):
        self.assertEqual(gc.classify_decoupling(1.5, 0.9), "absolute")
        self.assertEqual(gc.classify_decoupling(1.5, 1.2), "relative")
        self.assertEqual(gc.classify_decoupling(1.2, 1.5), "none")
        self.assertEqual(gc.classify_decoupling(0.9, 0.8), "none")

    def test_ols_recovers_an_exact_line(self):
        x = np.arange(10.0)
        slope, se, r2 = gc.ols_slope(x, 2 + 0.5 * x)
        self.assertAlmostEqual(slope, 0.5)
        self.assertAlmostEqual(se, 0.0, places=9)
        self.assertAlmostEqual(r2, 1.0)


class Returns(unittest.TestCase):
    def test_flat_price_twelve_percent_dividend_flat_cpi(self):
        idx = gc.real_total_return_index([100.0] * 13, [12.0] * 13, [1.0] * 13)
        self.assertAlmostEqual(idx.iloc[-1], 1.01**12)

    def test_inflation_is_removed(self):
        cpi = [1.02**k for k in range(13)]
        price = [100 * c for c in cpi]
        idx = gc.real_total_return_index(price, [0.0] * 13, cpi)
        self.assertAlmostEqual(idx.iloc[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
