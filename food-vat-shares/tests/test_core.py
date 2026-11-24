"""The core functions against answers known by construction."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vat_core as vc  # noqa: E402


def synthetic():
    """Two countries, five fifths: food falls from 20% to 10% of the budget, imputed rent is 10% everywhere."""
    rows, spend = [], []
    for geo in ("AA", "BB"):
        for k, q in enumerate(vc.FIFTHS):
            food = [200, 175, 150, 125, 100][k]
            for code, value in (("CP01", food), ("CP011", food - 20), ("CP111", 20 + 10 * k), ("CP042", 100)):
                rows.append({"geo": geo, "time": 2015, "quant_inc": q, "coicop": code, "value": value})
            spend.append({"geo": geo, "time": 2015, "quant_inc": q, "unit": "PPS_AE", "value": 10000.0 * (k + 1)})
            spend.append({"geo": geo, "time": 2015, "quant_inc": q, "unit": "PPS_HH", "value": 20000.0 * (k + 1)})
    return pd.DataFrame(rows), pd.DataFrame(spend)


class TestCore(unittest.TestCase):
    def test_price_fall(self):
        self.assertAlmostEqual(vc.price_fall(0.06, 0.0), 6 / 106)
        self.assertAlmostEqual(vc.price_fall(0.12, 0.06), 6 / 112)
        self.assertAlmostEqual(vc.price_fall(0.06, 0.23), -17 / 106)
        self.assertEqual(vc.price_fall(0.1, 0.1), 0.0)

    def test_concentration_equal_and_extreme(self):
        self.assertAlmostEqual(vc.concentration_coefficient([1, 1, 1, 1, 1]), 0.0)
        self.assertAlmostEqual(vc.concentration_coefficient([0, 0, 0, 0, 1]), 0.8)
        self.assertAlmostEqual(vc.concentration_coefficient([1, 0, 0, 0, 0]), -0.8)
        self.assertTrue(math.isnan(vc.concentration_coefficient([0, 0, 0, 0, 0])))

    def test_kakwani_proportional_is_zero(self):
        spend = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
        self.assertAlmostEqual(vc.kakwani(0.1 * spend, spend), 0.0)
        # A saving that is a falling share of spending is spread more equally than spending: negative.
        self.assertLess(vc.kakwani(spend * np.array([0.2, 0.18, 0.16, 0.14, 0.1]), spend), 0)
        self.assertGreater(vc.kakwani(spend * np.array([0.02, 0.03, 0.04, 0.05, 0.06]), spend), 0)

    def test_ratio_change_sign(self):
        self.assertLess(vc.top_bottom_ratio_change([0.2, 0.18, 0.16, 0.14, 0.1], 0.05), 0)
        self.assertAlmostEqual(vc.top_bottom_ratio_change([0.15] * 5, 0.05), 0.0)
        self.assertAlmostEqual(vc.top_bottom_ratio_change([0.2, 0, 0, 0, 0.1], 0.1), 1.01 / 1.02 - 1)

    def test_percentile_from_cuts(self):
        cuts = np.arange(1, 10) * 1000.0
        self.assertAlmostEqual(vc.percentile_from_cuts(5000, cuts), 50.0)
        self.assertAlmostEqual(vc.percentile_from_cuts(500, cuts), 5.0)
        self.assertAlmostEqual(vc.percentile_from_cuts(5500, cuts), 55.0)
        self.assertTrue(math.isnan(vc.percentile_from_cuts(9500, cuts)))
        with self.assertRaises(ValueError):
            vc.percentile_from_cuts(1, [1, 2, 3])
        with self.assertRaises(ValueError):
            vc.percentile_from_cuts(1, [9, 8, 7, 6, 5, 4, 3, 2, 1])

    def test_equivalised(self):
        self.assertAlmostEqual(vc.equivalised(18000, 1, 0), 18000)
        self.assertAlmostEqual(vc.equivalised(42000, 2, 0), 28000)
        self.assertAlmostEqual(vc.equivalised(42000, 2, 2), 20000)
        with self.assertRaises(ValueError):
            vc.equivalised(1000, 0, 1)

    def test_budget_table_cash_basis(self):
        s, e = synthetic()
        t = vc.budget_table(s, e, 2015, cash=True)
        self.assertAlmostEqual(t.loc[("AA", "QU1"), "food_drink"], 0.2 / 0.9)
        self.assertAlmostEqual(t.loc[("AA", "QU5"), "spend_ae"], 50000 * 0.9)
        self.assertTrue(t["cash_basis"].all())
        published = vc.budget_table(s, e, 2015, cash=False)
        self.assertAlmostEqual(published.loc[("AA", "QU1"), "food_drink"], 0.2)
        self.assertFalse(published["cash_basis"].any())
        self.assertEqual(vc.complete_countries(t, "food_drink"), [])  # synthetic codes are not member states

    def test_fifth_profile(self):
        s, e = synthetic()
        t = vc.budget_table(s, e, 2015, cash=False)
        prof = vc.fifth_profile(t, "AA", "food_drink", 0.01)
        self.assertAlmostEqual(prof["share_of_total"].sum(), 1.0)
        self.assertAlmostEqual(prof["spend_share_of_total"].sum(), 1.0)
        # food spending per adult equivalent: 0.2*10000=2000 ... 0.1*50000=5000, so euros rise 2.5 times
        self.assertAlmostEqual(prof["euros_ae"].iloc[-1] / prof["euros_ae"].iloc[0], 2.5)
        # while the saving as a share of the budget halves
        self.assertAlmostEqual(prof["pct_budget"].iloc[0] / prof["pct_budget"].iloc[-1], 2.0)
        self.assertLess(vc.kakwani(prof["euros_hh"], t.loc["AA"]["spend_hh"]), 0)


if __name__ == "__main__":
    unittest.main()
