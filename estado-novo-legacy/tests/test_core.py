"""The core functions against answers known by construction."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import legacy_core as lc  # noqa: E402


class TestCore(unittest.TestCase):
    def test_relative_to_group(self):
        years = [2000, 2001]
        gdp = pd.DataFrame({"A": [100.0, 200.0], "B": [300.0, 300.0], "X": [50.0, 80.0]}, index=years)
        pop = pd.DataFrame({"A": [1.0, 1.0], "B": [3.0, 3.0], "X": [1.0, 1.0]}, index=years)
        rel = lc.relative_to_group(gdp, pop, ["A", "B"])
        # group per head: 400/4 = 100 in 2000, 500/4 = 125 in 2001
        self.assertAlmostEqual(rel.loc[2000, "X"], 50.0)
        self.assertAlmostEqual(rel.loc[2001, "X"], 64.0)
        gdp.loc[2001, "B"] = np.nan
        rel = lc.relative_to_group(gdp, pop, ["A", "B"])
        self.assertTrue(math.isnan(rel.loc[2001, "X"]))
        self.assertAlmostEqual(lc.relative_to_group(gdp, pop, ["A", "B"], min_members=1).loc[2001, "X"], 40.0)

    def test_annual_change(self):
        s = pd.Series({2000: 100.0, 2010: 100.0 * 1.02**10})
        self.assertAlmostEqual(lc.annual_change(s, 2000, 2010), 2.0)
        self.assertTrue(math.isnan(lc.annual_change(s, 2000, 2020)))

    def test_catch_up_residuals(self):
        start = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
        end = start + (5.0 - 0.5 * start)  # gain = 5 - 0.5 * start exactly
        end["d"] += 1.0
        cu = lc.catch_up_residuals(start, end)
        self.assertAlmostEqual(cu["residual"].sum(), 0.0)
        self.assertGreater(cu.loc["d", "residual"], 0)
        self.assertLess(cu.attrs["slope"], 0)

    def test_mincer(self):
        self.assertAlmostEqual(float(lc.mincer_phi(4.0)), 0.536)
        self.assertAlmostEqual(float(lc.mincer_phi(8.0)), 0.536 + 0.404)
        self.assertAlmostEqual(float(lc.mincer_phi(10.0)), 0.536 + 0.404 + 0.136)
        self.assertAlmostEqual(float(lc.human_capital_ratio(3.0, 3.0)), 1.0)
        self.assertAlmostEqual(float(lc.human_capital_ratio(5.0, 3.0)), math.exp(0.134 + 0.101))

    def test_accounting_adds_up(self):
        r = lc.accounting(y=60.0, ky=3.0, h=2.0, y_ref=100.0, ky_ref=2.5, h_ref=3.0)
        self.assertAlmostEqual(r["capital"] + r["human_capital"] + r["productivity"], r["total"])
        self.assertAlmostEqual(r["relative_output"], 0.6)
        self.assertAlmostEqual(r["relative_output_with_ref_h"], 0.6 * 3.0 / 2.0)
        same = lc.accounting(5.0, 2.0, 3.0, 5.0, 2.0, 3.0)
        self.assertAlmostEqual(same["total"], 0.0)

    def test_geometric_mean(self):
        self.assertAlmostEqual(lc.geometric_mean([1.0, 4.0]), 2.0)
        self.assertAlmostEqual(lc.geometric_mean([2.0, np.nan, 8.0]), 4.0)

    def test_convergence_fit_recovers_coefficients(self):
        rng = np.random.default_rng(1)
        y0 = pd.Series(rng.uniform(30, 130, 40))
        s = pd.Series(rng.uniform(6, 13, 40))
        y1 = y0 * np.exp(2.0 - 0.6 * np.log(y0) + 0.05 * s)
        fit = lc.convergence_fit(y0, y1, s)
        np.testing.assert_allclose(fit["beta"], [2.0, -0.6, 0.05], atol=1e-10)
        self.assertAlmostEqual(lc.predict_level(fit["beta"], 80.0, 10.0), 80.0 * math.exp(2.0 - 0.6 * math.log(80.0) + 0.5))
        boot = lc.bootstrap_convergence(fit["data"], 50, 3)
        np.testing.assert_allclose(boot[:, 2], 0.05, atol=1e-8)

    def test_first_year_reaching(self):
        s = pd.Series({1950: 5.0, 1960: 7.0, 1970: 9.5})
        self.assertEqual(lc.first_year_reaching(s, 7.0), 1960.0)
        self.assertTrue(math.isnan(lc.first_year_reaching(s, 10.0)))


if __name__ == "__main__":
    unittest.main()
