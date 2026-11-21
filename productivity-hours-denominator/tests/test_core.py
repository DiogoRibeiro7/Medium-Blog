"""Checks of the definitions on cases with a known answer."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_data  # noqa: E402
import hours_core as hc  # noqa: E402


def quarterly(first, values):
    k = hc.q(first)
    return {k + i: v for i, v in enumerate(values)}


class Quarters(unittest.TestCase):
    def test_round_trip(self):
        for text in ("1995-Q1", "2021-Q1", "2026-Q2"):
            self.assertEqual(hc.label(hc.q(text)), text)

    def test_consecutive(self):
        self.assertEqual(hc.q("2021-Q1") - hc.q("2020-Q4"), 1)
        self.assertEqual(hc.year_of(hc.q("2019-Q4")), 2019)


class Growth(unittest.TestCase):
    def test_index_base_year_is_100(self):
        s = quarterly("2009-Q1", [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertAlmostEqual(hc.annual_mean(hc.index_to(s, 2009), 2009), 100.0)

    def test_annual_mean_needs_four_quarters(self):
        s = quarterly("2019-Q1", [1, 2, 3])
        self.assertIsNone(hc.annual_mean(s, 2019))

    def test_constant_growth(self):
        # 2% a year, compounded quarterly: annual means grow 2% a year as well.
        s = quarterly("2010-Q1", [1.02 ** (i / 4) for i in range(40)])
        self.assertAlmostEqual(hc.growth(s, 2011, 2018), 2.0, places=9)

    def test_productivity_gap_is_the_hours_gap(self):
        # The same output over two hours series: the log growth gap equals the hours gap with the sign reversed.
        rng = np.random.default_rng(1)
        y = quarterly("2018-Q1", list(100 * np.exp(np.cumsum(rng.normal(0.004, 0.01, 40)))))
        h1 = quarterly("2018-Q1", list(50 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 40)))))
        h2 = quarterly("2018-Q1", list(70 * np.exp(np.cumsum(rng.normal(0.000, 0.01, 40)))))
        g1, g2 = hc.growth_ratio(y, h1, 2019, 2025), hc.growth_ratio(y, h2, 2019, 2025)
        prod_gap = math.log(1 + g2 / 100) - math.log(1 + g1 / 100)
        hours_gap = (
            math.log(hc.annual_mean(h1, 2025) / hc.annual_mean(h1, 2019))
            - math.log(hc.annual_mean(h2, 2025) / hc.annual_mean(h2, 2019))
        ) / 6
        self.assertAlmostEqual(prod_gap, hours_gap, places=12)

    def test_last_years(self):
        s = quarterly("2020-Q1", [1.03 ** (i / 4) for i in range(30)])
        end = hc.q("2026-Q2")
        self.assertAlmostEqual(hc.growth_last_years(s, end), 3.0, places=9)


class Trend(unittest.TestCase):
    def test_exact_exponential(self):
        s = quarterly("1997-Q1", [80 * 1.018 ** (i / 4) for i in range(44)])
        g, f = hc.log_trend(s, hc.q("1997-Q1"), hc.q("2007-Q4"))
        self.assertAlmostEqual(g, 1.8, places=9)
        self.assertAlmostEqual(f(hc.q("2026-Q2")), 80 * 1.018 ** ((hc.q("2026-Q2") - hc.q("1997-Q1")) / 4))


class Hac(unittest.TestCase):
    def test_slope_matches_ols(self):
        rng = np.random.default_rng(7)
        x = np.arange(60) / 4
        y = 0.3 - 0.25 * x + rng.normal(0, 0.5, 60)
        slope, se = hc.hac_slope(x, y)
        self.assertAlmostEqual(slope, float(np.polyfit(x, y, 1)[0]), places=12)
        self.assertGreater(se, 0)

    def test_no_noise_no_error(self):
        x = np.arange(20, dtype=float)
        slope, se = hc.hac_slope(x, 2 * x + 1)
        self.assertAlmostEqual(slope, 2.0)
        self.assertLess(se, 1e-10)

    def test_zero_lags_is_white_sandwich(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=50)
        y = 1 + x + rng.normal(size=50) * (1 + np.abs(x))
        X = np.column_stack([np.ones(50), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        u = y - X @ b
        bread = np.linalg.inv(X.T @ X)
        white = bread @ (X.T * u**2) @ X @ bread
        self.assertAlmostEqual(hc.hac_slope(x, y, lags=0)[1], math.sqrt(white[1, 1]), places=12)


class Ranks(unittest.TestCase):
    def test_ranks_highest_first(self):
        self.assertEqual(hc.ranks({"a": 1.0, "b": 3.0, "c": 2.0}), {"b": 1, "c": 2, "a": 3})

    def test_spearman_monotone(self):
        self.assertAlmostEqual(hc.spearman([1, 2, 3, 4], [10, 20, 30, 400]), 1.0)
        self.assertAlmostEqual(hc.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


class Tidy(unittest.TestCase):
    def test_keeps_and_sorts(self):
        text = (
            "DATAFLOW,LAST UPDATE,freq,age,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG,CONF_STATUS\n"
            "X,1,Q,Y15-74,PT,2021-Q1,101.5,b,\n"
            "X,1,Q,Y15-74,AT,2020-Q4,99.0,,\n"
            "X,1,Q,Y15-74,AT,2021-Q1,,,\n"
        )
        out = fetch_data.tidy(text, ("age",)).splitlines()
        self.assertEqual(out[0], "country,period,age,value,flag")
        self.assertEqual(out[1:], ["AT,2020-Q4,Y15-74,99.0,", "PT,2021-Q1,Y15-74,101.5,b"])


if __name__ == "__main__":
    unittest.main()
