"""Checks of the core functions against answers known by construction.

python -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prices_core import (  # noqa: E402
    block_identity,
    covariance_shares,
    gap,
    ols_slope,
    pct,
    per_year,
    ratio_range,
    rmse,
    substitute_productivity,
    tornqvist_log_volume,
    within,
)

YEARS = list(range(2000, 2011))


def industry(wage_g, prod_g, hours_g, share, emp_share=0.8):
    """One industry with constant growth rates and a constant adjusted labour share."""
    t = np.arange(len(YEARS))
    hours = 100 * np.exp(hours_g * t)
    emp_hours = emp_share * hours
    wage = 20 * np.exp(wage_g * t)
    volume = 2000 * np.exp((prod_g + hours_g) * t)
    # adjusted labour share s = wage * hours / nominal, so nominal = wage * hours / s
    nominal = wage * hours / share
    d1 = wage * emp_hours
    return {"cp": nominal, "clv": volume, "d1": d1, "hours": hours, "emp_hours": emp_hours}


def tables(*industries):
    cols = [chr(ord("A") + i) for i in range(len(industries))]
    out = {}
    for k in ("cp", "clv", "d1", "hours", "emp_hours"):
        out[k] = pd.DataFrame({c: ind[k] for c, ind in zip(cols, industries, strict=True)}, index=YEARS)
    return out, cols


class Aggregation(unittest.TestCase):
    def test_single_industry_volume_is_its_own_log_change(self):
        t, _ = tables(industry(0.03, 0.01, 0.0, 0.6))
        v = tornqvist_log_volume(t["clv"], t["cp"])
        self.assertAlmostEqual(v.iloc[-1], math.log(t["clv"]["A"].iloc[-1] / t["clv"]["A"].iloc[0]), places=12)

    def test_equal_growth_gives_that_growth(self):
        t, _ = tables(industry(0.03, 0.02, 0.0, 0.6), industry(0.01, 0.02, 0.0, 0.5))
        v = tornqvist_log_volume(t["clv"], t["cp"])
        self.assertAlmostEqual(v.iloc[-1], 0.02 * (len(YEARS) - 1), places=12)


class Identity(unittest.TestCase):
    def test_price_is_wage_minus_productivity_minus_share(self):
        t, cols = tables(industry(0.03, 0.01, 0.005, 0.6))
        b = block_identity(t, cols, YEARS[0], YEARS[-1])
        n = len(YEARS) - 1
        self.assertAlmostEqual(b.loc[YEARS[-1], "price"], 0.02 * n, places=12)
        self.assertAlmostEqual(b.loc[YEARS[-1], "wage"], 0.03 * n, places=12)
        self.assertAlmostEqual(b.loc[YEARS[-1], "prod"], 0.01 * n, places=12)
        self.assertAlmostEqual(b.loc[YEARS[-1], "lshare"], 0.0, places=12)
        resid = b.price - (b.wage - b["prod"] - b.lshare)
        self.assertLess(resid.abs().max(), 1e-12)

    def test_labour_share_term_is_the_adjusted_share(self):
        ind = industry(0.03, 0.01, 0.0, 0.6)
        ind["cp"] = ind["cp"] * np.exp(0.004 * np.arange(len(YEARS)))  # margins widen: nominal grows faster
        t, cols = tables(ind)
        b = block_identity(t, cols, YEARS[0], YEARS[-1])
        s = t["d1"]["A"] / t["emp_hours"]["A"] * t["hours"]["A"] / t["cp"]["A"]
        self.assertAlmostEqual(b.loc[YEARS[-1], "lshare"], math.log(s.iloc[-1] / s.iloc[0]), places=12)
        self.assertAlmostEqual(b.loc[YEARS[-1], "price"], 0.024 * (len(YEARS) - 1), places=12)

    def test_gap_contributions_add_up(self):
        t, cols = tables(industry(0.03, 0.005, 0.01, 0.6), industry(0.035, 0.03, -0.01, 0.55))
        a = block_identity(t, ["A"], YEARS[0], YEARS[-1])
        b = block_identity(t, ["B"], YEARS[0], YEARS[-1])
        g = gap(a, b).loc[YEARS[-1]]
        n = len(YEARS) - 1
        self.assertAlmostEqual(g.price, g.wage + g.productivity + g.margins, places=12)
        self.assertAlmostEqual(g.productivity, (0.03 - 0.005) * n, places=12)
        self.assertAlmostEqual(g.wage, -0.005 * n, places=12)

    def test_missing_years_are_refused(self):
        t, cols = tables(industry(0.03, 0.01, 0.0, 0.6))
        t["hours"].loc[2005, "A"] = np.nan
        with self.assertRaises(ValueError):
            block_identity(t, cols, YEARS[0], YEARS[-1])

    def test_substituted_productivity_moves_only_the_price(self):
        t, cols = tables(industry(0.03, 0.0, 0.01, 0.6))
        path = pd.Series(0.02 * np.arange(len(YEARS)), index=YEARS)
        t2 = substitute_productivity(t, cols, path)
        b1 = block_identity(t, cols, YEARS[0], YEARS[-1]).loc[YEARS[-1]]
        b2 = block_identity(t2, cols, YEARS[0], YEARS[-1]).loc[YEARS[-1]]
        n = len(YEARS) - 1
        self.assertAlmostEqual(b2["prod"], 0.02 * n, places=12)
        self.assertAlmostEqual(b1.price - b2.price, 0.02 * n, places=12)
        self.assertAlmostEqual(b1.wage, b2.wage, places=12)
        pd.testing.assert_frame_equal(t["cp"], t2["cp"])


class Statistics(unittest.TestCase):
    def test_covariance_shares_sum_to_one(self):
        rng = np.random.default_rng(7)
        a, b, c = rng.normal(size=(3, 50))
        price = a - b + 0.5 * c
        shares = covariance_shares(price, {"a": a, "b": -b, "c": 0.5 * c})
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=12)
        w = rng.uniform(0.1, 1.0, size=50)
        shares_w = covariance_shares(price, {"a": a, "b": -b, "c": 0.5 * c}, w)
        self.assertAlmostEqual(sum(shares_w.values()), 1.0, places=12)

    def test_ols_on_an_exact_line(self):
        x = np.arange(10.0)
        slope, se, r2 = ols_slope(x, 3 - 0.5 * x)
        self.assertAlmostEqual(slope, -0.5, places=12)
        self.assertLess(se, 1e-10)
        self.assertAlmostEqual(r2, 1.0, places=12)

    def test_within_removes_group_means(self):
        v = within([1.0, 3.0, 10.0, 14.0], ["a", "a", "b", "b"])
        np.testing.assert_allclose(v, [-1.0, 1.0, -2.0, 2.0])

    def test_units(self):
        self.assertAlmostEqual(pct(math.log(1.25)), 25.0, places=10)
        self.assertAlmostEqual(per_year(0.30, 15), 2.0, places=12)
        self.assertAlmostEqual(ratio_range([2, 8, 4]), 4.0)
        self.assertAlmostEqual(rmse([3.0, -4.0]), math.sqrt(12.5))


if __name__ == "__main__":
    unittest.main()
