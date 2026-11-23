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

from materials_core import (  # noqa: E402
    breakeven_growth,
    circular_rate,
    contributions,
    intensity_split,
    log_change,
    material_use_allowed,
    max_window_increase,
    ols_slope,
    pct,
    per_year,
    recycled_from_rate,
    recycled_needed,
    within_slope,
)


class Decoupling(unittest.TestCase):
    def test_intensity_split_adds_up(self):
        m, g, i = intensity_split(100.0, 90.0, 50.0, 70.0)
        self.assertAlmostEqual(m, g + i, places=12)
        self.assertAlmostEqual(i, math.log(0.9 / 1.4), places=12)

    def test_contributions_sum_to_one(self):
        c = contributions({"a": 10.0, "b": 5.0, "c": 1.0}, {"a": 7.0, "b": 6.0, "c": 1.0})
        self.assertAlmostEqual(sum(c.values()), 1.0, places=12)
        self.assertAlmostEqual(c["a"], 1.5, places=12)
        self.assertAlmostEqual(c["b"], -0.5, places=12)

    def test_units(self):
        self.assertAlmostEqual(log_change(2.0, 4.0), math.log(2.0), places=12)
        self.assertAlmostEqual(per_year(0.24, 24), 1.0, places=12)
        self.assertAlmostEqual(pct(math.log(1.1)), 10.0, places=10)


class DriftAndGrowth(unittest.TestCase):
    def test_ols_recovers_line(self):
        x = np.linspace(-0.05, 0.05, 30)
        slope, se, r2, intercept = ols_slope(x, -0.02 + 1.3 * x)
        self.assertAlmostEqual(slope, 1.3, places=10)
        self.assertAlmostEqual(intercept, -0.02, places=10)
        self.assertAlmostEqual(r2, 1.0, places=10)
        self.assertLess(se, 1e-8)

    def test_within_slope_with_group_intercepts(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(["a", "b", "c"], 40)
        x = rng.normal(0.02, 0.02, size=120)
        drift = {"a": -0.03, "b": -0.01, "c": 0.0}
        y = np.array([drift[g] for g in groups]) + 1.5 * x
        b, se, intercepts = within_slope(y, x, groups)
        self.assertAlmostEqual(b, 1.5, places=10)
        for g, v in drift.items():
            self.assertAlmostEqual(intercepts[g], v, places=10)
        self.assertLess(se, 1e-8)

    def test_breakeven(self):
        self.assertAlmostEqual(breakeven_growth(-0.026, 1.3), 0.02, places=12)


class Circularity(unittest.TestCase):
    def test_rate_and_recycled_are_inverse(self):
        u = recycled_from_rate(0.12, 6.0)
        self.assertAlmostEqual(circular_rate(u, 6.0), 0.12, places=12)

    def test_targets(self):
        self.assertAlmostEqual(recycled_needed(0.25, 8.0), 2.0, places=12)
        self.assertAlmostEqual(material_use_allowed(0.25, 2.0), 8.0, places=12)

    def test_max_window_increase(self):
        s = pd.Series([1.0, 2.0, 1.5, 4.0, 3.0], index=[2010, 2011, 2012, 2013, 2014])
        self.assertEqual(max_window_increase(s, 2), (2.0, 2011))
        self.assertIsNone(max_window_increase(s, 10))


if __name__ == "__main__":
    unittest.main()
