"""Contracts for part4_threshold_crossings.py helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import part4_threshold_crossings as p4  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_change_distribution_cdf_and_quantile(self) -> None:
        dist = p4.ChangeDist(np.array([-0.04, -0.02, 0.0, 0.02]), np.array([1.0, 1.0, 1.0, 1.0]), 0.02)
        self.assertAlmostEqual(float(dist.cdf(-0.04)[0]), 0.0)
        self.assertAlmostEqual(float(dist.cdf(0.0)[0]), 0.5)
        self.assertAlmostEqual(float(dist.cdf(0.01)[0]), 0.625)
        self.assertAlmostEqual(float(dist.cdf(0.5)[0]), 1.0)
        self.assertAlmostEqual(dist.quantile(0.5), 0.0, places=9)

    def test_count_above_interpolates_in_log(self) -> None:
        n, lo, hi = np.array([10.0, 4.0]), np.array([1000.0, 2000.0]), np.array([2000.0, 4000.0])
        self.assertAlmostEqual(p4.above(n, lo, hi, 1000.0), 14.0)
        self.assertAlmostEqual(p4.above(n, lo, hi, 2000.0), 4.0)
        self.assertAlmostEqual(p4.above(n, lo, hi, np.sqrt(2) * 1000.0), 9.0)

    def test_pareto_alpha_recovers_grouped_tail(self) -> None:
        edges = np.array([1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0])
        alpha = 1.2
        surv = (edges / 1000.0) ** -alpha
        counts = np.append(-np.diff(surv), surv[-1]) * 100000
        his = np.append(edges[1:], edges[-1] * 10)
        est = p4.pareto_alpha(edges, his, counts, np.arange(len(edges)) == len(edges) - 1)
        self.assertAlmostEqual(est, alpha, places=3)

    def test_country_components_add_up(self) -> None:
        bands = pd.DataFrame({"year": [2010, 2010, 2011], "country_label": ["X", "X", "X"],
                              "band_lo_musd": [1000, 5000, 1200], "n": [2, 1, 2], "worth_musd": [2100, 5000, 2500]})
        flows = pd.DataFrame({
            "year": [2010] * 4, "country_label": ["X"] * 4, "flow": ["stay", "exit", "exit", "entry"],
            "band_lo_musd": [1000, 1000, 5000, 1200], "n": [1, 1, 1, 1], "n_linked_elsewhere": [0, 0, 0, 0]})
        c = p4.country_components(flows, bands, 1500)
        row = c.loc[(2010, "X")]
        self.assertEqual((row.near, row.far, row.d), (0, -1, -1))


if __name__ == "__main__":
    unittest.main()
