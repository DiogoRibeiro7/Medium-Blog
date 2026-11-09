"""Checks of the helpers in drifttools.py against known values and against river."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drifttools as dt  # noqa: E402


class CusumARL(unittest.TestCase):
    def test_textbook_values(self):
        # Two-sided CUSUM, k = 0.5: in-control ARL 168 at h = 4 and 465 at h = 5 (Montgomery, Introduction to Statistical Quality Control).
        self.assertAlmostEqual(dt.cusum_arl(0.5, 4.0), 168, delta=2)
        self.assertAlmostEqual(dt.cusum_arl(0.5, 5.0), 465, delta=4)
        # Out of control, one-sigma shift: about 8.4 at h = 4 and 10.4 at h = 5.
        self.assertAlmostEqual(dt.cusum_arl(0.5, 4.0, shift=1.0), 8.38, delta=0.2)
        self.assertAlmostEqual(dt.cusum_arl(0.5, 5.0, shift=1.0), 10.4, delta=0.2)

    def test_empirical_matches_normal_for_a_large_sample(self):
        z = np.random.default_rng(1).standard_normal(200_000)
        self.assertAlmostEqual(dt.cusum_arl(0.5, 4.0, sample=z) / dt.cusum_arl(0.5, 4.0), 1.0, delta=0.05)

    def test_h_for_arl_inverts(self):
        h = dt.cusum_h_for_arl(0.5, 365)
        self.assertAlmostEqual(dt.cusum_arl(0.5, h), 365, delta=1)

    def test_guarantee_widens_the_limit(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(60)
        self.assertGreater(dt.guaranteed_cusum_h(x, 0.5, 365, 0.1, rng, B=60), dt.cusum_h_for_arl(0.5, 365))

    def test_alarms_restart(self):
        self.assertEqual(dt.cusum_alarms([3, 3, 3, 3, 3, 3], 0.5, 4.0), [1, 3, 5])


class PageHinkley(unittest.TestCase):
    def test_matches_river(self):
        from river import drift
        rng = np.random.default_rng(3)
        x = rng.standard_normal(2000) + np.r_[np.zeros(1000), np.ones(1000)]
        stat = dt.page_hinkley_statistic(x)
        mine = next(t for t in range(len(x)) if t + 1 >= 30 and stat[t] > 50)
        theirs = dt.run_detector(drift.PageHinkley(mode="up"), x)[0]
        self.assertEqual(mine, theirs)


class Scores(unittest.TestCase):
    def test_psi_null_mean(self):
        rng = np.random.default_rng(4)
        v = [dt.psi_quantile_bins(rng.standard_normal(5000), rng.standard_normal(1000)) for _ in range(300)]
        self.assertAlmostEqual(np.mean(v) / dt.psi_null_mean(10, 5000, 1000), 1.0, delta=0.1)

    def test_wasserstein_floor(self):
        self.assertAlmostEqual(dt.wasserstein_normed(np.zeros(10), np.full(10, 0.01)), 10.0)

    def test_nannyml_threshold(self):
        s = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(dt.nannyml_upper_threshold(s), 2.0 + 3 * math.sqrt(2 / 3))


class Streams(unittest.TestCase):
    def test_markov_bernoulli(self):
        x = dt.markov_bernoulli(0.3, 0.5, 200_000, np.random.default_rng(5)).astype(float)
        self.assertAlmostEqual(x.mean(), 0.3, delta=0.01)
        self.assertAlmostEqual(np.corrcoef(x[:-1], x[1:])[0, 1], 0.5, delta=0.02)

    def test_ar1(self):
        x = dt.ar1(0.6, 200_000, np.random.default_rng(6))
        self.assertAlmostEqual(x.var(), 1.0, delta=0.03)
        self.assertAlmostEqual(np.corrcoef(x[:-1], x[1:])[0, 1], 0.6, delta=0.01)


if __name__ == "__main__":
    unittest.main()
