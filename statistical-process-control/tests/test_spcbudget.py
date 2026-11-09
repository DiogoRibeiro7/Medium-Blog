"""Tests for spcbudget. Run from statistical-process-control/:  python -m unittest discover -s tests"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spcbudget as sb  # noqa: E402


class GuaranteedLimits(unittest.TestCase):
    def test_guarantee_matches_simulation(self):
        rng = np.random.default_rng(11)
        for m, alpha, eps in ((20, 0.05, 0.05), (100, 0.0027, 0.10)):
            k = sb.guaranteed_k(m, alpha, eps)
            z = rng.normal(0, 1 / math.sqrt(m), 400_000)
            s = np.sqrt(rng.chisquare(m - 1, 400_000) / (m - 1))
            share = np.mean(sb.conditional_fap(z, s, k) <= alpha)
            self.assertAlmostEqual(share, 1 - eps, delta=0.003)

    def test_close_to_howe_tolerance_factor(self):
        # Howe (1969) approximation to the two-sided normal tolerance factor, accurate to about 0.5%
        for n, p, g in ((10, 0.99, 0.95), (20, 0.95, 0.95), (100, 0.99, 0.95)):
            howe = math.sqrt((n - 1) * (1 + 1 / n) * stats.norm.isf((1 - p) / 2) ** 2 / stats.chi2.ppf(1 - g, n - 1))
            self.assertAlmostEqual(sb.guaranteed_k(n, 1 - p, 1 - g) / howe, 1, delta=0.005)

    def test_large_phase1_tends_to_known_parameter_limit(self):
        self.assertAlmostEqual(sb.guaranteed_k(1_000_000, 0.0027, 0.1), stats.norm.isf(0.00135), delta=0.01)

    def test_parametric_bootstrap_close_to_exact(self):
        rng = np.random.default_rng(5)
        x = rng.normal(10, 2, 100)
        kb = sb.bootstrap_guaranteed_k(x, 0.0027, 0.1, rng, B=20_000)
        self.assertAlmostEqual(kb, sb.guaranteed_k(100, 0.0027, 0.1), delta=0.03)

    def test_line_k_with_one_chart_is_the_single_chart_k(self):
        rng = np.random.default_rng(8)
        k1 = sb.line_guaranteed_k(50, 1, 0.0027, 0.1, rng, reps=100_000)
        self.assertAlmostEqual(k1, sb.guaranteed_k(50, 0.0027, 0.1), delta=0.01)

    def test_sidak(self):
        a = sb.sidak_alpha(0.0027, 50)
        self.assertAlmostEqual(1 - (1 - a) ** 50, 0.0027, places=12)
        self.assertAlmostEqual(sb.sidak_k(0.0027, 1), 3.0, places=3)


class DistributionFree(unittest.TestCase):
    def test_min_calibration_size(self):
        self.assertEqual(sb.min_calibration_size(0.0027, 0.1), 852)
        self.assertEqual(sb.min_calibration_size(0.01, 0.1), 230)
        n = sb.min_calibration_size(0.01, 0.1, rank=2)
        self.assertGreaterEqual(stats.binom.sf(1, n, 0.01), 0.9)
        self.assertLess(stats.binom.sf(1, n - 1, 0.01), 0.9)

    def test_calibrated_threshold_guarantee(self):
        rng = np.random.default_rng(3)
        alpha, eps, n = 0.01, 0.1, 600
        ok = []
        for _ in range(4000):
            u = rng.uniform(size=n)
            t = sb.calibrated_threshold(u, alpha, eps)
            ok.append(1 - t.threshold <= alpha)      # exceedance probability of a uniform score
        self.assertGreaterEqual(np.mean(ok), 1 - eps - 0.015)
        with self.assertRaises(ValueError):
            sb.calibrated_threshold(rng.uniform(size=100), alpha, eps)

    def test_conformal_pvalues(self):
        p = sb.conformal_pvalues([1, 2, 3, 4], [0.5, 2, 5])
        np.testing.assert_allclose(p, [1.0, 4 / 5, 1 / 5])

    def test_benjamini_hochberg(self):
        np.testing.assert_array_equal(sb.benjamini_hochberg([0.01, 0.04, 0.03, 0.2], 0.05), [True, False, False, False])
        np.testing.assert_array_equal(sb.benjamini_hochberg([0.01, 0.02, 0.03, 0.2], 0.05), [True, True, True, False])


class Adaptive(unittest.TestCase):
    def test_long_run_rate_within_bound(self):
        rng = np.random.default_rng(1)
        a = sb.AdaptiveThreshold(alpha=0.02, gamma=0.01, window=500)
        x = np.concatenate([rng.normal(size=5000), rng.normal(3, 1, size=5000)])   # a shift halfway
        alarms = np.array([a.update(v) for v in x])
        bound = (max(0.02, 0.98) + 0.01) / (0.01 * len(x))
        self.assertLessEqual(abs(alarms.mean() - 0.02), bound)


if __name__ == "__main__":
    unittest.main()
