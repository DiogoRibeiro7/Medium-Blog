"""Tests of the methods on systems where the answer is known.

python -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chaos_core as cc  # noqa: E402


class TestIntegration(unittest.TestCase):
    def test_rk4_matches_the_exponential(self):
        x = np.array([1.0])
        for _ in range(100):
            x = cc.rk4(lambda v: v, x, 0.01)
        self.assertAlmostEqual(float(x[0]), math.e, places=8)

    def test_two_scale_matches_a_direct_loop(self):
        m = cc.TwoScale(k=6, j=4)
        s = np.random.default_rng(0).standard_normal(6 + 24)
        x, y = s[:6], s[6:]
        dx = np.empty(6)
        dy = np.empty(24)
        for k in range(6):
            dx[k] = (x[(k + 1) % 6] - x[k - 2]) * x[k - 1] - x[k] + m.forcing - m.coupling * y[4 * k : 4 * k + 4].sum()
        for i in range(24):
            dy[i] = -m.cb * y[(i + 1) % 24] * (y[(i + 2) % 24] - y[i - 1]) - m.c * y[i] + m.coupling * x[i // 4]
        np.testing.assert_allclose(m(s), np.concatenate([dx, dy]), rtol=1e-12, atol=1e-12)


class TestGrowth(unittest.TestCase):
    def test_lyapunov_of_a_linear_system(self):
        """dx/dt = diag(0.5, -1) x around its fixed point at zero: the largest exponent is 0.5."""
        rate = np.array([0.5, -1.0])
        lam = cc.lyapunov(lambda v: rate * v, np.zeros(2), 0.01, 20_000)
        self.assertAlmostEqual(lam, 0.5, delta=0.01)

    def test_lyapunov_of_lorenz63(self):
        """The published value is about 0.906."""
        x = cc.run(cc.lorenz63, np.array([1.0, 1.0, 1.0]), 0.01, 2000)
        self.assertAlmostEqual(cc.lyapunov(cc.lorenz63, x, 0.01, 100_000), 0.906, delta=0.03)

    def test_horizon_gain_is_log10_over_rate_for_pure_growth(self):
        """Errors growing as exp(r t): each tenfold reduction of the start adds ln(10)/r."""
        r = 2.0
        starts = np.zeros((4000, 3))  # around the fixed point, so the error is the perturbed state itself
        eps = 10.0 ** -np.arange(6, 0, -1)
        hz, _, _ = cc.horizons(lambda v: r * v, starts, 1.0, eps, 0.002, 4000, 1, slice(None), 1.0, 0)
        gains = -np.diff(hz)
        np.testing.assert_allclose(gains, math.log(10) / r, rtol=0.02)
        np.testing.assert_allclose(cc.days_per_decade(eps, hz), math.log(10) / r * cc.DAYS_PER_UNIT, rtol=0.02)

    def test_first_crossing_interpolates(self):
        t = np.array([0.0, 1.0, 2.0])
        self.assertAlmostEqual(cc.first_crossing(t, np.array([0.0, 1.0, 3.0]), 2.0), 1.5)
        self.assertTrue(math.isnan(cc.first_crossing(t, np.zeros(3), 1.0)))


class TestScores(unittest.TestCase):
    def test_fair_crps_is_unbiased_for_the_parent_distribution(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(20_000)
        members = rng.standard_normal((20_000, 10))
        self.assertAlmostEqual(cc.crps_fair(members, y).mean(), cc.crps_normal(0.0, 1.0, y).mean(), delta=0.01)

    def test_crps_normal_at_the_mean(self):
        """CRPS of N(0, 1) at its mean is (sqrt(2) - 1) / sqrt(pi)."""
        self.assertAlmostEqual(float(cc.crps_normal(0.0, 1.0, np.array(0.0))), (math.sqrt(2) - 1) / math.sqrt(math.pi))

    def test_anomaly_correlation_of_a_perfect_forecast(self):
        x = np.random.default_rng(2).standard_normal((50, 4))
        self.assertAlmostEqual(cc.anomaly_correlation(x, x, 0.0), 1.0)


class TestAnalogues(unittest.TestCase):
    def test_a_planted_repeat_is_found_and_stays_at_zero(self):
        rng = np.random.default_rng(3)
        n = 400
        a = rng.standard_normal((n, 30))
        a[300:330] = a[100:130]  # day 300 repeats day 100 for 30 days
        doy = np.zeros(n)
        curves, _, _ = cc.analogue_divergence(a, doy, horizon=20)
        self.assertTrue(np.allclose(curves[100], 0.0))

    def test_linear_surrogate_keeps_the_memory(self):
        """An AR(1) with coefficient 0.8: the copy has lag-one autocorrelation near 0.8."""
        rng = np.random.default_rng(4)
        z = np.zeros((5000, 2))
        for t in range(1, 5000):
            z[t] = 0.8 * z[t - 1] + rng.standard_normal(2)
        q = cc.var1_surrogate(z, 5000, seed=5)
        r = np.corrcoef(q[:-1, 0], q[1:, 0])[0, 1]
        self.assertAlmostEqual(r, 0.8, delta=0.03)


if __name__ == "__main__":
    unittest.main()
