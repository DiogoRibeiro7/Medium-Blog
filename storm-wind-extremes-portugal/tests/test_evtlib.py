"""Checks of the extreme value tools against scipy and against simulated data."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evtlib as ev  # noqa: E402


class Distribution(unittest.TestCase):
    def test_cdf_and_quantile_match_scipy(self):
        for xi in (-0.2, 0.0, 0.15):
            x = np.linspace(5, 30, 11)
            ref = stats.genextreme.cdf(x, c=-xi, loc=12, scale=2.5)
            np.testing.assert_allclose(ev.gev_cdf(x, 12, 2.5, xi), ref, atol=1e-6)
            p = np.array([0.1, 0.5, 0.98])
            np.testing.assert_allclose(ev.gev_quantile(p, 12, 2.5, xi),
                                       stats.genextreme.ppf(p, c=-xi, loc=12, scale=2.5), rtol=1e-5)

    def test_return_period_beyond_end_point(self):
        # xi < 0: upper end point at mu - sigma / xi = 12 + 2.5 / 0.2 = 24.5
        self.assertTrue(np.isinf(ev.return_period(30, 12, 2.5, -0.2)))
        self.assertAlmostEqual(float(ev.return_period(ev.return_level(50, 12, 2.5, -0.2), 12, 2.5, -0.2)), 50, 6)

    def test_gradient(self):
        rng = np.random.default_rng(1)
        x = stats.genextreme.rvs(c=0.1, loc=12, scale=2, size=60, random_state=rng)
        Xm = np.column_stack([np.ones(60), np.linspace(-1, 1, 60)])
        Xs = np.ones((60, 1))
        th = np.array([11.8, 0.1, np.log(2.1), -0.08])
        _, g = ev.gev_nll(th, x, Xm, Xs)
        num = np.zeros_like(th)
        for i in range(len(th)):
            e = np.zeros_like(th)
            e[i] = 1e-6
            num[i] = (ev.gev_nll(th + e, x, Xm, Xs)[0] - ev.gev_nll(th - e, x, Xm, Xs)[0]) / 2e-6
        np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-5)


class Fitting(unittest.TestCase):
    def test_fit_matches_scipy(self):
        rng = np.random.default_rng(2)
        x = stats.genextreme.rvs(c=0.1, loc=12, scale=2, size=200, random_state=rng)
        f = ev.fit_gev(x)
        c, loc, scale = stats.genextreme.fit(x)
        ref = -stats.genextreme.logpdf(x, c, loc, scale).sum()
        self.assertLessEqual(f.nll, ref + 1e-4)
        self.assertAlmostEqual(f.xi, -c, delta=0.02)

    def test_trend_recovered(self):
        rng = np.random.default_rng(3)
        t = np.linspace(-3.75, 3.75, 400)
        x = stats.genextreme.rvs(c=0.05, loc=12 + 0.4 * t, scale=2, size=400, random_state=rng)
        f = ev.fit_gev(x, Xmu=t[:, None])
        self.assertAlmostEqual(f.beta_mu[1], 0.4, delta=0.12)

    def test_profile_interval_covers(self):
        rng = np.random.default_rng(4)
        x = stats.genextreme.rvs(c=0.1, loc=12, scale=2, size=76, random_state=rng)
        z50 = stats.genextreme.ppf(0.98, c=0.1, loc=12, scale=2)
        T, lo, hi, _ = ev.profile_return_period(x, z50)
        self.assertTrue(lo < T < hi)
        zhat, zlo, zhi = ev.profile_return_level(x, 50)
        self.assertTrue(zlo < zhat < zhi)

    def test_pooled_shape(self):
        rng = np.random.default_rng(5)
        M = np.column_stack([stats.genextreme.rvs(c=0.1, loc=10 + j, scale=1.5 + 0.1 * j, size=80, random_state=rng)
                             for j in range(6)])
        f = ev.fit_pooled_shape(M)
        self.assertAlmostEqual(f["xi"], -0.1, delta=0.07)
        np.testing.assert_allclose(f["mu"], 10 + np.arange(6), atol=0.6)


class PeaksOverThreshold(unittest.TestCase):
    def test_declustering(self):
        x = np.array([0, 5, 6, 0, 7, 0, 0, 0, 8, 0, 0], float)
        peaks, n_exc, n_cl = ev.runs_decluster(x, 4, r=1)
        self.assertEqual(n_exc, 4)
        self.assertEqual(list(peaks), [2, 4, 8])
        peaks2, _, _ = ev.runs_decluster(x, 4, r=2)
        self.assertEqual(list(peaks2), [4, 8])

    def test_gpd_pooled(self):
        rng = np.random.default_rng(7)
        Y = [stats.genpareto.rvs(c=-0.15, scale=1 + 0.2 * j, size=400, random_state=rng) for j in range(5)]
        f = ev.fit_gpd_pooled(Y)
        self.assertAlmostEqual(f["xi"], -0.15, delta=0.06)
        self.assertAlmostEqual(float(ev.gpd_sf(1.0, 1.0, -0.15)), stats.genpareto.sf(1.0, c=-0.15), places=6)

    def test_extremal_index_independent(self):
        rng = np.random.default_rng(8)
        x = rng.uniform(size=20000)
        self.assertAlmostEqual(ev.intervals_extremal_index(x, 0.98), 1.0, delta=0.12)


class Dependence(unittest.TestCase):
    def test_extremal_coefficient_limits(self):
        rng = np.random.default_rng(6)
        a = rng.uniform(size=5000)
        b = rng.uniform(size=5000)
        self.assertAlmostEqual(ev.madogram_theta(a, a), 1.0, places=6)
        self.assertAlmostEqual(ev.madogram_theta(a, b), 2.0, delta=0.08)
        U = rng.uniform(size=(20000, 5))
        self.assertAlmostEqual(ev.field_theta(U), 5.0, delta=0.15)
        self.assertAlmostEqual(ev.field_theta(np.repeat(a[:, None], 5, axis=1)), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
