"""Tests of the shared tools in tariffcal.py. Run from the series folder: python -m unittest discover -s tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tariffcal import (StepRecalibration, gini, lorenz, lower_convex_hull, murphy, poisson_deviance,  # noqa: E402
                       premium_gini)


def portfolio(n=20_000, seed=1):
    rng = np.random.default_rng(seed)
    e = np.where(rng.random(n) < 0.5, 1.0, rng.uniform(0.05, 1.0, n))
    mu = np.exp(rng.normal(np.log(0.1), 0.5, n))
    N = rng.poisson(e * mu).astype(float)
    return e, mu, N


class TestGini(unittest.TestCase):
    def test_invariant_to_increasing_transformations(self):
        e, mu, N = portfolio()
        g = gini(mu, e, N)
        for f in (lambda x: 3 * x, np.sqrt, lambda x: x ** 2, np.log, lambda x: np.exp(5 * x)):
            self.assertEqual(gini(f(mu), e, N), g)

    def test_flat_rate_scores_zero_and_ties_form_blocks(self):
        e, _, N = portfolio()
        self.assertAlmostEqual(gini(np.ones_like(e), e, N), 0.0, places=12)
        cw, cc = lorenz(np.ones_like(e), e, N)
        self.assertEqual(cw.size, 2)

    def test_premium_gini_of_flat_rate(self):
        e, _, _ = portfolio()
        self.assertAlmostEqual(premium_gini(np.full_like(e, 0.1), e), 0.0, places=12)


class TestIsotonicGeometry(unittest.TestCase):
    def test_recalibration_is_the_convex_minorant(self):
        e, mu, N = portfolio(3000, seed=2)
        noisy = mu * np.exp(np.random.default_rng(3).normal(0, 0.6, mu.size))
        m = murphy(noisy, e, N)
        cw, cc = lorenz(noisy, e, N)
        hx, hy = lower_convex_hull(cw, cc)
        rx, ry = lorenz(m["rc"], e, N)
        self.assertLess(np.max(np.abs(np.interp(rx, hx, hy) - ry)), 1e-12)
        self.assertGreaterEqual(gini(m["rc"], e, N), gini(noisy, e, N) - 1e-12)

    def test_murphy_identity_and_signs(self):
        e, mu, N = portfolio()
        m = murphy(mu * 1.3, e, N)
        self.assertAlmostEqual(m["D"], m["UNC"] - m["DSC"] + m["MCB"], places=6)
        self.assertGreaterEqual(m["MCB"], 0.0)
        self.assertGreaterEqual(m["DSC"], 0.0)

    def test_discrimination_is_a_kl_divergence(self):
        e, mu, N = portfolio()
        m = murphy(mu, e, N)
        _, inv = np.unique(m["rc"], return_inverse=True)
        cb, eb = np.bincount(inv, weights=N), np.bincount(inv, weights=e)
        nz = cb > 0
        kl = np.sum(cb[nz] / N.sum() * np.log((cb[nz] / N.sum()) / (eb[nz] / e.sum())))
        self.assertAlmostEqual(m["DSC"], 2 * N.sum() * kl, places=6)


class TestStepRecalibration(unittest.TestCase):
    def test_steps_are_monotone_balanced_and_large_enough(self):
        e, mu, N = portfolio()
        st = StepRecalibration(min_exposure=500).fit(mu, e, N)
        self.assertTrue(np.all(np.diff(st.rates_) >= 0))
        self.assertTrue(np.all(st.exposure_ >= 500))
        self.assertAlmostEqual((e * st.predict(mu)).sum(), N.sum(), places=6)

    def test_plain_fit_matches_isotonic_in_sample(self):
        e, mu, N = portfolio(5000, seed=4)
        st = StepRecalibration(0.0).fit(mu, e, N)
        self.assertLess(np.max(np.abs(st.predict(mu) - murphy(mu, e, N)["rc"])), 1e-12)


class TestDeviance(unittest.TestCase):
    def test_zero_claims_and_perfect_fit(self):
        self.assertAlmostEqual(poisson_deviance(np.array([0.0, 2.0]), np.array([0.5, 2.0])), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
