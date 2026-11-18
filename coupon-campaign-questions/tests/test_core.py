"""Checks of the estimators on synthetic panels with known answers. They need no download.

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coupon_core as cc  # noqa: E402


def panel(y_by_unit, treated, weeks, n_dur):
    """A minimal Panel around a units-by-weeks array, for the estimator functions."""
    n = len(treated)
    return cc.Panel("x", "Type B", np.array([str(i) for i in range(n)]), np.asarray(treated, bool), weeks,
                    n_dur, 0, n_dur, {"cov_trips": y_by_unit, "all_spend": y_by_unit, "all_trips": y_by_unit},
                    np.zeros(n), np.zeros(n), np.zeros(n, bool), int(np.sum(treated)), 0, 0)


class TestDid(unittest.TestCase):
    def test_recovers_an_injected_effect(self):
        rng = np.random.default_rng(1)
        n, weeks = 4000, np.arange(-8, 5)
        treated = rng.random(n) < 0.3
        mu = rng.gamma(2.0, 0.5, n) + 0.5 * treated            # recipients buy more, by level only
        y = rng.poisson(mu[:, None] * np.ones(len(weeks)) + 0.4 * treated[:, None] * (weeks >= 0)).astype(float)
        p = panel(y, treated, weeks, 5)
        d = cc.window_mean(p, "cov_trips", 0, 4) - cc.window_mean(p, "cov_trips", -8, -1)
        W = cc.bootstrap_weights(n, 200, 2)
        att, _ = cc.did(d, treated, np.zeros(n, int), np.arange(n), W)
        lo, hi = np.quantile(att[1:], [0.005, 0.995])
        self.assertTrue(lo < 0.4 < hi, (att[0], lo, hi))

    def test_cells_reweight_controls_to_the_recipients(self):
        # two cells with different effects of being in the cell but no treatment effect: the matched estimate is zero
        treated = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
        cells = np.array([0, 0, 1, 0, 1, 1, 1, 1])
        d = np.array([1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 5.0, 5.0])
        att, n_t = cc.did(d, treated, cells, np.arange(8), np.ones((1, 8)))
        self.assertAlmostEqual(att[0], 0.0)
        self.assertEqual(n_t[0], 3)
        raw, _ = cc.did(d, treated, np.zeros(8, int), np.arange(8), np.ones((1, 8)))
        self.assertLess(raw[0], 0)                               # without cells the composition difference shows up


class TestMatchingOnTheBaseline(unittest.TestCase):
    """Matching on a noisy baseline when the groups differ in their long-run means manufactures an effect."""

    def test_regression_to_the_mean_appears_without_any_treatment(self):
        rng = np.random.default_rng(3)
        n, weeks = 20000, np.arange(-8, 5)
        treated = rng.random(n) < 0.2
        mu = np.where(treated, rng.gamma(2.0, 0.15, n), rng.gamma(2.0, 0.06, n))    # no effect anywhere
        y = rng.poisson(mu[:, None] * np.ones(len(weeks))).astype(float)
        p = panel(y, treated, weeks, 5)
        d = cc.window_mean(p, "cov_trips", 0, 4) - cc.window_mean(p, "cov_trips", -8, -1)
        cells = np.digitize(y[:, weeks < 0].sum(1), [1, 2, 3, 6])
        one = np.ones((1, n))
        matched, _ = cc.did(d, treated, cells, np.arange(n), one)
        plain, _ = cc.did(d, treated, np.zeros(n, int), np.arange(n), one)
        self.assertGreater(matched[0], 0.03)                     # a spurious "lift"
        self.assertLess(abs(plain[0]), 0.01)                     # the unmatched change is unbiased here


class TestHelpers(unittest.TestCase):
    def test_trend_residual_is_zero_on_straight_lines(self):
        weeks = np.arange(-16, 5)
        y = 2.0 + 0.1 * weeks[None, :] * np.array([[1.0], [-2.0], [0.5]])
        p = panel(y, [1, 0, 0], weeks, 5)
        r, nw = cc.trend_residual(p, "cov_trips", -16, -1, 0, 4)
        self.assertEqual(nw, 5)
        np.testing.assert_allclose(r, 0.0, atol=1e-12)

    def test_benjamini_hochberg(self):
        p = np.array([0.01, 0.04, 0.03, 0.20])
        np.testing.assert_allclose(cc.bh(p), [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)

    def test_bootstrap_first_row_is_the_estimate(self):
        W = cc.bootstrap_weights(50, 10, 0)
        self.assertTrue((W[0] == 1).all())
        self.assertTrue((W[1:].sum(1) == 50).all())


if __name__ == "__main__":
    unittest.main()
