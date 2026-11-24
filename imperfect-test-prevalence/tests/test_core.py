"""Tests of the methods on cases where the answer is known.

python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prevalence_core as C  # noqa: E402


class TestData(unittest.TestCase):
    def test_pooled_counts_match_version_2(self):
        """Bendavid et al. (2020), version 2: specificity 3,308/3,324 and sensitivity 130/157."""
        self.assertEqual(C.pooled(C.SPEC_STUDIES), (3308, 3324))
        self.assertEqual(C.pooled(C.SENS_STUDIES), (130, 157))
        self.assertEqual(len(C.SPEC_STUDIES), 13)

    def test_version_1_counts(self):
        self.assertEqual(C.V1_SPEC, (369 + 30, 371 + 30))
        self.assertEqual(C.V1_SENS, (78 + 25, 85 + 37))


class TestPosterior(unittest.TestCase):
    def test_perfect_test_gives_beta_posterior(self):
        """With specificity and sensitivity exactly 1, the posterior of pi is Beta(y + 1, n - y + 1)."""
        ones = np.ones(10)
        post = C.posterior_pi(50, 3330, ones, ones)
        for q in (0.05, 0.5, 0.95):
            self.assertAlmostEqual(float(post.quantile(q)), stats.beta.ppf(q, 51, 3281), delta=2e-5)

    def test_mass_sums_to_one_and_shortest_interval_holds_its_level(self):
        post, _, _ = C.pooled_posterior(C.V1_SPEC, C.V1_SENS, np.random.default_rng(1), k=5000)
        self.assertAlmostEqual(float(post.mass.sum()), 1.0)
        lo, hi = post.shortest(0.9)
        self.assertAlmostEqual(post.cdf(hi) - post.cdf(lo), 0.9, delta=0.005)

    def test_pooled_v2_reproduces_the_journal_interval(self):
        """The 2021 journal version reports 1.2% (0.7% to 1.8%) unweighted and test-adjusted."""
        post, _, _ = C.pooled_posterior(C.pooled(C.SPEC_STUDIES), C.pooled(C.SENS_STUDIES), np.random.default_rng(2))
        lo, med, hi = post.quantile([0.025, 0.5, 0.975])
        self.assertAlmostEqual(med, 0.012, delta=0.001)
        self.assertAlmostEqual(lo, 0.007, delta=0.001)
        self.assertAlmostEqual(hi, 0.018, delta=0.001)

    def test_hierarchical_matches_gelman_carpenter_strong_prior(self):
        """Gelman and Carpenter (2020), stronger hyperpriors: 0.3%, 1.2%, 2.1% (2.5%, median, 97.5%)."""
        post, *_ = C.hierarchical_posterior(0.2, 0.2, np.random.default_rng(3), k=50_000)
        lo, med, hi = post.quantile([0.025, 0.5, 0.975])
        self.assertAlmostEqual(lo, 0.003, delta=0.001)
        self.assertAlmostEqual(med, 0.012, delta=0.001)
        self.assertAlmostEqual(hi, 0.021, delta=0.0015)


class TestHierarchy(unittest.TestCase):
    def test_marginal_likelihood_at_zero_spread_is_binomial(self):
        mu, sig = np.array([3.0]), np.array([0.0])
        got = C.study_marginal_loglik(mu, sig, 95, 100)[0]
        p = special.expit(3.0)
        self.assertAlmostEqual(got, 95 * np.log(p) + 5 * np.log1p(-p), places=10)

    def test_marginal_likelihood_matches_monte_carlo(self):
        rng = np.random.default_rng(4)
        z = rng.normal(2.0, 0.8, 400_000)
        p = special.expit(z)
        mc = np.log(np.mean(np.exp(18 * np.log(p) + 2 * np.log1p(-p))))
        got = C.study_marginal_loglik(np.array([2.0]), np.array([0.8]), 18, 20)[0]
        self.assertAlmostEqual(got, mc, delta=0.01)


class TestDesignAndChecks(unittest.TestCase):
    def test_design_quantiles_are_ordered_and_cover(self):
        rng = np.random.default_rng(5)
        q05, q95 = C.design_widths(800, 122, 2931, 0.03, 0.995, 0.83, rng, reps=40)
        self.assertTrue(np.all(q05 < q95))
        self.assertGreater(np.mean((q05 < 0.03) & (q95 > 0.03)), 0.7)

    def test_ppc_flags_the_heterogeneity(self):
        ppc = C.pooled_ppc(np.random.default_rng(6), k=4000)
        self.assertLess(ppc["p_value"], 0.01)
        self.assertEqual(ppc["largest_n"], 1102)

    def test_engine_is_calibrated(self):
        u = C.calibration(np.random.default_rng(7), sims=80, k=1500)
        self.assertGreater(stats.kstest(u, "uniform").pvalue, 0.01)


if __name__ == "__main__":
    unittest.main()
