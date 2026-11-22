"""Checks of the methods and of the transcribed snapshot, on cases where the answer is known.

python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_data as fd  # noqa: E402
import ranking_core as rc  # noqa: E402
import snapshot as sn  # noqa: E402

# Digest recorded in DiogoRibeiro7/data, external/eurotoptech-happiness-taxes-europe/metadata.yaml
SNAPSHOT_SHA256 = "7bb9db9c13991e50adeb860a8e2f8403fe0ff85687f1862e582dbcdc64707140"


class TestSnapshot(unittest.TestCase):
    def test_counts_match_the_page(self):
        d = sn.table()
        self.assertEqual(len(d), 29)
        self.assertEqual(int(d.n.sum()), 744)
        self.assertLessEqual(int(d.n.sum()), sn.RATED_REPORTS)
        self.assertEqual(int(d.thin.sum()), 12)
        self.assertTrue((d.n >= 5).all())

    def test_page_order_is_by_happiness(self):
        h = sn.table().happiness.to_numpy()
        self.assertTrue(np.all(np.diff(h) <= 0))

    def test_ratings_in_range(self):
        d = sn.table()
        for col in ("happiness", "services"):
            self.assertTrue(d[col].between(0, 10).all())

    def test_digest_matches_registry(self):
        self.assertEqual(sn.canonical_sha256(), SNAPSHOT_SHA256)

    def test_correlations_quoted_by_the_page(self):
        """The page's two country-level correlations follow from the table to two decimals (one for services)."""
        d = sn.table()
        self.assertAlmostEqual(rc.pearson(d.happiness, d.tax), -0.46, delta=0.01)
        self.assertAlmostEqual(rc.pearson(d.services, d.tax), 0.06, delta=0.02)


class TestEvidence(unittest.TestCase):
    def test_matches_numerical_integration_over_the_mean(self):
        rng = np.random.default_rng(0)
        n = np.array([5, 8, 20, 60, 110])
        y = 6.5 + rng.normal(0, 0.8, len(n))
        X = rc.design(len(n))
        for sigma, tau in ((2.0, 0.5), (1.2, 1.1), (3.0, 0.05)):
            v = tau**2 + sigma**2 / n

            def integrand(mu, v=v):
                return np.exp(np.sum(stats.norm.logpdf(y, mu, np.sqrt(v))))

            brute = np.log(integrate.quad(integrand, -20, 30, points=[y.mean()], limit=200)[0])
            ours = rc.log_evidence(y, n, X, sigma, tau)
            # equal up to the constant -N/2 log(2 pi) + 1/2 log(2 pi) dropped in log_evidence
            const = -0.5 * len(n) * np.log(2 * np.pi) + 0.5 * np.log(2 * np.pi)
            self.assertAlmostEqual(float(ours + const), float(brute), places=6)

    def test_grid_shape_and_normalisation(self):
        d = sn.table()
        p = rc.grid_posterior(d.services, d.n, rc.design(29))
        self.assertEqual(p.shape, (len(rc.SIGMA_GRID), len(rc.TAU_GRID)))
        self.assertAlmostEqual(float(p.sum()), 1.0)

    def test_recovers_sigma_and_tau_with_many_groups(self):
        rng = np.random.default_rng(1)
        n = rng.integers(5, 120, 600)
        theta = 7 + 0.6 * rng.standard_normal(len(n))
        y = theta + 2.2 / np.sqrt(n) * rng.standard_normal(len(n))
        f = rc.posterior_draws(y, n, rc.design(len(n)), 4000, rng)
        # sigma is learnt only from how the spread changes with n, so it is the less precise of the two
        for draws, truth in ((f["sigma"], 2.2), (f["tau"], 0.6)):
            lo, hi = rc.interval(draws, 0.95)
            self.assertLess(lo, truth)
            self.assertGreater(hi, truth)
        self.assertLess(float(np.subtract(*rc.interval(f["tau"], 0.9)[::-1])), 0.2)


class TestShrinkage(unittest.TestCase):
    def test_huge_samples_are_not_shrunk(self):
        rng = np.random.default_rng(2)
        y = np.array([5.0, 6.0, 7.5, 8.0])
        n = np.full(4, 1e7)
        f = rc.posterior_draws(y, n, rc.design(4), 2000, rng)
        np.testing.assert_allclose(f["theta"].mean(0), y, atol=0.01)

    def test_weight_formula(self):
        self.assertAlmostEqual(float(rc.shrinkage_weight(4, 2.0, 1.0)), 0.5)
        self.assertAlmostEqual(float(rc.shrinkage_weight(100, 2.0, 0.0)), 0.0)

    def test_posterior_mean_is_the_weighted_average(self):
        """With sigma and tau pinned, E[theta | y] = w y + (1 - w) mu-hat on average over the draws."""
        rng = np.random.default_rng(3)
        y = np.array([4.0, 6.0, 9.0])
        n = np.array([4, 16, 64])
        sig = np.array([2.0, 2.0 + 1e-9])
        tau = np.array([1.0, 1.0 + 1e-9])
        f = rc.posterior_draws(y, n, rc.design(3), 200_000, rng, sigma_grid=sig, tau_grid=tau)
        v = 1.0 + 4.0 / n
        mu = np.sum(y / v) / np.sum(1 / v)
        w = rc.shrinkage_weight(n, 2.0, 1.0)
        np.testing.assert_allclose(f["theta"].mean(0), w * y + (1 - w) * mu, atol=0.01)


class TestRanksAndExtremes(unittest.TestCase):
    def test_ranks(self):
        np.testing.assert_array_equal(rc.ranks_desc([3.0, 9.0, 1.0, 5.0]), [3, 1, 4, 2])
        np.testing.assert_array_equal(rc.ranks_desc([[1, 2], [2, 1]]), [[2, 1], [1, 2]])

    def test_rank_interval_rounds_outwards(self):
        draws = np.repeat(np.arange(1, 11)[:, None], 10, axis=1).T.reshape(-1, 1)
        lo, hi = rc.rank_interval(draws, 0.9)
        self.assertEqual(int(lo[0]), 1)
        self.assertEqual(int(hi[0]), 10)

    def test_thin_at_extremes(self):
        means = np.array([9, 1, 5, 6, 4, 8, 2, 7])
        thin = np.array([1, 1, 0, 0, 0, 0, 0, 1], bool)
        # top 3: 9 (thin), 8, 7 (thin); bottom 3: 4, 2, 1 (thin)
        self.assertEqual(int(rc.thin_at_extremes(means, thin, 3)), 3)

    def test_funnel_limits(self):
        lo, hi = rc.funnel_limits(7.0, 2.0, np.array([4, 100]), 1.96)
        np.testing.assert_allclose(hi - 7.0, [1.96, 0.392])
        np.testing.assert_allclose(7.0 - lo, [1.96, 0.392])

    def test_noise_alone_puts_small_groups_at_the_ends(self):
        """Equal true means: the largest of many noisy means mostly belongs to a small group."""
        rng = np.random.default_rng(4)
        n = np.array([5] * 10 + [100] * 10)
        y, _ = rc.simulate_means(n, np.zeros(5000), np.ones(5000), np.zeros(5000), rng)
        top_small = (np.argmax(y, 1) < 10).mean()
        self.assertGreater(top_small, 0.95)


class TestDiscreteRatings(unittest.TestCase):
    def test_noise_has_mean_zero_and_set_sd(self):
        e = rc.skewed_noise(400_000, 2.0, np.random.default_rng(5))
        self.assertAlmostEqual(float(e.mean()), 0.0, delta=0.02)
        self.assertAlmostEqual(float(e.std()), 2.0, delta=0.02)
        self.assertLess(float(stats.skew(e)), -0.8)

    def test_expected_rating_curve(self):
        locs = np.arange(-2, 13, 0.5)
        c = rc.expected_rating_curve(locs, 2.0, np.random.default_rng(6), size=50_000)
        self.assertTrue(np.all(np.diff(c) >= 0))
        self.assertTrue(np.all((c >= 0) & (c <= 10)))

    def test_ratings_are_whole_numbers_in_range(self):
        r = rc.ratings(7.0, 1000, 2.5, np.random.default_rng(7))
        self.assertTrue(np.all(r == np.rint(r)))
        self.assertTrue(np.all((r >= 0) & (r <= 10)))


class TestOecdTable(unittest.TestCase):
    def test_build_keeps_2023_countries_with_all_three(self):
        rows = [
            "DATAFLOW,FREQ,REF_AREA,MEASURE,UNIT_MEASURE,SECTOR,EDITION,SCALE,CATEGORY,TIME_PERIOD,OBS_VALUE",
            "x,A,PRT,CS_HC,PT_RESP,_Z,2025,SAT,SPS,2023,40",
            "x,A,PRT,CS_ES,PT_RESP,_Z,2025,SAT,SPS,2023,50",
            "x,A,PRT,TRUST_S_AS,PT_RESP,_Z,2025,SAT,SPS,2023,60",
            "x,A,PRT,CS_HC,PT_RESP,_Z,2025,SAT,SPS,2021,10",
            "x,A,AUT,CS_HC,PT_RESP,_Z,2025,SAT,SPS,2021,70",
            "x,A,OECD_REP,CS_HC,PT_RESP,_Z,2025,SAT,SPS,2023,55",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.csv"
            p.write_text("\n".join(rows) + "\n")
            t = fd.build_oecd(p)
        self.assertEqual(t.iso3.tolist(), ["PRT"])
        self.assertAlmostEqual(float(t.composite.iloc[0]), 50.0)


if __name__ == "__main__":
    unittest.main()
