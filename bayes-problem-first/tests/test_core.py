"""Tests of the methods on cases where the answer is known.

python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bayes_core as B  # noqa: E402

A, Bp = B.PROTOCOL_PRIOR


class TestVaccineArithmetic(unittest.TestCase):
    def test_ve_theta_round_trip(self):
        for r in (1.0, B.SURV_RATIO):
            ve = np.array([-2.0, 0.0, 0.3, 0.95])
            np.testing.assert_allclose(B.ve_from_theta(B.theta_from_ve(ve, r), r), ve)

    def test_prior_mean_is_thirty_percent(self):
        """The protocol chose the prior so that its mean of theta corresponds to VE = 30%."""
        self.assertAlmostEqual(float(B.ve_from_theta(A / (A + Bp))), 0.30, places=3)

    def test_published_interval(self):
        """Beta-binomial with the surveillance-time ratio gives 90.3% to 97.6% (Polack et al. 2020)."""
        lo, hi = B.ve_quantiles(B.VACCINE_CASES, B.TOTAL_CASES, A, Bp, [0.025, 0.975], r=B.SURV_RATIO)
        self.assertEqual(round(100 * float(lo), 1), 90.3)
        self.assertEqual(round(100 * float(hi), 1), 97.6)

    def test_published_boundaries(self):
        """Success boundaries reproduce the protocol's VE thresholds of 76.9%, 62.7% and 58.8%."""
        for n, ve in ((32, 0.769), (92, 0.627), (120, 0.588)):
            x = B.success_boundary(n)
            self.assertAlmostEqual(1 - x / (n - x), ve, places=3)

    def test_grid_matches_conjugate(self):
        th, w = B.grid_posterior(B.VACCINE_CASES, B.TOTAL_CASES, A, Bp)
        pa, pb = B.conjugate(B.VACCINE_CASES, B.TOTAL_CASES, A, Bp)
        for q in (0.05, 0.5, 0.95):
            self.assertAlmostEqual(B.weighted_quantile(th, w, q), stats.beta.ppf(q, pa, pb), places=4)

    def test_sequential_equals_batch(self):
        th, w = B.grid_posterior(0, 0, A, Bp)
        for c in (1, 0, 0, 1, 0, 0, 0):
            w = B.grid_update(th, w, bool(c))
        np.testing.assert_allclose(w, B.grid_posterior(2, 7, A, Bp)[1], rtol=1e-9, atol=1e-15)

    def test_sampler_matches_conjugate(self):
        rng = np.random.default_rng(1)
        draws, _ = B.metropolis_theta(np.full(4, 8), 170, A, Bp, rng=rng, iters=12_000, step=0.6, start=0.0, burn=1_000)
        pa, pb = B.conjugate(8, 170, A, Bp)
        self.assertAlmostEqual(float(np.median(draws)), stats.beta.ppf(0.5, pa, pb), delta=0.002)
        self.assertLess(B.split_rhat(draws), 1.01)

    def test_false_success_under_the_bar(self):
        """At exactly VE = 30% the final-look rule alone succeeds rarely, by construction."""
        rate = B.false_success_rate(np.random.default_rng(2), (164,), (B.THRESHOLD_FINAL,), trials=50_000)
        self.assertLess(rate, 0.03)


class TestOrings(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.temp = np.repeat(np.linspace(50, 85, 12), 4)
        self.flights = B.Flights(self.temp, rng.binomial(B.RINGS, B.p_at(-2.5, -1.0, self.temp)))

    def test_mle_recovers_known_curve(self):
        big = B.Flights(
            np.repeat(self.temp, 200),
            np.random.default_rng(4).binomial(B.RINGS, B.p_at(-2.5, -1.0, np.repeat(self.temp, 200))),
        )
        a, b = B.mle(big)
        self.assertAlmostEqual(a, -2.5, delta=0.1)
        self.assertAlmostEqual(b, -1.0, delta=0.1)

    def test_vague_posterior_mode_near_mle(self):
        g = B.grid_posterior_2d(self.flights, B.VAGUE)
        i, j = np.unravel_index(np.argmax(g.w), g.w.shape)
        a, b = B.mle(self.flights)
        self.assertAlmostEqual(g.a[i], a, delta=0.05)
        self.assertAlmostEqual(g.b[j], b, delta=0.05)

    def test_grid_is_wide_enough(self):
        self.assertLess(B.grid_edge_mass(B.grid_posterior_2d(self.flights, B.WEAKLY_INFORMATIVE)), 1e-6)

    def test_vague_prior_is_extreme(self):
        share = B.prior_extreme_share(B.VAGUE, np.random.default_rng(5))
        expected = 2 * stats.norm.sf(special_logit(0.99) / 10)
        self.assertAlmostEqual(share, expected, delta=0.005)

    def test_load_drops_missing_count(self):
        rows = "rownames,FlightNumber,Temperature,Pressure,Fail,nFailures,Damage\n1,1,66,50,no,0,0\n4,4,80,50,,,\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write(rows)
        f = B.load_flights(Path(fh.name))
        self.assertEqual(f.temp.tolist(), [66.0])


def special_logit(p):
    return float(np.log(p / (1 - p)))


class TestCalibration(unittest.TestCase):
    def test_correct_sampler_is_calibrated(self):
        ranks, covered, _ = B.sbc(B.SAMPLERS[0], np.random.default_rng(6), sims=600)
        self.assertGreater(B.rank_uniformity(ranks)[1], 0.001)
        self.assertAlmostEqual(covered.mean(), 0.9, delta=0.04)

    def test_double_counting_is_caught(self):
        ranks, covered, _ = B.sbc(B.SAMPLERS[2], np.random.default_rng(7), sims=600)
        self.assertLess(B.rank_uniformity(ranks)[1], 1e-6)
        self.assertLess(covered.mean(), 0.82)


if __name__ == "__main__":
    unittest.main()
