"""Tests for the small estimators in panel_stats (known answers and simulations with fixed seeds)."""

import math
import random
import unittest

import panel_stats as S


class LinearAlgebraTests(unittest.TestCase):
    def test_solve_inverse_and_singularity(self) -> None:
        a = [[4.0, 2.0], [1.0, 3.0]]
        x = S.solve(a, [10.0, 5.0])
        self.assertAlmostEqual(x[0], 2.0)
        self.assertAlmostEqual(x[1], 1.0)
        inv = S.inverse(a)
        self.assertAlmostEqual(inv[0][0], 0.3)
        self.assertAlmostEqual(inv[1][0], -0.1)
        with self.assertRaises(ValueError):
            S.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])

    def test_quantiles(self) -> None:
        self.assertAlmostEqual(S.normal_quantile(0.975), 1.959964, places=5)
        self.assertAlmostEqual(S.normal_quantile(0.01), -2.326348, places=5)
        self.assertAlmostEqual(S.t_quantile(0.975, 25), 2.059539, places=3)
        self.assertAlmostEqual(S.t_quantile(0.975, 10), 2.228139, places=3)
        self.assertEqual(S.quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)


class OlsTests(unittest.TestCase):
    def test_exact_fit_and_hand_computed_cluster_variance(self) -> None:
        x = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        y = [1.0, 3.0, 5.0, 7.0]
        fit = S.ols(x, y, [1, 1, 2, 2], ["c", "x"])
        self.assertAlmostEqual(fit.beta[0], 1.0)
        self.assertAlmostEqual(fit.beta[1], 2.0)
        self.assertAlmostEqual(fit.se[1], 0.0, places=9)
        # Residuals (1, -1, -1, 1) by hand; slope variance from the sandwich formula.
        y2 = [2.0, 2.0, 4.0, 8.0]
        fit2 = S.ols(x, y2, [1, 1, 2, 2], ["c", "x"])
        a = S.inverse(S.crossprod(x))
        resid = fit2.resid
        s1 = [sum(x[i][c] * resid[i] for i in (0, 1)) for c in range(2)]
        s2 = [sum(x[i][c] * resid[i] for i in (2, 3)) for c in range(2)]
        meat = [[s1[p] * s1[q] + s2[p] * s2[q] for q in range(2)] for p in range(2)]
        v11 = (2 / 1) * (3 / 2) * sum(a[1][p] * meat[p][q] * a[q][1] for p in range(2) for q in range(2))
        self.assertAlmostEqual(fit2.se[1], math.sqrt(v11))

    def test_two_way_within_and_recovery_of_a_known_slope(self) -> None:
        rng = random.Random(7)
        units, years = list(range(60)), list(range(2015, 2021))
        alpha = {u: rng.gauss(0, 2) for u in units}
        gamma = {t: rng.gauss(0, 1) for t in years}
        x = {(u, t): rng.gauss(0, 1) + alpha[u] for u in units for t in years}
        y = {(u, t): alpha[u] + gamma[t] + 0.8 * x[(u, t)] + rng.gauss(0, 0.5) for u in units for t in years}
        wx, wy = S.two_way_within(x, units, years), S.two_way_within(y, units, years)
        for u in units:
            self.assertAlmostEqual(sum(wx[(u, t)] for t in years), 0.0, places=9)
        for t in years:
            self.assertAlmostEqual(sum(wx[(u, t)] for u in units), 0.0, places=9)
        keys = [(u, t) for u in units for t in years]
        fit = S.ols([[wx[k]] for k in keys], [wy[k] for k in keys], [k[0] for k in keys], ["x"])
        lo, hi = fit.ci(0)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        boot = S.wild_cluster_bootstrap([[wx[k]] for k in keys], [wy[k] for k in keys], [k[0] % 12 for k in keys], 0, reps=499)
        self.assertLess(boot["ci95_low"], 0.8)
        self.assertGreater(boot["ci95_high"], 0.8)
        self.assertLess(boot["p_value"], 0.01)
        self.assertLess(boot["p_value_restricted"], 0.01)
        # Under a true null the restricted p-value is not small.
        y0 = {(u, t): alpha[u] + gamma[t] + rng.gauss(0, 0.5) for u in units for t in years}
        wy0 = S.two_way_within(y0, units, years)
        null = S.wild_cluster_bootstrap([[wx[k]] for k in keys], [wy0[k] for k in keys], [k[0] % 12 for k in keys], 0, reps=499)
        self.assertGreater(null["p_value_restricted"], 0.05)
        # With two regressors the restricted bootstrap drops only the tested one.
        two = S.wild_cluster_bootstrap([[wx[k], wy0[k]] for k in keys], [wy[k] for k in keys], [k[0] % 12 for k in keys], 0, reps=199)
        self.assertLess(two["p_value_restricted"], 0.05)
        self.assertGreater(S.within_share(x, units, years), 0.0)
        with self.assertRaises(ValueError):
            S.two_way_within({(0, 2015): 1.0}, [0], [2015, 2016])

    def test_wooldridge_distinguishes_iid_from_random_walk_errors(self) -> None:
        rng = random.Random(11)
        units, years = list(range(300)), list(range(8))
        iid = {(u, t): rng.gauss(0, 1) for u in units for t in years}
        walk: dict[tuple[int, int], float] = {}
        for u in units:
            level = 0.0
            for t in years:
                level += rng.gauss(0, 1)
                walk[(u, t)] = level

        def fd(e: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
            return {(u, t): e[(u, t)] - e[(u, t - 1)] for u in units for t in years[1:]}

        self.assertAlmostEqual(S.wooldridge_serial(fd(iid), units, years[1:])["slope"], -0.5, delta=0.06)
        self.assertAlmostEqual(S.wooldridge_serial(fd(walk), units, years[1:])["slope"], 0.0, delta=0.06)


class CorrelationTests(unittest.TestCase):
    def test_correlations_bootstrap_and_fisher(self) -> None:
        rng = random.Random(3)
        x = [rng.gauss(0, 1) for _ in range(200)]
        y = [0.6 * v + rng.gauss(0, 0.8) for v in x]
        out = S.cluster_bootstrap_corr(x, y, [i // 8 for i in range(200)], reps=499)
        self.assertLess(out["pearson_ci95_low"], out["pearson"])
        self.assertGreater(out["pearson_ci95_high"], out["pearson"])
        self.assertGreater(out["spearman"], 0.4)
        self.assertEqual(S.ranks([3.0, 1.0, 3.0]), [2.5, 1.0, 2.5])
        lo, hi = S.fisher_ci(0.5, 30)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertEqual(S.fisher_ci(0.5, 3), (-1.0, 1.0))
        trend = [float(t) + rng.gauss(0, 0.1) for t in range(30)]
        self.assertLess(S.effective_n(trend, trend), 5)
        self.assertGreater(S.autocorrelation(trend), 0.8)


if __name__ == "__main__":
    unittest.main()
