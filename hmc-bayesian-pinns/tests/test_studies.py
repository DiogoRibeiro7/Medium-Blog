"""Tests for the diagnostics, design, variational and identification additions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import design  # noqa: E402
import diagnostics  # noqa: E402
import studies  # noqa: E402
from experiment import (  # noqa: E402
    OBSERVATION_SIGMA,
    Posterior,
    adaptive_grid,
    full_rank_objective,
    full_rank_variational,
    hmc,
    posterior_grid,
    simulate,
)

try:  # optional cross-check
    import arviz
except ImportError:  # pragma: no cover - depends on the environment
    arviz = None


def ar1_chains(seed: int, phi: float, chains: int = 4, draws: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((chains, draws))
    for c in range(chains):
        noise = rng.normal(size=draws)
        for i in range(1, draws):
            out[c, i] = phi * out[c, i - 1] + noise[i]
    return out


class DiagnosticsTests(unittest.TestCase):
    def test_iid_draws_have_full_ess_and_unit_rhat(self) -> None:
        draws = np.random.default_rng(5).normal(size=(4, 1000))
        self.assertGreater(diagnostics.ess_bulk(draws), 3200)
        self.assertGreater(diagnostics.ess_tail(draws), 2500)
        self.assertLess(abs(diagnostics.rhat_rank(draws) - 1), 0.01)

    def test_ar1_ess_matches_theory(self) -> None:
        """For AR(1) with coefficient phi the integrated time is (1 + phi) / (1 - phi)."""
        phi = 0.8
        draws = ar1_chains(11, phi, draws=20000)
        expected = draws.size * (1 - phi) / (1 + phi)
        self.assertLess(abs(diagnostics.ess_bulk(draws) / expected - 1), 0.12)

    def test_shifted_chain_inflates_rhat(self) -> None:
        draws = np.random.default_rng(9).normal(size=(4, 800))
        draws[0] += 1.5
        self.assertGreater(diagnostics.rhat_rank(draws), 1.1)

    @unittest.skipIf(arviz is None, "ArviZ not installed")
    def test_agrees_with_arviz(self) -> None:
        draws = np.exp(ar1_chains(3, 0.9) / 3)
        self.assertAlmostEqual(diagnostics.ess_bulk(draws), float(arviz.ess(draws, method="bulk")), places=6)
        self.assertAlmostEqual(diagnostics.ess_tail(draws),
                               float(arviz.ess(draws, method="tail", prob=(0.05, 0.95))), places=6)
        self.assertAlmostEqual(diagnostics.rhat_rank(draws), float(arviz.rhat(draws, method="rank")), places=6)
        for p in (0.05, 0.95):
            ours = diagnostics.mcse_quantile(draws, p)
            theirs = float(arviz.mcse(draws, method="quantile", prob=p))
            self.assertLess(abs(ours / theirs - 1), 0.05)


class DesignTests(unittest.TestCase):
    x = np.linspace(0.18, 0.82, 24)

    def test_fisher_matches_jacobian(self) -> None:
        t = np.linspace(0.06, 0.57, 24)
        amp, kap = 1.15, 0.26
        u = amp * np.sin(np.pi * self.x) * np.exp(-kap * np.pi**2 * t)
        jac = np.column_stack((u, -kap * np.pi**2 * t * u))
        expected = jac.T @ jac / OBSERVATION_SIGMA**2
        np.testing.assert_allclose(design.fisher_information(self.x, t, amp, kap, OBSERVATION_SIGMA), expected,
                                   rtol=1e-12)

    def test_box_lucas_maximises_determinant(self) -> None:
        kap = 0.26
        mask = design.early_mask(self.x, 12)
        late = np.linspace(0.07, 1.5, 4000)
        dets = [np.linalg.det(design.fisher_information(self.x, np.where(mask, 0.06, tl), 1.15, kap,
                                                        OBSERVATION_SIGMA)) for tl in late]
        self.assertAlmostEqual(late[int(np.argmax(dets))], design.box_lucas_second_time(0.06, kap), delta=2e-3)

    def test_kappa_targeted_design_without_prior(self) -> None:
        """With a negligible prior the best split is about a fifth early, the late time ~1.28 decay times on."""
        kap = 0.26
        plan = design.two_time_design(self.x, 0.06, 1.15, kap, OBSERVATION_SIGMA, np.array([1e3, 1e3]))
        self.assertIn(plan.n_early, (5, 6))
        self.assertAlmostEqual((plan.late - plan.early) * kap * np.pi**2, 1.28, delta=0.06)

    def test_early_mask_counts(self) -> None:
        for k in range(1, 24):
            self.assertEqual(int(design.early_mask(self.x, k).sum()), k)


class VariationalTests(unittest.TestCase):
    def test_full_rank_gradient(self) -> None:
        post = Posterior(simulate(92, "narrow"))
        params = np.array([0.05, -1.5, np.log(0.08), np.log(0.03), 0.12])
        _, grad = full_rank_objective(post, params)
        numeric = np.empty(5)
        for j in range(5):
            step = np.zeros(5)
            step[j] = 1e-6
            numeric[j] = (full_rank_objective(post, params + step)[0]
                          - full_rank_objective(post, params - step)[0]) / 2e-6
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-5)

    def test_full_rank_recovers_near_gaussian_posterior(self) -> None:
        """Under the wide design the posterior is close to Gaussian, so full-rank VI should match it."""
        post = Posterior(simulate(20261116, "wide"))
        approx = full_rank_variational(post, post.map_estimate())
        _, cov = posterior_grid(post).log_moments()
        np.testing.assert_allclose(np.sqrt(np.diag(approx.covariance)), np.sqrt(np.diag(cov)), rtol=0.05)
        corr_q = approx.covariance[0, 1] / np.sqrt(np.prod(np.diag(approx.covariance)))
        corr_p = cov[0, 1] / np.sqrt(np.prod(np.diag(cov)))
        self.assertLess(abs(corr_q - corr_p), 0.02)


class IdentificationTests(unittest.TestCase):
    def test_profile_amplitude_is_the_least_squares_optimum(self) -> None:
        data = simulate(4, "narrow")
        kappa = 0.21
        delta, _ = studies.profile_likelihood(data, np.array([kappa, 0.05, 0.5]))
        post = Posterior(data, prior_sd=np.array([1e6, 1e6]))
        amps = np.linspace(0.3, 2.5, 20001)
        nll = [float(post.potential(np.array([np.log(a), np.log(kappa)]))) for a in amps]
        # the curve is relative to its own minimum, so compare differences between two kappas
        nll2 = [float(post.potential(np.array([np.log(a), np.log(0.05)]))) for a in amps]
        self.assertAlmostEqual(delta[0] - delta[1], min(nll) - min(nll2), places=4)

    def test_identity_metric_widths_are_hessian_eigenvalues(self) -> None:
        post = Posterior(simulate(20261116, "narrow"))
        grid = posterior_grid(post, resolution=200)
        points = studies.ridge_points(post, grid, (0.5,))
        rows = studies.ridge_in_metric(points, np.eye(2))
        vals = np.linalg.eigvalsh(points[0].hessian)
        self.assertAlmostEqual(rows[0]["thin_sd"], 1 / np.sqrt(vals[1]), places=10)
        self.assertAlmostEqual(rows[0]["long_sd"], 1 / np.sqrt(vals[0]), places=10)


class GridTests(unittest.TestCase):
    def test_kappa_quantiles_stable_under_refinement(self) -> None:
        """CDF-interpolated quantiles should barely move when the grid is refined."""
        for name in ("narrow", "wide"):
            post = Posterior(simulate(20261116, name))
            coarse = adaptive_grid(post, resolution=230).kappa_quantiles((0.05, 0.95))
            fine = adaptive_grid(post, resolution=960).kappa_quantiles((0.05, 0.95))
            np.testing.assert_allclose(coarse, fine, rtol=1e-3)

    def test_adaptive_grid_contains_posterior(self) -> None:
        grid = adaptive_grid(Posterior(simulate(31, "wide")), resolution=200)
        self.assertLess(grid.boundary_mass, 1e-8)


class SamplerTests(unittest.TestCase):
    def test_default_path_range_matches_original_design(self) -> None:
        self.assertEqual(studies.path_range(15), (9, 21))

    def test_divergences_are_counted(self) -> None:
        """A step size far above the stability limit must produce divergent transitions."""
        post = Posterior(simulate(20261116, "narrow"))
        mode = post.map_estimate()
        result = hmc(post, mode, mass=np.eye(2), seed=1, warmup=20, draws=40, chains=1, target_accept=0.0,
                     leapfrog_range=(10, 10))
        self.assertGreater(result.divergences, 0)


if __name__ == "__main__":
    unittest.main()
