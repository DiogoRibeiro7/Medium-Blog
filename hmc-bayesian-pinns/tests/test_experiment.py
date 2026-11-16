"""Standard-library tests for the deterministic inverse-diffusion benchmark."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import (  # noqa: E402
    OBSERVATION_SIGMA,
    Posterior,
    effective_sample_size,
    mean_field_variational,
    posterior_grid,
    rhat,
    simulate,
    solution,
)


class ExperimentTests(unittest.TestCase):
    """Validate derivatives, physical identity, quadrature and VI optimisers."""

    def test_exact_pde_residual_and_boundaries(self) -> None:
        """The analytic forward model satisfies the PDE and Dirichlet BCs."""
        x = np.linspace(0.0, 1.0, 51)
        t = np.linspace(0.0, 0.7, 51)
        a, k = 1.1, 0.27
        u = solution(x, t, a, k)
        u_t = -(k * np.pi**2) * u
        u_xx = -np.pi**2 * u
        np.testing.assert_allclose(u_t - k * u_xx, np.zeros_like(u), atol=1e-12)
        self.assertLess(abs(float(solution(np.array(0), np.array(.2), a, k))), 1e-12)
        self.assertLess(abs(float(solution(np.array(1), np.array(.2), a, k))), 1e-12)

    def test_posterior_gradient(self) -> None:
        """Analytic score agrees with central differences."""
        post = Posterior(simulate(109, "narrow"))
        candidate = np.log(np.array([1.17, 0.27]))
        num = np.empty(2)
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = 1e-5
            num[j] = (post.potential(candidate + delta) - post.potential(candidate - delta)) / 2e-5
        np.testing.assert_allclose(post.gradient(candidate), num, rtol=1e-7, atol=1e-6)

    def test_grid_normalisation_and_positive_variance(self) -> None:
        """Quadrature mass and parameter intervals are valid."""
        grid = posterior_grid(Posterior(simulate(77, "narrow")), resolution=160)
        self.assertAlmostEqual(float(grid.probabilities.sum()), 1, places=12)
        self.assertLess(grid.boundary_mass, 1e-6)
        summary = grid.summary()
        self.assertGreater(summary["sd_diffusivity"], 0)
        self.assertLess(summary["interval90_diffusivity"][0], summary["interval90_diffusivity"][1])

    def test_vi_and_map_finite(self) -> None:
        """Mean-field VI returns a positive scale around a finite mode."""
        post = Posterior(simulate(92, "narrow"))
        mode = post.map_estimate()
        approximation = mean_field_variational(post, mode)
        self.assertTrue(np.all(np.isfinite(approximation.mean)))
        self.assertTrue(np.all(approximation.sd > 0))
        self.assertTrue(np.isfinite(approximation.objective))

    def test_mcmc_diagnostic_for_independent_normal_draws(self) -> None:
        """ESS approximates draw count and Rhat approaches one for IID chains."""
        rng = np.random.default_rng(335)
        draws = rng.normal(size=(4, 600))
        self.assertGreater(effective_sample_size(draws), 1100)
        self.assertLess(abs(rhat(draws) - 1.0), 0.1)


if __name__ == "__main__":
    unittest.main()
