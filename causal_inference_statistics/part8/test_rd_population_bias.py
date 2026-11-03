"""Exact-check tests for Part 8's population smoothing bias."""

from __future__ import annotations

import unittest

import numpy as np

from rd_population_bias import (
    population_target,
    quadratic_boundary_intercept,
    triangular_moment,
)


class RDPopulationBiasTests(unittest.TestCase):
    """Compare exact moments with independent numerical weighted regression."""

    def test_exact_moments(self) -> None:
        self.assertAlmostEqual(triangular_moment(0), 0.5)
        self.assertAlmostEqual(triangular_moment(3), 0.05)
        with self.assertRaises(ValueError):
            triangular_moment(-1)

    def test_boundary_intercept_against_dense_quadrature(self) -> None:
        for bandwidth in (0.15, 0.30, 0.60, 1.0):
            scaled = (np.arange(20_000, dtype=float) + 0.5) / 20_000
            x = scaled * bandwidth
            weight = 1.0 - scaled
            design = np.column_stack((np.ones(x.size), x))
            estimate = np.linalg.lstsq(
                design * np.sqrt(weight)[:, None],
                2.0 * x**2 * np.sqrt(weight),
                rcond=None,
            )[0][0]
            self.assertAlmostEqual(
                float(estimate), quadratic_boundary_intercept(2.0, bandwidth),
                places=8,
            )

    def test_population_targets(self) -> None:
        for bandwidth in (0.15, 0.30, 0.60, 1.0):
            self.assertAlmostEqual(population_target("smooth", bandwidth).fitted_jump, -1)
            self.assertAlmostEqual(population_target("sorting", bandwidth).fitted_jump, -0.3)
            self.assertAlmostEqual(
                population_target("curvature", bandwidth).fitted_jump,
                -1 - 0.2 * bandwidth**2,
            )


if __name__ == "__main__":
    unittest.main()
