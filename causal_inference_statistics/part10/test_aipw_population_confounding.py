"""Independent numerical checks of Part 10's naive-bias decomposition."""

from __future__ import annotations

import unittest

from aipw_population_confounding import population_contrast


class PopulationConfoundingTests(unittest.TestCase):
    """Check quadrature convergence and opposing selection components."""

    def test_components_have_opposing_signs(self) -> None:
        result = population_contrast()
        self.assertGreater(result.linear_component, 0)
        self.assertLess(result.quadratic_component, 0)
        self.assertLess(abs(result.total_bias), 0.1)
        self.assertAlmostEqual(result.naive_contrast, 1.0 + result.total_bias)

    def test_quadrature_resolution_is_stable(self) -> None:
        coarse = population_contrast(80)
        fine = population_contrast(160)
        self.assertAlmostEqual(coarse.total_bias, fine.total_bias, places=8)
        self.assertAlmostEqual(coarse.treatment_probability, fine.treatment_probability, places=8)


if __name__ == "__main__":
    unittest.main()
