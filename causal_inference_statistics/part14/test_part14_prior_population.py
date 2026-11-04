"""Tests for Part 14's expected-design Bayesian regularization benchmarks."""

from __future__ import annotations

import unittest

from part14_prior_population import population_ridge


class PopulationRidgeTests(unittest.TestCase):
    """Verify exact algebraic benchmarks and their limits."""

    def test_unadjusted_likelihood_targets_two_as_prior_weakens(self) -> None:
        """Removing shrinkage recovers the confounded association, not tau=1."""
        slopes = population_ridge(5000, 0.000001)
        self.assertAlmostEqual(slopes.unadjusted, 2.0, places=8)
        self.assertAlmostEqual(slopes.adjusted, 1.0, places=8)

    def test_adjusted_coincidence_at_n_equals_precision(self) -> None:
        """At lambda=n, adjusted shrinkage happens to cancel exactly."""
        for n in (100, 1000, 5000):
            with self.subTest(n=n):
                slopes = population_ridge(n, float(n))
                self.assertAlmostEqual(slopes.adjusted, 1.0, places=12)
                self.assertAlmostEqual(slopes.unadjusted, 4.0 / 3.0, places=12)

    def test_adjusted_bias_is_nonmonotone_in_penalty(self) -> None:
        """The adjusted point estimate may first rise then fall as λ increases."""
        self.assertLess(population_ridge(1000, 0.1).adjusted, population_ridge(1000, 100).adjusted)
        self.assertGreater(population_ridge(1000, 100).adjusted, population_ridge(1000, 1000).adjusted)

    def test_invalid_inputs(self) -> None:
        """Reject nonsensical sample sizes and prior precisions."""
        for n, precision in ((0, 1.0), (100, 0.0), (100, float("inf"))):
            with self.subTest(n=n, precision=precision):
                with self.assertRaises(ValueError):
                    population_ridge(n, precision)


if __name__ == "__main__":
    unittest.main()
