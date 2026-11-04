"""Tests for Part 16's analytical precision and causal-coverage benchmark."""

from __future__ import annotations

import unittest

from part16_omitted_precision import observed_only_precision


class OmittedPrecisionTests(unittest.TestCase):
    """Check the shrinking standard errors despite increasing causal bias."""

    def test_precision_matches_simulated_design(self) -> None:
        """Population large-sample SEs reproduce the Monte Carlo scale."""
        for strength, expected in ((0.0, 0.04031129), (0.25, 0.0383531),
                                   (0.5, 0.0334664), (1.0, 0.02304886)):
            with self.subTest(strength=strength):
                result = observed_only_precision(strength)
                self.assertAlmostEqual(result.standard_error, expected, delta=0.0001)

    def test_stronger_hidden_assignment_can_increase_bias_and_precision(self) -> None:
        """Narrower conditional intervals need not cover the causal target."""
        zero = observed_only_precision(0.0)
        strong = observed_only_precision(1.0)
        self.assertGreater(strong.coefficient, zero.coefficient)
        self.assertLess(strong.standard_error, zero.standard_error)
        self.assertAlmostEqual(zero.approximate_causal_coverage, 0.95, delta=0.001)
        self.assertLess(strong.approximate_causal_coverage, 1e-10)

    def test_invalid_inputs(self) -> None:
        """Reject invalid nuisance parameters and sample sizes."""
        for strength, n in ((float("nan"), 2000), (float("inf"), 2000), (0.0, 9)):
            with self.subTest(strength=strength, n=n):
                with self.assertRaises(ValueError):
                    observed_only_precision(strength, n)


if __name__ == "__main__":
    unittest.main()
