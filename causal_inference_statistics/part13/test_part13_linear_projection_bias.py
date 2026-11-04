"""Unit checks for the exact population projection-bias calculation."""

from __future__ import annotations

import unittest

from part13_linear_projection_bias import nonlinear_linear_projection_limit


class LinearProjectionBiasTests(unittest.TestCase):
    """Check independent Gaussian moments and the misspecified DML limit."""

    def test_nonlinear_design_has_positive_population_bias(self) -> None:
        """Residual nonlinear confounding shifts the coefficient above one."""
        result = nonlinear_linear_projection_limit()
        self.assertGreater(result.sine_residual_variance, 0.0)
        self.assertGreater(result.omitted_treatment_variance, 0.0)
        self.assertGreater(result.omitted_treatment_outcome_covariance, 0.0)
        self.assertAlmostEqual(result.coefficient, 1.9505905792805183, places=12)

    def test_more_treatment_noise_reduces_confounded_projection_bias(self) -> None:
        """Larger independent treatment variation dilutes omitted covariance."""
        baseline = nonlinear_linear_projection_limit()
        noisier = nonlinear_linear_projection_limit(treatment_noise_variance=4.0)
        self.assertGreater(noisier.coefficient, 1.0)
        self.assertLess(noisier.coefficient, baseline.coefficient)

    def test_rejects_nonpositive_treatment_noise_variance(self) -> None:
        """Reject an unsupported zero-variance treatment disturbance."""
        with self.assertRaises(ValueError):
            nonlinear_linear_projection_limit(0.0)


if __name__ == "__main__":
    unittest.main()
