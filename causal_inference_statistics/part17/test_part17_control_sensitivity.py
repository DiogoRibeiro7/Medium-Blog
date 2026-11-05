"""Independent numerical invariants for Part 17 control-overlap sensitivity."""

from __future__ import annotations

import unittest

from part17_control_sensitivity import control_sensitivity


class ControlSensitivityTests(unittest.TestCase):
    """Check both archived limiting cases and continuous power behaviour."""

    def test_original_endpoint_population_coefficients(self) -> None:
        """The two original confounded designs have the same target bias."""
        invisible = control_sensitivity(0.0)
        detected = control_sensitivity(1.0)
        for result in (invisible, detected):
            self.assertAlmostEqual(result.target_causal_effect, 1.0)
            self.assertAlmostEqual(result.target_observational_slope, 1.75)
            self.assertAlmostEqual(result.target_bias, 0.75)
        self.assertAlmostEqual(invisible.negative_control_slope, 0.0)
        self.assertAlmostEqual(detected.negative_control_slope, 0.5)
        self.assertAlmostEqual(invisible.approximate_rejection_probability, 0.05)
        self.assertGreater(detected.approximate_rejection_probability, 0.999)

    def test_derived_residual_variance_and_standard_error(self) -> None:
        """Population conditioning agrees with independently known endpoints."""
        at_zero = control_sensitivity(0.0, sample_size=1200)
        at_one = control_sensitivity(1.0, sample_size=1200)
        self.assertAlmostEqual(at_zero.negative_control_residual_variance, 2.0)
        self.assertAlmostEqual(at_one.negative_control_residual_variance, 1.5)
        self.assertAlmostEqual(at_zero.approximate_standard_error**2, 1 / 1200)
        self.assertAlmostEqual(at_one.approximate_standard_error, 0.025)

    def test_control_power_can_be_low_despite_fixed_large_target_bias(self) -> None:
        """A weak proxy offers little detection even when the causal bias is 0.75."""
        results = [control_sensitivity(rho) for rho in (0.0, 0.05, 0.1, 0.2)]
        self.assertTrue(
            all(left.target_bias == right.target_bias for left, right in zip(results, results[1:]))
        )
        self.assertTrue(
            all(
                left.approximate_rejection_probability < right.approximate_rejection_probability
                for left, right in zip(results, results[1:])
            )
        )
        self.assertLess(results[1].approximate_rejection_probability, 0.2)
        self.assertGreater(results[-1].approximate_rejection_probability, 0.85)

    def test_sign_symmetry_and_more_information(self) -> None:
        """Two-sided power is symmetric and improves with larger samples."""
        positive = control_sensitivity(0.1)
        negative = control_sensitivity(-0.1)
        larger = control_sensitivity(0.1, sample_size=4800)
        self.assertAlmostEqual(
            positive.approximate_rejection_probability,
            negative.approximate_rejection_probability,
        )
        self.assertAlmostEqual(
            positive.negative_control_slope, -negative.negative_control_slope
        )
        self.assertGreater(
            larger.approximate_rejection_probability,
            positive.approximate_rejection_probability,
        )

    def test_invalid_design_parameters(self) -> None:
        """Reject non-finite and non-sensical arguments before computation."""
        for alignment in (float("nan"), float("inf"), -1.01, 1.01, True, "0.5"):
            with self.subTest(alignment=alignment):
                with self.assertRaises(ValueError):
                    control_sensitivity(alignment)  # type: ignore[arg-type]
        for sample_size in (0, 29, 1200.0, True):
            with self.subTest(sample_size=sample_size):
                with self.assertRaises(ValueError):
                    control_sensitivity(0.5, sample_size=sample_size)  # type: ignore[arg-type]
        for alpha in (0.0, 1.0, -0.1, float("nan"), False):
            with self.subTest(significance=alpha):
                with self.assertRaises(ValueError):
                    control_sensitivity(0.5, significance=alpha)


if __name__ == "__main__":
    unittest.main()
