"""Exact tests of the partial-overlap transportability counterexample."""

import unittest

from part19_extrapolation_sensitivity import target_effect_with_curvature


class ExtrapolationSensitivityTests(unittest.TestCase):
    """Verify in-support equivalence and out-of-support target uncertainty."""

    def test_original_linear_effect_matches_part19(self) -> None:
        result = target_effect_with_curvature(0.0)
        self.assertEqual(result.trial_ate, 1.0)
        self.assertEqual(result.supported_target_ate, 1.75)
        self.assertEqual(result.full_target_ate, 2.5)

    def test_curvature_does_not_change_trial_or_supported_target(self) -> None:
        for curvature in (-2.0, 0.0, 1.0, 3.0):
            with self.subTest(curvature=curvature):
                result = target_effect_with_curvature(curvature)
                self.assertEqual(result.trial_ate, 1.0)
                self.assertEqual(result.supported_target_ate, 1.75)

    def test_quadratic_target_integration(self) -> None:
        result = target_effect_with_curvature(1.0)
        self.assertEqual(result.full_target_ate, 3.0625)
        self.assertEqual(result.linear_extrapolation_bias, -0.5625)

    def test_extrapolation_bias_changes_sign(self) -> None:
        self.assertGreater(target_effect_with_curvature(-1.0).linear_extrapolation_bias, 0)
        self.assertLess(target_effect_with_curvature(1.0).linear_extrapolation_bias, 0)

    def test_nonfinite_curvature_rejected(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                target_effect_with_curvature(value)


if __name__ == "__main__":
    unittest.main()
