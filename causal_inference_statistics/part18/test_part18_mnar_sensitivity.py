"""Tests for Part 18's selected-population analytical calculation."""

import unittest

from part18_mnar_sensitivity import population_complete_case_effect


class MnarSensitivityTests(unittest.TestCase):
    """Check analytical selection effects and numerical convergence."""

    def test_mar_is_unbiased_given_correct_conditional_outcome_model(self) -> None:
        result = population_complete_case_effect(0.0)
        self.assertAlmostEqual(result.complete_case_effect, 1.0, places=11)

    def test_original_mnar_scenario_matches_monte_carlo(self) -> None:
        result = population_complete_case_effect(0.9)
        # Original experiment: mean=0.9724, observed fraction=0.5842.
        self.assertAlmostEqual(result.complete_case_effect, 0.9724, delta=0.006)
        self.assertAlmostEqual(result.observed_fraction, 0.5842, delta=0.006)

    def test_selection_bias_is_not_monotone_in_sensitivity_parameter(self) -> None:
        baseline = population_complete_case_effect(0.0)
        weak = population_complete_case_effect(0.3)
        cancellation = population_complete_case_effect(0.6)
        moderate = population_complete_case_effect(0.9)
        stronger = population_complete_case_effect(1.5)
        self.assertGreater(weak.complete_case_effect, baseline.complete_case_effect)
        self.assertAlmostEqual(cancellation.complete_case_effect, 1.0, places=9)
        self.assertGreater(baseline.complete_case_effect, moderate.complete_case_effect)
        self.assertGreater(moderate.complete_case_effect, stronger.complete_case_effect)

    def test_quadrature_convergence(self) -> None:
        low = population_complete_case_effect(0.9, order=30)
        high = population_complete_case_effect(0.9, order=50)
        self.assertAlmostEqual(low.complete_case_effect, high.complete_case_effect, places=7)

    def test_invalid_parameters(self) -> None:
        for gamma in (-0.1, float("nan"), float("inf")):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                population_complete_case_effect(gamma)
        with self.assertRaises(ValueError):
            population_complete_case_effect(0.9, order=2)
        with self.assertRaises(ValueError):
            population_complete_case_effect(0.9, order=True)


if __name__ == "__main__":
    unittest.main()
