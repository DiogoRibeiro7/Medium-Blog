"""Deterministic tests for Part 17's analytical OLS benchmarks."""

import math
import unittest

from part17_population_moments import population_moments


class PopulationMomentsTests(unittest.TestCase):
    """Check causal bias, placebo nulls, and scale of sampling uncertainty."""

    def test_target_slopes_identify_structural_bias(self) -> None:
        self.assertEqual(population_moments("no_confounding")["target"].slope, 1.0)
        for scenario in ("shared_confounder", "outcome_specific_confounder"):
            self.assertEqual(population_moments(scenario)["target"].slope, 1.75)

    def test_control_only_detects_shared_confounder(self) -> None:
        self.assertEqual(population_moments("shared_confounder")["negative_control_outcome"].slope, 0.5)
        self.assertEqual(population_moments("outcome_specific_confounder")["negative_control_outcome"].slope, 0.0)

    def test_independent_placebo_is_null_in_all_scenarios(self) -> None:
        for scenario in ("no_confounding", "shared_confounder", "outcome_specific_confounder"):
            with self.subTest(scenario=scenario):
                self.assertEqual(population_moments(scenario)["placebo_exposure"].slope, 0.0)

    def test_asymptotic_se_scales_with_inverse_square_root_n(self) -> None:
        small = population_moments("shared_confounder", n=1200)
        large = population_moments("shared_confounder", n=4800)
        for test_name in small:
            self.assertAlmostEqual(small[test_name].asymptotic_se / large[test_name].asymptotic_se, 2.0)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            population_moments("missing")
        with self.assertRaises(ValueError):
            population_moments("no_confounding", n=2)
        with self.assertRaises(TypeError):
            population_moments("no_confounding", n=True)

    def test_standard_errors_match_population_formulae(self) -> None:
        shared = population_moments("shared_confounder")
        self.assertTrue(math.isclose(shared["target"].asymptotic_se, math.sqrt(3.125 / 2400)))
        self.assertTrue(math.isclose(shared["negative_control_outcome"].asymptotic_se, 0.025))


if __name__ == "__main__":
    unittest.main()
