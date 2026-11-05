"""Regression checks for the Part 20 policy-allocation stress test."""

from __future__ import annotations

import unittest

import numpy as np

from validate_policy import CAPACITY, N_TEST, SEED, SCENARIOS, simulate


class PolicyValidationTests(unittest.TestCase):
    """Verify deterministic simulation and meaningful decision comparisons."""

    def test_all_declared_scenarios_produce_finite_policy_values(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                scores = simulate(np.random.default_rng(SEED), scenario)
                self.assertEqual(
                    set(scores),
                    {"random", "baseline_risk", "oracle", "linear_cate", "quadratic_cate"},
                )
                self.assertTrue(all(np.isfinite(value) for value in scores.values()))

    def test_seed_reproducibility(self) -> None:
        first = simulate(np.random.default_rng(SEED), "nonlinear")
        second = simulate(np.random.default_rng(SEED), "nonlinear")
        self.assertEqual(first, second)

    def test_oracle_never_underperforms_other_rankings(self) -> None:
        for scenario in SCENARIOS:
            result = simulate(np.random.default_rng(SEED), scenario)
            for policy, value in result.items():
                self.assertLessEqual(value, result["oracle"] + 1e-10, msg=f"{scenario}/{policy}")

    def test_shifted_linear_model_can_be_harmful(self) -> None:
        result = simulate(np.random.default_rng(SEED), "shifted")
        self.assertLess(result["linear_cate"], 0)
        self.assertGreater(result["quadratic_cate"], 0)
        self.assertGreater(result["quadratic_cate"], result["linear_cate"])

    def test_capacity_matches_held_out_population(self) -> None:
        self.assertEqual(N_TEST * CAPACITY, 400.0)

    def test_unknown_scenario_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate(np.random.default_rng(SEED), "unknown")


if __name__ == "__main__":
    unittest.main()
