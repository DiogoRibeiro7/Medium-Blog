"""Regression tests for the Part 12 population targeting benchmark."""

from __future__ import annotations

import unittest

from part12_policy_population import CAPACITY, STANDARD_NORMAL, linear_policy, oracle_policy


class PopulationPolicyTests(unittest.TestCase):
    """Verify Gaussian tail allocation and the oracle's tiny improvement."""

    def test_linear_policy_selects_exact_capacity(self) -> None:
        """The one-sided Gaussian threshold selects the desired mass."""
        policy = linear_policy()
        self.assertAlmostEqual(1.0 - STANDARD_NORMAL.cdf(policy.cutoff), CAPACITY, places=12)
        self.assertAlmostEqual(policy.mean_selected_effect, 2.651015316, places=8)

    def test_oracle_selects_two_tails_and_dominates(self) -> None:
        """The nonlinear oracle uses both equal-effect tail boundaries."""
        oracle = oracle_policy()
        linear = linear_policy()
        selection_mass = 1.0 - STANDARD_NORMAL.cdf(oracle.cutoff) + oracle.left_tail_probability
        self.assertAlmostEqual(selection_mass, CAPACITY, places=12)
        self.assertGreater(oracle.left_tail_probability, 0.0)
        self.assertLess(oracle.left_tail_probability, 0.0001)
        self.assertGreater(oracle.mean_selected_effect, linear.mean_selected_effect)
        self.assertAlmostEqual(oracle.mean_selected_effect, 2.651161470, places=8)

    def test_invalid_capacity_is_rejected(self) -> None:
        """Policies require a nonempty, non-universal selected group."""
        for invalid in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(capacity=invalid):
                with self.assertRaises(ValueError):
                    linear_policy(invalid)
                with self.assertRaises(ValueError):
                    oracle_policy(invalid)


if __name__ == "__main__":
    unittest.main()
