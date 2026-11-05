"""Scientific invariants for the Part 20 oracle population capacity frontier."""

from __future__ import annotations

import unittest

from part20_capacity_frontier import population_capacity_frontier


class CapacityFrontierTests(unittest.TestCase):
    """Verify analytic endpoint values, break-even capacity and welfare loss."""

    def test_zero_and_full_capacity(self) -> None:
        """No treatment gives zero; treating everyone gives the population ATE."""
        none = population_capacity_frontier(0.0)
        all_people = population_capacity_frontier(1.0)
        self.assertEqual(none.exact_population_gain, 0.0)
        self.assertEqual(none.optional_population_gain, 0.0)
        self.assertAlmostEqual(all_people.exact_population_gain, 0.5)
        self.assertLess(all_people.optional_treated_share, 1.0)
        self.assertGreater(all_people.avoidable_population_loss, 0.3)

    def test_twenty_percent_population_gain_matches_archived_scale(self) -> None:
        """Independent Gaussian integral matches original Monte Carlo mean gain."""
        result = population_capacity_frontier(0.2)
        archived_monte_carlo_gain: float = 0.5197891495118
        self.assertAlmostEqual(
            result.exact_population_gain, archived_monte_carlo_gain, delta=0.002
        )
        self.assertAlmostEqual(
            result.exact_population_gain, result.optional_population_gain
        )
        self.assertEqual(result.forced_negative_benefit_share, 0.0)

    def test_break_even_treatment_share(self) -> None:
        """Population benefit crosses zero at X=-1/3, not at capacity 100%."""
        result = population_capacity_frontier(1.0)
        self.assertAlmostEqual(result.positive_benefit_share, 0.6305586598, delta=1e-9)
        self.assertAlmostEqual(result.optional_treated_share, result.positive_benefit_share)
        self.assertAlmostEqual(
            result.forced_negative_benefit_share,
            1.0 - result.positive_benefit_share,
        )

    def test_optional_policy_does_not_treat_negative_benefit(self) -> None:
        """Exact-capacity welfare eventually falls; optional welfare plateaus."""
        at_peak = population_capacity_frontier(0.6305586598)
        above = population_capacity_frontier(0.8)
        full = population_capacity_frontier(1.0)
        self.assertAlmostEqual(
            at_peak.optional_population_gain, above.optional_population_gain, delta=1e-10
        )
        self.assertAlmostEqual(above.optional_population_gain, full.optional_population_gain)
        self.assertGreater(at_peak.exact_population_gain, above.exact_population_gain)
        self.assertGreater(above.exact_population_gain, full.exact_population_gain)
        self.assertGreater(above.avoidable_population_loss, 0.0)

    def test_optional_welfare_dominates_exact_welfare(self) -> None:
        """More available capacity cannot worsen the oracle at-most policy."""
        previous_optional: float = -1.0
        for i in range(101):
            q: float = i / 100
            result = population_capacity_frontier(q)
            self.assertGreaterEqual(result.optional_population_gain + 1e-12, previous_optional)
            self.assertGreaterEqual(
                result.optional_population_gain + 1e-12, result.exact_population_gain
            )
            previous_optional = result.optional_population_gain

    def test_invalid_capacities(self) -> None:
        """Reject percentages outside range and invalid scalar types."""
        for capacity in (-0.01, 1.01, float("inf"), float("nan"), True, "0.2"):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    population_capacity_frontier(capacity)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
