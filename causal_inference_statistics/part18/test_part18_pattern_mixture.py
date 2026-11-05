"""Unit tests for exact outcome-observation mixture identities."""

from __future__ import annotations

import unittest

from part18_pattern_mixture import (
    ObservedArm,
    completed_arm_mean,
    completed_contrast,
    contrast_bounds,
    treated_shift_for_zero_contrast,
)


class PatternMixtureTests(unittest.TestCase):
    """Check exact reconstruction, interval extrema, and input safeguards."""

    def test_known_complete_cohorts_reconstruct_exactly(self) -> None:
        """Known unobserved means give exact full-sample arithmetic contrasts."""
        # Treated outcomes [3, 5, 1, 2]: 2 observed, 2 missing.
        treated = ObservedArm(respondent_mean=4.0, response_fraction=0.5)
        # Control outcomes [1, 2, 3, -1, 0]: 3 observed, 2 missing.
        control = ObservedArm(respondent_mean=2.0, response_fraction=0.6)
        self.assertAlmostEqual(completed_arm_mean(treated, -2.5), 11.0 / 4.0)
        self.assertAlmostEqual(completed_arm_mean(control, -2.5), 5.0 / 5.0)
        self.assertAlmostEqual(completed_contrast(treated, control, -2.5, -2.5), 1.75)

    def test_zero_shifts_recover_respondent_contrast(self) -> None:
        """Equal observed/missing means imply no mixture correction."""
        treated = ObservedArm(2.0, 0.55)
        control = ObservedArm(1.2, 0.7)
        self.assertAlmostEqual(completed_contrast(treated, control), 0.8)
        self.assertAlmostEqual(completed_contrast(treated, control, 0.0, 0.0), 0.8)

    def test_extreme_shift_intervals_and_tipping_point(self) -> None:
        """Signed extrema and the zero tipping point follow the same identity."""
        treated = ObservedArm(2.0, 0.55)
        control = ObservedArm(1.2, 0.7)
        interval = contrast_bounds(treated, control, (-2.0, 0.0), (-2.0, 0.0))
        self.assertAlmostEqual(interval.respondent_contrast, 0.8)
        self.assertAlmostEqual(interval.lower, -0.1)
        self.assertAlmostEqual(interval.upper, 1.4)
        shift = treated_shift_for_zero_contrast(treated, control)
        self.assertGreaterEqual(shift, -2.0)
        self.assertLessEqual(shift, 0.0)
        self.assertAlmostEqual(completed_contrast(treated, control, shift, 0.0), 0.0)

    def test_no_missing_outcomes_makes_shifts_ineffective(self) -> None:
        """At complete response, a missing-outcome parameter cannot change means."""
        treated = ObservedArm(2.0, 1.0)
        control = ObservedArm(1.2, 1.0)
        self.assertAlmostEqual(completed_contrast(treated, control, -5.0, 7.0), 0.8)
        with self.assertRaises(ValueError):
            treated_shift_for_zero_contrast(treated, control)

    def test_shift_in_control_changes_treated_tipping_point(self) -> None:
        """An assumed control missingness shift affects the tipping threshold."""
        treated = ObservedArm(2.0, 0.5)
        control = ObservedArm(1.0, 0.5)
        shift = treated_shift_for_zero_contrast(treated, control, -1.0)
        self.assertAlmostEqual(shift, -3.0)
        self.assertAlmostEqual(completed_contrast(treated, control, shift, -1.0), 0.0)

    def test_invalid_summaries_and_bounds_are_rejected(self) -> None:
        """Use finite response fractions, means, shifts and ordered intervals."""
        for response_fraction in (0.0, -0.1, 1.01, float("nan"), True):
            with self.subTest(response_fraction=response_fraction):
                with self.assertRaises(ValueError):
                    ObservedArm(2.0, response_fraction)
        with self.assertRaises(ValueError):
            ObservedArm(float("inf"), 0.5)
        treated = ObservedArm(2.0, 0.5)
        control = ObservedArm(1.2, 0.7)
        for shifts in ((2.0, -2.0), (float("nan"), 1.0), (0.0, float("inf"))):
            with self.subTest(shifts=shifts):
                with self.assertRaises(ValueError):
                    contrast_bounds(treated, control, shifts, (-2.0, 0.0))
        for invalid_shift in (float("nan"), float("inf"), True):
            with self.subTest(invalid_shift=invalid_shift):
                with self.assertRaises(ValueError):
                    completed_contrast(treated, control, invalid_shift, 0.0)


if __name__ == "__main__":
    unittest.main()
