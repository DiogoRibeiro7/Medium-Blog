"""Deterministic checks for the Part 7 weak-IV inference extension."""

from __future__ import annotations

import unittest

import numpy as np

from scipy.stats import t as student_t

from weak_iv_inference import (
    invert_anderson_rubin,
    run,
    test_statistics as compute_statistics,
)


class WeakIVInferenceTests(unittest.TestCase):
    """Check the finite-sample statistics and qualitative coverage behavior."""

    def test_statistics_are_finite_under_a_valid_strong_design(self) -> None:
        """A well-identified normal sample gives finite inference statistics."""
        rng: np.random.Generator = np.random.default_rng(17)
        z, u, v, error = rng.standard_normal((4, 1000))
        x = z + u + v
        y = x + u + error
        estimate, se, ar_t = compute_statistics(z, x, y)
        self.assertTrue(np.isfinite([estimate, se, ar_t]).all())
        self.assertGreater(se, 0)

    def test_analytical_inversion_matches_direct_ar_test(self) -> None:
        """Set membership and t-test acceptance agree across randomized cases."""
        rng: np.random.Generator = np.random.default_rng(38)
        cutoff: float = float(student_t.ppf(0.975, 998))
        for strength in (0.0, 0.01, 0.1, 1.0):
            for _ in range(40):
                z, u, v, error = rng.standard_normal((4, 1000))
                x = strength * z + u + v
                y = x + u + error
                _, _, ar_t = compute_statistics(z, x, y)
                conf_set = invert_anderson_rubin(z, x, y, cutoff)
                self.assertEqual(conf_set.contains(1.0), bool(abs(ar_t) <= cutoff))

    def test_weak_design_has_unbounded_sets(self) -> None:
        """Weak relevance exposes disconnected and all-real AR confidence sets."""
        results = {str(row["scenario"]): row for row in run()}
        weak = results["valid_weak"]
        self.assertGreater(float(weak["ar_disconnected_fraction"]), 0.1)
        self.assertGreater(float(weak["ar_all_real_fraction"]), 0.1)
        for row in results.values():
            self.assertAlmostEqual(
                float(row["ar_nonrejection_of_true_effect"]),
                float(row["ar_set_coverage_of_true_effect"]),
            )

    def test_fixed_seed_coverage_and_invalid_design(self) -> None:
        """AR maintains approximate size for valid cases and rejects invalidity."""
        results = {str(row["scenario"]): row for row in run()}
        for scenario in ("valid_strong", "valid_moderate", "valid_weak"):
            self.assertGreater(float(results[scenario]["ar_nonrejection_of_true_effect"]), 0.90)
        self.assertLess(float(results["invalid_strong"]["ar_nonrejection_of_true_effect"]), 0.05)
        self.assertGreater(float(results["valid_weak"]["fraction_first_stage_f_below_10"]), 0.75)


if __name__ == "__main__":
    unittest.main()
