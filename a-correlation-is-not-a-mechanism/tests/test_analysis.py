"""Numerical and design-contract tests for the first causal studies."""

from __future__ import annotations

import unittest

import numpy as np

from analysis import (
    STUDY_IDS,
    collider_theoretical_slope,
    confounding_theoretical_bias,
    simpson_exact_marginal_difference,
    simulate,
    summarize,
)


class CausalStudyTests(unittest.TestCase):
    """Check signs, analytical limits, determinism and summary estimates."""

    def test_analytic_identities(self) -> None:
        self.assertAlmostEqual(confounding_theoretical_bias(), 1.8 / 2.44)
        self.assertAlmostEqual(collider_theoretical_slope(), -0.5)
        self.assertAlmostEqual(simpson_exact_marginal_difference(), -0.39)

    def test_study_seeds_are_deterministic(self) -> None:
        for study in STUDY_IDS:
            self.assertEqual(simulate(study, 3), simulate(study, 3))

    def test_estimators_have_expected_bias_directions(self) -> None:
        estimates = {study: {row["method"]: row for row in summarize(simulate(study, 300))} for study in STUDY_IDS}
        self.assertAlmostEqual(float(estimates["confounding"]["naive"]["bias"]), confounding_theoretical_bias(), delta=0.08)
        self.assertAlmostEqual(float(estimates["confounding"]["adjusted"]["bias"]), 0.0, delta=0.05)
        self.assertAlmostEqual(float(estimates["simpson"]["marginal"]["mean_estimate"]), simpson_exact_marginal_difference(), delta=0.04)
        self.assertAlmostEqual(float(estimates["simpson"]["standardized"]["bias"]), 0.0, delta=0.03)
        self.assertAlmostEqual(float(estimates["collider"]["conditioned"]["mean_estimate"]), collider_theoretical_slope(), delta=0.05)
        self.assertAlmostEqual(float(estimates["collider"]["marginal"]["bias"]), 0.0, delta=0.05)

    def test_replicate_keys_unique(self) -> None:
        for study in STUDY_IDS:
            rows = simulate(study, 4)
            keys = [(r.study, r.scenario, r.method, r.replicate) for r in rows]
            self.assertEqual(len(keys), len(set(keys)))

    def test_errors_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate("unknown")
        with self.assertRaises(ValueError):
            simulate("simpson", 1)
        with self.assertRaises(ValueError):
            collider_theoretical_slope(-1)


if __name__ == "__main__":
    unittest.main()
