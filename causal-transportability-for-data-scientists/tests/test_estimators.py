"""Randomization, sampling weights, nuisance models, and augmentation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import (  # noqa: E402
    Cohorts, Design, METHODS, augmented_ate, effective_sample_size, estimate_all,
    fit_logit, oracle_effects, outcome_predictions, sample_cohorts,
    trial_target_weights, weighted_ate,
)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = Design(n_trial=1800, n_target=3500)
        cls.cohorts = sample_cohorts(cls.design, np.random.default_rng(12))

    def test_sample_sizes(self) -> None:
        self.assertEqual(self.cohorts.trial_x.size, 1800)
        self.assertEqual(self.cohorts.target_x.size, 3500)

    def test_treatment_randomization_allows_both_arms(self) -> None:
        self.assertGreater(int(self.cohorts.trial_a.sum()), 700)
        self.assertLess(int(self.cohorts.trial_a.sum()), 1100)

    def test_treatment_unrelated_to_trial_x(self) -> None:
        correlation = np.corrcoef(self.cohorts.trial_x, self.cohorts.trial_a)[0, 1]
        self.assertLess(abs(float(correlation)), .075)

    def test_conditional_source_target_shift(self) -> None:
        self.assertLess(float(self.cohorts.trial_x.mean()),
                        float(self.cohorts.target_x.mean()) - .3)

    def test_seeded_generation_is_bitwise_repeatable(self) -> None:
        d = Design(n_trial=80, n_target=160)
        a = sample_cohorts(d, np.random.default_rng(99))
        b = sample_cohorts(d, np.random.default_rng(99))
        np.testing.assert_array_equal(a.trial_y, b.trial_y)
        np.testing.assert_array_equal(a.target_x, b.target_x)

    def test_logit_intercept_only_matches_prevalence(self) -> None:
        x = np.ones((10, 1))
        y = np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])
        from scipy.special import expit
        coefficient = fit_logit(x, y)
        self.assertAlmostEqual(float(expit(coefficient[0])), .3, places=5)

    def test_logit_rejects_nonbinary_labels(self) -> None:
        with self.assertRaises(ValueError):
            fit_logit(np.ones((5, 1)), np.array([0., .5, 1., 0., 1.]))

    def test_logit_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            fit_logit(np.ones((5, 2)), np.ones(4))

    def test_rich_selection_weight_concentrates(self) -> None:
        w = trial_target_weights(self.cohorts, rich=True)
        self.assertLess(effective_sample_size(w), self.design.n_trial * .8)

    def test_selection_weights_favor_high_x_in_trial(self) -> None:
        w = trial_target_weights(self.cohorts, rich=True)
        self.assertGreater(float(np.corrcoef(w, self.cohorts.trial_x)[0, 1]), .6)

    def test_reduced_weights_are_constant_within_z(self) -> None:
        w = trial_target_weights(self.cohorts, rich=False)
        for z in (0., 1.):
            self.assertLess(float(np.std(w[self.cohorts.trial_z == z])), 1e-12)

    def test_rich_outcomes_predict_finite_arrays(self) -> None:
        pred = outcome_predictions(self.cohorts, rich=True)
        self.assertEqual([len(v) for v in pred], [1800, 1800, 3500, 3500])
        self.assertTrue(all(np.isfinite(v).all() for v in pred))

    def test_rich_outcome_model_adjusts_quadratic_response(self) -> None:
        estimates, _ = estimate_all(self.cohorts)
        truth = oracle_effects()["target_ate"]
        self.assertLess(abs(estimates["standardization_rich"] - truth), .13)

    def test_result_names_are_stable(self) -> None:
        estimates, diagnostics = estimate_all(self.cohorts)
        self.assertEqual(tuple(estimates), METHODS)
        self.assertIn("rich_ess", diagnostics)

    def test_all_estimates_are_finite(self) -> None:
        estimates, diagnostics = estimate_all(self.cohorts)
        self.assertTrue(all(np.isfinite(v) for v in estimates.values()))
        self.assertTrue(all(np.isfinite(v) for v in diagnostics.values()))

    def test_uniform_weighted_means_equal_naive_trial(self) -> None:
        c = self.cohorts
        naive = c.trial_y[c.trial_a == 1].mean() - c.trial_y[c.trial_a == 0].mean()
        self.assertAlmostEqual(weighted_ate(c, np.ones(c.trial_x.size)), float(naive))

    def test_weighted_means_invariant_to_multiplier(self) -> None:
        w = trial_target_weights(self.cohorts, True)
        self.assertAlmostEqual(weighted_ate(self.cohorts, w),
                               weighted_ate(self.cohorts, w * 1000), places=10)

    def test_augmented_estimator_invariant_to_weight_scale(self) -> None:
        w = trial_target_weights(self.cohorts, True)
        m = outcome_predictions(self.cohorts, True)
        self.assertAlmostEqual(augmented_ate(self.cohorts, w, m),
                               augmented_ate(self.cohorts, w * 1e-3, m), places=10)

    def test_augmented_rejects_invalid_randomization_probability(self) -> None:
        with self.assertRaises(ValueError):
            augmented_ate(self.cohorts, np.ones(self.design.n_trial),
                          outcome_predictions(self.cohorts, True), 0.)

    def test_augmented_zero_residual_equals_target_standardization(self) -> None:
        c = self.cohorts
        # Synthetic perfect outcome predictions make the correction exactly 0.
        pred = (np.zeros_like(c.trial_y), c.trial_y.copy(),
                np.zeros_like(c.target_x), np.full_like(c.target_x, 1.5))
        # Only treated units use m1; only controls use m0, so fix m0 on controls.
        m0 = pred[0].copy()
        m1 = pred[1].copy()
        m0[c.trial_a == 0] = c.trial_y[c.trial_a == 0]
        self.assertAlmostEqual(augmented_ate(c, np.ones(c.trial_x.size),
                               (m0, m1, pred[2], pred[3])), 1.5)


if __name__ == "__main__":
    unittest.main()
