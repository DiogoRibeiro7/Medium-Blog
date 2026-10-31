"""Analytic identification, population moments, and input contracts."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import (  # noqa: E402
    Design, effective_sample_size, feature_matrix, oracle_effects,
    participation_matrix, selection_probability, treatment_effect,
)


class DesignTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        self.assertEqual(Design().n_trial, 550)

    def test_rejects_boolean_seed(self) -> None:
        with self.assertRaises(TypeError):
            Design(seed=True)

    def test_rejects_float_sample_size(self) -> None:
        with self.assertRaises(TypeError):
            Design(n_trial=500.5)  # type: ignore[arg-type]

    def test_rejects_small_trial(self) -> None:
        with self.assertRaises(ValueError):
            Design(n_trial=35)

    def test_rejects_bad_treatment_probability(self) -> None:
        with self.assertRaises(ValueError):
            Design(p_treatment=1.0)

    def test_rejects_bad_binary_probability(self) -> None:
        with self.assertRaises(ValueError):
            Design(p_binary=0.0)

    def test_rejects_bad_hidden_probability(self) -> None:
        with self.assertRaises(ValueError):
            Design(p_hidden=-0.1)

    def test_rejects_negative_selection_slope(self) -> None:
        with self.assertRaises(ValueError):
            Design(selection_slope=-0.1)

    def test_rejects_nonpositive_noise(self) -> None:
        with self.assertRaises(ValueError):
            Design(noise_sd=0.0)

    def test_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            Design(hidden_effect=math.nan)


class FormulaTests(unittest.TestCase):
    def test_base_treatment_effect_at_origin(self) -> None:
        self.assertAlmostEqual(float(treatment_effect(np.array([0.]), np.array([0.]))[0]), .4)

    def test_interaction_changes_effect_slope(self) -> None:
        z0 = treatment_effect(np.array([0., 1.]), np.array([0., 0.]))
        z1 = treatment_effect(np.array([0., 1.]), np.array([1., 1.]))
        self.assertAlmostEqual(float(z0[1] - z0[0]), .5)
        self.assertAlmostEqual(float(z1[1] - z1[0]), .85)

    def test_hidden_effect_additive(self) -> None:
        effect = treatment_effect(np.array([0.]), np.array([0.]), np.array([1.]), 1.2)
        self.assertAlmostEqual(float(effect[0]), 1.6)

    def test_effect_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            treatment_effect(np.array([1.]), np.array([0., 1.]))

    def test_selection_probability_bounds(self) -> None:
        p = selection_probability(np.linspace(-10, 10, 100), np.zeros(100), .8)
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_selection_slope_is_negative_in_log_odds(self) -> None:
        p = selection_probability(np.array([-1., 1.]), np.zeros(2), .8)
        self.assertGreater(float(p[0]), float(p[1]))

    def test_oracle_probabilities_partition_population(self) -> None:
        o = oracle_effects()
        self.assertAlmostEqual(o["trial_participation_probability"] +
                               o["target_nonparticipation_probability"], 1., places=12)

    def test_target_differs_from_trial(self) -> None:
        o = oracle_effects()
        self.assertGreater(o["target_ate"] - o["trial_ate"], .4)

    def test_oracle_values_reproducible_across_quadrature_resolutions(self) -> None:
        a, b = oracle_effects(n_nodes=64), oracle_effects(n_nodes=100)
        self.assertAlmostEqual(a["target_ate"], b["target_ate"], places=9)

    def test_rejects_insufficient_quadrature(self) -> None:
        with self.assertRaises(ValueError):
            oracle_effects(n_nodes=5)

    def test_no_latent_selection_means_no_oracle_transport_bias(self) -> None:
        o = oracle_effects(hidden_selection=0., hidden_effect=1.2)
        self.assertAlmostEqual(o["oracle_transport_bias"], 0., places=11)

    def test_no_latent_effect_means_no_oracle_transport_bias(self) -> None:
        o = oracle_effects(hidden_selection=2., hidden_effect=0.)
        self.assertAlmostEqual(o["oracle_transport_bias"], 0., places=11)

    def test_unobserved_modifier_and_selection_create_bias(self) -> None:
        o = oracle_effects(hidden_selection=2., hidden_effect=1.2)
        self.assertGreater(abs(o["oracle_transport_bias"]), .1)

    def test_latent_bias_reverses_with_effect_sign(self) -> None:
        a = oracle_effects(hidden_selection=2., hidden_effect=1.)
        b = oracle_effects(hidden_selection=2., hidden_effect=-1.)
        self.assertAlmostEqual(a["oracle_transport_bias"], -b["oracle_transport_bias"], places=11)

    def test_oracle_ate_no_dependence_on_hidden_if_no_latent_selection(self) -> None:
        a = oracle_effects(hidden_effect=0.)
        b = oracle_effects(hidden_effect=1.)
        self.assertAlmostEqual(b["target_ate"] - a["target_ate"], .5, places=10)

    def test_rich_features_have_five_columns(self) -> None:
        d = feature_matrix(np.arange(10, dtype=float), np.zeros(10), True)
        self.assertEqual(d.shape, (10, 5))

    def test_reduced_features_have_two_columns(self) -> None:
        d = feature_matrix(np.arange(10, dtype=float), np.zeros(10), False)
        self.assertEqual(d.shape, (10, 2))

    def test_participation_features_have_three_columns(self) -> None:
        d = participation_matrix(np.arange(10, dtype=float), np.zeros(10), True)
        self.assertEqual(d.shape, (10, 3))

    def test_features_reject_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            feature_matrix(np.zeros((2, 2)), np.zeros((2, 2)), True)


class EffectiveInformationTests(unittest.TestCase):
    def test_uniform_weight_ess_equals_sample_size(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.ones(100)), 100.)

    def test_weight_ess_drops_under_concentration(self) -> None:
        self.assertLess(effective_sample_size(np.array([20., 1., 1., 1.])), 2.)

    def test_weight_ess_invariant_to_rescaling(self) -> None:
        weights = np.array([1., 2., 4., 8.])
        self.assertAlmostEqual(effective_sample_size(weights),
                               effective_sample_size(weights * 25.))

    def test_weight_ess_cannot_exceed_sample_size(self) -> None:
        self.assertLessEqual(effective_sample_size(np.arange(1., 10.)), 9.)

    def test_weight_ess_rejects_negative_weight(self) -> None:
        with self.assertRaises(ValueError):
            effective_sample_size(np.array([1., -1.]))

    def test_weight_ess_rejects_zero_weight_total(self) -> None:
        with self.assertRaises(ValueError):
            effective_sample_size(np.zeros(5))

    def test_weight_ess_rejects_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            effective_sample_size(np.array([1., float("inf")]))


if __name__ == "__main__":
    unittest.main()
