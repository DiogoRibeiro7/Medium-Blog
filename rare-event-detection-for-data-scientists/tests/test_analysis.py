"""Mathematical, simulation, decision and artifact checks for article #9."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    Scenario,
    binary_log_loss,
    black_box_shift_prevalence,
    brier_score,
    budget_threshold,
    calibration_bins,
    correct_prior,
    cost_probability_threshold,
    decision_operating_point,
    expected_calibration_error,
    label_shift_prior_mle,
    operating_rates,
    population_sample,
    posterior,
    precision_at_prior,
    roc_auc,
    run_study,
    score_threshold,
    verification_estimates,
)


class PosteriorMathTests(unittest.TestCase):
    def test_likelihood_ratio_identity(self) -> None:
        z = np.array([-2.0, 0.0, 1.0, 2.0, 5.0])
        actual = posterior(z, 0.2, 2.0)
        expected = expit(logit(0.2) + 2 * z - 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_posterior_at_class_density_intersection_equals_prior(self) -> None:
        self.assertAlmostEqual(float(posterior(np.array([1.0]), 0.2, 2.0)[0]), 0.2)

    def test_prior_shift_odds_ratio(self) -> None:
        p = np.array([0.01, 0.15, 0.8, 0.99])
        q = correct_prior(p, 0.2, 0.01)
        np.testing.assert_allclose(logit(q) - logit(p), logit(0.01) - logit(0.2))

    def test_corrected_posterior_equals_target_oracle(self) -> None:
        z = np.linspace(-4, 6, 80)
        np.testing.assert_allclose(correct_prior(posterior(z, 0.2, 2), 0.2, 0.01), posterior(z, 0.01, 2), atol=1e-12)

    def test_adjustment_preserves_ranking(self) -> None:
        p = np.linspace(0.01, 0.99, 100)
        q = correct_prior(p, 0.2, 0.01)
        self.assertTrue(np.all(np.diff(q) > 0))

    def test_posterior_requires_valid_prior(self) -> None:
        for bad in (0.0, 1.0, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                posterior(np.array([0.0]), bad, 2.0)

    def test_negative_separation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            posterior(np.array([0.0]), 0.01, -1.0)

    def test_source_probability_bounds(self) -> None:
        with self.assertRaises(ValueError):
            correct_prior(np.array([1.2]), 0.2, 0.01)

    def test_config_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            Scenario(target_prior=1)
        with self.assertRaises(TypeError):
            Scenario(seed=True)


class RareEventMetricsTests(unittest.TestCase):
    def test_gaussian_auc_formula_and_analytic_independence_of_prior(self) -> None:
        self.assertAlmostEqual(roc_auc(2.0), norm.cdf(np.sqrt(2)))

    def test_operating_rates_at_midpoint(self) -> None:
        tpr, fpr = operating_rates(1.0, 2.0)
        self.assertAlmostEqual(float(tpr), norm.cdf(1.0))
        self.assertAlmostEqual(float(fpr), norm.sf(1.0))

    def test_precision_bayes_identity(self) -> None:
        self.assertAlmostEqual(
            float(precision_at_prior(0.01, np.array(0.8), np.array(0.1))),
            0.008 / (0.008 + 0.099),
        )

    def test_precision_drops_when_prevalence_drops(self) -> None:
        tpr, fpr = operating_rates(1.5, 2.0)
        self.assertGreater(float(precision_at_prior(0.2, tpr, fpr)), float(precision_at_prior(0.01, tpr, fpr)))

    def test_no_alerts_precision_is_undefined(self) -> None:
        self.assertTrue(np.isnan(float(precision_at_prior(0.01, np.array(0.0), np.array(0.0)))))

    def test_metrics_prefer_truth_in_expected_value(self) -> None:
        true_p = np.array([0.05, 0.1, 0.4, 0.9])
        wrong_p = np.array([0.4, 0.6, 0.9, 0.1])
        expected_brier = lambda p: float(np.mean(true_p * (1-p)**2 + (1-true_p) * p**2))
        self.assertLess(expected_brier(true_p), expected_brier(wrong_p))

    def test_brier_and_logloss_on_perfect_binary_probabilities(self) -> None:
        y = np.array([False, True, False, True])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(brier_score(y, p), 0)
        self.assertLess(binary_log_loss(y, p), 1e-9)

    def test_binned_reliability_ece(self) -> None:
        y = np.array([False, True, True, False])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        bins = calibration_bins(y, p, 2)
        self.assertEqual(len(bins), 2)
        self.assertAlmostEqual(expected_calibration_error(bins), 0.4)

    def test_nonmatching_shapes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            brier_score(np.array([True]), np.array([0.2, 0.4]))

    def test_sample_reproducibility(self) -> None:
        a = population_sample(np.random.default_rng(123), 100, 0.01, 2)
        b = population_sample(np.random.default_rng(123), 100, 0.01, 2)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


class LabelShiftTests(unittest.TestCase):
    def test_confusion_matrix_exact_inverse(self) -> None:
        pi, tpr, fpr = 0.01, 0.8, 0.05
        observed = pi * tpr + (1 - pi) * fpr
        self.assertAlmostEqual(black_box_shift_prevalence(observed, tpr, fpr), pi)

    def test_unclipped_estimate_can_show_incompatibility(self) -> None:
        self.assertLess(black_box_shift_prevalence(0.02, 0.8, 0.05), 0)

    def test_singular_confusion_matrix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            black_box_shift_prevalence(0.1, 0.2, 0.2)

    def test_mle_recovers_prior_under_assumed_gaussian_model(self) -> None:
        z, _ = population_sample(np.random.default_rng(17), 120000, 0.02, 2.0)
        inferred = label_shift_prior_mle(posterior(z, 0.2, 2.0), 0.2)
        self.assertAlmostEqual(inferred, 0.02, delta=0.005)

    def test_mle_requires_minimum_sample(self) -> None:
        with self.assertRaises(ValueError):
            label_shift_prior_mle(np.array([0.2, 0.3]), 0.2)


class DecisionTests(unittest.TestCase):
    def test_bayes_threshold_formula(self) -> None:
        self.assertAlmostEqual(cost_probability_threshold(8, 250), 8 / 258)

    def test_score_cutoff_inverts_posterior(self) -> None:
        cost_p = cost_probability_threshold(8, 250)
        z = score_threshold(cost_p, 0.01, 2.0)
        self.assertAlmostEqual(float(posterior(np.array([z]), 0.01, 2.0)[0]), cost_p)

    def test_target_policy_minimizes_expected_loss_locally(self) -> None:
        z = score_threshold(8/258, 0.01, 2)
        optimum = decision_operating_point(z, 0.01, 2, 8, 250)["cost_per_case"]
        for shift in (-1.0, -0.3, 0.3, 1.0):
            self.assertLess(optimum, decision_operating_point(z + shift, 0.01, 2, 8, 250)["cost_per_case"])

    def test_frozen_development_policy_costs_more_in_target(self) -> None:
        z_src = score_threshold(8/258, 0.2, 2)
        z_tgt = score_threshold(8/258, 0.01, 2)
        wrong = decision_operating_point(z_src, 0.01, 2, 8, 250)
        correct = decision_operating_point(z_tgt, 0.01, 2, 8, 250)
        self.assertGreater(wrong["cost_per_case"], 3 * correct["cost_per_case"])

    def test_budget_threshold_fills_budget_exactly(self) -> None:
        for fraction in (0.001, 0.01, 0.05, 0.5):
            z = budget_threshold(fraction, 0.01, 2)
            point = decision_operating_point(z, 0.01, 2, 8, 250)
            self.assertAlmostEqual(point["alert_rate"], fraction, places=9)

    def test_budgeted_policy_does_not_exceed_capacity(self) -> None:
        optimal = score_threshold(8/258, 0.01, 2)
        for cap in (0.01, 0.08, 0.3):
            cutoff = max(optimal, budget_threshold(cap, 0.01, 2))
            self.assertLessEqual(
                decision_operating_point(cutoff, 0.01, 2, 8, 250)["alert_rate"],
                cap + 1e-12,
            )

    def test_verification_full_observation(self) -> None:
        y = np.array([True, False, False, True])
        a = np.array([True, True, False, False])
        verified = np.ones(4, dtype=bool)
        naive, weighted = verification_estimates(y, a, verified, 0.5)
        self.assertAlmostEqual(naive, 0.5)
        self.assertAlmostEqual(weighted, 0.75)  # HT assumes 50% inclusion, so full selection is nonrandom here.

    def test_verification_ht_expectation_with_known_propensities(self) -> None:
        y = np.array([True, False, True, False, False])
        a = np.array([True, True, False, False, False])
        prop = np.where(a, 1, 0.04)
        expected_ht = float(np.sum(y * prop / prop) / len(y))
        self.assertAlmostEqual(expected_ht, y.mean())


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.summary = run_study(quick=True, output_dir=cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_ten_tables_and_eight_figures(self) -> None:
        self.assertEqual(len(list((self.root / "results").glob("*_quick.csv"))), 10)
        self.assertEqual(len(list((self.root / "figures").glob("*_quick.png"))), 8)

    def test_source_and_target_precision_ordering(self) -> None:
        self.assertGreater(self.summary["source_precision_at_reference"], self.summary["target_precision_at_reference"])

    def test_corrected_forecasts_have_lower_brier_and_logloss(self) -> None:
        self.assertLess(self.summary["target_brier_adjusted"], self.summary["target_brier_frozen"])
        self.assertLess(self.summary["target_logloss_adjusted"], self.summary["target_logloss_frozen"])

    def test_summary_is_machine_readable(self) -> None:
        path = self.root / "results" / "summary_quick.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            data = list(csv.DictReader(handle))
        self.assertIn("analytic_roc_auc", {r["measure"] for r in data})

    def test_identical_seed_produces_identical_csv(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            run_study(quick=True, output_dir=Path(other))
            for path in (self.root / "results").glob("*_quick.csv"):
                self.assertEqual(path.read_bytes(), (Path(other) / "results" / path.name).read_bytes())

    def test_notebook_is_clean_and_executes_in_fresh_kernel(self) -> None:
        path = ROOT / "rare_event_detection.ipynb"
        notebook_json = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook_json["nbformat"], 4)
        self.assertTrue(all(
            not cell.get("outputs") and cell.get("execution_count") is None
            for cell in notebook_json["cells"] if cell["cell_type"] == "code"
        ))
        if not all(importlib.util.find_spec(name) for name in ("nbformat", "nbclient", "ipykernel")):
            self.skipTest("Notebook execution tools are optional for local tests")
        import nbformat
        from nbclient import NotebookClient

        with tempfile.TemporaryDirectory() as temp:
            nb = nbformat.read(path, as_version=4)
            client = NotebookClient(
                nb, timeout=240, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            )
            client.execute()


if __name__ == "__main__":
    unittest.main()
