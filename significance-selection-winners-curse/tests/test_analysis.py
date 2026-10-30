"""Numerical, analytical and reproducibility tests for significance article #11."""

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
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    Design,
    benjamini_hochberg,
    critical_value,
    one_multiple_testing_family,
    posterior_mean_normal,
    rejection_summary,
    run_study,
    significant_interval_coverage,
    two_sided_power,
    type_m_error,
    type_s_error,
)


class DesignTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        self.assertEqual(Design().seed, 20261030)

    def test_reject_boolean_seed(self) -> None:
        with self.assertRaises(TypeError):
            Design(seed=True)

    def test_reject_invalid_alpha(self) -> None:
        for alpha in (0.0, 1.0, float("nan")):
            with self.assertRaises((TypeError, ValueError)):
                Design(alpha=alpha)

    def test_reject_nonpositive_effect(self) -> None:
        with self.assertRaises(ValueError):
            Design(true_effect=0.0)

    def test_reject_nonpositive_standard_error(self) -> None:
        with self.assertRaises(ValueError):
            Design(study_se=-1.0)

    def test_reject_nan_prior(self) -> None:
        with self.assertRaises(TypeError):
            Design(shrinkage_prior_sd=float("nan"))


class SelectionTheoryTests(unittest.TestCase):
    def test_two_sided_critical_value(self) -> None:
        self.assertAlmostEqual(critical_value(.05), norm.ppf(.975))

    def test_bad_critical_probability(self) -> None:
        for alpha in (-.1, 0.0, 1.0):
            with self.assertRaises(ValueError):
                critical_value(alpha)

    def test_power_is_alpha_under_null(self) -> None:
        self.assertAlmostEqual(two_sided_power(0, .05), .05, places=12)

    def test_power_grows_with_positive_signal(self) -> None:
        self.assertLess(two_sided_power(.5), two_sided_power(1.0))
        self.assertLess(two_sided_power(1.0), two_sided_power(3.0))

    def test_power_is_even_function(self) -> None:
        for delta in (.1, .5, 1, 3):
            self.assertAlmostEqual(two_sided_power(-delta), two_sided_power(delta))

    def test_type_s_needs_nonzero_effect(self) -> None:
        with self.assertRaises(ValueError):
            type_s_error(0.0)

    def test_type_m_needs_nonzero_effect(self) -> None:
        with self.assertRaises(ValueError):
            type_m_error(0.0)

    def test_type_s_decreases_with_signal(self) -> None:
        self.assertGreater(type_s_error(.25), type_s_error(1))
        self.assertGreater(type_s_error(1), type_s_error(2))

    def test_type_m_exaggeration_is_large_at_low_power(self) -> None:
        self.assertGreater(type_m_error(.5), 4)
        self.assertAlmostEqual(type_m_error(1), 2.490928, delta=1e-5)

    def test_type_m_converges_to_one(self) -> None:
        self.assertAlmostEqual(type_m_error(10), 1.0, delta=1e-4)

    def test_type_s_m_sign_symmetric(self) -> None:
        self.assertAlmostEqual(type_s_error(-1), type_s_error(1))
        self.assertAlmostEqual(type_m_error(-1), type_m_error(1))

    def test_power_type_s_m_agree_with_monte_carlo(self) -> None:
        z = np.random.default_rng(10).normal(1.0, 1.0, 150000)
        selected = np.abs(z) > critical_value()
        self.assertAlmostEqual(selected.mean(), two_sided_power(1), delta=.004)
        self.assertAlmostEqual((z[selected] < 0).mean(), type_s_error(1), delta=.003)
        self.assertAlmostEqual(np.abs(z[selected]).mean(), type_m_error(1), delta=.025)

    def test_selected_null_interval_never_covers_zero(self) -> None:
        self.assertEqual(significant_interval_coverage(0.0), 0.0)

    def test_selected_coverage_is_lower_at_delta_one(self) -> None:
        self.assertAlmostEqual(significant_interval_coverage(1.0), .843961, delta=1e-5)
        self.assertLess(significant_interval_coverage(1.0), .95)

    def test_selected_coverage_sign_symmetric(self) -> None:
        self.assertAlmostEqual(
            significant_interval_coverage(-1.0), significant_interval_coverage(1.0)
        )

    def test_selected_coverage_approaches_nominal_for_large_signal(self) -> None:
        self.assertAlmostEqual(significant_interval_coverage(20), .95, places=10)

    def test_selected_coverage_matches_simulation(self) -> None:
        delta = 1.0
        innovations = np.random.default_rng(11).standard_normal(220000)
        z = delta + innovations
        selected = np.abs(z) > critical_value()
        interval_contains_true = np.abs(z - delta) <= critical_value()
        empirical = float(np.mean(interval_contains_true[selected]))
        self.assertAlmostEqual(empirical, significant_interval_coverage(delta), delta=.008)


class MultiplicityTests(unittest.TestCase):
    def test_bh_no_rejections(self) -> None:
        np.testing.assert_array_equal(
            benjamini_hochberg(np.array([.3, .5, .8])), [False, False, False]
        )

    def test_bh_step_up_case(self) -> None:
        np.testing.assert_array_equal(
            benjamini_hochberg(np.array([.8, .001, .02, .6]), .05),
            [False, True, True, False],
        )

    def test_bh_all_rejected_when_all_p_zero(self) -> None:
        self.assertTrue(np.all(benjamini_hochberg(np.zeros(100))))

    def test_bh_rejects_invalid_p(self) -> None:
        with self.assertRaises(ValueError):
            benjamini_hochberg(np.array([-.1, .5]))

    def test_bh_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            benjamini_hochberg(np.array([]))

    def test_bh_never_fewer_than_bonferroni(self) -> None:
        p = np.random.default_rng(12).random(1000)
        self.assertTrue(np.all(benjamini_hochberg(p)[p < .05/1000]))

    def test_rejection_summary_handles_no_discoveries(self) -> None:
        out = rejection_summary(np.array([False, False]), np.array([True, False]))
        self.assertEqual(out["false_discovery_proportion"], 0)
        self.assertEqual(out["alternative_detection_fraction"], 0)

    def test_rejection_summary_counts_false_and_true(self) -> None:
        out = rejection_summary(np.array([True, True, False]), np.array([False, True, True]))
        self.assertEqual(out["false_discoveries"], 1.0)
        self.assertEqual(out["true_discoveries"], 1.0)
        self.assertEqual(out["false_discovery_proportion"], 0.5)
        self.assertEqual(out["any_false_discovery"], 1.0)

    def test_rejection_summary_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            rejection_summary(np.array([True]), np.array([False, True]))

    def test_one_family_is_reproducible(self) -> None:
        a = one_multiple_testing_family(np.random.default_rng(13))
        b = one_multiple_testing_family(np.random.default_rng(13))
        self.assertEqual(a, b)

    def test_one_family_preserves_method_order_constraints(self) -> None:
        out = one_multiple_testing_family(np.random.default_rng(14), m=2000, pi0=.90)
        self.assertGreaterEqual(out["uncorrected"]["discoveries"],
                                out["benjamini_hochberg"]["discoveries"])
        self.assertGreaterEqual(out["benjamini_hochberg"]["discoveries"],
                                out["bonferroni"]["discoveries"])


class ShrinkageTests(unittest.TestCase):
    def test_normal_posterior_mean_formula(self) -> None:
        y = np.array([-1.0, 0.0, 2.0])
        out = posterior_mean_normal(y, .3, .2)
        np.testing.assert_allclose(out, (.2**2/(.3**2+.2**2))*y)

    def test_bad_prior_scale_rejected(self) -> None:
        with self.assertRaises(ValueError):
            posterior_mean_normal(np.ones(3), .3, 0.0)

    def test_bad_data_rejected(self) -> None:
        with self.assertRaises(ValueError):
            posterior_mean_normal(np.array([1.0, float("nan")]), .3, .2)

    def test_correct_prior_shrinkage_reduces_selected_mse(self) -> None:
        rng = np.random.default_rng(15)
        theta = rng.normal(0, .2, 80000)
        y = theta + .3*rng.standard_normal(len(theta))
        selected = np.abs(y/.3) > critical_value()
        posterior = posterior_mean_normal(y[selected], .3, .2)
        self.assertLess(np.mean((posterior - theta[selected])**2),
                        np.mean((y[selected]-theta[selected])**2) / 3)


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.summary = run_study(quick=True, output_dir=cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def test_eight_figure_and_ten_result_files(self) -> None:
        self.assertEqual(len(list((self.root/"figures").glob("*_quick.png"))), 8)
        self.assertEqual(len(list((self.root/"results").glob("*_quick.csv"))), 10)

    def test_summary_matches_analytic_type_m(self) -> None:
        self.assertAlmostEqual(self.summary["type_m_at_delta_1"], type_m_error(1))
        self.assertAlmostEqual(self.summary["type_s_at_delta_1"], type_s_error(1))

    def test_observed_power_near_analytic_power(self) -> None:
        self.assertAlmostEqual(self.summary["selected_fraction_simulation"],
                               two_sided_power(1), delta=.02)

    def test_independent_replication_mean_near_truth(self) -> None:
        self.assertAlmostEqual(self.summary["replication_signed_mean"], .3, delta=.025)

    def test_bh_fdr_smaller_than_naive(self) -> None:
        self.assertLess(self.summary["bh_fdr_pi0_0_95"],
                        self.summary["uncorrected_fdr_pi0_0_95"])

    def test_shrinkage_improves_selected_mse_under_true_prior(self) -> None:
        self.assertLess(self.summary["selected_oracle_shrinkage_mse"],
                        self.summary["selected_naive_mse"])

    def test_summary_csv_contains_key_metrics(self) -> None:
        with (self.root/"results"/"summary_quick.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            names = {row["measure"] for row in csv.DictReader(handle)}
        self.assertIn("type_m_at_delta_1", names)
        self.assertIn("selected_interval_coverage_delta_1", names)

    def test_deterministic_results_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_study(quick=True, output_dir=Path(temp))
            for path in (self.root/"results").glob("*_quick.csv"):
                self.assertEqual(path.read_bytes(), (Path(temp)/"results"/path.name).read_bytes())

    def test_notebook_clean_and_fresh_kernel(self) -> None:
        path = ROOT/"significance_selection.ipynb"
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["nbformat"], 4)
        self.assertTrue(all(
            not c.get("outputs") and c.get("execution_count") is None
            for c in document["cells"] if c["cell_type"] == "code"
        ))
        if not all(importlib.util.find_spec(name) for name in
                   ("nbformat", "nbclient", "ipykernel")):
            self.skipTest("Notebook execution dependencies not available locally")
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(path, as_version=4)
        with tempfile.TemporaryDirectory():
            NotebookClient(
                notebook, timeout=240, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()


if __name__ == "__main__":
    unittest.main()
