"""Mathematical and reproducibility checks for dependent-data article #10."""

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
    Scenario,
    asymptotic_long_run_variance,
    bartlett_hac_long_run_variance,
    basic_bootstrap_interval,
    changed_mean_ar1,
    contains,
    coverage_and_width,
    default_bandwidth,
    effective_sample_size,
    exact_mean_variance,
    fit_ar_predictor,
    forecast_mse,
    hac_mean_se,
    iid_bootstrap_means,
    iid_mean_se,
    moving_block_means,
    normal_mean_interval,
    one_forecast_validation,
    run_study,
    sample_acf,
    stationary_ar1,
    validate_phi,
)


class StationaryARTests(unittest.TestCase):
    def test_initial_and_later_marginal_variance_are_one(self) -> None:
        paths = stationary_ar1(np.random.default_rng(40), 90, 0.8, 18000)
        self.assertAlmostEqual(float(paths[:, 0].var()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(paths[:, 89].var()), 1.0, delta=0.05)

    def test_lag_one_correlation_matches_phi(self) -> None:
        paths = stationary_ar1(np.random.default_rng(41), 2, -0.5, 25000)
        self.assertAlmostEqual(float(np.corrcoef(paths.T)[0, 1]), -0.5, delta=0.025)

    def test_random_seed_is_reproducible(self) -> None:
        a = stationary_ar1(np.random.default_rng(123), 100, 0.7)
        b = stationary_ar1(np.random.default_rng(123), 100, 0.7)
        np.testing.assert_array_equal(a, b)

    def test_reject_unit_root_and_explosive_models(self) -> None:
        for phi in (-1.0, 1.0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                validate_phi(phi)

    def test_reject_bad_sample_size(self) -> None:
        with self.assertRaises(ValueError):
            stationary_ar1(np.random.default_rng(1), 1, 0.5)

    def test_config_rejects_noninteger_seed(self) -> None:
        with self.assertRaises(TypeError):
            Scenario(seed=True)

    def test_autocorrelation_lag_zero_is_one(self) -> None:
        x = stationary_ar1(np.random.default_rng(2), 200, 0.8)[0]
        acf = sample_acf(x, 10)
        self.assertAlmostEqual(float(acf[0]), 1)
        self.assertEqual(len(acf), 11)

    def test_constant_acf_undefined(self) -> None:
        with self.assertRaises(ValueError):
            sample_acf(np.ones(100), 3)


class EffectiveSampleTests(unittest.TestCase):
    def test_iid_exact_mean_variance(self) -> None:
        for n in (2, 10, 240):
            self.assertAlmostEqual(exact_mean_variance(0, n), 1 / n)
            self.assertAlmostEqual(effective_sample_size(0, n), n)

    def test_finite_variance_matches_ar1_two_sample_formula(self) -> None:
        self.assertAlmostEqual(exact_mean_variance(0.8, 2), (1 + 0.8) / 2)

    def test_strong_positive_dependence_reduces_effective_n(self) -> None:
        n_eff = effective_sample_size(0.8, 240)
        self.assertAlmostEqual(n_eff, 27.1698113208, delta=1e-5)
        self.assertLess(n_eff, 30)

    def test_negative_autocorrelation_increases_precision(self) -> None:
        self.assertGreater(effective_sample_size(-0.5, 240), 240)

    def test_asymptotic_long_run_variance_formula(self) -> None:
        self.assertAlmostEqual(asymptotic_long_run_variance(0.8), 9.0)
        self.assertAlmostEqual(asymptotic_long_run_variance(-0.5), 1/3)

    def test_finite_and_asymptotic_approach_each_other(self) -> None:
        self.assertAlmostEqual(
            effective_sample_size(0.5, 100000, finite=True)
            / effective_sample_size(0.5, 100000, finite=False),
            1.0, delta=1e-4,
        )

    def test_analytic_mean_variance_matches_simulation(self) -> None:
        means = stationary_ar1(np.random.default_rng(4), 180, 0.8, 3000).mean(axis=1)
        self.assertAlmostEqual(
            float(np.var(means)), exact_mean_variance(0.8, 180), delta=0.006
        )


class UncertaintyTests(unittest.TestCase):
    def test_white_noise_hac_lag_zero_nearly_equals_iid(self) -> None:
        x = np.array([0.0, 2.0, -1.0, 1.0, 3.0, -2.0])
        expected = np.var(x, ddof=0)
        self.assertAlmostEqual(bartlett_hac_long_run_variance(x, 0), expected)

    def test_hac_estimate_nonnegative(self) -> None:
        x = stationary_ar1(np.random.default_rng(5), 240, -0.5)[0]
        for lag in (0, 5, 13, 50):
            self.assertGreaterEqual(bartlett_hac_long_run_variance(x, lag), 0)

    def test_hac_se_is_finite(self) -> None:
        x = stationary_ar1(np.random.default_rng(6), 240, 0.7)[0]
        self.assertTrue(np.isfinite(hac_mean_se(x, 13)))
        self.assertGreater(hac_mean_se(x, 13), 0)

    def test_bad_hac_bandwidth_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bartlett_hac_long_run_variance(np.arange(10.0), 10)

    def test_bandwidth_rule_is_valid(self) -> None:
        self.assertEqual(default_bandwidth(240), 13)

    def test_normal_interval_width(self) -> None:
        x = np.array([-1.0, 1.0])
        low, high = normal_mean_interval(x, 0.2)
        self.assertAlmostEqual(high - low, 2 * norm.ppf(0.975) * 0.2)
        self.assertTrue(contains((low, high), 0.0))

    def test_invalid_interval_rejected(self) -> None:
        with self.assertRaises(ValueError):
            contains((3.0, 2.0))

    def test_basic_bootstrap_interval_centered(self) -> None:
        x = np.array([-1.0, 1.0])
        low, high = basic_bootstrap_interval(x, np.array([-1, 0, 1], dtype=float))
        self.assertAlmostEqual(low, -0.95)
        self.assertAlmostEqual(high, 0.95)

    def test_block_length_one_matches_iid_bootstrap(self) -> None:
        x = np.arange(10.0)
        a = iid_bootstrap_means(np.random.default_rng(7), x, 60)
        b = moving_block_means(np.random.default_rng(7), x, 1, 60)
        np.testing.assert_array_equal(a, b)

    def test_one_full_block_yields_original_mean(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
        means = moving_block_means(np.random.default_rng(8), x, len(x), 50)
        np.testing.assert_array_equal(means, np.full(50, x.mean()))

    def test_block_length_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            moving_block_means(np.random.default_rng(8), np.arange(10.0), 11, 10)

    def test_iid_bootstrap_se_matches_iid_standard_error(self) -> None:
        x = np.random.default_rng(10).standard_normal(180)
        means = iid_bootstrap_means(np.random.default_rng(11), x, 3500)
        self.assertAlmostEqual(float(means.std()), iid_mean_se(x), delta=0.015)

    def test_aggregated_coverage_and_width(self) -> None:
        cover, width = coverage_and_width(
            [{"test_covered": 1, "test_width": 2.0}, {"test_covered": 0, "test_width": 4.0}], "test"
        )
        self.assertEqual(cover, 0.5)
        self.assertEqual(width, 3.0)


class ForecastValidationTests(unittest.TestCase):
    def test_mean_shift_generator_has_future_higher_mean(self) -> None:
        x = changed_mean_ar1(np.random.default_rng(12), 2200, 0.5, 1000, 2.5)
        self.assertGreater(x[1200:].mean() - x[:800].mean(), 1.8)

    def test_least_squares_recovers_exact_linear_relation(self) -> None:
        x = np.linspace(-3, 3, 100)
        intercept, slope = fit_ar_predictor(x, 2.0 + 0.7*x)
        self.assertAlmostEqual(intercept, 2.0)
        self.assertAlmostEqual(slope, 0.7)
        self.assertAlmostEqual(forecast_mse(x, 2.0+0.7*x, intercept, slope), 0, delta=1e-28)

    def test_forward_validation_is_worse_after_known_shift_on_average(self) -> None:
        outcomes = [
            one_forecast_validation(np.random.default_rng(100 + i))[0]
            for i in range(20)
        ]
        self.assertGreater(
            np.mean([r["genuine_future_after_shift_rmse"] for r in outcomes]),
            np.mean([r["random_mixed_holdout_rmse"] for r in outcomes]) + 0.2,
        )

    def test_forecast_reproducibility(self) -> None:
        a, xa = one_forecast_validation(np.random.default_rng(15))
        b, xb = one_forecast_validation(np.random.default_rng(15))
        self.assertEqual(a, b)
        np.testing.assert_array_equal(xa, xb)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.summary = run_study(quick=True, output_dir=cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_eight_figures_and_ten_tables(self) -> None:
        self.assertEqual(len(list((self.root/"figures").glob("*_quick.png"))), 8)
        self.assertEqual(len(list((self.root/"results").glob("*_quick.csv"))), 10)

    def test_summary_values_match_mathematical_definitions(self) -> None:
        self.assertAlmostEqual(self.summary["long_run_variance_phi_0_8"], 9.0)
        self.assertAlmostEqual(self.summary["effective_n_phi_0_8"], effective_sample_size(0.8, 240))

    def test_iid_undercoverage_when_positive_dependence_is_strong(self) -> None:
        self.assertLess(self.summary["naive_coverage_phi_0_8"], 0.7)
        self.assertGreater(self.summary["hac_coverage_phi_0_8"], self.summary["naive_coverage_phi_0_8"])

    def test_summary_csv_has_expected_columns(self) -> None:
        with (self.root/"results"/"summary_quick.csv").open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        self.assertIn("effective_n_phi_0_8", {r["measure"] for r in records})

    def test_repeated_run_produces_identical_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_study(quick=True, output_dir=Path(temp))
            for path in (self.root/"results").glob("*_quick.csv"):
                self.assertEqual(
                    path.read_bytes(), (Path(temp)/"results"/path.name).read_bytes()
                )

    def test_notebook_is_clean_and_executes_in_fresh_kernel(self) -> None:
        path = ROOT/"dependent_data_uncertainty.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertTrue(all(
            not cell.get("outputs") and cell.get("execution_count") is None
            for cell in nb["cells"] if cell["cell_type"] == "code"
        ))
        if not all(importlib.util.find_spec(name) for name in ("nbformat", "nbclient", "ipykernel")):
            self.skipTest("Optional notebook execution tools not installed locally")
        import nbformat
        from nbclient import NotebookClient

        with tempfile.TemporaryDirectory():
            notebook = nbformat.read(path, as_version=4)
            NotebookClient(
                notebook, timeout=240, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()


if __name__ == "__main__":
    unittest.main()
