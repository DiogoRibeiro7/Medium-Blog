"""Mathematical and evidence-contract checks for the article's experiments."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    Design,
    ar1,
    cointegrated_pair,
    dickey_fuller_t,
    ecm_fit,
    engle_granger_t,
    independent_iid,
    independent_walks,
    ols_levels,
    run_study,
)


def csv_rows(path: Path) -> list[dict[str, str]]:
    """Read one generated evidence table without implicit type coercion."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class DesignTests(unittest.TestCase):
    """Public parameter validation and reproducible defaults."""

    def test_seed_is_fixed(self) -> None:
        self.assertEqual(Design().seed, 20261031)

    def test_default_horizon_is_fixed(self) -> None:
        self.assertEqual(Design().horizon, 250)

    def test_default_adjustment_is_stationary(self) -> None:
        self.assertLess(abs(Design().adjustment_rho), 1)

    def test_seed_requires_integer(self) -> None:
        with self.assertRaises(TypeError):
            Design(seed=1.4)

    def test_boolean_seed_is_invalid(self) -> None:
        with self.assertRaises(TypeError):
            Design(seed=True)

    def test_horizon_cannot_be_too_short(self) -> None:
        with self.assertRaises(ValueError):
            Design(horizon=39)

    def test_horizon_requires_integer(self) -> None:
        with self.assertRaises(TypeError):
            Design(horizon=250.0)

    def test_rho_cannot_equal_one(self) -> None:
        with self.assertRaises(ValueError):
            Design(adjustment_rho=1.)

    def test_stationary_rho_cannot_equal_minus_one(self) -> None:
        with self.assertRaises(ValueError):
            Design(stationary_rho=-1.)

    def test_noise_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            Design(residual_noise=0.)

    def test_alpha_must_be_probability(self) -> None:
        with self.assertRaises(ValueError):
            Design(nominal_alpha=1.)

    def test_nonfinite_parameter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Design(long_run_beta=float("nan"))

    def test_boolean_numeric_parameter_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Design(long_run_beta=True)


class LeastSquaresTests(unittest.TestCase):
    """Check the regression implementation independently of the plots."""

    def setUp(self) -> None:
        self.x = np.arange(100, dtype=np.float64)
        self.y = 2.0 + 3.0 * self.x + np.sin(self.x)
        self.fit = ols_levels(self.y, self.x)

    def test_slope_close_to_known(self) -> None:
        self.assertAlmostEqual(self.fit.slope, 3, delta=.01)

    def test_intercept_close_to_known(self) -> None:
        # The finite sample includes a sinusoidal disturbance with nonzero
        # sample mean; the intercept is not expected to equal two exactly.
        self.assertAlmostEqual(self.fit.intercept, 2, delta=.08)

    def test_fitted_shape(self) -> None:
        self.assertEqual(self.fit.fitted.shape, self.x.shape)

    def test_residual_mean_near_zero(self) -> None:
        self.assertAlmostEqual(float(self.fit.residuals.mean()), 0, places=10)

    def test_residual_orthogonal_to_predictor(self) -> None:
        self.assertAlmostEqual(float(self.x @ self.fit.residuals), 0, places=8)

    def test_r_squared_between_zero_and_one(self) -> None:
        self.assertTrue(0 <= self.fit.r_squared <= 1)

    def test_p_value_between_zero_and_one(self) -> None:
        self.assertTrue(0 <= self.fit.p_naive <= 1)

    def test_standard_error_positive(self) -> None:
        self.assertGreater(self.fit.slope_se_naive, 0)

    def test_t_stat_equals_slope_over_se(self) -> None:
        self.assertAlmostEqual(self.fit.t_naive,
                               self.fit.slope / self.fit.slope_se_naive, places=10)

    def test_durbin_watson_nonnegative(self) -> None:
        self.assertGreaterEqual(self.fit.durbin_watson, 0)

    def test_offset_in_y_preserves_slope(self) -> None:
        self.assertAlmostEqual(ols_levels(self.y + 100, self.x).slope,
                               self.fit.slope, places=10)

    def test_offset_in_x_preserves_slope(self) -> None:
        self.assertAlmostEqual(ols_levels(self.y, self.x + 100).slope,
                               self.fit.slope, places=10)

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ols_levels(self.y[:-1], self.x)

    def test_nonvarying_x_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ols_levels(self.y, np.ones_like(self.x))

    def test_nonvarying_y_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ols_levels(np.ones_like(self.x), self.x)

    def test_nan_series_rejected(self) -> None:
        corrupt = self.x.copy()
        corrupt[10] = np.nan
        with self.assertRaises(ValueError):
            ols_levels(self.y, corrupt)

    def test_two_dimensional_series_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ols_levels(self.y, self.x[:, None])


class StochasticConstructionTests(unittest.TestCase):
    """Verify simulated objects are generated with the stated identities."""

    def test_random_walk_lengths(self) -> None:
        x, y = independent_walks(100, np.random.default_rng(1))
        self.assertEqual(x.shape, y.shape)

    def test_random_walk_seed_reproducibility(self) -> None:
        one = independent_walks(150, np.random.default_rng(19))
        two = independent_walks(150, np.random.default_rng(19))
        np.testing.assert_array_equal(one[0], two[0])
        np.testing.assert_array_equal(one[1], two[1])

    def test_walk_increments_are_finite(self) -> None:
        x, y = independent_walks(150, np.random.default_rng(1))
        self.assertTrue(np.isfinite(np.diff(x)).all())
        self.assertTrue(np.isfinite(np.diff(y)).all())

    def test_independent_innovations_low_sample_correlation(self) -> None:
        x, y = independent_walks(5000, np.random.default_rng(20))
        self.assertLess(abs(float(np.corrcoef(np.diff(x), np.diff(y))[0, 1])), .06)

    def test_independent_iid_lengths(self) -> None:
        x, y = independent_iid(120, np.random.default_rng(7))
        self.assertEqual((x.size, y.size), (120, 120))

    def test_independent_iid_reproducible(self) -> None:
        a = independent_iid(120, np.random.default_rng(7))
        b = independent_iid(120, np.random.default_rng(7))
        np.testing.assert_array_equal(a[0], b[0])

    def test_short_random_walk_rejected(self) -> None:
        with self.assertRaises(ValueError):
            independent_walks(30, np.random.default_rng(5))

    def test_short_iid_rejected(self) -> None:
        with self.assertRaises(ValueError):
            independent_iid(30, np.random.default_rng(5))

    def test_ar1_initial_stationary_variance(self) -> None:
        rng = np.random.default_rng(3)
        initial = np.array([ar1(40, .7, 2., rng)[0] for _ in range(1200)])
        self.assertAlmostEqual(float(initial.var()), 4 / (1 - .7 ** 2), delta=.8)

    def test_ar1_reproducible(self) -> None:
        one = ar1(150, .6, .7, np.random.default_rng(13))
        two = ar1(150, .6, .7, np.random.default_rng(13))
        np.testing.assert_array_equal(one, two)

    def test_ar1_rejects_unstable_rho(self) -> None:
        with self.assertRaises(ValueError):
            ar1(100, 1., .7, np.random.default_rng(1))

    def test_ar1_rejects_negative_sd(self) -> None:
        with self.assertRaises(ValueError):
            ar1(100, .5, -.7, np.random.default_rng(1))

    def test_cointegration_exact_identity(self) -> None:
        p = cointegrated_pair(400, np.random.default_rng(22), Design())
        np.testing.assert_allclose(p.y - 1.6 * p.x, p.latent_residual, atol=1e-14)

    def test_cointegration_beta_constant_without_break(self) -> None:
        p = cointegrated_pair(400, np.random.default_rng(22), Design())
        self.assertTrue(np.all(p.beta_path == 1.6))

    def test_cointegration_true_residual_variance_stationary(self) -> None:
        p = cointegrated_pair(8000, np.random.default_rng(22), Design())
        expected = .7 ** 2 / (1 - .65 ** 2)
        self.assertAlmostEqual(float(np.var(p.latent_residual)), expected, delta=.10)

    def test_cointegration_break_exact(self) -> None:
        p = cointegrated_pair(420, np.random.default_rng(2), Design(), 252, 2.3)
        self.assertTrue(np.all(p.beta_path[:252] == 1.6))
        self.assertTrue(np.all(p.beta_path[252:] == 2.3))
        np.testing.assert_allclose(p.y - p.beta_path * p.x, p.latent_residual)

    def test_break_rejects_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            cointegrated_pair(100, np.random.default_rng(3), Design(), 95, 2.3)

    def test_break_requires_post_beta(self) -> None:
        with self.assertRaises(ValueError):
            cointegrated_pair(100, np.random.default_rng(3), Design(), 50)

    def test_break_rejects_nan_post_beta(self) -> None:
        with self.assertRaises(ValueError):
            cointegrated_pair(100, np.random.default_rng(3), Design(), 50, float("nan"))


class UnitRootAndErrorCorrectionTests(unittest.TestCase):
    """Check test definitions and the known exact ECM identity."""

    def setUp(self) -> None:
        self.design = Design()
        self.x, self.y = independent_walks(250, np.random.default_rng(85))

    def test_df_level_finite(self) -> None:
        self.assertTrue(math.isfinite(dickey_fuller_t(self.x)))

    def test_df_without_intercept_finite(self) -> None:
        self.assertTrue(math.isfinite(dickey_fuller_t(self.x, False)))

    def test_df_deterministic_terms_change_statistic(self) -> None:
        self.assertNotAlmostEqual(dickey_fuller_t(self.x), dickey_fuller_t(self.x, False))

    def test_eg_is_df_on_fitted_residual_without_constant(self) -> None:
        u = ols_levels(self.y, self.x).residuals
        self.assertAlmostEqual(engle_granger_t(self.x, self.y),
                               dickey_fuller_t(u, intercept=False), places=12)

    def test_eg_under_no_cointegration_finite(self) -> None:
        self.assertTrue(math.isfinite(engle_granger_t(self.x, self.y)))

    def test_df_short_series_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dickey_fuller_t(np.arange(30, dtype=float))

    def test_df_nonfinite_series_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dickey_fuller_t(np.array([np.nan] * 60))

    def test_ecm_exact_generating_identity(self) -> None:
        p = cointegrated_pair(800, np.random.default_rng(1), self.design)
        innovation = (np.diff(p.y) - self.design.long_run_beta * np.diff(p.x)
                      - (self.design.adjustment_rho - 1) * p.latent_residual[:-1])
        np.testing.assert_allclose(innovation,
                                   p.latent_residual[1:] - self.design.adjustment_rho * p.latent_residual[:-1],
                                   atol=1e-12)

    def test_ecm_estimated_adjustment_near_truth(self) -> None:
        p = cointegrated_pair(4000, np.random.default_rng(1), self.design)
        estimated = ecm_fit(p.x, p.y)
        self.assertAlmostEqual(estimated["adjustment_lambda"], -.35, delta=.05)

    def test_ecm_estimated_short_run_slope_near_truth(self) -> None:
        p = cointegrated_pair(4000, np.random.default_rng(1), self.design)
        estimated = ecm_fit(p.x, p.y)
        self.assertAlmostEqual(estimated["delta_x_slope"], 1.6, delta=.07)

    def test_ecm_estimated_long_run_slope_near_truth(self) -> None:
        p = cointegrated_pair(4000, np.random.default_rng(1), self.design)
        self.assertAlmostEqual(ecm_fit(p.x, p.y)["longrun_slope"], 1.6, delta=.02)

    def test_ecm_residual_sd_positive(self) -> None:
        p = cointegrated_pair(500, np.random.default_rng(1), self.design)
        self.assertGreater(ecm_fit(p.x, p.y)["residual_sd"], 0)

    def test_ecm_mismatched_lengths_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ecm_fit(self.x, self.y[:-1])


class GeneratedEvidenceTests(unittest.TestCase):
    """Verify the quick full-output contract once per test class."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="spurious-regression-test-")
        cls.root = Path(cls.tmp.name)
        cls.summary = run_study(cls.root, quick=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_exactly_eight_figures(self) -> None:
        self.assertEqual(len(list((self.root / "figures").glob("*.png"))), 8)

    def test_exactly_fifteen_csv_tables(self) -> None:
        self.assertEqual(len(list((self.root / "results").glob("*.csv"))), 15)

    def test_every_output_has_quick_suffix(self) -> None:
        paths = list((self.root / "figures").iterdir()) + list((self.root / "results").iterdir())
        self.assertTrue(all(path.stem.endswith("_quick") for path in paths))

    def test_all_pngs_have_nonzero_size(self) -> None:
        for path in (self.root / "figures").glob("*.png"):
            with self.subTest(path=path.name):
                self.assertGreater(path.stat().st_size, 1000)

    def test_all_csvs_have_data_rows(self) -> None:
        for path in (self.root / "results").glob("*.csv"):
            with self.subTest(path=path.name):
                self.assertGreater(len(csv_rows(path)), 0)

    def test_null_replicate_count(self) -> None:
        rows = csv_rows(self.root / "results/null_regression_replicates_quick.csv")
        self.assertEqual(len(rows), 90 * 3)

    def test_null_categories_distinct(self) -> None:
        rows = csv_rows(self.root / "results/null_regression_replicates_quick.csv")
        self.assertEqual({r["case"] for r in rows},
                         {"random_walk_levels", "random_walk_differences", "independent_iid"})

    def test_null_summary_count(self) -> None:
        rows = csv_rows(self.root / "results/null_regression_summary_quick.csv")
        self.assertEqual(len(rows), 3)

    def test_unrelated_walks_naive_rejection_is_inflated(self) -> None:
        self.assertGreater(self.summary["rw_levels_false_rejection"], .6)

    def test_differences_not_inflated(self) -> None:
        self.assertLess(self.summary["rw_differences_false_rejection"], .2)

    def test_iid_not_inflated(self) -> None:
        self.assertLess(self.summary["iid_false_rejection"], .2)

    def test_df_eg_critical_values_not_same(self) -> None:
        self.assertNotAlmostEqual(self.summary["df_null_critical_5pct"],
                                  self.summary["eg_null_critical_5pct"], delta=.15)

    def test_eg_cutoff_more_negative_than_df_in_fixed_design(self) -> None:
        self.assertLess(self.summary["eg_null_critical_5pct"],
                        self.summary["df_null_critical_5pct"])

    def test_cointegrated_ecm_adjustment_is_negative(self) -> None:
        self.assertLess(self.summary["cointegrated_ecm_lambda"], 0)

    def test_cointegrated_ecm_short_run_slope_positive(self) -> None:
        self.assertGreater(self.summary["cointegrated_ecm_beta"], 0)

    def test_genuine_cointegration_rejects_no_relation(self) -> None:
        self.assertLess(self.summary["cointegration_residual_t"],
                        self.summary["eg_null_critical_5pct"])

    def test_illustration_selected_independently(self) -> None:
        self.assertGreaterEqual(self.summary["illustration_attempt"], 1)

    def test_illustrative_trajectory_length(self) -> None:
        rows = csv_rows(self.root / "results/illustrative_null_path_quick.csv")
        self.assertEqual(len(rows), Design().horizon)

    def test_unit_root_null_replicate_count(self) -> None:
        rows = csv_rows(self.root / "results/unit_root_null_replicates_quick.csv")
        self.assertEqual(len(rows), 150)

    def test_calibrations_are_two_distinct_rows(self) -> None:
        rows = csv_rows(self.root / "results/calibrated_critical_values_quick.csv")
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["test"], rows[1]["test"])

    def test_cointegrated_path_has_horizon_rows(self) -> None:
        rows = csv_rows(self.root / "results/cointegrated_path_quick.csv")
        self.assertEqual(len(rows), Design().horizon)

    def test_scaling_has_eight_groups(self) -> None:
        rows = csv_rows(self.root / "results/slope_scaling_summary_quick.csv")
        self.assertEqual(len(rows), 4 * 2)

    def test_scaling_replicates_count(self) -> None:
        rows = csv_rows(self.root / "results/slope_scaling_replicates_quick.csv")
        self.assertEqual(len(rows), 4 * 60 * 2)

    def test_break_path_length(self) -> None:
        rows = csv_rows(self.root / "results/broken_relation_path_quick.csv")
        self.assertEqual(len(rows), 420)

    def test_break_rolling_windows_length(self) -> None:
        rows = csv_rows(self.root / "results/rolling_break_estimates_quick.csv")
        self.assertEqual(len(rows), 420 - 95 + 1)

    def test_break_replicates_count(self) -> None:
        rows = csv_rows(self.root / "results/break_replicates_quick.csv")
        self.assertEqual(len(rows), 65)

    def test_break_is_exactly_point_seven(self) -> None:
        self.assertAlmostEqual(self.summary["structural_break_true_beta_change"], .7)

    def test_summary_csv_matches_returned_dictionary(self) -> None:
        rows = csv_rows(self.root / "results/study_summary_quick.csv")
        for row in rows:
            with self.subTest(metric=row["metric"]):
                self.assertAlmostEqual(float(row["value"]), self.summary[row["metric"]])

    def test_all_numeric_null_statistics_finite(self) -> None:
        rows = csv_rows(self.root / "results/null_regression_replicates_quick.csv")
        for row in rows:
            for key in ("slope", "r_squared", "t_naive", "p_naive", "durbin_watson"):
                with self.subTest(case=row["case"], replicate=row["replicate"], metric=key):
                    self.assertTrue(math.isfinite(float(row[key])))

    def test_invalid_output_root_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            run_study("not-a-path", quick=True)


if __name__ == "__main__":
    unittest.main()
