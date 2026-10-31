"""Independent numerical checks for fixed-parameter model comparisons."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis import alternative_gallup, fit_model, read_elections, read_scenario
from model_extensions import (
    MODEL_NAMES, RIDGE_ALPHA, RECENCY_HALF_LIFE, STUDENT_DF,
    _standardized_design, _student_t_loss_gradient, compute_model_evidence,
    fit_recency_wls, fit_ridge, fit_student_t, forward_model_folds,
    save_model_evidence,
)


class ModelEstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = read_elections()
        cls.scenario = read_scenario()

    def test_zero_ridge_penalty_reproduces_ols(self) -> None:
        for train in (self.rows[:8], self.rows[:12], self.rows):
            actual = fit_ridge(train, alpha=0.0)
            expected = fit_model(train)
            np.testing.assert_allclose(actual.coefficients, expected.coefficients,
                                       rtol=1e-11, atol=1e-10)

    def test_ridge_standardization_is_fold_local(self) -> None:
        train = self.rows[:8]
        z, y, means, scales = _standardized_design(train)
        penalized = z.T @ z + np.diag([0.0, RIDGE_ALPHA, RIDGE_ALPHA])
        estimated = np.linalg.solve(penalized, z.T @ y)
        test = self.rows[8]
        manual = float(np.r_[1.0, ([test.party_margin, test.seats] - means) / scales] @ estimated)
        self.assertAlmostEqual(fit_ridge(train).predict(test.party_margin, test.seats), manual)

    def test_student_t_analytic_gradient_matches_finite_difference(self) -> None:
        z, y, _, _ = _standardized_design(self.rows[:8])
        theta = np.array([-25.0, 5.0, -4.0, math.log(7.0)])
        _, gradient = _student_t_loss_gradient(theta, z, y, STUDENT_DF)
        numerical = []
        for j in range(len(theta)):
            offset = np.zeros_like(theta)
            offset[j] = 1e-6
            upper = _student_t_loss_gradient(theta + offset, z, y, STUDENT_DF)[0]
            lower = _student_t_loss_gradient(theta - offset, z, y, STUDENT_DF)[0]
            numerical.append((upper - lower) / 2e-6)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-6)

    def test_student_t_fit_converges_deterministically(self) -> None:
        a = fit_student_t(self.rows)
        b = fit_student_t(self.rows)
        np.testing.assert_array_equal(a.coefficients, b.coefficients)
        self.assertGreater(a.residual_scale, 0)
        self.assertAlmostEqual(a.predict(-8, 220), -33.75755, delta=0.02)

    def test_wls_long_half_life_converges_to_ols(self) -> None:
        old = fit_model(self.rows)
        recent = fit_recency_wls(self.rows, 2026, half_life_years=1e12)
        np.testing.assert_allclose(recent.coefficients, old.coefficients,
                                   rtol=1e-9, atol=1e-8)
        self.assertLess(fit_recency_wls(self.rows, 2026).effective_sample_size,
                        len(self.rows))
        self.assertGreater(fit_recency_wls(self.rows, 2026).effective_sample_size, 8)

    def test_invalid_parameters_fail_before_fit(self) -> None:
        with self.assertRaises(ValueError):
            fit_ridge(self.rows, alpha=-1.0)
        with self.assertRaises(ValueError):
            fit_student_t(self.rows, df=2.0)
        with self.assertRaises(ValueError):
            fit_recency_wls(self.rows, 2026, half_life_years=0)
        with self.assertRaises(ValueError):
            fit_recency_wls(self.rows, 2022, half_life_years=RECENCY_HALF_LIFE)


class ModelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = read_elections()

    def test_all_models_share_exact_forward_targets(self) -> None:
        folds = forward_model_folds(self.rows)
        self.assertEqual(len(folds), 10)
        self.assertEqual([row.year for row in folds],
                         [1982, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022])
        self.assertEqual([row.train_count for row in folds], list(range(8, 18)))
        self.assertTrue(all(math.isfinite(getattr(row, name))
                            for row in folds for name in MODEL_NAMES))

    def test_held_out_outcome_does_not_influence_its_prediction(self) -> None:
        original = forward_model_folds(self.rows)
        changed = tuple(replace(row, change=-48) if row.year == 1994 else row
                        for row in self.rows)
        mutated = forward_model_folds(changed)
        first = next(row for row in original if row.year == 1994)
        second = next(row for row in mutated if row.year == 1994)
        for name in MODEL_NAMES:
            self.assertAlmostEqual(getattr(first, name), getattr(second, name), places=9)

    def test_source_sensitivity_keeps_targets_and_fixed_parameters(self) -> None:
        first = forward_model_folds(self.rows)
        second = forward_model_folds(alternative_gallup(self.rows))
        self.assertEqual([(f.year, f.actual, f.train_count) for f in first],
                         [(f.year, f.actual, f.train_count) for f in second])

    def test_scores_and_candidate_ranking_are_reported_not_selected(self) -> None:
        evidence = compute_model_evidence()
        main = evidence.summary["primary"]
        alternate = evidence.summary["alternative_gallup"]
        self.assertEqual(evidence.summary["forward_count"], 10)
        self.assertEqual((evidence.summary["student_t_df_fixed"],
                          evidence.summary["ridge_alpha_fixed"],
                          evidence.summary["recency_half_life_years_fixed"]),
                         (4.0, 2.0, 24.0))
        self.assertAlmostEqual(main["ols"]["rmse"], 8.69580, places=4)
        self.assertAlmostEqual(main["student_t"]["rmse"], 8.48140, delta=0.02)
        self.assertAlmostEqual(main["ridge"]["rmse"], 12.17911, delta=0.02)
        self.assertAlmostEqual(main["recency_wls"]["rmse"], 9.12774, delta=0.02)
        self.assertGreater(alternate["student_t"]["rmse"], alternate["ols"]["rmse"])
        self.assertEqual(len(evidence.coefficient_path), 11)
        self.assertEqual(evidence.coefficient_path[-1].train_end_year, 2022)

    def test_saved_evidence_has_one_row_per_election_and_path_origin(self) -> None:
        evidence = compute_model_evidence()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_model_evidence(evidence, directory)
            for name, expected_count in (
                ("alternative_model_forward.csv", 10),
                ("alternative_model_forward_gallup.csv", 10),
                ("alternative_model_path.csv", 11),
            ):
                with (directory / name).open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), expected_count)
            self.assertTrue((directory / "alternative_models_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
