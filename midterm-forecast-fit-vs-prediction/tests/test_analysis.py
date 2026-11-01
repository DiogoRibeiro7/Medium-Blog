"""Source integrity, mathematical identities and reproducible forecast tests."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis import (
    Election, alternative_gallup, bootstrap_future, compute_evidence,
    expanding_backtest, expanding_reduced_backtest, fit_model, influence,
    leave_one_out, prediction_interval, predictor, read_elections,
    read_scenario, reduced_predict, save_tables,
)
from figures import FIGURE_FILES, generate_figures


class DataTests(unittest.TestCase):
    def test_canonical_18_midterms_are_historical_and_sorted(self) -> None:
        rows = read_elections()
        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[0].year, 1950)
        self.assertEqual(rows[-1].year, 2022)
        self.assertNotIn(1986, {r.year for r in rows})
        self.assertNotIn(2026, {r.year for r in rows})

    def test_transcription_spot_checks(self) -> None:
        rows = {r.year: r for r in read_elections()}
        self.assertEqual((rows[1950].seats, rows[1950].change), (263, -28))
        self.assertEqual((rows[1994].seats, rows[1994].change), (258, -54))
        self.assertEqual((rows[2002].seats, rows[2002].change), (220, 8))
        self.assertEqual((rows[2010].seats, rows[2010].change), (257, -63))
        self.assertEqual(rows[1998].dem_margin, 4.2)
        self.assertEqual(rows[2006].dem_margin, 11.5)
        self.assertEqual(rows[2022].dem_margin, -2.5)

    def test_presidential_party_sign_is_consistent(self) -> None:
        rows = {r.year: r for r in read_elections()}
        self.assertEqual(rows[1958].party_margin, -14.0)  # Eisenhower (R)
        self.assertEqual(rows[2010].party_margin, -9.4)  # Obama (D)
        self.assertEqual(rows[2018].party_margin, -7.3)  # Trump (R)
        self.assertEqual(rows[1998].party_margin, 4.2)  # Clinton (D)

    def test_scenario_is_not_labeled_with_a_result(self) -> None:
        scenario = read_scenario()
        self.assertEqual((scenario.year, scenario.party, scenario.seats), (2026, "R", 220))
        self.assertEqual(scenario.party_margin, -8)

    def test_rejects_malformed_data(self) -> None:
        original = read_elections()[0]
        with self.assertRaises(ValueError):
            fit_model([Election(original.year, "X", original.seats, original.change, 2, "bad")] * 5)
        with self.assertRaises(ValueError):
            predictor(math.inf, 220)
        with self.assertRaises(TypeError):
            predictor(True, 220)
        with self.assertRaises(ValueError):
            predictor(5.0, 500)

    def test_alternative_gallup_changes_only_two_elections(self) -> None:
        rows = read_elections()
        revised = alternative_gallup(rows)
        changed = [original.year for original, alt in zip(rows, revised) if original != alt]
        self.assertEqual(changed, [1962, 1998])
        self.assertEqual(revised[3].dem_margin, 9.0)
        self.assertEqual(revised[11].dem_margin, 0.0)


class EstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = read_elections()
        self.model = fit_model(self.rows)
        self.scenario = read_scenario()

    def test_ols_normal_equations(self) -> None:
        np.testing.assert_allclose(self.model.design.T @ self.model.residuals,
                                   np.zeros(3), atol=1e-9)
        self.assertEqual(self.model.degrees_of_freedom, 15)

    def test_identification_of_r_squared_and_fitted_forecast(self) -> None:
        self.assertAlmostEqual(self.model.r_squared, 0.893447684694887, places=10)
        self.assertAlmostEqual(self.model.predict(-8, 220), -32.3228880263, places=6)
        self.assertGreater(self.model.r_squared, 0.88)

    def test_result_matches_independent_linear_algebra_formula(self) -> None:
        independent = np.linalg.inv(self.model.design.T @ self.model.design) @ self.model.design.T @ self.model.outcomes
        np.testing.assert_allclose(self.model.coefficients, independent, atol=1e-10)

    def test_add_one_ballot_point_has_a_linear_effect(self) -> None:
        before = self.model.predict(-8, 220)
        after = self.model.predict(-9, 220)
        self.assertAlmostEqual(after - before, -self.model.coefficients[1])

    def test_more_seats_defended_increases_expected_losses(self) -> None:
        self.assertLess(self.model.predict(-8, 225), self.model.predict(-8, 220))

    def test_prediction_interval_wider_than_mean_interval(self) -> None:
        mean_lo, mean_hi = prediction_interval(self.model, -8, 220, 0.95,
                                                future_observation=False)
        pred_lo, pred_hi = prediction_interval(self.model, -8, 220, 0.95)
        self.assertGreater(pred_hi - pred_lo, mean_hi - mean_lo)
        self.assertLess(pred_lo, self.model.predict(-8, 220))
        self.assertGreater(pred_hi, self.model.predict(-8, 220))

    def test_intervals_have_correct_nesting_and_parameter_validation(self) -> None:
        lo90, hi90 = prediction_interval(self.model, -8, 220, 0.90)
        lo95, hi95 = prediction_interval(self.model, -8, 220, 0.95)
        self.assertLess(lo95, lo90)
        self.assertGreater(hi95, hi90)
        with self.assertRaises(ValueError):
            prediction_interval(self.model, -8, 220, 1.0)

    def test_influence_diagnostics_are_bounded(self) -> None:
        leverage, standardized, cook = influence(self.model)
        self.assertEqual(len(leverage), 18)
        self.assertAlmostEqual(float(sum(leverage)), 3.0, places=9)
        self.assertTrue(np.all(leverage < 1))
        self.assertTrue(np.all(cook >= 0))
        self.assertTrue(np.isfinite(standardized).all())


class ForecastValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = read_elections()

    def test_loocv_excludes_each_holdout(self) -> None:
        folds = leave_one_out(self.rows)
        self.assertEqual(len(folds), 18)
        self.assertTrue(all(fold.train_count == 17 for fold in folds))
        for row, fold in zip(self.rows, folds):
            self.assertEqual(row.year, fold.year)
            self.assertEqual(row.change, fold.actual)

    def test_loocv_matches_analytic_ols_press_formula(self) -> None:
        model = fit_model(self.rows)
        leverage, _, _ = influence(model)
        folds = leave_one_out(self.rows)
        analytic_error = model.residuals / (1 - leverage)
        empirical_error = np.array([fold.actual - fold.predicted for fold in folds])
        np.testing.assert_allclose(empirical_error, analytic_error, atol=1e-9)

    def test_forward_test_has_no_future_outcomes(self) -> None:
        rolls = expanding_backtest(self.rows, minimum_train=8)
        self.assertEqual(len(rolls), 10)
        self.assertEqual([r.train_count for r in rolls], list(range(8, 18)))
        self.assertEqual(rolls[0].year, 1982)
        self.assertEqual(rolls[-1].year, 2022)
        for i, roll in enumerate(rolls, start=8):
            expected = fit_model(self.rows[:i]).predict(self.rows[i].party_margin, self.rows[i].seats)
            self.assertAlmostEqual(roll.predicted, expected)

    def test_reduced_models_use_identical_forward_folds(self) -> None:
        full = expanding_backtest(self.rows)
        reduced = expanding_reduced_backtest(self.rows)
        self.assertEqual(
            [(r.year, r.actual, r.train_count) for r in full],
            [(r.year, r.actual, r.train_count) for r in reduced],
        )
        self.assertEqual(len(reduced), 10)
        training = self.rows[:8]
        first = self.rows[8]
        independent = np.linalg.lstsq(
            np.array([[1.0, r.party_margin] for r in training]),
            np.array([r.change for r in training], dtype=float), rcond=None,
        )[0]
        self.assertAlmostEqual(reduced[0].generic_only,
                               float(np.array([1.0, first.party_margin]) @ independent))

    def test_reduced_features_are_separate_and_validated(self) -> None:
        scenario = read_scenario()
        ballot = reduced_predict(self.rows, "generic", margin=scenario.party_margin,
                                 seats=scenario.seats)
        seats = reduced_predict(self.rows, "seats", margin=scenario.party_margin,
                                seats=scenario.seats)
        self.assertAlmostEqual(ballot, -31.5560783367, places=6)
        self.assertAlmostEqual(seats, -23.5344316141, places=6)
        self.assertAlmostEqual(
            ballot, reduced_predict(self.rows, "generic", margin=-8, seats=230),
        )
        self.assertAlmostEqual(
            seats, reduced_predict(self.rows, "seats", margin=-5, seats=220),
        )
        with self.assertRaises(ValueError):
            reduced_predict(self.rows, "unsupported", margin=-8, seats=220)

    def test_reduced_backtest_cannot_see_its_held_out_outcome(self) -> None:
        before = expanding_reduced_backtest(self.rows)
        updated = tuple(
            replace(row, change=-48) if row.year == 1994 else row
            for row in self.rows
        )
        after = expanding_reduced_backtest(updated)
        # A change to 1994's outcome must not alter its own predictions.
        original = next(row for row in before if row.year == 1994)
        altered = next(row for row in after if row.year == 1994)
        self.assertEqual((original.generic_only, original.seats_only),
                         (altered.generic_only, altered.seats_only))

    def test_reduced_backtest_rejects_invalid_training_window(self) -> None:
        with self.assertRaises(ValueError):
            expanding_reduced_backtest(self.rows, minimum_train=18)
        with self.assertRaises(TypeError):
            expanding_reduced_backtest(self.rows, minimum_train=True)

    def test_rolling_window_bounds(self) -> None:
        with self.assertRaises(ValueError):
            expanding_backtest(self.rows, minimum_train=18)
        with self.assertRaises(ValueError):
            expanding_backtest(self.rows, minimum_train=4)

    def test_2026_bootstrap_is_seeded_and_does_not_use_future_outcomes(self) -> None:
        scenario = read_scenario()
        a = bootstrap_future(self.rows, scenario, draws=300, seed=42)
        b = bootstrap_future(self.rows, scenario, draws=300, seed=42)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(float(np.std(a)), 0)

    def test_alternative_values_do_not_destroy_historical_fit(self) -> None:
        primary = fit_model(self.rows)
        alternate = fit_model(alternative_gallup(self.rows))
        self.assertGreater(alternate.r_squared, 0.88)
        self.assertLess(abs(primary.predict(-8, 220) - alternate.predict(-8, 220)), 2)


class ReproducibilityTests(unittest.TestCase):
    def test_summary_metrics_and_tables(self) -> None:
        evidence = compute_evidence(draws=200)
        self.assertEqual(evidence.metrics["observations"], 18)
        self.assertEqual(evidence.metrics["forward_test_count"], 10)
        self.assertLess(evidence.metrics["forward_rmse"],
                        evidence.metrics["forward_mean_baseline_rmse"])
        self.assertAlmostEqual(evidence.metrics["forward_generic_only_rmse"],
                               20.094, delta=0.01)
        self.assertAlmostEqual(evidence.metrics["forward_seats_only_rmse"],
                               23.003, delta=0.01)
        self.assertEqual(evidence.metrics["forward_full_beats_generic_absolute_count"], 9)
        self.assertAlmostEqual(evidence.metrics["alternative_forward_rmse"],
                               8.97356, delta=0.01)
        self.assertAlmostEqual(evidence.metrics["alternative_forward_generic_only_rmse"],
                               20.62549, delta=0.01)
        self.assertEqual(evidence.metrics["alternative_forward_full_beats_generic_absolute_count"], 9)
        with tempfile.TemporaryDirectory() as temporary:
            dest = Path(temporary) / "results"
            save_tables(evidence, dest)
            summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["bootstrap_draws"], 200)
            with (dest / "historical_fits.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 18)
            with (dest / "scenario_grid.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 45)
            with (dest / "forward_model_comparison.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 10)
                self.assertEqual([int(row["year"]) for row in rows],
                                 [r.year for r in evidence.forward])
            with (dest / "source_sensitivity.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 2)
                self.assertIn("forward_generic_only_rmse", rows[0])
                self.assertGreater(float(rows[1]["forward_generic_only_rmse"]),
                                   float(rows[1]["forward_rmse"]))

    def test_thirteen_publication_figures_regenerate(self) -> None:
        evidence = compute_evidence(draws=200)
        with tempfile.TemporaryDirectory() as temporary:
            paths = generate_figures(evidence, Path(temporary) / "figures")
            self.assertEqual(len(paths), 13)
            self.assertEqual([path.name for path in paths], list(FIGURE_FILES))
            self.assertTrue(all(path.stat().st_size > 10_000 for path in paths))


if __name__ == "__main__":
    unittest.main()
