"""Temporal price thresholds and out-of-year predictive checks, not migration inference."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import load_panel  # noqa: E402
from commuting_network import load_weights  # noqa: E402
from sequential_pressure import (  # noqa: E402
    FIGURES,
    RESULTS,
    SEED,
    annual_log_growth,
    first_passage,
    lagged_design,
    network_shuffle_benchmark,
    ridge_forecast,
    rolling_predictions,
    run,
    summarize_forecasts,
    threshold_table,
)


class ObservedThresholdTests(unittest.TestCase):
    """Thresholds are computed from medians and missing years are never filled."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names, cls.weights = load_weights()
        cls.panel = load_panel()

    def test_fixed_crossings_and_explicit_censoring(self) -> None:
        rows = {r["municipality"]: r for r in threshold_table(self.panel, self.names)}
        self.assertEqual(len(rows), 17)
        self.assertEqual({r["new_contract_rent_plus_25_first_year"] for r in rows.values()}, {2023})
        self.assertEqual(rows["Porto"]["sold_price_plus_50_first_year"], 2023)
        self.assertEqual(rows["Porto"]["new_contract_rent_plus_50_first_year"], 2025)
        self.assertEqual(rows["Porto"]["sold_price_plus_100_first_year"], "")
        self.assertEqual(sum(r["sold_price_plus_100_first_year"] == "" for r in rows.values()), 7)

    def test_specific_first_passage_and_invalid_threshold(self) -> None:
        self.assertEqual(first_passage(self.panel, municipality="Vila Nova de Gaia",
                                       field="price_m2", base_year=2019, threshold_pct=50), 2022)
        self.assertIsNone(first_passage(self.panel, municipality="Porto",
                                        field="price_m2", base_year=2019, threshold_pct=100))
        for threshold in (0, -10, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                first_passage(self.panel, municipality="Porto", field="price_m2",
                              base_year=2019, threshold_pct=threshold)

    def test_missing_baselines_not_imputed(self) -> None:
        self.assertIsNone(self.panel["Porto", 2019].rent_m2)
        with self.assertRaisesRegex(ValueError, "Missing"):
            first_passage(self.panel, municipality="Porto", field="rent_m2",
                          base_year=2019, threshold_pct=25)
        rent_changes = annual_log_growth(self.panel, self.names, field="rent_m2", base_year=2020)
        self.assertEqual(set(rent_changes), {2021, 2022, 2023, 2024, 2025})

    def test_annual_log_growth_is_exact_median_ratio(self) -> None:
        changes = annual_log_growth(self.panel, self.names, field="price_m2", base_year=2019)
        porto_index = self.names.index("Porto")
        observed = self.panel["Porto", 2020].price_m2 / self.panel["Porto", 2019].price_m2
        self.assertAlmostEqual(changes[2020][porto_index], 100 * math.log(observed))
        self.assertEqual(set(changes), {2020, 2021, 2022, 2023, 2024, 2025})

    def test_bad_field_and_nonpositive_median_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            annual_log_growth(self.panel, self.names, field="population", base_year=2019)
        changed = dict(self.panel)
        changed["Porto", 2021] = replace(changed["Porto", 2021], rent_m2=0)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            annual_log_growth(changed, self.names, field="rent_m2", base_year=2020)


class RollingForecastTests(unittest.TestCase):
    """Known-lag, fixed-complexity prediction on a short panel."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names, cls.weights = load_weights()
        cls.panel = load_panel()

    def test_network_feature_excludes_own_municipality_and_mean(self) -> None:
        weights = np.array([[0., 2., 4.], [2., 0., 3.], [4., 3., 0.]])
        growth = {2022: np.array([10., 20., 30.])}
        x = lagged_design(growth, 2023, weights)
        self.assertAlmostEqual(x[0, 0], 10)
        self.assertAlmostEqual(x[0, 1], 20)
        self.assertAlmostEqual(x[0, 2], (2 * 20 + 4 * 30) / 6 - 20)
        shifted = {2022: np.array([1000., 20., 30.])}
        self.assertAlmostEqual(lagged_design(shifted, 2023, weights)[0, 2],
                               (2 * 20 + 4 * 30) / 6 - shifted[2022].mean())

    def test_invalid_network_and_missing_lag_rejected(self) -> None:
        weights = np.array([[0., 2.], [2., 0.]])
        growth = {2022: np.array([1., 2.])}
        with self.assertRaisesRegex(ValueError, "prior"):
            lagged_design(growth, 2022, weights)
        for invalid in (np.eye(2), np.array([[0., -2.], [-2., 0.]]),
                        np.array([[0., 2.], [1., 0.]])):
            with self.assertRaisesRegex(ValueError, "network|Network"):
                lagged_design(growth, 2023, invalid)

    def test_ridge_training_only_standardization_and_penalty_checks(self) -> None:
        train_x = np.array([[2., 7.], [3., 7.], [4., 7.], [5., 7.]])
        train_y = np.array([2., 3., 4., 5.])
        prediction = ridge_forecast(train_x, train_y, np.array([[6., 7.]]))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreater(prediction[0], train_y.mean())
        self.assertAlmostEqual(prediction[0], ridge_forecast(train_x, train_y,
                                                             np.array([[6., 700.]]))[0])
        with self.assertRaisesRegex(ValueError, "penalty"):
            ridge_forecast(train_x, train_y, train_x, penalty=0)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            ridge_forecast(train_x, train_y, np.ones((1, 3)))

    def test_only_three_heldout_years_and_two_outcomes(self) -> None:
        predicted = rolling_predictions(self.panel, self.names, self.weights)
        self.assertEqual(len(predicted), 102)
        self.assertEqual({row.year for row in predicted}, {2023, 2024, 2025})
        self.assertEqual({row.outcome for row in predicted}, {"sold_price", "new_contract_rent"})
        summary = summarize_forecasts(predicted)
        self.assertEqual(len(summary), 8)
        self.assertTrue(all(row["n_municipality_years"] in {17, 51} for row in summary))

    def test_holdout_2025_outcomes_do_not_leak_back_into_2023(self) -> None:
        original = [r for r in rolling_predictions(self.panel, self.names, self.weights) if r.year == 2023]
        changed = dict(self.panel)
        for municipality in self.names:
            row = changed[municipality, 2025]
            changed[municipality, 2025] = replace(row, price_m2=row.price_m2 * 7,
                                                   rent_m2=row.rent_m2 * 5)
        perturbed = [r for r in rolling_predictions(changed, self.names, self.weights) if r.year == 2023]
        self.assertEqual(original, perturbed)

    def test_2024_outcome_does_not_inform_2024_prediction(self) -> None:
        original = [r for r in rolling_predictions(self.panel, self.names, self.weights) if r.year == 2024]
        changed = dict(self.panel)
        for municipality in self.names:
            row = changed[municipality, 2024]
            changed[municipality, 2024] = replace(row, price_m2=row.price_m2 * 2,
                                                   rent_m2=row.rent_m2 * 2)
        perturbed = [r for r in rolling_predictions(changed, self.names, self.weights) if r.year == 2024]
        for before, after in zip(original, perturbed):
            self.assertAlmostEqual(before.baseline_prediction_pct, after.baseline_prediction_pct)
            self.assertAlmostEqual(before.network_prediction_pct, after.network_prediction_pct)
            self.assertNotAlmostEqual(before.actual_log_growth_pct, after.actual_log_growth_pct)

    def test_network_label_shuffle_reproducible_but_not_pvalue(self) -> None:
        a = network_shuffle_benchmark(self.panel, self.names, self.weights,
                                      permutations=99, seed=SEED)
        b = network_shuffle_benchmark(self.panel, self.names, self.weights,
                                      permutations=99, seed=SEED)
        self.assertEqual(a, b)
        for stat in a.values():
            self.assertEqual(stat["shuffles"], 99)
            self.assertGreaterEqual(stat["fraction_shuffled_at_least_observed"], .01)
            self.assertLessEqual(stat["fraction_shuffled_at_least_observed"], 1.0)
        with self.assertRaisesRegex(ValueError, "99"):
            network_shuffle_benchmark(self.panel, self.names, self.weights, permutations=10)

    def test_published_artifacts_and_limitations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run(output, shuffles=99)
            self.assertEqual(report["municipality_count"], 17)
            self.assertEqual(report["rolling_test_years"], [2023, 2024, 2025])
            self.assertIn("not a residential migration OD", str(report["limits"]))
            self.assertEqual(set(report["network_label_shuffle"]), {"sold_price", "new_contract_rent"})
            for figure in FIGURES:
                self.assertGreater((output / "figures" / figure).stat().st_size, 5000)
            for result in RESULTS:
                self.assertGreater((output / "results" / result).stat().st_size, 100)
            with (output / "results" / RESULTS[1]).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 102)
            with (output / "results" / RESULTS[0]).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 17)
            saved = json.loads((output / "results" / RESULTS[3]).read_text(encoding="utf-8"))
            self.assertEqual(saved["source_panel_sha256"], report["source_panel_sha256"])


if __name__ == "__main__":
    unittest.main()
