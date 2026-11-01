"""Falsification-style forecast checks with no household relocation inference."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import load_panel  # noqa: E402
from commuting_network import load_weights  # noqa: E402
from forecast_robustness import (  # noqa: E402
    FIGURES, RESULTS, PREDICTORS, benchmark_predictions, benchmark_table,
    first_fold_geometry, forecast_gain, mean_absolute_error, municipality_deletions,
    porto_training_ablation, run, year_deletions,
)
from sequential_pressure import METRICS, TEST_YEARS, rolling_predictions  # noqa: E402


class RobustnessForecastTests(unittest.TestCase):
    """Audit naive comparators and deterministic evaluation-only deletions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names, cls.network = load_weights()
        cls.panel = load_panel()
        cls.rows = benchmark_predictions(cls.panel, cls.names, cls.network)

    def test_exact_same_heldout_ridge_forecasts(self) -> None:
        original = rolling_predictions(self.panel, self.names, self.network)
        self.assertEqual(len(original), len(self.rows))
        self.assertEqual(len(self.rows), 102)
        for first, second in zip(original, self.rows):
            self.assertEqual((first.outcome, first.year, first.municipality),
                             (second.outcome, second.year, second.municipality))
            self.assertEqual(first.actual_log_growth_pct, second.actual)
            self.assertEqual(first.baseline_prediction_pct, second.baseline_ridge)
            self.assertEqual(first.network_prediction_pct, second.network_ridge)

    def test_naive_comparators_use_only_prior_year(self) -> None:
        porto = next(row for row in self.rows if row.outcome == "new_contract_rent"
                     and row.year == 2023 and row.municipality == "Porto")
        prior_2021 = self.panel["Porto", 2021].rent_m2
        prior_2022 = self.panel["Porto", 2022].rent_m2
        self.assertEqual(porto.no_change, 0.0)
        self.assertAlmostEqual(porto.persistence, 100 * math.log(prior_2022 / prior_2021))
        self.assertAlmostEqual(porto.amp_prior_mean, sum(r.persistence for r in self.rows
                               if r.outcome == "new_contract_rent" and r.year == 2023) / 17)
        self.assertEqual({r.year for r in self.rows}, set(TEST_YEARS))

    def test_incomplete_or_repeated_geography_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            benchmark_predictions(self.panel, ["Porto", "Porto"], np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            benchmark_table(self.rows[:-1])

    def test_invalid_loss_inputs_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            mean_absolute_error([], "baseline_ridge")
        with self.assertRaisesRegex(ValueError, "registered"):
            mean_absolute_error(self.rows, "unpublished_model")

    def test_five_pooled_comparators_match_fixed_source(self) -> None:
        table = benchmark_table(self.rows)
        self.assertEqual(len(table), 8)
        self.assertEqual(len(PREDICTORS), 5)
        rent = next(r for r in table if r["outcome"] == "new_contract_rent"
                    and r["target_year"] == "pooled")
        sale = next(r for r in table if r["outcome"] == "sold_price"
                    and r["target_year"] == "pooled")
        self.assertAlmostEqual(float(rent["amp_prior_mean_mae_log_points"]), 3.854540)
        self.assertAlmostEqual(float(rent["baseline_ridge_mae_log_points"]), 5.789929)
        self.assertAlmostEqual(float(rent["network_ridge_mae_log_points"]), 8.141219)
        self.assertAlmostEqual(float(sale["amp_prior_mean_mae_log_points"]), 5.597173)
        self.assertAlmostEqual(float(sale["network_ridge_mae_log_points"]), 5.187891)

    def test_year_deletions_account_for_34_remaining_cells(self) -> None:
        values = year_deletions(self.rows)
        self.assertEqual(len(values), 6)
        self.assertTrue(all(row["n_remaining"] == 34 for row in values))
        sales = [float(r["gain_after_evaluation_deletion"]) for r in values
                 if r["outcome"] == "sold_price"]
        self.assertGreater(max(sales), 0)
        self.assertLess(min(sales), 0)
        self.assertTrue(all(float(r["gain_after_evaluation_deletion"]) < 0 for r in values
                            if r["outcome"] == "new_contract_rent"))

    def test_municipality_deletions_preserve_error_arithmetic(self) -> None:
        table = municipality_deletions(self.rows)
        self.assertEqual(len(table), 34)
        for outcome in METRICS:
            subset = [r for r in self.rows if r.outcome == outcome]
            for deletion in (r for r in table if r["outcome"] == outcome):
                name = deletion["excluded_municipality"]
                retained = [r for r in subset if r.municipality != name]
                self.assertEqual(len(retained), deletion["n_remaining"])
                self.assertEqual(len(retained), 48)
                self.assertAlmostEqual(forecast_gain(retained),
                                       float(deletion["gain_after_evaluation_deletion"]), places=6)

    def test_every_municipality_rent_network_contribution_negative(self) -> None:
        rental = [r for r in municipality_deletions(self.rows)
                  if r["outcome"] == "new_contract_rent"]
        self.assertEqual(len(rental), 17)
        self.assertTrue(all(float(r["excluded_municipality_own_gain"]) < 0 for r in rental))

    def test_porto_refit_is_not_evaluation_only_deletion(self) -> None:
        report = porto_training_ablation(self.panel, self.names, self.network, self.rows)
        for outcome in METRICS:
            self.assertEqual(report[outcome]["held_out_municipality_years_after_refit"], 48)
            self.assertNotEqual(report[outcome]["original_models_evaluation_without_porto_gain"],
                                report[outcome]["refitted_16_and_network_without_porto_gain"])
        self.assertLess(report["sold_price"]["refitted_16_and_network_without_porto_gain"], 0)
        self.assertLess(report["new_contract_rent"]["refitted_16_and_network_without_porto_gain"], 0)
        retained = [i for i, name in enumerate(self.names) if name != "Porto"]
        with self.assertRaisesRegex(ValueError, "requires Porto"):
            porto_training_ablation(self.panel, [self.names[i] for i in retained],
                                    self.network[np.ix_(retained, retained)], self.rows)

    def test_single_training_year_common_lag_is_not_identifiable(self) -> None:
        for statistic in first_fold_geometry(self.panel, self.names, self.network).values():
            self.assertEqual(statistic["train_targets"], 17)
            self.assertEqual(statistic["amp_prior_lag_train_sd"], 0)
            self.assertEqual(statistic["baseline_intercept_augmented_rank"], 2)
            self.assertEqual(statistic["network_intercept_augmented_rank"], 3)

    def test_new_outputs_reproducible_and_explicitly_descriptive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report = run(folder)
            self.assertEqual(report["municipalities"], 17)
            self.assertIn("not an exogenous", " ".join(report["limitations"]))
            self.assertIn("not_residential_od", report["network_source"])
            self.assertEqual(report["network_source_provenance"]["source_commit"],
                             "a3bd023b8eab56962cb377d1f07c6279cde67007")
            self.assertEqual(report["predictor_set"], list(PREDICTORS))
            for name in FIGURES:
                self.assertGreater((folder / "figures" / name).stat().st_size, 5000)
            for name in RESULTS:
                self.assertGreater((folder / "results" / name).stat().st_size, 100)
            with (folder / "results" / RESULTS[0]).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 8)
            with (folder / "results" / RESULTS[1]).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 34)
            with (folder / "results" / RESULTS[2]).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)
            stored = json.loads((folder / "results" / RESULTS[3]).read_text(encoding="utf-8"))
            self.assertEqual(stored, report)


if __name__ == "__main__":
    unittest.main()
