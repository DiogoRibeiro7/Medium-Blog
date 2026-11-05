"""Numerical invariants, experimental contracts, archival checks, and notebooks."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analysis
import notebook_builder

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    nbformat = None
    NotebookClient = None


ROOT = Path(__file__).resolve().parents[1]

# Explicit, independent specifications: no deriving expected conditions from
# the implementation being tested. Cubic previously flagged that mistake.
CONTRACTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "iid": (("120", "400", "900"),
            ("cv_brier", "fresh_test_brier", "training_brier")),
    "search": (("1", "5", "20", "100", "500"),
               ("selected_validation_brier", "independent_test_brier",
                "selection_optimism")),
    "nested": (("tuned_model",),
               ("selected_inner_cv_brier", "nested_outer_cv_brier", "fresh_test_brier")),
    "preprocess": (("30", "150", "600"),
                   ("leaky_cv_brier", "fold_safe_cv_brier", "fresh_test_brier")),
    "temporal": (("0.0", "1.5", "3.0"),
                 ("random_history_cv_brier", "forward_history_cv_brier",
                  "future_period_brier")),
    "lookahead": (("1", "3", "7", "14"),
                  ("causal_feature_brier", "lookahead_feature_brier",
                   "optimism_from_lookahead")),
    "groups": (("0.0", "0.8", "1.8"),
               ("random_row_cv_brier", "unseen_site_cv_brier",
                "site_generalization_gap")),
    "uncertainty": (("5fold",),
                    ("cv_brier", "external_brier", "naive_fold_se",
                     "absolute_cv_external_gap", "naive_interval_covers_external")),
    "adaptive": (("0", "20", "60", "120"),
                 ("public_holdout_accuracy", "fresh_independent_accuracy",
                  "recovered_label_fraction", "public_queries_used")),
}


def expected_keys(study: str) -> set[tuple[str, str]]:
    """Return exact expected condition-metric pairs for a study."""

    conditions, metrics = CONTRACTS[study]
    return {(condition, metric) for condition in conditions for metric in metrics}


def strict_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    """Reject malformed headers/rows and every nonfinite required numeric cell."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fields):
            raise ValueError("wrong or duplicated CSV header")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    for row in rows:
        if set(row) != set(fields) or any(value is None for value in row.values()):
            raise ValueError("missing or surplus CSV columns")
        if not row["condition"].strip() or not row["metric"].strip():
            raise ValueError("empty observation label")
        for field in fields:
            if field in ("condition", "metric", "theory"):
                continue
            if not row[field].strip():
                raise ValueError(f"empty numeric field: {field}")
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite numeric field: {field}")
            if field in ("sd", "se") and value < 0:
                raise ValueError("negative uncertainty")
            if field in ("replicate", "n"):
                if str(int(value)) != row[field] or value < (1 if field == "n" else 0):
                    raise ValueError("invalid integer")
        if row["theory"] != "" and not math.isfinite(float(row["theory"])):
            raise ValueError("nonfinite theory")
        if "mean" in row and not (float(row["lower95"]) <= float(row["mean"])
                                 <= float(row["upper95"])):
            raise ValueError("interval excludes its mean")
    return rows


class DesignAndSeedTests(unittest.TestCase):
    def test_nine_studies(self) -> None:
        self.assertEqual(len(analysis.EXPERIMENTS), 9)

    def test_nine_distinct_studies(self) -> None:
        self.assertEqual(len(set(analysis.EXPERIMENTS)), 9)

    def test_nine_figures(self) -> None:
        self.assertEqual(len(analysis.FIGURES), 9)

    def test_distinct_figures(self) -> None:
        self.assertEqual(len(set(analysis.FIGURES)), 9)

    def test_figures_numbered(self) -> None:
        self.assertTrue(all(v.startswith(f"fig{i}_") for i, v
                            in enumerate(analysis.FIGURES, start=1)))

    def test_mapping_matches_experiments(self) -> None:
        self.assertEqual(set(analysis.STUDIES), set(analysis.EXPERIMENTS))

    def test_repetitions_match_experiments(self) -> None:
        self.assertEqual(set(analysis.REPETITIONS), set(analysis.EXPERIMENTS))

    def test_full_repetitions_at_least_twenty(self) -> None:
        self.assertTrue(all(r >= 20 for r in analysis.REPETITIONS.values()))

    def test_seed_default(self) -> None:
        self.assertEqual(analysis.Design().seed, analysis.SEED)

    def test_root_default(self) -> None:
        self.assertEqual(analysis.Design().root, ROOT)

    def test_quick_default_false(self) -> None:
        self.assertFalse(analysis.Design().quick)

    def test_stream_repeats_exactly(self) -> None:
        a = analysis.stream(123, "iid", 0).normal(size=20)
        b = analysis.stream(123, "iid", 0).normal(size=20)
        np.testing.assert_array_equal(a, b)

    def test_stream_differs_by_study(self) -> None:
        a = analysis.stream(123, "iid", 0).normal(size=20)
        b = analysis.stream(123, "search", 0).normal(size=20)
        self.assertFalse(np.array_equal(a, b))

    def test_stream_differs_by_repetition(self) -> None:
        a = analysis.stream(123, "iid", 0).normal(size=20)
        b = analysis.stream(123, "iid", 1).normal(size=20)
        self.assertFalse(np.array_equal(a, b))

    def test_stream_differs_by_seed(self) -> None:
        a = analysis.stream(123, "iid", 0).normal(size=20)
        b = analysis.stream(124, "iid", 0).normal(size=20)
        self.assertFalse(np.array_equal(a, b))

    def test_stream_rejects_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(-1, "iid", 0)

    def test_stream_rejects_float_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1.5, "iid", 0)  # type: ignore[arg-type]

    def test_stream_rejects_bad_study(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "not_a_study", 0)

    def test_stream_rejects_negative_repetition(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "iid", -1)

    def test_stream_rejects_float_repetition(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "iid", 1.5)  # type: ignore[arg-type]

    def test_run_rejects_root_string(self) -> None:
        with self.assertRaises(TypeError):
            analysis.run(analysis.Design(root="bad", quick=True))  # type: ignore[arg-type]

    def test_run_rejects_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(seed=-1, quick=True))

    def test_run_guards_canonical_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(seed=analysis.SEED+1))


class DataAndLossTests(unittest.TestCase):
    def test_logistic_population_shape(self) -> None:
        x, y = analysis.logistic_sample(np.random.default_rng(1), 100)
        self.assertEqual(x.shape, (100, 2))
        self.assertEqual(y.shape, (100,))

    def test_logistic_population_extra_dimensions(self) -> None:
        x, _ = analysis.logistic_sample(np.random.default_rng(1), 100, 12)
        self.assertEqual(x.shape, (100, 12))

    def test_logistic_population_binary(self) -> None:
        _, y = analysis.logistic_sample(np.random.default_rng(1), 200)
        self.assertTrue(np.isin(y, (0, 1)).all())

    def test_logistic_population_integer(self) -> None:
        _, y = analysis.logistic_sample(np.random.default_rng(1), 100)
        self.assertEqual(y.dtype, np.int64)

    def test_logistic_population_reproducible(self) -> None:
        a = analysis.logistic_sample(np.random.default_rng(1), 100)
        b = analysis.logistic_sample(np.random.default_rng(1), 100)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)

    def test_logistic_population_rejects_short(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_sample(np.random.default_rng(1), 19)

    def test_logistic_population_rejects_dimension_one(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_sample(np.random.default_rng(1), 100, 1)

    def test_brier_perfect(self) -> None:
        y = np.array([0, 1, 0, 1], dtype=int)
        self.assertEqual(analysis.brier(y, y.astype(float)), 0.)

    def test_brier_constant_half(self) -> None:
        y = np.array([0, 1, 0, 1], dtype=int)
        self.assertEqual(analysis.brier(y, np.full(4, .5)), .25)

    def test_brier_opposite_prediction(self) -> None:
        y = np.array([0, 1, 0, 1], dtype=int)
        self.assertEqual(analysis.brier(y, 1-y.astype(float)), 1.)

    def test_brier_rejects_negative_probability(self) -> None:
        with self.assertRaises(ValueError):
            analysis.brier(np.array([0, 1]), np.array([-.1, .8]))

    def test_brier_rejects_above_one(self) -> None:
        with self.assertRaises(ValueError):
            analysis.brier(np.array([0, 1]), np.array([.1, 1.1]))

    def test_brier_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.brier(np.array([0, 1]), np.array([float("nan"), .8]))

    def test_brier_rejects_inf(self) -> None:
        with self.assertRaises(ValueError):
            analysis.brier(np.array([0, 1]), np.array([float("inf"), .8]))

    def test_brier_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            analysis.brier(np.array([0, 1]), np.array([.1, .2, .3]))

    def test_brier_rejects_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.brier(np.array([0, 1]), np.array([[.2], [.8]]))

    def test_model_accepts_positive_c(self) -> None:
        self.assertAlmostEqual(analysis.logistic_model(.3).C, .3)

    def test_model_rejects_zero_c(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_model(0.)

    def test_model_rejects_negative_c(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_model(-1.)

    def test_model_rejects_nan_c(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_model(float("nan"))

    def test_model_rejects_infinite_c(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_model(float("inf"))

    def test_model_probabilities_finite(self) -> None:
        x, y = analysis.logistic_sample(np.random.default_rng(1), 100)
        model = analysis.logistic_model().fit(x, y)
        self.assertTrue(np.isfinite(model.predict_proba(x)).all())

    def test_ar1_shape(self) -> None:
        self.assertEqual(analysis.ar1_path(np.random.default_rng(1), 100).shape, (100,))

    def test_ar1_reproducible(self) -> None:
        a = analysis.ar1_path(np.random.default_rng(1), 100)
        b = analysis.ar1_path(np.random.default_rng(1), 100)
        np.testing.assert_array_equal(a, b)

    def test_ar1_positive_correlation(self) -> None:
        x = analysis.ar1_path(np.random.default_rng(1), 500)
        self.assertGreater(np.corrcoef(x[1:], x[:-1])[0, 1], .7)

    def test_ar1_rejects_short(self) -> None:
        with self.assertRaises(ValueError):
            analysis.ar1_path(np.random.default_rng(1), 19)

    def test_ar1_rejects_zero_coefficient(self) -> None:
        with self.assertRaises(ValueError):
            analysis.ar1_path(np.random.default_rng(1), 100, 0.)

    def test_ar1_rejects_unit_coefficient(self) -> None:
        with self.assertRaises(ValueError):
            analysis.ar1_path(np.random.default_rng(1), 100, 1.)

    def test_ar1_rejects_negative_coefficient(self) -> None:
        with self.assertRaises(ValueError):
            analysis.ar1_path(np.random.default_rng(1), 100, -.1)


class ObservationSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = analysis.observation(0, "condition", "loss", .2)
        self.b = analysis.observation(1, "condition", "loss", .4)

    def test_observation_fields(self) -> None:
        self.assertEqual(set(self.a), set(analysis.CSV_FIELDS))

    def test_observation_replicate(self) -> None:
        self.assertEqual(self.a["replicate"], 0)

    def test_observation_value(self) -> None:
        self.assertEqual(self.a["value"], .2)

    def test_observation_empty_theory(self) -> None:
        self.assertIsNone(self.a["theory"])

    def test_observation_with_theory(self) -> None:
        row = analysis.observation(0, "c", "m", .25, .25)
        self.assertEqual(row["theory"], .25)

    def test_observation_rejects_negative_replicate(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(-1, "c", "m", .2)

    def test_observation_rejects_empty_condition(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "", "m", .2)

    def test_observation_rejects_empty_metric(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "c", "", .2)

    def test_observation_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "c", "m", float("nan"))

    def test_observation_rejects_inf(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "c", "m", float("inf"))

    def test_observation_rejects_nan_theory(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "c", "m", .2, float("nan"))

    def test_summarize_mean(self) -> None:
        self.assertAlmostEqual(analysis.summarize([self.a, self.b])[0]["mean"], .3)

    def test_summarize_n(self) -> None:
        self.assertEqual(analysis.summarize([self.a, self.b])[0]["n"], 2)

    def test_summarize_sd(self) -> None:
        self.assertAlmostEqual(analysis.summarize([self.a, self.b])[0]["sd"],
                               math.sqrt(.02))

    def test_summarize_se(self) -> None:
        self.assertAlmostEqual(analysis.summarize([self.a, self.b])[0]["se"], .1)

    def test_summarize_interval(self) -> None:
        row = analysis.summarize([self.a, self.b])[0]
        self.assertAlmostEqual(row["lower95"], .3-1.96*.1)
        self.assertAlmostEqual(row["upper95"], .3+1.96*.1)

    def test_summarize_single(self) -> None:
        row = analysis.summarize([self.a])[0]
        self.assertEqual((row["n"], row["sd"], row["se"]), (1, 0., 0.))

    def test_summarize_theory(self) -> None:
        a = analysis.observation(0, "c", "m", .2, .25)
        b = analysis.observation(1, "c", "m", .4, .25)
        self.assertEqual(analysis.summarize([a, b])[0]["theory"], .25)

    def test_summarize_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([])

    def test_summarize_rejects_duplicate_replicate(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([self.a, self.a])

    def test_summarize_rejects_inconsistent_theory(self) -> None:
        a = analysis.observation(0, "c", "m", .2, .25)
        b = analysis.observation(1, "c", "m", .4, .30)
        with self.assertRaises(ValueError):
            analysis.summarize([a, b])

    def test_summarize_two_distinct_keys(self) -> None:
        different = analysis.observation(0, "other", "loss", .3)
        self.assertEqual(len(analysis.summarize([self.a, different])), 2)

    def test_lookup_mean(self) -> None:
        rows = analysis.summarize([self.a, self.b])
        self.assertAlmostEqual(analysis.lookup(rows, "condition", "loss"), .3)

    def test_lookup_se(self) -> None:
        rows = analysis.summarize([self.a, self.b])
        self.assertAlmostEqual(analysis.lookup(rows, "condition", "loss", "se"), .1)

    def test_lookup_rejects_missing_key(self) -> None:
        with self.assertRaises(KeyError):
            analysis.lookup(analysis.summarize([self.a]), "missing", "loss")

    def test_lookup_rejects_missing_field(self) -> None:
        with self.assertRaises(KeyError):
            analysis.lookup(analysis.summarize([self.a]), "condition", "loss", "fake")

    def test_write_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/"some"/"data.csv"
            analysis.write_csv(path, [self.a, self.b], analysis.CSV_FIELDS)
            rows = strict_csv(path, analysis.CSV_FIELDS)
            self.assertEqual(len(rows), 2)

    def test_write_csv_refuses_empty(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(dirname)/"x.csv", [], analysis.CSV_FIELDS)

    def test_write_csv_refuses_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            wrong = dict(self.a)
            wrong.pop("theory")
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(dirname)/"x.csv", [wrong], analysis.CSV_FIELDS)

    def test_write_csv_uses_lf(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/"x.csv"
            analysis.write_csv(path, [self.a], analysis.CSV_FIELDS)
            self.assertNotIn(b"\r\n", path.read_bytes())


class StudyContractTests(unittest.TestCase):
    def test_contract_keys_match_names(self) -> None:
        self.assertEqual(set(CONTRACTS), set(analysis.EXPERIMENTS))

    def test_expected_keys_nonempty(self) -> None:
        self.assertTrue(all(expected_keys(name) for name in analysis.EXPERIMENTS))

    def test_expected_keys_exact(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = analysis.STUDIES[name](analysis.stream(77, name, 1), 1, True)
                observed = [(str(r["condition"]), str(r["metric"])) for r in rows]
                self.assertEqual(len(observed), len(set(observed)))
                self.assertEqual(set(observed), expected_keys(name))

    def test_all_study_values_finite(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = analysis.STUDIES[name](analysis.stream(7, name, 1), 1, True)
                self.assertTrue(all(math.isfinite(float(r["value"])) for r in rows))

    def test_studies_reproducible(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                a = analysis.STUDIES[name](analysis.stream(7, name, 1), 1, True)
                b = analysis.STUDIES[name](analysis.stream(7, name, 1), 1, True)
                self.assertEqual(a, b)

    def test_iid_training_more_optimistic_at_small_n(self) -> None:
        rows = analysis.iid_study(analysis.stream(7, "iid", 1), 1, True)
        metrics = {(r["condition"], r["metric"]): r["value"] for r in rows}
        self.assertLess(metrics[("120", "training_brier")],
                        metrics[("120", "cv_brier")])

    def test_search_larger_pool_never_worsens_selected_score(self) -> None:
        rows = analysis.search_study(analysis.stream(7, "search", 1), 1, True)
        values = {r["condition"]: float(r["value"]) for r in rows if
                  r["metric"] == "selected_validation_brier"}
        self.assertLessEqual(values["500"], values["1"])

    def test_search_null_baseline_is_not_perfect(self) -> None:
        rows = analysis.search_study(analysis.stream(7, "search", 1), 1, True)
        selected = [float(r["value"]) for r in rows if
                    r["metric"] == "independent_test_brier"]
        self.assertTrue(all(.20 < value < .35 for value in selected))

    def test_nested_exposes_all_three_estimators(self) -> None:
        rows = analysis.nested_study(analysis.stream(7, "nested", 1), 1, True)
        self.assertEqual(len(rows), 3)

    def test_nested_probability_loss_bounded(self) -> None:
        rows = analysis.nested_study(analysis.stream(7, "nested", 1), 1, True)
        self.assertTrue(all(0 <= float(r["value"]) <= 1 for r in rows))

    def test_preprocess_leakage_hurts_at_large_p(self) -> None:
        rows = analysis.preprocess_study(analysis.stream(7, "preprocess", 1), 1, True)
        metrics = {(r["condition"], r["metric"]): float(r["value"]) for r in rows}
        self.assertLess(metrics[("600", "leaky_cv_brier")],
                        metrics[("600", "fold_safe_cv_brier")])

    def test_temporal_strong_drift_hurts_future(self) -> None:
        rows = analysis.temporal_study(analysis.stream(7, "temporal", 1), 1, True)
        metrics = {(r["condition"], r["metric"]): float(r["value"]) for r in rows}
        self.assertGreater(metrics[("3.0", "future_period_brier")],
                           metrics[("3.0", "random_history_cv_brier")])

    def test_lookahead_strong_horizon_leak(self) -> None:
        rows = analysis.lookahead_study(analysis.stream(7, "lookahead", 1), 1, True)
        metrics = {(r["condition"], r["metric"]): float(r["value"]) for r in rows}
        self.assertGreater(metrics[("14", "causal_feature_brier")],
                           metrics[("14", "lookahead_feature_brier")])

    def test_group_unseen_site_gap(self) -> None:
        rows = analysis.groups_study(analysis.stream(7, "groups", 1), 1, True)
        metrics = {(r["condition"], r["metric"]): float(r["value"]) for r in rows}
        self.assertGreater(metrics[("1.8", "unseen_site_cv_brier")],
                           metrics[("1.8", "random_row_cv_brier")])

    def test_uncertainty_coverage_is_binary(self) -> None:
        rows = analysis.uncertainty_study(analysis.stream(7, "uncertainty", 1),
                                          1, True)
        value = next(float(r["value"]) for r in rows if
                     r["metric"] == "naive_interval_covers_external")
        self.assertIn(value, (0., 1.))

    def test_uncertainty_positive_fold_se(self) -> None:
        rows = analysis.uncertainty_study(analysis.stream(7, "uncertainty", 1),
                                          1, True)
        value = next(float(r["value"]) for r in rows if
                     r["metric"] == "naive_fold_se")
        self.assertGreater(value, 0.)

    def test_adaptive_reconstructs_entire_public_set(self) -> None:
        rows = analysis.adaptive_study(analysis.stream(7, "adaptive", 1), 1, True)
        value = next(r["value"] for r in rows if
                     r["condition"] == "120" and
                     r["metric"] == "public_holdout_accuracy")
        self.assertEqual(value, 1.)

    def test_adaptive_query_count_includes_baseline(self) -> None:
        rows = analysis.adaptive_study(analysis.stream(7, "adaptive", 1), 1, True)
        value = next(r["value"] for r in rows if
                     r["condition"] == "120" and r["metric"] == "public_queries_used")
        self.assertEqual(value, 121)

    def test_adaptive_fresh_score_not_perfect(self) -> None:
        rows = analysis.adaptive_study(analysis.stream(7, "adaptive", 1), 1, True)
        value = next(float(r["value"]) for r in rows if
                     r["condition"] == "120" and
                     r["metric"] == "fresh_independent_accuracy")
        self.assertLess(value, .6)


class ArchivedEvidenceTests(unittest.TestCase):
    def invalid_summary(self, replacements: dict[str, str]) -> None:
        """Build one intentionally bad summary row and demand rejection."""

        row = {"condition": "c", "metric": "m", "n": "2", "mean": "0.2",
               "sd": "0.1", "se": "0.07", "lower95": "0.0",
               "upper95": "0.4", "theory": ""}
        row.update(replacements)
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/"bad.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=analysis.SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaises(ValueError):
                strict_csv(path, analysis.SUMMARY_FIELDS)

    def test_eighteen_canonical_csvs(self) -> None:
        actual = {p.name for p in (ROOT/"results").glob("*.csv")
                  if not p.name.endswith("_quick.csv")}
        expected = {f"{study}_{kind}.csv" for study in analysis.EXPERIMENTS
                    for kind in ("replicates", "summary")}
        self.assertEqual(actual, expected)

    def test_nine_canonical_figures(self) -> None:
        actual = {p.name for p in (ROOT/"figures").glob("*.png")
                  if not p.name.endswith("_quick.png")}
        self.assertEqual(actual, {f"{name}.png" for name in analysis.FIGURES})

    def test_png_signatures(self) -> None:
        """Validate each archived figure without reading editorial Markdown."""
        for name in analysis.FIGURES:
            with self.subTest(figure=name):
                path: Path = ROOT/"figures"/f"{name}.png"
                contents: bytes = path.read_bytes()
                self.assertGreaterEqual(len(contents), 24)
                self.assertEqual(contents[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(contents[12:16], b"IHDR")
                # The PNG header stores unsigned, big-endian pixel dimensions.
                width: int = int.from_bytes(contents[16:20], "big")
                height: int = int.from_bytes(contents[20:24], "big")
                self.assertGreater(width, 0)
                self.assertGreater(height, 0)

    def test_pngs_nonempty(self) -> None:
        self.assertTrue(all((ROOT/"figures"/f"{name}.png").stat().st_size > 25_000
                            for name in analysis.FIGURES))

    def test_replication_csv_exact_schema(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = strict_csv(ROOT/"results"/f"{name}_replicates.csv",
                                  analysis.CSV_FIELDS)
                self.assertEqual({(r["condition"], r["metric"]) for r in rows},
                                 expected_keys(name))

    def test_summary_csv_exact_schema(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = strict_csv(ROOT/"results"/f"{name}_summary.csv",
                                  analysis.SUMMARY_FIELDS)
                self.assertEqual({(r["condition"], r["metric"]) for r in rows},
                                 expected_keys(name))
                self.assertEqual(len(rows), len(expected_keys(name)))

    def test_repetitions_complete_for_each_metric(self) -> None:
        for name in analysis.EXPERIMENTS:
            rows = strict_csv(ROOT/"results"/f"{name}_replicates.csv",
                              analysis.CSV_FIELDS)
            grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
            for row in rows:
                grouped[row["condition"], row["metric"]].append(int(row["replicate"]))
            with self.subTest(study=name):
                self.assertEqual(set(grouped), expected_keys(name))
                for values in grouped.values():
                    self.assertEqual(sorted(values), list(range(analysis.REPETITIONS[name])))

    def test_summary_n_matches_repetitions(self) -> None:
        for name in analysis.EXPERIMENTS:
            rows = strict_csv(ROOT/"results"/f"{name}_summary.csv",
                              analysis.SUMMARY_FIELDS)
            self.assertTrue(all(int(row["n"]) == analysis.REPETITIONS[name]
                                for row in rows))

    def test_summary_means_match_replicates(self) -> None:
        for name in analysis.EXPERIMENTS:
            raw = strict_csv(ROOT/"results"/f"{name}_replicates.csv",
                             analysis.CSV_FIELDS)
            summary = strict_csv(ROOT/"results"/f"{name}_summary.csv",
                                 analysis.SUMMARY_FIELDS)
            for row in summary:
                values = [float(r["value"]) for r in raw if
                          (r["condition"], r["metric"]) ==
                          (row["condition"], row["metric"])]
                with self.subTest(study=name, condition=row["condition"],
                                  metric=row["metric"]):
                    self.assertAlmostEqual(float(row["mean"]), np.mean(values))

    def test_summary_sd_matches_replicates(self) -> None:
        for name in analysis.EXPERIMENTS:
            raw = strict_csv(ROOT/"results"/f"{name}_replicates.csv",
                             analysis.CSV_FIELDS)
            summary = strict_csv(ROOT/"results"/f"{name}_summary.csv",
                                 analysis.SUMMARY_FIELDS)
            for row in summary:
                values = [float(r["value"]) for r in raw if
                          (r["condition"], r["metric"]) ==
                          (row["condition"], row["metric"])]
                self.assertAlmostEqual(float(row["sd"]), np.std(values, ddof=1))

    def test_summary_se_formula(self) -> None:
        for name in analysis.EXPERIMENTS:
            rows = strict_csv(ROOT/"results"/f"{name}_summary.csv",
                              analysis.SUMMARY_FIELDS)
            for row in rows:
                self.assertAlmostEqual(float(row["se"]),
                                       float(row["sd"])/math.sqrt(int(row["n"])))

    def test_summary_intervals_formula(self) -> None:
        for name in analysis.EXPERIMENTS:
            rows = strict_csv(ROOT/"results"/f"{name}_summary.csv",
                              analysis.SUMMARY_FIELDS)
            for row in rows:
                self.assertAlmostEqual(float(row["lower95"]),
                                       float(row["mean"])-1.96*float(row["se"]))
                self.assertAlmostEqual(float(row["upper95"]),
                                       float(row["mean"])+1.96*float(row["se"]))

    def test_adaptive_theory_half(self) -> None:
        rows = strict_csv(ROOT/"results"/"adaptive_summary.csv",
                          analysis.SUMMARY_FIELDS)
        theory = {r["theory"] for r in rows if
                  r["metric"] == "fresh_independent_accuracy"}
        self.assertEqual(theory, {"0.5"})

    def test_iid_cv_and_test_close_at_large_n(self) -> None:
        rows = strict_csv(ROOT/"results"/"iid_summary.csv", analysis.SUMMARY_FIELDS)
        values = {(r["condition"], r["metric"]): float(r["mean"]) for r in rows}
        self.assertLess(abs(values[("900", "cv_brier")]-
                            values[("900", "fresh_test_brier")]), .02)

    def test_preprocessing_leakage_archive(self) -> None:
        rows = strict_csv(ROOT/"results"/"preprocess_summary.csv",
                          analysis.SUMMARY_FIELDS)
        values = {(r["condition"], r["metric"]): float(r["mean"]) for r in rows}
        self.assertLess(values[("600", "leaky_cv_brier")],
                        values[("600", "fold_safe_cv_brier")])

    def test_adaptive_public_rises_but_fresh_does_not(self) -> None:
        rows = strict_csv(ROOT/"results"/"adaptive_summary.csv",
                          analysis.SUMMARY_FIELDS)
        values = {(r["condition"], r["metric"]): float(r["mean"]) for r in rows}
        self.assertEqual(values[("120", "public_holdout_accuracy")], 1.)
        self.assertLess(values[("120", "fresh_independent_accuracy")], .6)

    def test_quick_mode_preserves_all_canonical_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            base = Path(dirname)
            result_dir, figure_dir = base/"results", base/"figures"
            result_dir.mkdir()
            figure_dir.mkdir()
            original: dict[Path, bytes] = {}
            for study in analysis.EXPERIMENTS:
                for kind in ("replicates", "summary"):
                    path = result_dir/f"{study}_{kind}.csv"
                    original[path] = f"immutable {path.name}\n".encode("utf-8")
            for figure in analysis.FIGURES:
                path = figure_dir/f"{figure}.png"
                original[path] = f"immutable {path.name}\n".encode("utf-8")
            for path, content in original.items():
                path.write_bytes(content)
            returned = analysis.run(analysis.Design(root=base, quick=True))
            self.assertEqual(set(returned), set(analysis.EXPERIMENTS))
            expected_csv = {f"{study}_{kind}_quick.csv" for study in analysis.EXPERIMENTS
                            for kind in ("replicates", "summary")}
            expected_png = {f"{figure}_quick.png" for figure in analysis.FIGURES}
            self.assertEqual({p.name for p in result_dir.iterdir()},
                             expected_csv | {p.name for p in original if p.parent == result_dir})
            self.assertEqual({p.name for p in figure_dir.iterdir()},
                             expected_png | {p.name for p in original if p.parent == figure_dir})
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)

    def test_strict_csv_rejects_nan_sd(self) -> None:
        self.invalid_summary({"sd": "nan"})

    def test_strict_csv_rejects_inf_se(self) -> None:
        self.invalid_summary({"se": "inf"})

    def test_strict_csv_rejects_negative_sd(self) -> None:
        self.invalid_summary({"sd": "-0.1"})

    def test_strict_csv_rejects_nan_theory(self) -> None:
        self.invalid_summary({"theory": "NaN"})

    def test_strict_csv_rejects_text_mean(self) -> None:
        self.invalid_summary({"mean": "not-a-number"})

    def test_strict_csv_rejects_missing_mean(self) -> None:
        self.invalid_summary({"mean": ""})

    def test_strict_csv_rejects_noninteger_n(self) -> None:
        self.invalid_summary({"n": "2.5"})

    def test_strict_csv_rejects_zero_n(self) -> None:
        self.invalid_summary({"n": "0"})

    def test_strict_csv_rejects_bad_interval(self) -> None:
        self.invalid_summary({"lower95": "0.3"})

    def test_strict_csv_rejects_missing_header(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/"bad.csv"
            path.write_text("condition,metric\nc,m\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                strict_csv(path, analysis.SUMMARY_FIELDS)

    def test_strict_csv_rejects_surplus_column(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/"bad.csv"
            path.write_text("replicate,condition,metric,value,theory\n0,c,m,0.2,,extra\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                strict_csv(path, analysis.CSV_FIELDS)

    def test_strict_csv_rejects_blank_csv(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/"bad.csv"
            path.write_text(",".join(analysis.SUMMARY_FIELDS)+"\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                strict_csv(path, analysis.SUMMARY_FIELDS)


class NotebookAndDocumentationTests(unittest.TestCase):
    """Protect portable notebook execution, scope, and the long-form archive."""

    def notebook_path(self) -> Path:
        """Return the one committed unexecuted notebook."""

        return ROOT/notebook_builder.FILENAME

    def test_notebook_builder_deterministic(self) -> None:
        payload = json.dumps(notebook_builder.notebook(), indent=2,
                             ensure_ascii=False)+"\n"
        self.assertEqual(self.notebook_path().read_text(encoding="utf-8"), payload)

    def test_notebook_fourth_major_schema(self) -> None:
        notebook = notebook_builder.notebook()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(notebook["nbformat_minor"], 5)

    def test_notebook_schema_valid(self) -> None:
        if nbformat is None:
            self.skipTest("nbformat is not installed")
        nbformat.validate(nbformat.read(self.notebook_path(), as_version=4))

    def test_notebook_unique_cell_ids(self) -> None:
        cells = notebook_builder.notebook()["cells"]
        identifiers = [cell["id"] for cell in cells]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_notebook_no_cached_outputs(self) -> None:
        cells = notebook_builder.notebook()["cells"]
        for cell in cells:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_notebook_uses_isolated_temporary_directory(self) -> None:
        source = self.notebook_path().read_text(encoding="utf-8")
        self.assertIn("TemporaryDirectory()", source)
        self.assertIn("Design(root=destination, quick=True)", source)

    def test_notebook_executes_nine_studies(self) -> None:
        source = self.notebook_path().read_text(encoding="utf-8")
        self.assertIn("len(EXPERIMENTS) == len(FIGURES) == 9", source)
        self.assertIn("zip(EXPERIMENTS, FIGURES)", source)

    def test_notebook_checks_eighteen_csvs(self) -> None:
        source = self.notebook_path().read_text(encoding="utf-8")
        self.assertIn("len(csv_files) == 18", source)

    def test_notebook_checks_nine_pngs(self) -> None:
        source = self.notebook_path().read_text(encoding="utf-8")
        self.assertIn("len(png_files) == 9", source)

    def test_notebook_external_path_hint(self) -> None:
        source = self.notebook_path().read_text(encoding="utf-8")
        self.assertIn("VALIDATION_ARTICLE_DIR", source)

    def test_notebook_no_embedded_absolute_workspace_path(self) -> None:
        source = self.notebook_path().read_text(encoding="utf-8")
        self.assertNotIn("/workspace/scratch", source)

    def test_public_article_is_private_manuscript_notice(self) -> None:
        """Preserve the historical path while owning only computational evidence."""
        notice: str = (ROOT/"article.md").read_text(encoding="utf-8")
        self.assertIn("research-articles-medium", notice)
        self.assertIn("the-test-set-is-not-an-oracle/article.md", notice)
        self.assertLess(len(notice.split()), 200)

    def test_readme_nine_studies(self) -> None:
        content = (ROOT/"README.md").read_text(encoding="utf-8")
        for name in analysis.FIGURES:
            self.assertIn(f"{name}.png", content)

    def test_readme_explicit_canonical_seed(self) -> None:
        self.assertIn(str(analysis.SEED),
                      (ROOT/"README.md").read_text(encoding="utf-8"))

    def test_requirements_science_versions(self) -> None:
        content = (ROOT/"requirements.txt").read_text(encoding="utf-8")
        for dependency in ("numpy==", "scipy==", "matplotlib==",
                           "scikit-learn==", "nbformat>=", "nbclient>=",
                           "ipykernel>="):
            self.assertIn(dependency, content)

    def test_workflow_article_scoped(self) -> None:
        path = ROOT.parent/".github"/"workflows"/"the-test-set-is-not-an-oracle.yml"
        content = path.read_text(encoding="utf-8")
        self.assertIn("the-test-set-is-not-an-oracle/**", content)
        self.assertIn(".github/workflows/the-test-set-is-not-an-oracle.yml", content)

    def test_workflow_contents_read_permission(self) -> None:
        path = ROOT.parent/".github"/"workflows"/"the-test-set-is-not-an-oracle.yml"
        self.assertIn("contents: read", path.read_text(encoding="utf-8"))

    def test_workflow_tests_fresh_kernel(self) -> None:
        path = ROOT.parent/".github"/"workflows"/"the-test-set-is-not-an-oracle.yml"
        content = path.read_text(encoding="utf-8")
        self.assertIn("RUN_NOTEBOOK_EXECUTION: '1'", content)
        self.assertIn("python -m unittest discover -s tests", content)

    def test_workflow_quick_smoke(self) -> None:
        path = ROOT.parent/".github"/"workflows"/"the-test-set-is-not-an-oracle.yml"
        self.assertIn("python analysis.py --quick",
                      path.read_text(encoding="utf-8"))

    def _execute_clean_kernel(self, working_directory: Path,
                              explicit_hint: bool) -> None:
        """Run all nine studies in a fresh kernel without notebook path metadata."""

        if nbformat is None or NotebookClient is None:
            self.skipTest("notebook runtime dependencies are not installed")
        if os.getenv("RUN_NOTEBOOK_EXECUTION") != "1":
            self.skipTest("set RUN_NOTEBOOK_EXECUTION=1 for kernel tests")
        notebook = nbformat.read(self.notebook_path(), as_version=4)
        old_directory = Path.cwd()
        hint = {"VALIDATION_ARTICLE_DIR": str(ROOT)} if explicit_hint else {}
        try:
            os.chdir(working_directory)
            with patch.dict(os.environ, hint, clear=False):
                if not explicit_hint:
                    os.environ.pop("VALIDATION_ARTICLE_DIR", None)
                executed = NotebookClient(
                    notebook, timeout=240, kernel_name="python3",
                    resources={"metadata": {}}, allow_errors=False,
                ).execute()
        finally:
            os.chdir(old_directory)
        for cell in executed.cells:
            if cell.cell_type == "code":
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(output.output_type == "error"
                                     for output in cell.outputs))
        rendered_images = sum(
            "image/png" in output.get("data", {})
            for cell in executed.cells if cell.cell_type == "code"
            for output in cell.outputs
        )
        self.assertEqual(rendered_images, 9)

    def test_fresh_kernel_from_repository_root(self) -> None:
        self._execute_clean_kernel(ROOT.parent, explicit_hint=False)

    def test_fresh_kernel_external_directory(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            self._execute_clean_kernel(Path(dirname), explicit_hint=True)



