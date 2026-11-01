"""Mathematical, evidence, notebook, and regeneration contract tests."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

import analysis
import notebook_builder

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    nbformat = None
    NotebookClient = None


ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    """Read archived evidence as text so empty theory cells stay visible."""

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class InputValidation(unittest.TestCase):
    def test_valid_integer_matrix_casts_float(self) -> None:
        self.assertEqual(analysis.check_matrix(np.ones((4, 2), dtype=int)).dtype, float)

    def test_matrix_preserves_shape(self) -> None:
        self.assertEqual(analysis.check_matrix(np.arange(12).reshape(4, 3)).shape, (4, 3))

    def test_reject_vector(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.arange(4))

    def test_reject_cube(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.zeros((3, 2, 2)))

    def test_reject_empty_columns(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.empty((4, 0)))

    def test_reject_one_row(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.zeros((1, 2)))

    def test_reject_two_rows(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.zeros((2, 2)))

    def test_reject_nan_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.array([[1., 2.], [2., math.nan], [3., 4.]]))

    def test_reject_inf_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.array([[1., 2.], [2., math.inf], [3., 4.]]))

    def test_good_vector(self) -> None:
        np.testing.assert_array_equal(analysis.check_vector(np.arange(4), 4), np.arange(4))

    def test_reject_wrong_response_length(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.arange(4), 5)

    def test_reject_nonnumeric_response_shape(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.ones((4, 1)), 4)

    def test_reject_nonfinite_response(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.array([1., 2., np.inf]), 3)

    def test_reject_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(-1, "shapley", 0)

    def test_reject_unknown_study(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(10, "unknown", 0)

    def test_reject_negative_rep(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(10, "shapley", -1)

    def test_reject_non_path_root(self) -> None:
        with self.assertRaises(TypeError):
            analysis.run(analysis.Design(root="/tmp", quick=True))  # type: ignore[arg-type]

    def test_reject_run_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(root=Path("/tmp"), quick=True, seed=-1))

    def test_reject_bad_rho(self) -> None:
        with self.assertRaises(ValueError):
            analysis.gaussian_pair(np.random.default_rng(1), 5, 1.0)

    def test_reject_small_gaussian_sample(self) -> None:
        with self.assertRaises(ValueError):
            analysis.gaussian_pair(np.random.default_rng(1), 2, 0.5)

    def test_reject_bad_conditional_rho(self) -> None:
        with self.assertRaises(ValueError):
            analysis.conditional_resample(np.random.default_rng(1), np.ones(5), -1.0)


class EstimandIdentities(unittest.TestCase):
    def test_mse_zero_for_exact_fit(self) -> None:
        self.assertEqual(analysis.mse(np.arange(10.), np.arange(10.)), 0.)

    def test_mse_matches_squared_error_definition(self) -> None:
        self.assertAlmostEqual(analysis.mse(np.array([1., 2., 3.]),
                                            np.zeros(3)), 14 / 3)

    def test_permutation_reproducible(self) -> None:
        rng = np.random.default_rng(13)
        x = rng.normal(size=(100, 2))
        y = x[:, 0]
        model = Ridge().fit(x, y)
        a = analysis.permutation_delta(model, x, y, 0, np.random.default_rng(4))
        b = analysis.permutation_delta(model, x, y, 0, np.random.default_rng(4))
        self.assertEqual(a, b)

    def test_permutation_preserves_model_coefficient(self) -> None:
        rng = np.random.default_rng(12)
        x = rng.normal(size=(50, 2))
        model = Ridge().fit(x, x[:, 0])
        original = model.coef_.copy()
        analysis.permutation_delta(model, x, x[:, 0], 0, rng)
        np.testing.assert_array_equal(model.coef_, original)

    def test_permutation_preserves_input(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(30, 2))
        xx = x.copy()
        model = Ridge().fit(x, x[:, 0])
        analysis.permutation_delta(model, x, x[:, 0], 0, rng)
        np.testing.assert_array_equal(x, xx)

    def test_permutation_nonnegative_for_exact_oracle(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(500, 2))
        y = x[:, 0]
        model = Ridge(alpha=0).fit(x, y)
        self.assertGreater(analysis.permutation_delta(model, x, y, 0, rng), 0.5)

    def test_permutation_ignored_feature_is_exact_zero(self) -> None:
        class Oracle:
            def predict(self, x: np.ndarray) -> np.ndarray:
                return x[:, 0]
        rng = np.random.default_rng(6)
        x = rng.normal(size=(100, 2))
        self.assertEqual(analysis.permutation_delta(Oracle(), x, x[:, 0], 1, rng), 0.)

    def test_permutation_rejects_invalid_column(self) -> None:
        class Oracle:
            def predict(self, x: np.ndarray) -> np.ndarray:
                return x[:, 0]
        with self.assertRaises(IndexError):
            analysis.permutation_delta(Oracle(), np.ones((10, 2)), np.ones(10),
                                       3, np.random.default_rng(1))

    def test_permutation_rejects_missing_predict(self) -> None:
        with self.assertRaises(TypeError):
            analysis.permutation_delta(object(), np.ones((10, 2)), np.ones(10),
                                       0, np.random.default_rng(1))

    def test_gaussian_pair_shape(self) -> None:
        self.assertEqual(analysis.gaussian_pair(np.random.default_rng(5), 500, .7).shape, (500, 2))

    def test_gaussian_pair_approximately_correlated(self) -> None:
        x = analysis.gaussian_pair(np.random.default_rng(7), 10000, .7)
        self.assertAlmostEqual(np.corrcoef(x.T)[0, 1], .7, delta=.025)

    def test_gaussian_pair_standard_deviations(self) -> None:
        x = analysis.gaussian_pair(np.random.default_rng(7), 10000, .2)
        np.testing.assert_allclose(x.std(axis=0), 1., atol=.035)

    def test_conditional_mean_matches_formula(self) -> None:
        rng = np.random.default_rng(2)
        x2 = np.full(30000, 2.)
        draws = analysis.conditional_resample(rng, x2, .8)
        self.assertAlmostEqual(draws.mean(), 1.6, delta=.02)

    def test_conditional_variance_matches_formula(self) -> None:
        rng = np.random.default_rng(2)
        x2 = np.full(30000, 2.)
        draws = analysis.conditional_resample(rng, x2, .8)
        self.assertAlmostEqual(draws.var(), .36, delta=.015)

    def test_independent_streams_repeat(self) -> None:
        a = analysis.stream(99, "shift", 3).normal(size=20)
        b = analysis.stream(99, "shift", 3).normal(size=20)
        np.testing.assert_array_equal(a, b)

    def test_independent_streams_differ_by_replicate(self) -> None:
        a = analysis.stream(99, "shift", 2).normal(size=20)
        b = analysis.stream(99, "shift", 3).normal(size=20)
        self.assertFalse(np.array_equal(a, b))

    def test_independent_streams_differ_by_study(self) -> None:
        a = analysis.stream(99, "shift", 2).normal(size=20)
        b = analysis.stream(99, "shapley", 2).normal(size=20)
        self.assertFalse(np.array_equal(a, b))


class AttributionGames(unittest.TestCase):
    def test_interventional_unused_feature_zero(self) -> None:
        self.assertEqual(analysis.exact_two_feature_shapley(2., 3., .9, "interventional")[1], 0.)

    def test_interventional_used_feature(self) -> None:
        self.assertEqual(analysis.exact_two_feature_shapley(2., 3., .9, "interventional")[0], 2.)

    def test_observational_unused_feature_credit(self) -> None:
        self.assertAlmostEqual(
            analysis.exact_two_feature_shapley(2., 3., .8, "observational")[1], 1.2)

    def test_observational_used_feature_credit(self) -> None:
        self.assertAlmostEqual(
            analysis.exact_two_feature_shapley(2., 3., .8, "observational")[0], .8)

    def test_efficiency_interventional(self) -> None:
        a, b = analysis.exact_two_feature_shapley(-1.5, 4., .8, "interventional")
        self.assertAlmostEqual(a + b, -1.5)

    def test_efficiency_observational(self) -> None:
        a, b = analysis.exact_two_feature_shapley(-1.5, 4., .8, "observational")
        self.assertAlmostEqual(a + b, -1.5)

    def test_observational_zero_correlation_agrees(self) -> None:
        a = analysis.exact_two_feature_shapley(2., 3., 0., "observational")
        b = analysis.exact_two_feature_shapley(2., 3., 0., "interventional")
        self.assertEqual(a, b)

    def test_negative_correlation_allows_negative_credit(self) -> None:
        a, b = analysis.exact_two_feature_shapley(1., 1., -.7, "observational")
        self.assertLess(b, 0.)
        self.assertGreater(a, 1.)

    def test_reject_unknown_game(self) -> None:
        with self.assertRaises(ValueError):
            analysis.exact_two_feature_shapley(1., 2., .5, "causal")

    def test_reject_boundary_rho(self) -> None:
        with self.assertRaises(ValueError):
            analysis.exact_two_feature_shapley(1., 2., 1., "observational")


class SummaryContract(unittest.TestCase):
    def test_observation_fields(self) -> None:
        self.assertEqual(set(analysis.observe(1, "x", "risk", .5)),
                         {"replicate", "condition", "metric", "value", "theory"})

    def test_reject_nonnumeric_observation(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observe(1, "x", "risk", float("nan"))

    def test_reject_empty_metric(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observe(1, "x", "", 1.)

    def test_reject_empty_condition(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observe(1, "", "risk", 1.)

    def test_reject_nonnumeric_theory(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observe(1, "x", "risk", 1., float("inf"))

    def test_summary_mean(self) -> None:
        data = [analysis.observe(0, "a", "x", 1.), analysis.observe(1, "a", "x", 3.)]
        self.assertEqual(analysis.summarize(data)[0]["mean"], 2.)

    def test_summary_se(self) -> None:
        data = [analysis.observe(0, "a", "x", 1.), analysis.observe(1, "a", "x", 3.)]
        self.assertAlmostEqual(float(analysis.summarize(data)[0]["se"]), 1.)

    def test_summary_normal_interval(self) -> None:
        data = [analysis.observe(0, "a", "x", 1.), analysis.observe(1, "a", "x", 3.)]
        one = analysis.summarize(data)[0]
        self.assertAlmostEqual(float(one["lower95"]), 2 - 1.96)
        self.assertAlmostEqual(float(one["upper95"]), 2 + 1.96)

    def test_summary_reject_empty(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([])

    def test_summary_reject_inconsistent_theory(self) -> None:
        a = analysis.observe(0, "x", "y", 1., 2.)
        b = analysis.observe(1, "x", "y", 1., 3.)
        with self.assertRaises(ValueError):
            analysis.summarize([a, b])

    def test_summary_multiple_conditions(self) -> None:
        vals = [analysis.observe(0, "a", "x", 1.),
                analysis.observe(0, "b", "x", 2.),
                analysis.observe(0, "a", "y", 3.)]
        self.assertEqual(len(analysis.summarize(vals)), 3)

    def test_lookup_value(self) -> None:
        summary = analysis.summarize([analysis.observe(0, "a", "x", 1.)])
        self.assertEqual(analysis.lookup(summary, "a", "x"), 1.)

    def test_lookup_missing_raises(self) -> None:
        summary = analysis.summarize([analysis.observe(0, "a", "x", 1.)])
        with self.assertRaises(KeyError):
            analysis.lookup(summary, "no", "x")

    def test_write_csv_fails_on_inconsistent_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(directory) / "a.csv", [{"a": 1}, {"b": 2}])

    def test_write_csv_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "a.csv"
            analysis.write_csv(target, [analysis.observe(1, "x", "y", 3.)])
            self.assertEqual(len(rows(target)), 1)


class ExperimentBehavior(unittest.TestCase):
    def test_conditional_has_ten_observations(self) -> None:
        self.assertEqual(len(analysis.conditional_study(analysis.stream(8, "conditional", 0), 0, True)), 10)

    def test_proxy_has_twelve_observations(self) -> None:
        self.assertEqual(len(analysis.proxy_study(analysis.stream(8, "proxies", 0), 0, True)), 12)

    def test_impurity_has_six_observations(self) -> None:
        self.assertEqual(len(analysis.impurity_study(analysis.stream(8, "impurity", 0), 0, True)), 6)

    def test_shapley_has_twelve_observations(self) -> None:
        self.assertEqual(len(analysis.shapley_study(analysis.stream(8, "shapley", 0), 0, True)), 12)

    def test_interaction_has_five_observations(self) -> None:
        self.assertEqual(len(analysis.interaction_study(analysis.stream(8, "interaction", 0), 0, True)), 5)

    def test_leakage_has_four_observations(self) -> None:
        self.assertEqual(len(analysis.leakage_study(analysis.stream(8, "leakage", 0), 0, True)), 4)

    def test_shift_has_six_observations(self) -> None:
        self.assertEqual(len(analysis.shift_study(analysis.stream(8, "shift", 0), 0, True)), 6)

    def test_subgroup_has_twelve_observations(self) -> None:
        self.assertEqual(len(analysis.subgroup_study(analysis.stream(8, "subgroup", 0), 0, True)), 12)

    def test_selection_has_nine_observations(self) -> None:
        self.assertEqual(len(analysis.selection_study(analysis.stream(8, "selection", 0), 0, True)), 9)

    def test_expected_study_names(self) -> None:
        self.assertEqual(tuple(analysis.STUDIES), analysis.EXPERIMENTS)

    def test_nine_unique_figure_names(self) -> None:
        self.assertEqual(len(set(analysis.FIGURES)), 9)

    def test_full_repetition_floor(self) -> None:
        self.assertTrue(all(n >= 18 for n in analysis.REPETITIONS.values()))

    def test_conditional_theory_decreases_with_correlation(self) -> None:
        one = analysis.conditional_study(analysis.stream(7, "conditional", 0), 0, True)
        vals = [float(r["theory"]) for r in one if r["metric"] == "conditional_delta"]
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])))

    def test_shapley_unused_interventional_zero(self) -> None:
        vals = analysis.shapley_study(analysis.stream(7, "shapley", 0), 0, True)
        self.assertTrue(all(r["value"] == 0.
                            for r in vals if r["metric"] == "interventional_abs_phi2"))

    def test_xor_tree_learns_interaction(self) -> None:
        vals = analysis.interaction_study(analysis.stream(7, "interaction", 0), 0, True)
        score = next(float(r["value"]) for r in vals if r["metric"] == "test_r2")
        self.assertGreater(score, .6)

    def test_leakage_offline_beats_deployment(self) -> None:
        vals = analysis.leakage_study(analysis.stream(7, "leakage", 0), 0, True)
        d = {str(r["condition"]): float(r["value"]) for r in vals}
        self.assertGreater(d["leak_offline"], d["leak_deployed"] + .5)

    def test_clean_availability_same_both_scenarios(self) -> None:
        vals = analysis.leakage_study(analysis.stream(7, "leakage", 0), 0, True)
        d = {str(r["condition"]): float(r["value"]) for r in vals}
        self.assertEqual(d["clean_offline"], d["clean_deployed"])

    def test_shift_theory_ranks_reverse(self) -> None:
        vals = analysis.shift_study(analysis.stream(7, "shift", 0), 0, True)
        d = {(str(r["condition"]), str(r["metric"])): float(r["theory"]) for r in vals}
        self.assertLess(d["low_x1", "x1_delta"], d["low_x1", "x2_delta"])
        self.assertGreater(d["high_x1", "x1_delta"], d["high_x1", "x2_delta"])

    def test_minor_feature_local_theory_greater(self) -> None:
        vals = analysis.subgroup_study(analysis.stream(7, "subgroup", 0), 0, True)
        d = {(str(r["condition"]), str(r["metric"])): float(r["theory"]) for r in vals}
        self.assertGreater(d["0.05", "minority_x2_delta"],
                           d["0.05", "global_x2_delta"] * 10)


class ArchivedEvidence(unittest.TestCase):
    def test_nine_pairs_of_canonical_csv(self) -> None:
        self.assertEqual(len(list((ROOT / "results").glob("*.csv"))), 18)

    def test_nine_original_png(self) -> None:
        self.assertEqual(len(list((ROOT / "figures").glob("*.png"))), 9)

    def test_all_figures_nontrivial_png(self) -> None:
        for name in analysis.FIGURES:
            p = ROOT / "figures" / f"{name}.png"
            with self.subTest(name=name):
                self.assertGreater(p.stat().st_size, 12000)
                self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_each_rep_table_expected_count(self) -> None:
        expected = {name: sum(len(analysis.STUDIES[name](
                                analysis.stream(5, name, 0), 0, True))
                                for _ in range(1)) * analysis.REPETITIONS[name]
                    for name in analysis.EXPERIMENTS}
        for name in analysis.EXPERIMENTS:
            with self.subTest(name=name):
                self.assertEqual(len(rows(ROOT / "results" / f"{name}_replicates.csv")),
                                 expected[name])

    def test_all_summary_n_match_full_design(self) -> None:
        for name in analysis.EXPERIMENTS:
            for row in rows(ROOT / "results" / f"{name}_summary.csv"):
                with self.subTest(name=name, metric=row["metric"]):
                    self.assertEqual(int(row["n"]), analysis.REPETITIONS[name])

    def test_all_numeric_summary_finite(self) -> None:
        for name in analysis.EXPERIMENTS:
            for row in rows(ROOT / "results" / f"{name}_summary.csv"):
                with self.subTest(name=name, condition=row["condition"], metric=row["metric"]):
                    for field in ("mean", "sd", "se", "lower95", "upper95"):
                        self.assertTrue(math.isfinite(float(row[field])))

    def test_all_summary_interval_contain_mean(self) -> None:
        for name in analysis.EXPERIMENTS:
            for row in rows(ROOT / "results" / f"{name}_summary.csv"):
                with self.subTest(name=name, metric=row["metric"]):
                    self.assertLessEqual(float(row["lower95"]), float(row["mean"]))
                    self.assertLessEqual(float(row["mean"]), float(row["upper95"]))

    def test_theory_close_marginal_correlation(self) -> None:
        summary = rows(ROOT / "results" / "conditional_summary.csv")
        vals = {r["condition"]: r for r in summary if r["metric"] == "marginal_delta"}
        self.assertAlmostEqual(float(vals["0.98"]["mean"]), 4.5, delta=.2)

    def test_theory_close_conditional_high_rho(self) -> None:
        summary = rows(ROOT / "results" / "conditional_summary.csv")
        vals = {r["condition"]: r for r in summary if r["metric"] == "conditional_delta"}
        self.assertAlmostEqual(float(vals["0.98"]["mean"]), .1782, delta=.03)

    def test_proxy_retrain_less_than_permute_at_high_rho(self) -> None:
        summary = rows(ROOT / "results" / "proxies_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in summary}
        self.assertGreater(d["0.1", "fixed_permutation_delta"],
                           d["0.1", "refit_ablation_delta"] * 10)

    def test_noise_continuous_gets_more_impurity(self) -> None:
        summary = rows(ROOT / "results" / "impurity_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in summary}
        self.assertGreater(d["noise_continuous", "mdi"],
                           d["noise_binary", "mdi"] * 3)

    def test_noise_permutation_near_zero(self) -> None:
        summary = rows(ROOT / "results" / "impurity_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in summary}
        self.assertLess(abs(d["noise_continuous", "heldout_delta"]), .08)
        self.assertLess(abs(d["noise_binary", "heldout_delta"]), .08)

    def test_interaction_recovered(self) -> None:
        summary = rows(ROOT / "results" / "interaction_summary.csv")
        d = {r["metric"]: float(r["mean"]) for r in summary}
        self.assertLess(d["abs_corr_x1"], .07)
        self.assertGreater(d["permutation_delta_x1"], 1.5)
        self.assertGreater(d["test_r2"], .75)

    def test_deployment_leakage_failure(self) -> None:
        summary = rows(ROOT / "results" / "leakage_summary.csv")
        d = {r["condition"]: float(r["mean"]) for r in summary}
        self.assertGreater(d["leak_offline"], .95)
        self.assertLess(d["leak_deployed"], 0.)
        self.assertGreater(d["clean_deployed"], .65)

    def test_evaluation_shift_reverses_rank(self) -> None:
        summary = rows(ROOT / "results" / "shift_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in summary}
        self.assertLess(d["low_x1", "x1_delta"], d["low_x1", "x2_delta"])
        self.assertGreater(d["high_x1", "x1_delta"], d["high_x1", "x2_delta"])

    def test_subgroup_local_reliance(self) -> None:
        summary = rows(ROOT / "results" / "subgroup_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in summary}
        self.assertGreater(d["0.05", "minority_x2_delta"],
                           d["0.05", "global_x2_delta"] * 12)

    def test_selected_winner_shrinks_out_of_sample(self) -> None:
        summary = rows(ROOT / "results" / "selection_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in summary}
        self.assertGreater(d["80", "selected_holdout_delta"], .035)
        self.assertLess(abs(d["80", "independent_delta"]), .02)


class NotebookContract(unittest.TestCase):
    def test_notebook_is_v4(self) -> None:
        self.assertEqual(notebook_builder.notebook()["nbformat"], 4)

    def test_notebook_minor5_has_ids(self) -> None:
        notebook = notebook_builder.notebook()
        self.assertEqual(notebook["nbformat_minor"], 5)
        self.assertTrue(all("id" in cell for cell in notebook["cells"]))

    def test_notebook_all_ids_unique(self) -> None:
        ids = [cell["id"] for cell in notebook_builder.notebook()["cells"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_code_outputs_unexecuted(self) -> None:
        for cell in notebook_builder.notebook()["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_notebook_archived_matches_builder(self) -> None:
        saved = json.loads((ROOT / notebook_builder.FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(saved, notebook_builder.notebook())

    def test_notebook_mentions_all_evidence(self) -> None:
        all_text = json.dumps(notebook_builder.notebook())
        self.assertIn("nine", all_text)
        self.assertIn("eighteen", all_text)
        self.assertIn("conditional", all_text)

    @unittest.skipIf(nbformat is None, "nbformat not installed locally")
    def test_nbformat_validation(self) -> None:
        nbformat.validate(nbformat.read(ROOT / notebook_builder.FILENAME, as_version=4))

    @unittest.skipUnless(os.getenv("RUN_NOTEBOOK_EXECUTION") == "1",
                         "set RUN_NOTEBOOK_EXECUTION=1 to execute a fresh kernel")
    def test_fresh_kernel_execution(self) -> None:
        if nbformat is None or NotebookClient is None:
            self.fail("nbformat and nbclient required for fresh-kernel execution")
        nb = nbformat.read(ROOT / notebook_builder.FILENAME, as_version=4)
        NotebookClient(nb, timeout=420, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        self.assertTrue(all(cell["execution_count"] is not None
                            for cell in nb["cells"] if cell["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
