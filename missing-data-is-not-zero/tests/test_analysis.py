"""Numerical identities, mechanism boundaries, archived evidence, notebook."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

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


def csv_rows(path: Path) -> list[dict[str, str]]:
    """Read versioned evidence, retaining empty theory cells."""

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class InputValidation(unittest.TestCase):
    def test_vector_valid_float(self) -> None:
        np.testing.assert_array_equal(analysis.check_vector(np.arange(4.)), np.arange(4.))

    def test_vector_casts_integer(self) -> None:
        self.assertEqual(analysis.check_vector(np.arange(4)).dtype, float)

    def test_vector_rejects_scalar(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.asarray(2.))

    def test_vector_rejects_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.ones((4, 1)))

    def test_vector_rejects_two_values(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.arange(2.))

    def test_vector_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.array([1., 2., np.nan]))

    def test_vector_rejects_infinity(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.array([1., 2., np.inf]))

    def test_mask_accepts_bool(self) -> None:
        r = np.array([True, True, True, False, False, False])
        self.assertEqual(analysis.check_mask(r, 6).dtype, bool)

    def test_mask_rejects_integer(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_mask(np.array([1, 1, 1, 0, 0, 0]), 6)

    def test_mask_rejects_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_mask(np.ones((6, 1), dtype=bool), 6)

    def test_mask_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_mask(np.array([True]*3+[False]*3), 7)

    def test_mask_rejects_too_few_observed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_mask(np.array([True, True]+[False]*4), 6)

    def test_mask_rejects_too_few_missing(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_mask(np.array([True]*5+[False]), 6)

    def test_run_rejects_bad_root(self) -> None:
        with self.assertRaises(TypeError):
            analysis.run(analysis.Design(root="/tmp", quick=True))  # type: ignore[arg-type]

    def test_run_rejects_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(root=Path("/tmp"), quick=True, seed=-1))

    def test_stream_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "wrong", 0)

    def test_stream_rejects_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(-1, "zero", 0)

    def test_stream_rejects_negative_rep(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "zero", -1)

    def test_stream_repeat_same_sequence(self) -> None:
        a = analysis.stream(999, "zero", 3).normal(size=30)
        b = analysis.stream(999, "zero", 3).normal(size=30)
        np.testing.assert_array_equal(a, b)

    def test_stream_differs_across_studies(self) -> None:
        a = analysis.stream(999, "zero", 3).normal(size=30)
        b = analysis.stream(999, "variance", 3).normal(size=30)
        self.assertFalse(np.array_equal(a, b))

    def test_stream_differs_across_reps(self) -> None:
        a = analysis.stream(999, "zero", 3).normal(size=30)
        b = analysis.stream(999, "zero", 4).normal(size=30)
        self.assertFalse(np.array_equal(a, b))


class EstimationIdentities(unittest.TestCase):
    def setUp(self) -> None:
        self.y = np.array([1., 2., 3., 4., 5., 6.])
        self.r = np.array([True, False, True, False, True, False])
        self.pi = np.full(6, .5)

    def test_hajek_constant_probability_equals_cc(self) -> None:
        self.assertAlmostEqual(analysis.hajek_mean(self.y, self.r, self.pi), 3.)

    def test_hajek_allows_unobserved_nan(self) -> None:
        y = self.y.copy()
        y[~self.r] = np.nan
        self.assertAlmostEqual(analysis.hajek_mean(y, self.r, self.pi), 3.)

    def test_hajek_rejects_observed_nan(self) -> None:
        y = self.y.copy()
        y[0] = np.nan
        with self.assertRaises(ValueError):
            analysis.hajek_mean(y, self.r, self.pi)

    def test_hajek_rejects_bad_probability_length(self) -> None:
        with self.assertRaises(ValueError):
            analysis.hajek_mean(self.y, self.r, np.ones(4))

    def test_hajek_rejects_zero_probability(self) -> None:
        pi = self.pi.copy()
        pi[0] = 0.
        with self.assertRaises(ValueError):
            analysis.hajek_mean(self.y, self.r, pi)

    def test_hajek_rejects_over_one_probability(self) -> None:
        pi = self.pi.copy()
        pi[0] = 1.1
        with self.assertRaises(ValueError):
            analysis.hajek_mean(self.y, self.r, pi)

    def test_hajek_rejects_negative_floor(self) -> None:
        with self.assertRaises(ValueError):
            analysis.hajek_mean(self.y, self.r, self.pi, lower=-.1)

    def test_hajek_rejects_floor_one(self) -> None:
        with self.assertRaises(ValueError):
            analysis.hajek_mean(self.y, self.r, self.pi, lower=1.)

    def test_hajek_weights_change_mean(self) -> None:
        pi = np.array([.1, .5, .5, .5, .9, .5])
        self.assertLess(analysis.hajek_mean(self.y, self.r, pi), 3.)

    def test_hajek_clipping_changes_extreme_weights(self) -> None:
        pi = np.array([.01, .5, .5, .5, .9, .5])
        raw = analysis.hajek_mean(self.y, self.r, pi)
        clipped = analysis.hajek_mean(self.y, self.r, pi, lower=.2)
        self.assertGreater(clipped, raw)

    def test_ess_uniform_weights_equals_n(self) -> None:
        self.assertAlmostEqual(analysis.effective_sample_size(np.ones(20)), 20.)

    def test_ess_scale_invariant(self) -> None:
        w = np.array([1., 2., 3., 5.])
        self.assertAlmostEqual(analysis.effective_sample_size(w),
                               analysis.effective_sample_size(10*w))

    def test_ess_bounded_by_n(self) -> None:
        w = np.array([1., 2., 3., 5.])
        self.assertLessEqual(analysis.effective_sample_size(w), len(w))

    def test_ess_declines_when_one_weight_dominates(self) -> None:
        w = np.array([1., 1., 1., 100.])
        self.assertLess(analysis.effective_sample_size(w), 2.)

    def test_ess_rejects_nonpositive(self) -> None:
        with self.assertRaises(ValueError):
            analysis.effective_sample_size(np.array([1., 0., 2.]))

    def test_ess_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.effective_sample_size(np.array([1., 2., np.nan]))

    def test_mi_returns_positive_variance(self) -> None:
        rng = np.random.default_rng(2)
        mu, var = analysis.normal_multiple_imputation(rng, np.arange(10.), 5)
        self.assertTrue(math.isfinite(mu))
        self.assertGreater(var, 0.)

    def test_mi_is_seed_reproducible(self) -> None:
        x = np.arange(10.)
        a = analysis.normal_multiple_imputation(np.random.default_rng(13), x, 5)
        b = analysis.normal_multiple_imputation(np.random.default_rng(13), x, 5)
        self.assertEqual(a, b)

    def test_mi_rejects_zero_missing(self) -> None:
        with self.assertRaises(ValueError):
            analysis.normal_multiple_imputation(np.random.default_rng(1), np.arange(5.), 0)

    def test_mi_rejects_too_few_draws(self) -> None:
        with self.assertRaises(ValueError):
            analysis.normal_multiple_imputation(np.random.default_rng(1), np.arange(5.), 2, m=1)

    def test_mi_rejects_nonpositive_variance(self) -> None:
        with self.assertRaises(ValueError):
            analysis.normal_multiple_imputation(np.random.default_rng(1), np.arange(5.), 2,
                                                known_variance=0.)

    def test_mi_many_draws_center_near_observed_mean(self) -> None:
        x = np.array([2., 3., 4., 5., 6.])
        mu, _ = analysis.normal_multiple_imputation(np.random.default_rng(10),
                                                     x, 5, m=4000)
        self.assertAlmostEqual(mu, 4., delta=.06)

    def test_mi_proper_variance_exceeds_complete_data_variance(self) -> None:
        x = np.linspace(1., 3., 15)
        _, var = analysis.normal_multiple_imputation(np.random.default_rng(10),
                                                     x, 20, m=400)
        self.assertGreater(var, 1 / 35)


class SummaryContract(unittest.TestCase):
    def test_observation_fields(self) -> None:
        self.assertEqual(set(analysis.observation(1, "x", "y", 2.)),
                         {"replicate", "condition", "metric", "value", "theory"})

    def test_observation_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(1, "x", "y", math.nan)

    def test_observation_rejects_bad_theory(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(1, "x", "y", 1., math.inf)

    def test_observation_rejects_empty_condition(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(1, "", "y", 1.)

    def test_observation_rejects_empty_metric(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(1, "x", "", 1.)

    def test_summary_mean(self) -> None:
        x = [analysis.observation(0, "a", "b", 2.),
             analysis.observation(1, "a", "b", 4.)]
        self.assertEqual(analysis.summarize(x)[0]["mean"], 3.)

    def test_summary_se(self) -> None:
        x = [analysis.observation(0, "a", "b", 2.),
             analysis.observation(1, "a", "b", 4.)]
        self.assertAlmostEqual(float(analysis.summarize(x)[0]["se"]), 1.)

    def test_summary_interval_contains_mean(self) -> None:
        x = [analysis.observation(0, "a", "b", 2.),
             analysis.observation(1, "a", "b", 4.)]
        y = analysis.summarize(x)[0]
        self.assertLess(float(y["lower95"]), float(y["mean"]))
        self.assertGreater(float(y["upper95"]), float(y["mean"]))

    def test_summary_one_sample_sd_zero(self) -> None:
        x = [analysis.observation(0, "a", "b", 3.)]
        self.assertEqual(analysis.summarize(x)[0]["sd"], 0.)

    def test_summary_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([])

    def test_summary_mixed_theory_rejected(self) -> None:
        x = [analysis.observation(0, "a", "b", 3., 4.),
             analysis.observation(1, "a", "b", 3., 5.)]
        with self.assertRaises(ValueError):
            analysis.summarize(x)

    def test_lookup_metric(self) -> None:
        x = analysis.summarize([analysis.observation(0, "a", "b", 7.)])
        self.assertEqual(analysis.lookup(x, "a", "b"), 7.)

    def test_lookup_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            analysis.lookup([], "a", "b")

    def test_csv_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "more" / "x.csv"
            analysis.write_csv(target, [analysis.observation(1, "x", "y", 2.)])
            self.assertEqual(len(csv_rows(target)), 1)

    def test_csv_inconsistent_schema_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(directory) / "x.csv", [{"a": 1}, {"b": 2}])

    def test_csv_empty_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(directory) / "x.csv", [])


class StudyBehavior(unittest.TestCase):
    def test_nine_experiments(self) -> None:
        self.assertEqual(len(analysis.EXPERIMENTS), 9)

    def test_study_names_equal_mapping(self) -> None:
        self.assertEqual(tuple(analysis.STUDIES), analysis.EXPERIMENTS)

    def test_unique_figure_names(self) -> None:
        self.assertEqual(len(set(analysis.FIGURES)), 9)

    def test_repetition_floor(self) -> None:
        self.assertGreaterEqual(min(analysis.REPETITIONS.values()), 24)

    def test_coverage_study_has_300_repeats(self) -> None:
        self.assertEqual(analysis.REPETITIONS["intervals"], 300)

    def test_zero_has_nine_metrics(self) -> None:
        self.assertEqual(len(analysis.zero_study(analysis.stream(8, "zero", 0), 0, True)), 9)

    def test_mechanisms_has_nine_metrics(self) -> None:
        self.assertEqual(len(analysis.mechanisms_study(analysis.stream(8, "mechanisms", 0), 0, True)), 9)

    def test_mar_recovery_has_six_metrics(self) -> None:
        self.assertEqual(len(analysis.mar_recovery_study(analysis.stream(8, "mar_recovery", 0), 0, True)), 6)

    def test_variance_has_twelve_metrics(self) -> None:
        self.assertEqual(len(analysis.variance_study(analysis.stream(8, "variance", 0), 0, True)), 12)

    def test_intervals_has_twenty_seven_metrics(self) -> None:
        self.assertEqual(len(analysis.intervals_study(analysis.stream(8, "intervals", 0), 0, True)), 27)

    def test_conditional_has_eight_metrics(self) -> None:
        self.assertEqual(len(analysis.conditional_study(analysis.stream(8, "conditional", 0), 0, True)), 8)

    def test_sensitivity_has_eighteen_metrics(self) -> None:
        self.assertEqual(len(analysis.sensitivity_study(analysis.stream(8, "sensitivity", 0), 0, True)), 18)

    def test_indicator_has_four_metrics(self) -> None:
        self.assertEqual(len(analysis.indicator_study(analysis.stream(8, "indicator", 0), 0, True)), 4)

    def test_positivity_has_thirty_two_metrics(self) -> None:
        self.assertEqual(len(analysis.positivity_study(analysis.stream(8, "positivity", 0), 0, True)), 32)

    def test_zero_theory_declines(self) -> None:
        rs = analysis.zero_study(analysis.stream(8, "zero", 0), 0, True)
        theory = [float(r["theory"]) for r in rs if r["metric"] == "zero_filled_mean"]
        self.assertEqual(theory, [1.8, 1.4, .8])

    def test_mean_fill_variance_theory_declines(self) -> None:
        rs = analysis.variance_study(analysis.stream(8, "variance", 0), 0, True)
        th = [float(r["theory"]) for r in rs if r["metric"] == "mean_fill_variance"]
        self.assertTrue(all(a > b for a, b in zip(th, th[1:])))

    def test_interval_coverage_binary(self) -> None:
        rs = analysis.intervals_study(analysis.stream(8, "intervals", 0), 0, True)
        self.assertTrue(all(r["value"] in {0., 1.}
                            for r in rs if r["metric"] == "coverage"))

    def test_indicator_clean_unaffected_by_policy(self) -> None:
        rs = analysis.indicator_study(analysis.stream(8, "indicator", 0), 0, True)
        values = {str(r["condition"]): float(r["value"]) for r in rs}
        self.assertEqual(values["no_indicator_original"], values["no_indicator_shifted"])

    def test_indicator_original_beats_shifted(self) -> None:
        rs = analysis.indicator_study(analysis.stream(8, "indicator", 0), 0, True)
        values = {str(r["condition"]): float(r["value"]) for r in rs}
        self.assertGreater(values["indicator_original"], values["indicator_shifted"])

    def test_sensitivity_truth_identical_across_delta(self) -> None:
        rs = analysis.sensitivity_study(analysis.stream(8, "sensitivity", 0), 0, True)
        truths = [r["value"] for r in rs if r["metric"] == "full_population_mean"]
        self.assertEqual(len(set(truths)), 1)

    def test_positivity_ess_one_at_strength_zero(self) -> None:
        rs = analysis.positivity_study(analysis.stream(8, "positivity", 0), 0, True)
        one = next(r for r in rs if r["condition"] == "0.0" and r["metric"] == "relative_ess")
        self.assertAlmostEqual(float(one["value"]), 1.)


class ArchivedEvidence(unittest.TestCase):
    def test_eighteen_canonical_csv_files(self) -> None:
        self.assertEqual(len(list((ROOT / "results").glob("*.csv"))), 18)

    def test_nine_png_figures(self) -> None:
        self.assertEqual(len(list((ROOT / "figures").glob("*.png"))), 9)

    def test_png_signatures_and_sizes(self) -> None:
        for figure in analysis.FIGURES:
            with self.subTest(figure=figure):
                file = ROOT / "figures" / f"{figure}.png"
                self.assertGreater(file.stat().st_size, 12000)
                self.assertEqual(file.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_all_summary_counts_match_replicates(self) -> None:
        for name in analysis.EXPERIMENTS:
            for row in csv_rows(ROOT / "results" / f"{name}_summary.csv"):
                with self.subTest(study=name, metric=row["metric"]):
                    self.assertEqual(int(row["n"]), analysis.REPETITIONS[name])

    def test_all_rep_table_row_counts(self) -> None:
        for name in analysis.EXPERIMENTS:
            n_conditions = len(analysis.STUDIES[name](
                analysis.stream(3, name, 0), 0, True
            ))
            expected = analysis.REPETITIONS[name] * n_conditions
            with self.subTest(study=name):
                self.assertEqual(len(csv_rows(ROOT / "results" / f"{name}_replicates.csv")),
                                 expected)

    def test_all_summary_metrics_finite(self) -> None:
        for name in analysis.EXPERIMENTS:
            for row in csv_rows(ROOT / "results" / f"{name}_summary.csv"):
                with self.subTest(study=name, condition=row["condition"], metric=row["metric"]):
                    for field in ("mean", "sd", "se", "lower95", "upper95"):
                        self.assertTrue(math.isfinite(float(row[field])))

    def test_all_intervals_contain_mean(self) -> None:
        for name in analysis.EXPERIMENTS:
            for row in csv_rows(ROOT / "results" / f"{name}_summary.csv"):
                with self.subTest(study=name, metric=row["metric"]):
                    self.assertLessEqual(float(row["lower95"]), float(row["mean"]))
                    self.assertLessEqual(float(row["mean"]), float(row["upper95"]))

    def test_zero_fill_strong_bias_at_sixty_percent(self) -> None:
        rs = csv_rows(ROOT / "results" / "zero_summary.csv")
        a = {r["metric"]: float(r["mean"]) for r in rs if r["condition"] == "0.6"}
        self.assertLess(a["zero_filled_mean"], 1.)
        self.assertAlmostEqual(a["complete_case_mean"], 2., delta=.1)

    def test_mcar_complete_case_near_zero_bias(self) -> None:
        rs = csv_rows(ROOT / "results" / "mechanisms_summary.csv")
        score = next(float(r["mean"]) for r in rs
                     if r["condition"] == "MCAR" and r["metric"] == "cc_bias")
        self.assertLess(abs(score), .04)

    def test_mar_weighting_recovers_mean(self) -> None:
        rs = csv_rows(ROOT / "results" / "mechanisms_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in rs}
        self.assertLess(abs(d["MAR", "auxiliary_ipw_bias"]), .06)
        self.assertLess(d["MAR", "cc_bias"], -.2)

    def test_mnar_auxiliary_weighting_leaves_bias(self) -> None:
        rs = csv_rows(ROOT / "results" / "mechanisms_summary.csv")
        d = {(r["condition"], r["metric"]): float(r["mean"]) for r in rs}
        self.assertLess(d["MNAR", "auxiliary_ipw_bias"], -.2)

    def test_mar_regression_standardization_recovers(self) -> None:
        rs = csv_rows(ROOT / "results" / "mar_recovery_summary.csv")
        d = {r["condition"]: float(r["mean"]) for r in rs if r["metric"] == "error"}
        self.assertLess(abs(d["outcome_standardization"]), .05)
        self.assertLess(d["complete_case"], -.3)

    def test_mean_fill_variance_at_seventy_percent(self) -> None:
        rs = csv_rows(ROOT / "results" / "variance_summary.csv")
        d = {r["metric"]: float(r["mean"]) for r in rs if r["condition"] == "0.7"}
        self.assertAlmostEqual(d["mean_fill_variance"], .3, delta=.07)
        self.assertAlmostEqual(d["complete_case_variance"], 1., delta=.1)

    def test_mi_recovers_interval_coverage_at_high_missingness(self) -> None:
        rs = csv_rows(ROOT / "results" / "intervals_summary.csv")
        d = {r["condition"]: float(r["mean"]) for r in rs if r["metric"] == "coverage"}
        self.assertGreater(d["0.75:multiple"], .89)
        self.assertLess(d["0.75:single"], .50)

    def test_single_imputation_intervals_are_narrow(self) -> None:
        rs = csv_rows(ROOT / "results" / "intervals_summary.csv")
        d = {r["condition"]: float(r["mean"]) for r in rs if r["metric"] == "interval_width"}
        self.assertLess(d["0.50:single"], .6 * d["0.50:multiple"])

    def test_slope_versus_marginal_mean_under_mar(self) -> None:
        rs = csv_rows(ROOT / "results" / "conditional_summary.csv")
        d = {r["metric"]: float(r["mean"]) for r in rs if r["condition"] == "2.0"}
        self.assertLess(abs(d["slope_error"]), .05)
        self.assertLess(d["mean_error"], -.8)

    def test_mnar_oracle_delta_is_not_zero(self) -> None:
        rs = csv_rows(ROOT / "results" / "sensitivity_summary.csv")
        delta = next(float(r["mean"]) for r in rs
                     if r["metric"] == "oracle_pattern_delta")
        self.assertLess(delta, -.7)

    def test_mnar_delta_changes_sign_of_estimate(self) -> None:
        rs = csv_rows(ROOT / "results" / "sensitivity_summary.csv")
        d = {r["condition"]: float(r["mean"]) for r in rs
             if r["metric"] == "assumed_population_mean"}
        self.assertLess(d["-2.0"], 0.)
        self.assertGreater(d["0.0"], 0.)

    def test_indicator_collapse_on_policy_shift(self) -> None:
        rs = csv_rows(ROOT / "results" / "indicator_summary.csv")
        d = {r["condition"]: float(r["mean"]) for r in rs}
        self.assertGreater(d["indicator_original"], .5)
        self.assertLess(d["indicator_shifted"], 0.)

    def test_positivity_ess_decreases(self) -> None:
        rs = csv_rows(ROOT / "results" / "positivity_summary.csv")
        d = {r["condition"]: float(r["mean"]) for r in rs
             if r["metric"] == "relative_ess"}
        self.assertGreater(d["0.0"], d["2.4"] + .5)

    def test_positivity_clipping_bias(self) -> None:
        rs = csv_rows(ROOT / "results" / "positivity_summary.csv")
        d = {r["metric"]: float(r["mean"]) for r in rs if r["condition"] == "2.4"}
        self.assertLess(d["clipped_ipw_error"], -.08)


class NotebookContract(unittest.TestCase):
    def test_v4_format(self) -> None:
        self.assertEqual(notebook_builder.notebook()["nbformat"], 4)

    def test_v4_minor5(self) -> None:
        self.assertEqual(notebook_builder.notebook()["nbformat_minor"], 5)

    def test_unique_cell_ids(self) -> None:
        ids = [c["id"] for c in notebook_builder.notebook()["cells"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_cell_has_id(self) -> None:
        self.assertTrue(all("id" in c for c in notebook_builder.notebook()["cells"]))

    def test_code_cells_are_clean(self) -> None:
        for cell in notebook_builder.notebook()["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_archived_json_equals_builder(self) -> None:
        saved = json.loads((ROOT / notebook_builder.FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(saved, notebook_builder.notebook())

    def test_notebook_mentions_all_evidence(self) -> None:
        description = json.dumps(notebook_builder.notebook())
        self.assertIn("eighteen", description)
        self.assertIn("MCAR", description)
        self.assertIn("MNAR", description)

    @unittest.skipIf(nbformat is None, "nbformat unavailable locally")
    def test_nbformat_validates(self) -> None:
        nbformat.validate(nbformat.read(ROOT / notebook_builder.FILENAME, as_version=4))

    @unittest.skipUnless(os.getenv("RUN_NOTEBOOK_EXECUTION") == "1",
                         "fresh-kernel CI execution is opt-in locally")
    def test_fresh_kernel_executes(self) -> None:
        if nbformat is None or NotebookClient is None:
            self.fail("nbformat and nbclient are required for kernel execution")
        nb = nbformat.read(ROOT / notebook_builder.FILENAME, as_version=4)
        NotebookClient(nb, timeout=420, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        self.assertTrue(all(c["execution_count"] is not None
                            for c in nb["cells"] if c["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
