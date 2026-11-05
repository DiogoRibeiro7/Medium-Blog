"""Numerical invariants, estimand boundaries, evidence and fresh notebook CI."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

import analysis
import notebook_builder

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    nbformat = None
    NotebookClient = None


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class VectorAndSeedTests(unittest.TestCase):
    def test_accept_float_vector(self) -> None:
        np.testing.assert_array_equal(analysis.check_vector(np.arange(4.)), np.arange(4.))

    def test_cast_integer_vector(self) -> None:
        self.assertEqual(analysis.check_vector(np.arange(4)).dtype, float)

    def test_reject_scalar(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.asarray(2.))

    def test_reject_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.zeros((4, 1)))

    def test_reject_two_points(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.arange(2.))

    def test_reject_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.array([1., np.nan, 3.]))

    def test_reject_inf(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_vector(np.array([1., np.inf, 3.]))

    def test_stream_reproducible(self) -> None:
        np.testing.assert_array_equal(analysis.stream(4, "tails", 7).normal(size=100),
                                      analysis.stream(4, "tails", 7).normal(size=100))

    def test_stream_different_replicate(self) -> None:
        self.assertFalse(np.array_equal(analysis.stream(4, "tails", 7).normal(size=50),
                                        analysis.stream(4, "tails", 8).normal(size=50)))

    def test_stream_different_study(self) -> None:
        self.assertFalse(np.array_equal(analysis.stream(4, "tails", 7).normal(size=50),
                                        analysis.stream(4, "intervals", 7).normal(size=50)))

    def test_stream_reject_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(-1, "tails", 0)

    def test_stream_reject_negative_replicate(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "tails", -1)

    def test_stream_reject_unknown_study(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "unknown", 0)

    def test_run_reject_bad_root_type(self) -> None:
        with self.assertRaises(TypeError):
            analysis.run(analysis.Design(root="temp", quick=True))  # type: ignore[arg-type]

    def test_run_reject_bad_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(quick=True, seed=-1))

    def test_noncanonical_full_seed_requires_separate_directory(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(seed=analysis.SEED+1))

    def test_nine_unique_study_keys(self) -> None:
        self.assertEqual(len(set(analysis.EXPERIMENTS)), 9)

    def test_nine_unique_figure_keys(self) -> None:
        self.assertEqual(len(set(analysis.FIGURES)), 9)

    def test_all_studies_have_functions(self) -> None:
        self.assertEqual(set(analysis.EXPERIMENTS), set(analysis.STUDIES))

    def test_repetition_counts_positive(self) -> None:
        self.assertTrue(all(n >= 20 for n in analysis.REPETITIONS.values()))


class EstimatorMathTests(unittest.TestCase):
    def test_huber_constant(self) -> None:
        self.assertEqual(analysis.huber_location(np.full(10, 5.)), 5.)

    def test_huber_symmetric(self) -> None:
        self.assertAlmostEqual(analysis.huber_location(np.array([-9., -2., 0., 2., 9.])), 0.)

    def test_huber_translation_equivariance(self) -> None:
        x = np.asarray([-2., -.3, 0., .4, 18.])
        self.assertAlmostEqual(analysis.huber_location(x+7.),
                               analysis.huber_location(x)+7.)

    def test_huber_sign_reversal(self) -> None:
        x = np.asarray([-2., -.3, 0., .4, 18.])
        self.assertAlmostEqual(analysis.huber_location(-x), -analysis.huber_location(x))

    def test_huber_limits_single_shift(self) -> None:
        x = np.r_[np.zeros(99), 1e5]
        self.assertLess(analysis.huber_location(x), .02)
        self.assertGreater(x.mean(), 100.)

    def test_huber_reject_zero_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            analysis.huber_location(np.arange(5.), threshold=0.)

    def test_huber_reject_bad_scale(self) -> None:
        with self.assertRaises(ValueError):
            analysis.huber_location(np.arange(5.), scale=-1.)

    def test_huber_reject_nan_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            analysis.huber_location(np.arange(5.), threshold=float("nan"))

    def test_contamination_no_shift(self) -> None:
        a = analysis.contaminated_normal(np.random.default_rng(2), 50, 0., 20.)
        rng = np.random.default_rng(2)
        rng.random(50)
        b = rng.normal(size=50)
        np.testing.assert_allclose(a, b)

    def test_contamination_reject_negative_fraction(self) -> None:
        with self.assertRaises(ValueError):
            analysis.contaminated_normal(np.random.default_rng(1), 100, -.01, 2.)

    def test_contamination_reject_fraction_one(self) -> None:
        with self.assertRaises(ValueError):
            analysis.contaminated_normal(np.random.default_rng(1), 100, 1., 2.)

    def test_contamination_reject_small_n(self) -> None:
        with self.assertRaises(ValueError):
            analysis.contaminated_normal(np.random.default_rng(1), 2, .1, 2.)

    def test_mixture_median_zero(self) -> None:
        self.assertAlmostEqual(analysis.mixture_median(0., 20.), 0., places=10)

    def test_mixture_median_positive(self) -> None:
        self.assertGreater(analysis.mixture_median(.1, 20.), 0.)

    def test_mixture_median_below_mean_for_long_shift(self) -> None:
        self.assertLess(analysis.mixture_median(.1, 20.), 2.)

    def test_mixture_median_cdf_identity(self) -> None:
        m = analysis.mixture_median(.2, 7.)
        self.assertAlmostEqual(.8*norm.cdf(m)+.2*norm.cdf(m-7.), .5)

    def test_mixture_median_bad_probability(self) -> None:
        with self.assertRaises(ValueError):
            analysis.mixture_median(1., 20.)

    def test_mixture_median_bad_shift(self) -> None:
        with self.assertRaises(ValueError):
            analysis.mixture_median(.1, 0.)

    def test_median_interval_order(self) -> None:
        lo, hi = analysis.sign_median_interval(np.linspace(-5, 5, 100))
        self.assertLess(lo, hi)

    def test_median_interval_contains_symmetric_zero(self) -> None:
        lo, hi = analysis.sign_median_interval(np.linspace(-5, 5, 100))
        self.assertLessEqual(lo, 0.)
        self.assertGreaterEqual(hi, 0.)

    def test_median_sign_exact_tail(self) -> None:
        n = 80
        k = int(binom.ppf(.025, n, .5))
        self.assertLessEqual(float(binom.cdf(k-1, n, .5)), .025)

    def test_median_interval_reject_bad_alpha(self) -> None:
        with self.assertRaises(ValueError):
            analysis.sign_median_interval(np.arange(50.), alpha=0.)

    def test_median_interval_small_sample(self) -> None:
        with self.assertRaises(ValueError):
            analysis.sign_median_interval(np.arange(3.))

    def test_ols_exact_line(self) -> None:
        intercept, slope = analysis.ols_fit(np.arange(15.), 3.+2.*np.arange(15.))
        self.assertAlmostEqual(intercept, 3.)
        self.assertAlmostEqual(slope, 2.)

    def test_ols_reject_equal_x(self) -> None:
        with self.assertRaises(ValueError):
            analysis.ols_fit(np.ones(10), np.arange(10.))

    def test_ols_reject_unequal_length(self) -> None:
        with self.assertRaises(ValueError):
            analysis.ols_fit(np.arange(10.), np.arange(11.))

    def test_hat_leverages_sum_two(self) -> None:
        x = np.linspace(-3, 3, 30)
        h, _ = analysis.regression_diagnostics(x, 2.*x+np.sin(x))
        self.assertAlmostEqual(float(h.sum()), 2.)

    def test_hat_leverage_bounds(self) -> None:
        x = np.linspace(-3, 3, 30)
        h, _ = analysis.regression_diagnostics(x, 2.*x+np.sin(x))
        self.assertTrue(np.all((h > 0) & (h < 1)))

    def test_cooks_nonnegative(self) -> None:
        x = np.linspace(-3, 3, 30)
        _, d = analysis.regression_diagnostics(x, 2.*x+np.sin(x))
        self.assertTrue(np.all(d >= 0))

    def test_cooks_one_point_can_exceed_one(self) -> None:
        x = np.r_[np.linspace(-2, 2, 70), 12.]
        y = np.r_[1.+2.*x[:-1]+.03*np.sin(x[:-1]), -20.]
        _, d = analysis.regression_diagnostics(x, y)
        self.assertGreater(d[-1], 1.)

    def test_regression_diagnostics_reject_short(self) -> None:
        with self.assertRaises(ValueError):
            analysis.regression_diagnostics(np.arange(4.), np.arange(4.))

    def test_regression_diagnostics_reject_exact_line(self) -> None:
        with self.assertRaises(ValueError):
            analysis.regression_diagnostics(np.arange(20.), 1.+2.*np.arange(20.))

    def test_log_normal_es_greater_than_quantile(self) -> None:
        q, es = analysis.expected_shortfall_lognormal(.99, 1.1)
        self.assertGreater(es, q)

    def test_log_normal_quantile_identity(self) -> None:
        q, _ = analysis.expected_shortfall_lognormal(.99, 1.1)
        self.assertAlmostEqual(norm.cdf(np.log(q)/1.1), .99)

    def test_log_normal_invalid_p(self) -> None:
        with self.assertRaises(ValueError):
            analysis.expected_shortfall_lognormal(1., 1.1)

    def test_log_normal_invalid_scale(self) -> None:
        with self.assertRaises(ValueError):
            analysis.expected_shortfall_lognormal(.99, 0.)

    def test_log_normal_reject_infinite_scale(self) -> None:
        with self.assertRaises(ValueError):
            analysis.expected_shortfall_lognormal(.99, float("inf"))

    def test_log_normal_reject_nan_scale(self) -> None:
        with self.assertRaises(ValueError):
            analysis.expected_shortfall_lognormal(.99, float("nan"))

    def test_distance_outputs_positive(self) -> None:
        rng = np.random.default_rng(0)
        scores = analysis.population_distances(rng.normal(size=(50, 2)), False)
        self.assertTrue(np.all(scores >= 0))

    def test_distance_center_zero_under_symmetry(self) -> None:
        x = np.asarray([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]*15)
        d = analysis.population_distances(x, False)
        self.assertAlmostEqual(float(np.mean(d)), 2*(len(x)-1)/len(x))

    def test_distance_bad_dimension(self) -> None:
        with self.assertRaises(ValueError):
            analysis.population_distances(np.ones((30, 3)), False)

    def test_distance_nonfinite(self) -> None:
        x = np.zeros((30, 2))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            analysis.population_distances(x, False)

    def test_classical_distance_reject_singular_covariance(self) -> None:
        x = np.arange(30., dtype=float)
        with self.assertRaises(ValueError):
            analysis.population_distances(np.column_stack((x, 2*x)), False)

    def test_mcd_distance_reject_singular_covariance(self) -> None:
        x = np.arange(30., dtype=float)
        with self.assertRaises(ValueError):
            analysis.population_distances(np.column_stack((x, 2*x)), True)

    def test_normal_75th_absolute_quantile_is_not_signed_quantile(self) -> None:
        # P(|Z| <= q) = 2 Phi(q) - 1, so q=.875 normal quantile.
        q_abs = norm.ppf(.875)
        self.assertAlmostEqual(q_abs, 1.150349380376, places=10)
        self.assertAlmostEqual(float(2*norm.cdf(q_abs)-1), .75)
        self.assertGreater(q_abs, norm.ppf(.75))

    def test_threshold_exact_count(self) -> None:
        flags = analysis.threshold_alert(np.arange(100.), .05)
        self.assertEqual(int(flags.sum()), 5)

    def test_threshold_largest_selected(self) -> None:
        flags = analysis.threshold_alert(np.array([1., 100., 2., 3., 4.]), .2)
        self.assertEqual(np.where(flags)[0].tolist(), [1])

    def test_threshold_reject_zero_budget(self) -> None:
        with self.assertRaises(ValueError):
            analysis.threshold_alert(np.arange(10.), 0.)

    def test_threshold_reject_budget_one(self) -> None:
        with self.assertRaises(ValueError):
            analysis.threshold_alert(np.arange(10.), 1.)


class SummarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [analysis.observation(i, "a", "m", val, 2.)
                     for i, val in enumerate((1., 2., 3., 4.))]

    def test_mean(self) -> None:
        self.assertAlmostEqual(analysis.summarize(self.rows)[0]["mean"], 2.5)

    def test_standard_deviation(self) -> None:
        self.assertAlmostEqual(analysis.summarize(self.rows)[0]["sd"],
                               math.sqrt(5/3))

    def test_standard_error(self) -> None:
        self.assertAlmostEqual(analysis.summarize(self.rows)[0]["se"],
                               math.sqrt(5/3)/2)

    def test_theory_preserved(self) -> None:
        self.assertEqual(analysis.summarize(self.rows)[0]["theory"], 2.)

    def test_95_interval_order(self) -> None:
        row = analysis.summarize(self.rows)[0]
        self.assertLess(row["lower95"], row["mean"])
        self.assertGreater(row["upper95"], row["mean"])

    def test_lookup_exact(self) -> None:
        self.assertAlmostEqual(analysis.lookup(analysis.summarize(self.rows), "a", "m"), 2.5)

    def test_lookup_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            analysis.lookup(analysis.summarize(self.rows), "wrong", "m")

    def test_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([])

    def test_inconsistent_theory_rejected(self) -> None:
        bad = self.rows + [analysis.observation(5, "a", "m", 5., 3.)]
        with self.assertRaises(ValueError):
            analysis.summarize(bad)

    def test_duplicate_replication_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize(self.rows + [self.rows[0]])

    def test_observation_nonfinite_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "a", "m", float("nan"))

    def test_observation_empty_condition_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "", "m", 0.)

    def test_csv_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.write_csv(ROOT/"irrelevant.csv", [])


class StudyContractTests(unittest.TestCase):
    def test_single_replication_each_study_produces_finite(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = analysis.STUDIES[name](analysis.stream(4, name, 1), 1, True)
                self.assertTrue(rows)
                self.assertTrue(all(np.isfinite(r["value"]) for r in rows))

    def test_each_study_keys_unique_per_rep(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = analysis.STUDIES[name](analysis.stream(4, name, 1), 1, True)
                keys = [(r["condition"], r["metric"]) for r in rows]
                self.assertEqual(len(keys), len(set(keys)))

    def test_each_study_deterministic(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                a = analysis.STUDIES[name](analysis.stream(999, name, 0), 0, True)
                b = analysis.STUDIES[name](analysis.stream(999, name, 0), 0, True)
                self.assertEqual(a, b)

    def test_contamination_mean_theory(self) -> None:
        rows = analysis.contamination_study(analysis.stream(1, "contamination", 0), 0, True)
        row = [r for r in rows if r["condition"] == "0.20" and r["metric"] == "mean"]
        self.assertEqual(row[0]["theory"], 2.)

    def test_tails_all_location_errors_nonnegative(self) -> None:
        rows = analysis.tails_study(analysis.stream(1, "tails", 0), 0, True)
        self.assertTrue(all(r["value"] >= 0 for r in rows))

    def test_tails_metrics_are_explicit_about_aggregation(self) -> None:
        rows = analysis.tails_study(analysis.stream(1, "tails", 0), 0, True)
        self.assertTrue(all(str(r["metric"]).startswith("q75_abs_error_") for r in rows))

    def test_interval_exposes_two_distinct_targets(self) -> None:
        rows = analysis.intervals_study(analysis.stream(1, "intervals", 0), 0, True)
        self.assertEqual(set(r["condition"].split(":")[-1] for r in rows),
                         {"mean_student_t", "median_sign"})

    def test_interval_coverage_binary(self) -> None:
        rows = analysis.intervals_study(analysis.stream(1, "intervals", 0), 0, True)
        self.assertTrue(all(r["value"] in (0., 1.) for r in rows if r["metric"] == "coverage"))

    def test_regression_slope_and_intercept(self) -> None:
        rows = analysis.regression_study(analysis.stream(1, "regression", 0), 0, True)
        self.assertTrue(any("slope_error" in r["metric"] for r in rows))
        self.assertTrue(any("intercept_error" in r["metric"] for r in rows))

    def test_leverage_three_contexts(self) -> None:
        rows = analysis.leverage_study(analysis.stream(1, "leverage", 0), 0, True)
        self.assertEqual({r["condition"] for r in rows},
                         {"ordinary", "on_line", "off_line"})

    def test_multivariate_contrast(self) -> None:
        rows = analysis.multivariate_study(analysis.stream(1, "multivariate", 0), 0, True)
        self.assertEqual({r["metric"].split("_")[0] for r in rows},
                         {"classical", "mcd"})

    def test_minority_no_observed_error_label(self) -> None:
        rows = analysis.minority_study(analysis.stream(1, "minority", 0), 0, True)
        self.assertTrue(all("error_rate" not in str(r["metric"]) for r in rows))

    def test_minority_score_cutoff_is_experimental(self) -> None:
        rows = analysis.minority_study(analysis.stream(1, "minority", 0), 0, True)
        cutoffs = [r["value"] for r in rows if r["metric"] == "score_cutoff"]
        self.assertEqual(len(cutoffs), 4)
        self.assertTrue(all(float(v) > 0 for v in cutoffs))
        self.assertGreater(len(set(cutoffs)), 1)

    def test_tail_methods(self) -> None:
        rows = analysis.tail_risk_study(analysis.stream(1, "tail_risk", 0), 0, True)
        self.assertEqual({r["condition"] for r in rows},
                         {"raw", "drop_top_1pct", "winsor_top_1pct"})

    def test_shift_contains_both_thresholds(self) -> None:
        rows = analysis.shift_study(analysis.stream(1, "shift", 0), 0, True)
        metrics = {r["metric"] for r in rows}
        self.assertIn("fixed_background_alert_rate", metrics)
        self.assertIn("recalibrated_background_alert_rate", metrics)

    def test_shift_sd_name_is_not_variance_label(self) -> None:
        rows = analysis.shift_study(analysis.stream(1, "shift", 0), 0, True)
        self.assertIn("sd_1.5", {r["condition"] for r in rows})
        self.assertNotIn("variance_1.5", {r["condition"] for r in rows})

    def test_shift_fixed_threshold_has_no_fake_analytic_theory(self) -> None:
        rows = analysis.shift_study(analysis.stream(1, "shift", 0), 0, True)
        fixed = [r for r in rows if r["metric"] == "fixed_background_alert_rate"]
        self.assertTrue(all(r["theory"] is None for r in fixed))

    def test_shift_recalibrated_budget_has_exact_reference(self) -> None:
        rows = analysis.shift_study(analysis.stream(1, "shift", 0), 0, True)
        current = [r for r in rows if r["metric"] == "recalibrated_background_alert_rate"]
        self.assertTrue(all(r["theory"] == .01 for r in current))


class ArchivedEvidenceTests(unittest.TestCase):
    def test_18_canonical_tables(self) -> None:
        self.assertEqual(len(list((ROOT/"results").glob("*.csv"))) % 18, 0)
        self.assertEqual(len([p for p in (ROOT/"results").glob("*.csv")
                              if not p.name.endswith("_quick.csv")]), 18)

    def test_all_nine_full_figures(self) -> None:
        """Check committed scientific figure files independently of the manuscript."""
        for fig in analysis.FIGURES:
            with self.subTest(figure=fig):
                path: Path = ROOT/"figures"/f"{fig}.png"
                self.assertTrue(path.is_file())
                contents: bytes = path.read_bytes()
                self.assertGreaterEqual(len(contents), 24)
                self.assertEqual(contents[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(contents[12:16], b"IHDR")
                # The first PNG chunk records 32-bit big-endian pixel dimensions.
                width: int = int.from_bytes(contents[16:20], byteorder="big")
                height: int = int.from_bytes(contents[20:24], byteorder="big")
                self.assertGreater(width, 0)
                self.assertGreater(height, 0)

    def test_summary_columns(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_summary.csv")
                self.assertEqual(set(rows[0]), {"condition", "metric", "n", "mean",
                                                "sd", "se", "lower95", "upper95", "theory"})

    def test_replication_columns(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_replicates.csv")
                self.assertEqual(set(rows[0]), {"replicate", "condition", "metric",
                                                "value", "theory"})

    def test_summary_sample_counts(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_summary.csv")
                self.assertTrue(all(int(r["n"]) == analysis.REPETITIONS[name] for r in rows))

    def test_summary_matches_replicate_means(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                raw = read_csv(ROOT/"results"/f"{name}_replicates.csv")
                summary = read_csv(ROOT/"results"/f"{name}_summary.csv")
                for s in summary:
                    values = [float(r["value"]) for r in raw if
                              r["condition"] == s["condition"] and r["metric"] == s["metric"]]
                    self.assertAlmostEqual(np.mean(values), float(s["mean"]))

    def test_all_scores_finite(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_replicates.csv")
                self.assertTrue(all(math.isfinite(float(r["value"])) for r in rows))

    def test_95_mc_intervals_bound_means(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_summary.csv")
                for r in rows:
                    self.assertLessEqual(float(r["lower95"]), float(r["mean"]))
                    self.assertGreaterEqual(float(r["upper95"]), float(r["mean"]))

    def test_quick_does_not_overwrite_full_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            result = analysis.run(analysis.Design(root=Path(dirname), quick=True))
            self.assertEqual(set(result), set(analysis.EXPERIMENTS))
            self.assertEqual(len(list((Path(dirname)/"results").glob("*_quick.csv"))), 18)
            self.assertEqual(len(list((Path(dirname)/"figures").glob("*_quick.png"))), 9)
            self.assertFalse(list((Path(dirname)/"results").glob("*_summary.csv")))


class NotebookAndDocumentationTests(unittest.TestCase):
    def test_notebook_valid_json(self) -> None:
        json.loads((ROOT/notebook_builder.FILENAME).read_text(encoding="utf-8"))

    def test_notebook_builder_exactly_matches_file(self) -> None:
        obj = json.loads((ROOT/notebook_builder.FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(obj, notebook_builder.notebook())

    def test_notebook_unexecuted(self) -> None:
        obj = notebook_builder.notebook()
        self.assertTrue(all(cell.get("execution_count") is None for cell in obj["cells"]
                            if cell["cell_type"] == "code"))

    def test_notebook_cell_ids_unique(self) -> None:
        ids = [cell["id"] for cell in notebook_builder.notebook()["cells"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_notebook_references_all_nine(self) -> None:
        notebook_text = json.dumps(notebook_builder.notebook())
        self.assertIn("EXPERIMENTS, FIGURES", notebook_text)
        self.assertIn("zip(EXPERIMENTS, FIGURES)", notebook_text)

    def test_readme_discloses_synthetic_data(self) -> None:
        text = (ROOT/"README.md").read_text(encoding="utf-8")
        self.assertIn("synthetic", text.lower())

    def test_scientific_versions_are_pinned(self) -> None:
        lines = (ROOT/"requirements.txt").read_text(encoding="utf-8").splitlines()
        for name in ("numpy", "scipy", "matplotlib", "scikit-learn"):
            self.assertTrue(any(line.startswith(name+"==") for line in lines))

    def test_public_article_is_private_manuscript_notice(self) -> None:
        """The public path preserves links without owning editorial content."""
        notice: str = (ROOT/"article.md").read_text(encoding="utf-8")
        self.assertIn("research-articles-medium", notice)
        self.assertIn("outlier-is-not-an-error/article.md", notice)
        self.assertLess(len(notice.split()), 200)

    @unittest.skipIf(nbformat is None, "nbformat/nbclient optional locally")
    def test_jupyter_notebook_schema(self) -> None:
        notebook = nbformat.read(ROOT/notebook_builder.FILENAME, as_version=4)
        nbformat.validate(notebook)

    @unittest.skipUnless(os.environ.get("RUN_NOTEBOOK_EXECUTION") == "1",
                         "Fresh-kernel execution requested in CI")
    def test_notebook_executes_fresh_kernel(self) -> None:
        if nbformat is None or NotebookClient is None:
            self.fail("CI must install nbformat and nbclient")
        notebook = nbformat.read(ROOT/notebook_builder.FILENAME, as_version=4)
        executed = NotebookClient(notebook, timeout=600,
                                  kernel_name="python3",
                                  resources={"metadata": {"path": str(ROOT)}}).execute()
        self.assertTrue(all(not c.get("outputs") or
                            not any(x.get("output_type") == "error" for x in c["outputs"])
                            for c in executed.cells if c["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
