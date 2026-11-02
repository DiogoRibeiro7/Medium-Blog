"""Label-channel identities, synthetic contracts, archived evidence and CI notebook."""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read versioned evidence, preserving empty theoretical reference cells."""

    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class VectorAndSeedTests(unittest.TestCase):
    def test_binary_accepts_integer(self) -> None:
        np.testing.assert_array_equal(analysis.binary_labels(np.array([0, 1, 0])), [0, 1, 0])

    def test_binary_converts_integer_dtype(self) -> None:
        self.assertEqual(analysis.binary_labels(np.array([0, 1, 0], dtype=np.int8)).dtype,
                         np.dtype(np.int64))

    def test_binary_rejects_float(self) -> None:
        with self.assertRaises(ValueError):
            analysis.binary_labels(np.array([0., 1., 0.]))

    def test_binary_rejects_negatives(self) -> None:
        with self.assertRaises(ValueError):
            analysis.binary_labels(np.array([-1, 1, 0]))

    def test_binary_rejects_twos(self) -> None:
        with self.assertRaises(ValueError):
            analysis.binary_labels(np.array([2, 1, 0]))

    def test_binary_rejects_scalar(self) -> None:
        with self.assertRaises(ValueError):
            analysis.binary_labels(np.array(1))

    def test_binary_rejects_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.binary_labels(np.zeros((5, 1), dtype=int))

    def test_binary_rejects_short_vector(self) -> None:
        with self.assertRaises(ValueError):
            analysis.binary_labels(np.array([1, 0]))

    def test_probabilities_accept_endpoints(self) -> None:
        np.testing.assert_allclose(analysis.probabilities(np.array([0., .5, 1.])),
                                   [0., .5, 1.])

    def test_probabilities_reject_negative(self) -> None:
        with self.assertRaises(ValueError):
            analysis.probabilities(np.array([-.1, .2, .5]))

    def test_probabilities_reject_above_one(self) -> None:
        with self.assertRaises(ValueError):
            analysis.probabilities(np.array([.1, .2, 1.01]))

    def test_probabilities_reject_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.probabilities(np.array([.1, float("nan"), .3]))

    def test_probabilities_reject_inf(self) -> None:
        with self.assertRaises(ValueError):
            analysis.probabilities(np.array([.1, float("inf"), .3]))

    def test_probabilities_reject_matrix(self) -> None:
        with self.assertRaises(ValueError):
            analysis.probabilities(np.zeros((3, 1)))

    def test_stream_reproducible(self) -> None:
        np.testing.assert_array_equal(analysis.stream(44, "policy", 2).normal(size=40),
                                      analysis.stream(44, "policy", 2).normal(size=40))

    def test_stream_distinct_replicates(self) -> None:
        self.assertFalse(np.array_equal(analysis.stream(44, "policy", 2).normal(size=40),
                                        analysis.stream(44, "policy", 3).normal(size=40)))

    def test_stream_distinct_studies(self) -> None:
        self.assertFalse(np.array_equal(analysis.stream(44, "policy", 2).normal(size=40),
                                        analysis.stream(44, "instance", 2).normal(size=40)))

    def test_stream_reject_bad_name(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "typo", 0)

    def test_stream_reject_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(-1, "policy", 0)

    def test_stream_reject_negative_replicate(self) -> None:
        with self.assertRaises(ValueError):
            analysis.stream(1, "policy", -1)

    def test_run_reject_root_string(self) -> None:
        with self.assertRaises(TypeError):
            analysis.run(analysis.Design(root="bad", quick=True))  # type: ignore[arg-type]

    def test_run_reject_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(quick=True, seed=-1))

    def test_run_guard_noncanonical_full_seed(self) -> None:
        with self.assertRaises(ValueError):
            analysis.run(analysis.Design(seed=analysis.SEED+1))

    def test_nine_studies(self) -> None:
        self.assertEqual(len(set(analysis.EXPERIMENTS)), 9)

    def test_nine_figures(self) -> None:
        self.assertEqual(len(set(analysis.FIGURES)), 9)

    def test_studies_mapping_complete(self) -> None:
        self.assertEqual(set(analysis.STUDIES), set(analysis.EXPERIMENTS))

    def test_repetitions_at_least_twenty(self) -> None:
        self.assertTrue(all(v >= 20 for v in analysis.REPETITIONS.values()))


class NoiseChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=int)

    def test_no_flips(self) -> None:
        np.testing.assert_array_equal(analysis.flip_labels(np.random.default_rng(1),
                                                            self.y, 0., 0.), self.y)

    def test_flip_result_remains_binary(self) -> None:
        noisy = analysis.flip_labels(np.random.default_rng(1), self.y, .3, .2)
        self.assertTrue(np.isin(noisy, (0, 1)).all())

    def test_flip_seed_reproducible(self) -> None:
        a = analysis.flip_labels(np.random.default_rng(1), self.y, .3, .2)
        b = analysis.flip_labels(np.random.default_rng(1), self.y, .3, .2)
        np.testing.assert_array_equal(a, b)

    def test_flip_reject_negative(self) -> None:
        with self.assertRaises(ValueError):
            analysis.flip_labels(np.random.default_rng(1), self.y, -.1, 0.)

    def test_flip_reject_one(self) -> None:
        with self.assertRaises(ValueError):
            analysis.flip_labels(np.random.default_rng(1), self.y, .1, 1.)

    def test_flip_reject_nan(self) -> None:
        with self.assertRaises(ValueError):
            analysis.flip_labels(np.random.default_rng(1), self.y, float("nan"), .1)

    def test_invert_identity(self) -> None:
        self.assertAlmostEqual(analysis.invert_noisy_prevalence(.25, 0., 0.), .25)

    def test_invert_known_asymmetric_example(self) -> None:
        self.assertAlmostEqual(analysis.invert_noisy_prevalence(.2125, .05, .30), .25)

    def test_invert_round_trip(self) -> None:
        pi, eta0, eta1 = .31, .04, .27
        q = eta0+(1-eta0-eta1)*pi
        self.assertAlmostEqual(analysis.invert_noisy_prevalence(q, eta0, eta1), pi)

    def test_invert_reject_unidentified_channel(self) -> None:
        with self.assertRaises(ValueError):
            analysis.invert_noisy_prevalence(.5, .6, .4)

    def test_invert_reject_large_probability(self) -> None:
        with self.assertRaises(ValueError):
            analysis.invert_noisy_prevalence(1.1, .1, .2)

    def test_invert_reject_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            analysis.invert_noisy_prevalence(.5, float("inf"), .1)

    def test_invert_does_not_silently_clip_outside_probability(self) -> None:
        self.assertLess(analysis.invert_noisy_prevalence(0., .2, .1), 0.)

    def test_noisy_probability_identity(self) -> None:
        p = np.array([0., .5, 1.])
        np.testing.assert_allclose(analysis.noisy_probabilities(p, .1, .2), [.1, .45, .8])

    def test_noisy_probability_zero_channel(self) -> None:
        p = np.array([.1, .4, .9])
        np.testing.assert_allclose(analysis.noisy_probabilities(p, 0., 0.), p)

    def test_noisy_probability_reject_rates(self) -> None:
        with self.assertRaises(ValueError):
            analysis.noisy_probabilities(np.array([.1, .2, .3]), -.01, .1)

    def test_population_reproducible(self) -> None:
        a = analysis.logistic_population(np.random.default_rng(1), 100)
        b = analysis.logistic_population(np.random.default_rng(1), 100)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)

    def test_population_shapes(self) -> None:
        x, y, p = analysis.logistic_population(np.random.default_rng(1), 100)
        self.assertEqual(x.shape, (100, 2))
        self.assertEqual(y.shape, (100,))
        self.assertEqual(p.shape, (100,))

    def test_population_reject_short(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_population(np.random.default_rng(1), 2)

    def test_population_reject_bad_strength(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_population(np.random.default_rng(1), 100, -1.)

    def test_population_reject_nonfinite_strength(self) -> None:
        with self.assertRaises(ValueError):
            analysis.logistic_population(np.random.default_rng(1), 100, float("nan"))


class AnnotatorAndSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.truth = (rng.random(400) < .4).astype(int)
        self.rng = np.random.default_rng(11)

    def test_votes_shape(self) -> None:
        v = analysis.worker_votes(self.rng, self.truth, np.full(5, .8), np.full(5, .85))
        self.assertEqual(v.shape, (400, 5))

    def test_votes_binary(self) -> None:
        v = analysis.worker_votes(self.rng, self.truth, np.full(5, .8), np.full(5, .85))
        self.assertTrue(np.isin(v, [0, 1]).all())

    def test_votes_reject_bad_specificity(self) -> None:
        with self.assertRaises(ValueError):
            analysis.worker_votes(self.rng, self.truth, np.full(5, .8), np.full(5, 1.))

    def test_votes_reject_shape(self) -> None:
        with self.assertRaises(ValueError):
            analysis.worker_votes(self.rng, self.truth, np.full(5, .8), np.full(4, .8))

    def test_votes_reject_two_workers(self) -> None:
        with self.assertRaises(ValueError):
            analysis.worker_votes(self.rng, self.truth, np.full(2, .8), np.full(2, .8))

    def test_em_output_probabilities(self) -> None:
        v = analysis.worker_votes(self.rng, self.truth, np.full(5, .9), np.full(5, .9))
        q = analysis.dawid_skene_binary(v)
        self.assertEqual(q.shape, (400,))
        self.assertTrue(np.all((q >= 0) & (q <= 1)))

    def test_em_outperforms_chance_for_skilled_workers(self) -> None:
        v = analysis.worker_votes(self.rng, self.truth, np.full(5, .9), np.full(5, .9))
        q = analysis.dawid_skene_binary(v)
        self.assertGreater(float(np.mean((q >= .5) == self.truth)), .8)

    def test_em_invariant_to_worker_permutation(self) -> None:
        v = analysis.worker_votes(self.rng, self.truth, np.full(5, .9), np.full(5, .9))
        a = analysis.dawid_skene_binary(v)
        b = analysis.dawid_skene_binary(v[:, [4, 2, 0, 3, 1]])
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_em_rejects_two_workers(self) -> None:
        with self.assertRaises(ValueError):
            analysis.dawid_skene_binary(np.ones((30, 2), dtype=int))

    def test_em_rejects_float_vote(self) -> None:
        with self.assertRaises(ValueError):
            analysis.dawid_skene_binary(np.ones((30, 3), dtype=float))

    def test_em_rejects_nonbinary_vote(self) -> None:
        with self.assertRaises(ValueError):
            analysis.dawid_skene_binary(np.full((30, 3), 2, dtype=int))

    def test_em_rejects_zero_iterations(self) -> None:
        with self.assertRaises(ValueError):
            analysis.dawid_skene_binary(np.ones((30, 3), dtype=int), iterations=0)

    def test_em_rejects_bad_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            analysis.dawid_skene_binary(np.ones((30, 3), dtype=int), tolerance=0.)

    def test_weighted_review_all_reviewed_exact(self) -> None:
        r = np.ones(len(self.truth), dtype=bool)
        a = np.zeros(len(self.truth), dtype=bool)
        self.assertAlmostEqual(analysis.weighted_review_prevalence(self.truth, r, a, .1),
                               float(np.mean(self.truth)))

    def test_weighted_review_full_audit_exact(self) -> None:
        r = np.zeros(len(self.truth), dtype=bool)
        a = np.ones(len(self.truth), dtype=bool)
        self.assertAlmostEqual(analysis.weighted_review_prevalence(self.truth, r, a, 1.),
                               float(np.mean(self.truth)))

    def test_weighted_review_disjoint_half_exact(self) -> None:
        r = np.arange(len(self.truth)) % 2 == 0
        a = ~r
        self.assertAlmostEqual(analysis.weighted_review_prevalence(self.truth, r, a, 1.),
                               float(np.mean(self.truth)))

    def test_weighted_review_reject_overlap(self) -> None:
        r = np.ones(len(self.truth), dtype=bool)
        with self.assertRaises(ValueError):
            analysis.weighted_review_prevalence(self.truth, r, r, .5)

    def test_weighted_review_reject_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            analysis.weighted_review_prevalence(self.truth, np.ones(5, dtype=bool),
                                                np.zeros(5, dtype=bool), .5)

    def test_weighted_review_reject_non_boolean(self) -> None:
        with self.assertRaises(ValueError):
            analysis.weighted_review_prevalence(self.truth, np.ones(400, dtype=int),
                                                np.zeros(400, dtype=bool), .5)

    def test_weighted_review_reject_zero_probability(self) -> None:
        with self.assertRaises(ValueError):
            analysis.weighted_review_prevalence(self.truth, np.zeros(400, dtype=bool),
                                                np.ones(400, dtype=bool), 0.)

    def test_weighted_review_reject_above_one_probability(self) -> None:
        with self.assertRaises(ValueError):
            analysis.weighted_review_prevalence(self.truth, np.zeros(400, dtype=bool),
                                                np.ones(400, dtype=bool), 1.1)


class SummarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [analysis.observation(i, "a", "m", v, 2.)
                     for i, v in enumerate((1., 2., 3., 4.))]

    def test_mean(self) -> None:
        self.assertAlmostEqual(analysis.summarize(self.rows)[0]["mean"], 2.5)

    def test_sd(self) -> None:
        self.assertAlmostEqual(analysis.summarize(self.rows)[0]["sd"], math.sqrt(5/3))

    def test_mcse(self) -> None:
        self.assertAlmostEqual(analysis.summarize(self.rows)[0]["se"], math.sqrt(5/3)/2)

    def test_theory(self) -> None:
        self.assertEqual(analysis.summarize(self.rows)[0]["theory"], 2.)

    def test_interval(self) -> None:
        row = analysis.summarize(self.rows)[0]
        self.assertLess(row["lower95"], row["mean"])
        self.assertGreater(row["upper95"], row["mean"])

    def test_lookup(self) -> None:
        self.assertAlmostEqual(analysis.lookup(analysis.summarize(self.rows), "a", "m"), 2.5)

    def test_lookup_missing(self) -> None:
        with self.assertRaises(KeyError):
            analysis.lookup(analysis.summarize(self.rows), "x", "m")

    def test_empty_summary(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([])

    def test_inconsistent_theory(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize(self.rows+[analysis.observation(4, "a", "m", 5., 3.)])

    def test_duplicate_replicate(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize(self.rows+[self.rows[0]])

    def test_bad_observation_value(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "a", "m", float("inf"))

    def test_bad_observation_key(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "", "m", 0.)

    def test_bad_observation_theory(self) -> None:
        with self.assertRaises(ValueError):
            analysis.observation(0, "a", "m", 0., float("nan"))

    def test_empty_csv_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.write_csv(ROOT/"impossible.csv", [])


class StudyContractTests(unittest.TestCase):
    def test_all_studies_finite(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = analysis.STUDIES[name](analysis.stream(4, name, 1), 1, True)
                self.assertTrue(rows)
                self.assertTrue(all(np.isfinite(r["value"]) for r in rows))

    def test_study_keys_unique(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = analysis.STUDIES[name](analysis.stream(4, name, 1), 1, True)
                keys = [(r["condition"], r["metric"]) for r in rows]
                self.assertEqual(len(keys), len(set(keys)))

    def test_studies_reproducible(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                a = analysis.STUDIES[name](analysis.stream(4, name, 1), 1, True)
                b = analysis.STUDIES[name](analysis.stream(4, name, 1), 1, True)
                self.assertEqual(a, b)

    def test_symmetric_noise_changes_actual_flip_rates(self) -> None:
        rows = analysis.symmetric_study(analysis.stream(1, "symmetric", 0), 0, True)
        rates = {r["condition"]: r["value"] for r in rows if
                 r["metric"] == "empirical_train_flip_rate"}
        self.assertEqual(rates["0.00"], 0.)
        self.assertGreater(rates["0.30"], rates["0.05"])

    def test_asymmetric_oracle_theory(self) -> None:
        rows = analysis.asymmetric_study(analysis.stream(1, "asymmetric", 0), 0, True)
        targets = [r["theory"] for r in rows if r["metric"] == "oracle_corrected_prevalence"]
        self.assertTrue(all(v == .25 for v in targets))

    def test_instance_noise_equal_fraction(self) -> None:
        rows = analysis.instance_study(analysis.stream(1, "instance", 0), 0, True)
        flipped = [r["value"] for r in rows if r["metric"] == "observed_flip_fraction"]
        self.assertEqual(len(flipped), 3)
        self.assertTrue(all(abs(float(v)-.12) < 1e-3 for v in flipped))

    def test_annotators_three_scenarios(self) -> None:
        rows = analysis.annotators_study(analysis.stream(1, "annotators", 0), 0, True)
        self.assertEqual(len({r["condition"] for r in rows}), 3)

    def test_correlation_worker_theory_constant(self) -> None:
        rows = analysis.correlation_study(analysis.stream(1, "correlation", 0), 0, True)
        self.assertTrue(all(r["theory"] == .2 for r in rows if
                            r["metric"] == "worker_flip_fraction"))

    def test_selection_has_ht_and_naive(self) -> None:
        rows = analysis.selection_study(analysis.stream(1, "selection", 0), 0, True)
        self.assertIn("review_and_random_audit_ht", {r["metric"] for r in rows})
        self.assertIn("review_only_prevalence", {r["metric"] for r in rows})

    def test_benchmark_model_A_truly_better(self) -> None:
        rows = analysis.benchmark_study(analysis.stream(1, "benchmark", 0), 0, True)
        m = {(r["condition"], r["metric"]): r["value"] for r in rows}
        self.assertGreater(m[("0.00", "true_accuracy_A")], m[("0.00", "true_accuracy_B")])

    def test_benchmark_recorded_ranking_can_reverse(self) -> None:
        rows = analysis.benchmark_study(analysis.stream(1, "benchmark", 0), 0, True)
        m = {(r["condition"], r["metric"]): r["value"] for r in rows}
        self.assertGreater(m[("0.75", "observed_accuracy_B")],
                           m[("0.75", "observed_accuracy_A")])

    def test_leakage_has_zero_entity_overlap_on_holdout(self) -> None:
        rows = analysis.leakage_study(analysis.stream(1, "leakage", 0), 0, True)
        m = {(r["condition"], r["metric"]): r["value"] for r in rows}
        self.assertEqual(m[("entity_holdout", "test_entity_overlap")], 0.)
        self.assertGreater(m[("row_split", "test_entity_overlap")], .95)

    def test_leakage_row_split_more_optimistic(self) -> None:
        rows = analysis.leakage_study(analysis.stream(1, "leakage", 0), 0, True)
        m = {(r["condition"], r["metric"]): r["value"] for r in rows}
        self.assertGreater(m[("row_split", "test_accuracy")],
                           m[("entity_holdout", "test_accuracy")])

    def test_policy_latent_target_same_across_annotation_policy(self) -> None:
        rows = analysis.policy_study(analysis.stream(1, "policy", 0), 0, True)
        values = [r["value"] for r in rows if r["metric"] == "latent_outcome_prevalence"]
        self.assertEqual(len(set(values)), 1)

    def test_policy_recorded_target_changes(self) -> None:
        rows = analysis.policy_study(analysis.stream(1, "policy", 0), 0, True)
        m = {r["condition"]: r["value"] for r in rows if
             r["metric"] == "recorded_positive_fraction"}
        self.assertGreater(m["lenient"], m["baseline"])
        self.assertLess(m["strict"], m["baseline"])


class ArchivedEvidenceTests(unittest.TestCase):
    def test_eighteen_canonical_csvs(self) -> None:
        paths = [p for p in (ROOT/"results").glob("*.csv")
                 if not p.name.endswith("_quick.csv")]
        self.assertEqual(len(paths), 18)

    def test_nine_png_figures(self) -> None:
        for name in analysis.FIGURES:
            with self.subTest(figure=name):
                path = ROOT/"figures"/f"{name}.png"
                self.assertTrue(path.is_file())
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_replication_schema(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_replicates.csv")
                self.assertEqual(set(rows[0]), {"replicate", "condition", "metric",
                                                "value", "theory"})

    def test_summary_schema(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_summary.csv")
                self.assertEqual(set(rows[0]), {"condition", "metric", "n", "mean",
                                                "sd", "se", "lower95", "upper95", "theory"})

    def test_counts_match_outer_replications(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_summary.csv")
                self.assertTrue(all(int(r["n"]) == analysis.REPETITIONS[name] for r in rows))

    def test_summary_means_match_raw(self) -> None:
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

    def test_all_monte_carlo_intervals_bound_means(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_summary.csv")
                self.assertTrue(all(float(r["lower95"]) <= float(r["mean"]) <=
                                    float(r["upper95"]) for r in rows))

    def test_quick_writes_only_isolated_suffixed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as dirname:
            result = analysis.run(analysis.Design(root=Path(dirname), quick=True))
            self.assertEqual(set(result), set(analysis.EXPERIMENTS))
            self.assertEqual(len(list((Path(dirname)/"results").glob("*_quick.csv"))), 18)
            self.assertEqual(len(list((Path(dirname)/"figures").glob("*_quick.png"))), 9)
            self.assertFalse(list((Path(dirname)/"results").glob("*_summary.csv")))


class NotebookAndDocumentationTests(unittest.TestCase):
    def test_notebook_is_json(self) -> None:
        json.loads((ROOT/notebook_builder.FILENAME).read_text(encoding="utf-8"))

    def test_notebook_matches_builder(self) -> None:
        obj = json.loads((ROOT/notebook_builder.FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(obj, notebook_builder.notebook())

    def test_notebook_has_no_cached_execution(self) -> None:
        obj = notebook_builder.notebook()
        self.assertTrue(all(c.get("execution_count") is None for c in obj["cells"]
                            if c["cell_type"] == "code"))

    def test_notebook_cell_ids_unique(self) -> None:
        ids = [c["id"] for c in notebook_builder.notebook()["cells"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_notebook_runs_all_nine(self) -> None:
        obj = json.dumps(notebook_builder.notebook())
        self.assertIn("zip(EXPERIMENTS, FIGURES)", obj)

    def test_readme_discloses_synthetic_scope(self) -> None:
        text = (ROOT/"README.md").read_text(encoding="utf-8")
        self.assertIn("synthetic", text.lower())

    def test_scientific_version_pins(self) -> None:
        reqs = (ROOT/"requirements.txt").read_text(encoding="utf-8").splitlines()
        for name in ("numpy", "scipy", "matplotlib", "scikit-learn"):
            self.assertTrue(any(r.startswith(name+"==") for r in reqs))

    def test_manuscript_over_8000_words(self) -> None:
        text = (ROOT/"article.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(text.split()), 8000)

    def test_manuscript_nine_graphics(self) -> None:
        text = (ROOT/"article.md").read_text(encoding="utf-8")
        self.assertEqual(text.count("![Figure "), 9)

    def test_manuscript_primary_sources(self) -> None:
        text = (ROOT/"article.md").read_text(encoding="utf-8")
        self.assertIn("doi.org/10.2307/2346806", text)
        self.assertIn("papers.nips.cc/paper/5073", text)

    @unittest.skipIf(nbformat is None, "nbformat/nbclient optional locally")
    def test_jupyter_schema(self) -> None:
        notebook = nbformat.read(ROOT/notebook_builder.FILENAME, as_version=4)
        nbformat.validate(notebook)

    @unittest.skipUnless(os.environ.get("RUN_NOTEBOOK_EXECUTION") == "1",
                         "Fresh-kernel execution requested in CI")
    def test_notebook_runs_in_fresh_kernel(self) -> None:
        if nbformat is None or NotebookClient is None:
            self.fail("CI requires nbformat/nbclient")
        notebook = nbformat.read(ROOT/notebook_builder.FILENAME, as_version=4)
        executed = NotebookClient(notebook, timeout=600,
                                  kernel_name="python3",
                                  resources={"metadata": {"path": str(ROOT)}}).execute()
        self.assertTrue(all(not c.get("outputs") or
                            not any(x.get("output_type") == "error" for x in c["outputs"])
                            for c in executed.cells if c["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
