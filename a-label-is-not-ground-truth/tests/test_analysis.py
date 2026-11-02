"""Label-channel identities, synthetic contracts, archived evidence and CI notebook."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
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

# Exact study contracts guard both the generated observations and archived CSVs
# against silent loss, duplication, or renaming of conditions and metrics.
STUDY_CONTRACTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "symmetric": (
        ("0.00", "0.05", "0.15", "0.30"),
        ("clean_test_auc", "clean_test_brier", "empirical_train_flip_rate"),
    ),
    "asymmetric": (
        ("0.00:0.00", "0.05:0.10", "0.05:0.30", "0.15:0.40"),
        ("observed_positive_fraction", "oracle_corrected_prevalence",
         "latent_sample_prevalence"),
    ),
    "instance": (
        ("uniform", "boundary", "high_margin"),
        ("clean_test_auc", "clean_test_brier", "observed_flip_fraction"),
    ),
    "annotators": (
        ("3_similar", "5_heterogeneous", "5_weak_majority"),
        ("majority_accuracy", "em_accuracy", "em_brier"),
    ),
    "correlation": (
        ("0.00", "0.05", "0.10", "0.20"),
        ("majority_error_rate", "em_error_rate", "confident_wrong_fraction",
         "worker_flip_fraction"),
    ),
    "selection": (
        ("0.10", "0.20", "0.40"),
        ("review_only_prevalence", "review_and_random_audit_ht",
         "latent_population_prevalence", "audit_fraction"),
    ),
    "benchmark": (
        ("0.00", "0.10", "0.25", "0.50", "0.75"),
        ("true_accuracy_A", "true_accuracy_B", "observed_accuracy_A",
         "observed_accuracy_B", "label_error_fraction"),
    ),
    "leakage": (
        ("row_split", "entity_holdout"),
        ("test_accuracy", "test_entity_overlap"),
    ),
    "policy": (
        ("baseline", "lenient", "strict"),
        ("latent_outcome_prevalence", "recorded_positive_fraction",
         "true_model_accuracy", "recorded_label_accuracy"),
    ),
}
REPLICATION_FIELDS = ("replicate", "condition", "metric", "value", "theory")
SUMMARY_FIELDS = ("condition", "metric", "n", "mean", "sd", "se",
                  "lower95", "upper95", "theory")


def expected_keys(study: str) -> set[tuple[str, str]]:
    """The exhaustive condition-by-metric contract for one named study."""

    conditions, metrics = STUDY_CONTRACTS[study]
    return {(condition, metric) for condition in conditions for metric in metrics}


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

    def test_invert_negative_orientation_is_identifiable(self) -> None:
        pi, eta0, eta1 = .35, .8, .4
        q = eta0+(1-eta0-eta1)*pi
        self.assertAlmostEqual(analysis.invert_noisy_prevalence(q, eta0, eta1), pi)
        self.assertLess(q, eta0)

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
        self.assertEqual(set(STUDY_CONTRACTS), set(analysis.EXPERIMENTS))
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = analysis.STUDIES[name](analysis.stream(4, name, 1), 1, True)
                keys = [(r["condition"], r["metric"]) for r in rows]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(set(keys), expected_keys(name))

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
    def assert_strict_csv_rows(
        self,
        rows: list[dict[str, str]],
        expected_fields: tuple[str, ...],
        integer_fields: tuple[str, ...],
    ) -> None:
        """Reject missing, extra, malformed, or nonfinite archival data."""

        self.assertTrue(rows, "archived evidence must contain data rows")
        for row_number, row in enumerate(rows, start=2):
            with self.subTest(csv_row=row_number):
                # DictReader preserves header order and exposes surplus cells
                # with a None key. Checking every row catches ragged CSVs.
                self.assertEqual(tuple(row), expected_fields)
                for field in expected_fields:
                    self.assertIsInstance(row[field], str)
                    if field != "theory":
                        self.assertNotEqual(row[field], "")
                self.assertTrue(row["condition"].strip())
                self.assertTrue(row["metric"].strip())
                for field in expected_fields:
                    if field in ("condition", "metric", "theory"):
                        continue
                    value = float(row[field])
                    self.assertTrue(math.isfinite(value), f"nonfinite {field}")
                    if field in ("sd", "se"):
                        self.assertGreaterEqual(value, 0.)
                for field in integer_fields:
                    self.assertRegex(row[field], r"^(0|[1-9][0-9]*)$")
                if row["theory"] != "":
                    self.assertTrue(math.isfinite(float(row["theory"])))

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
                self.assert_strict_csv_rows(rows, REPLICATION_FIELDS, ("replicate",))
                actual = {(r["condition"], r["metric"]) for r in rows}
                self.assertEqual(actual, expected_keys(name))
                for condition, metric in actual:
                    repetitions = sorted(int(r["replicate"]) for r in rows
                                         if (r["condition"], r["metric"]) ==
                                         (condition, metric))
                    self.assertEqual(repetitions,
                                     list(range(analysis.REPETITIONS[name])))

    def test_summary_schema(self) -> None:
        for name in analysis.EXPERIMENTS:
            with self.subTest(study=name):
                rows = read_csv(ROOT/"results"/f"{name}_summary.csv")
                self.assert_strict_csv_rows(rows, SUMMARY_FIELDS, ("n",))
                self.assertEqual({(r["condition"], r["metric"]) for r in rows},
                                 expected_keys(name))
                self.assertEqual(len(rows), len(expected_keys(name)))

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
            result_dir = Path(dirname)/"results"
            figure_dir = Path(dirname)/"figures"
            result_dir.mkdir()
            figure_dir.mkdir()
            # Seed all canonical names to prove that quick runs preserve them.
            sentinels: dict[Path, bytes] = {}
            for name in analysis.EXPERIMENTS:
                for kind in ("replicates", "summary"):
                    path = result_dir/f"{name}_{kind}.csv"
                    sentinels[path] = f"immutable:{path.name}\n".encode("utf-8")
            for name in analysis.FIGURES:
                path = figure_dir/f"{name}.png"
                sentinels[path] = f"immutable:{path.name}\n".encode("utf-8")
            for path, data in sentinels.items():
                path.write_bytes(data)

            result = analysis.run(analysis.Design(root=Path(dirname), quick=True))
            self.assertEqual(set(result), set(analysis.EXPERIMENTS))
            quick_csv = {f"{name}_{kind}_quick.csv"
                         for name in analysis.EXPERIMENTS
                         for kind in ("replicates", "summary")}
            quick_png = {f"{name}_quick.png" for name in analysis.FIGURES}
            self.assertEqual({path.name for path in result_dir.iterdir()},
                             quick_csv | {p.name for p in sentinels if p.parent == result_dir})
            self.assertEqual({path.name for path in figure_dir.iterdir()},
                             quick_png | {p.name for p in sentinels if p.parent == figure_dir})
            for path, original in sentinels.items():
                self.assertEqual(path.read_bytes(), original, f"quick overwrote {path.name}")


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
        # No resources.metadata.path: kernel starts in the repository root.
        previous_cwd = Path.cwd()
        try:
            os.chdir(ROOT.parent)
            executed = NotebookClient(notebook, timeout=600,
                                      kernel_name="python3").execute()
        finally:
            os.chdir(previous_cwd)
        self.assertTrue(all(not c.get("outputs") or
                            not any(x.get("output_type") == "error" for x in c["outputs"])
                            for c in executed.cells if c["cell_type"] == "code"))

    @unittest.skipUnless(os.environ.get("RUN_NOTEBOOK_EXECUTION") == "1",
                         "Fresh-kernel execution requested in CI")
    def test_notebook_runs_from_external_directory_with_hint(self) -> None:
        if nbformat is None or NotebookClient is None:
            self.fail("CI requires nbformat/nbclient")
        notebook = nbformat.read(ROOT/notebook_builder.FILENAME, as_version=4)
        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as outside:
                with patch.dict(os.environ, {"LABEL_STUDY_DIR": str(ROOT)}):
                    os.chdir(outside)
                    executed = NotebookClient(notebook, timeout=600,
                                              kernel_name="python3").execute()
        finally:
            os.chdir(previous_cwd)
        self.assertTrue(all(not c.get("outputs") or
                            not any(x.get("output_type") == "error" for x in c["outputs"])
                            for c in executed.cells if c["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
