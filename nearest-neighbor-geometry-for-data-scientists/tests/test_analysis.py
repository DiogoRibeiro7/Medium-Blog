"""Invariants, simulation-evidence checks, and reproducibility smoke tests.

Run from the article directory with:
    python -m unittest discover -s tests -v

Set RUN_NOTEBOOK_EXECUTION=1 to execute the notebook in a fresh kernel.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import nbformat
import numpy as np
from scipy.stats import norm

import analysis


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class NumericalIdentityTests(unittest.TestCase):
    def test_distance_chi_one(self) -> None:
        mean, _, cv = analysis.gaussian_distance_moments(1)
        self.assertAlmostEqual(mean, 2 / np.sqrt(np.pi), places=12)
        self.assertAlmostEqual(cv, np.sqrt(2 - 4 / np.pi) / mean, places=12)

    def test_distance_chi_two(self) -> None:
        mean, _, _ = analysis.gaussian_distance_moments(2)
        self.assertAlmostEqual(mean, np.sqrt(np.pi), places=12)

    def test_distance_second_moment_all_dimensions(self) -> None:
        for dim in (1, *analysis.DIMS, 1024):
            mean, sd, _ = analysis.gaussian_distance_moments(dim)
            self.assertAlmostEqual(mean * mean + sd * sd, 2 * dim, places=7)

    def test_relative_concentration_improves(self) -> None:
        cv = [analysis.gaussian_distance_moments(d)[2] for d in analysis.DIMS]
        self.assertTrue(all(a > b for a, b in zip(cv, cv[1:])))

    def test_large_dim_asymptotic(self) -> None:
        dim = 10000
        _, _, cv = analysis.gaussian_distance_moments(dim)
        self.assertAlmostEqual(cv * np.sqrt(2 * dim), 1, delta=.001)

    def test_invalid_dimensions(self) -> None:
        for value in (0, -1, 1.2, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                analysis.gaussian_distance_moments(value)  # type: ignore[arg-type]

    def test_bayes_reference(self) -> None:
        self.assertAlmostEqual(norm.cdf(.85), .8023374568773076, places=10)

    def test_participation_is_rank_one(self) -> None:
        cov = np.diag([4., 0., 0.])
        self.assertEqual(analysis.participation_ratio(cov), 1.)

    def test_participation_is_full_rank(self) -> None:
        self.assertAlmostEqual(analysis.participation_ratio(np.eye(6)), 6.)

    def test_participation_mixed_spectrum(self) -> None:
        self.assertAlmostEqual(analysis.participation_ratio(np.diag([3., 1.])), 1.6)

    def test_participation_scale_invariance(self) -> None:
        cov = np.diag([8., 3., 1.])
        self.assertAlmostEqual(analysis.participation_ratio(cov),
                               analysis.participation_ratio(12 * cov))

    def test_participation_rejects_nonsymmetric(self) -> None:
        with self.assertRaises(ValueError):
            analysis.participation_ratio(np.array([[1., 1.], [0., 1.]]))

    def test_participation_rejects_non_psd(self) -> None:
        with self.assertRaises(ValueError):
            analysis.participation_ratio(np.diag([2., -1.]))

    def test_participation_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            analysis.participation_ratio(np.zeros((3, 3)))

    def test_participation_rejects_rectangular(self) -> None:
        with self.assertRaises(ValueError):
            analysis.participation_ratio(np.ones((2, 3)))


class NeighborRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = np.array([[0., 1.], [2., 0.], [0., 3.]])
        self.labels = np.array([0, 1, 0], dtype=np.int64)
        self.test = np.array([[0., 1.1], [1.9, 0.1]])
        self.truth = np.array([0, 1], dtype=np.int64)

    def test_one_neighbor_labels(self) -> None:
        score = analysis.knn_accuracy(self.train, self.labels, self.test,
                                      self.truth, neighbors=1)
        self.assertEqual(score, 1.)

    def test_three_neighbor_majority(self) -> None:
        score = analysis.knn_accuracy(self.train, self.labels, self.test,
                                      np.zeros(2, dtype=np.int64), neighbors=3)
        self.assertEqual(score, 1.)

    def test_norm_pseudodistance_has_zero_for_distinct_points(self) -> None:
        x = np.array([[1., 0.], [0., 1.]])
        self.assertNotEqual(np.linalg.norm(x[0] - x[1]), 0.)
        self.assertEqual(abs(np.linalg.norm(x[0]) - np.linalg.norm(x[1])), 0.)

    def test_cosine_ignores_positive_rescaling(self) -> None:
        original = analysis.knn_accuracy(self.train, self.labels, self.test,
                                         self.truth, neighbors=1, distance="cosine")
        scaled = analysis.knn_accuracy(self.train * 10, self.labels,
                                       self.test * 3, self.truth, neighbors=1,
                                       distance="cosine")
        self.assertEqual(original, scaled)

    def test_unknown_distance(self) -> None:
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(self.train, self.labels, self.test,
                                  self.truth, distance="invented")

    def test_even_neighbor_vote_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(self.train, self.labels, self.test,
                                  self.truth, neighbors=2)

    def test_oversized_k_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(self.train, self.labels, self.test,
                                  self.truth, neighbors=5)

    def test_wrong_binary_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(self.train, np.array([0, 1, 2]),
                                  self.test, self.truth)

    def test_length_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(self.train, self.labels[:-1],
                                  self.test, self.truth)

    def test_feature_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(self.train, self.labels, self.test[:, :1],
                                  self.truth)

    def test_nonfinite_rejected(self) -> None:
        bad = self.train.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(bad, self.labels, self.test, self.truth)

    def test_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(np.zeros((0, 2)), np.array([]),
                                  self.test, self.truth)

    def test_cosine_zero_vector_rejected(self) -> None:
        train = self.train.copy(); train[0] = 0
        with self.assertRaises(ValueError):
            analysis.knn_accuracy(train, self.labels, self.test, self.truth,
                                  distance="cosine")

    def test_no_self_neighbors_and_sum_invariant(self) -> None:
        counts = analysis.k_occurrences(np.arange(20, dtype=float)[:, None], 3)
        self.assertEqual(int(counts.sum()), 60)
        self.assertEqual(len(counts), 20)

    def test_k_occurrence_reproducible(self) -> None:
        x = np.random.default_rng(7).normal(size=(35, 9))
        np.testing.assert_array_equal(analysis.k_occurrences(x, 4),
                                      analysis.k_occurrences(x, 4))

    def test_k_occurrence_invalid_k(self) -> None:
        for value in (0, 3):
            with self.assertRaises(ValueError):
                analysis.k_occurrences(np.ones((3, 2)), value)


class GeneratorAndProjectionTests(unittest.TestCase):
    def test_signal_generator_has_no_leakage(self) -> None:
        a, ya, b, yb = analysis.generate_signal_and_nuisance(
            np.random.default_rng(2), 40, 30, 14)
        self.assertEqual(a.shape, (40, 14))
        self.assertEqual(b.shape, (30, 14))
        self.assertEqual(set(np.unique(np.r_[ya, yb])), {0, 1})

    def test_generator_is_seeded(self) -> None:
        lhs = analysis.generate_signal_and_nuisance(np.random.default_rng(9))
        rhs = analysis.generate_signal_and_nuisance(np.random.default_rng(9))
        for a, b in zip(lhs, rhs):
            np.testing.assert_array_equal(a, b)

    def test_jl_map_shape(self) -> None:
        result = analysis.jl_projection(np.ones((10, 25)), 7,
                                        np.random.default_rng(5))
        self.assertEqual(result.shape, (10, 7))

    def test_jl_map_second_moment_is_approximately_preserved(self) -> None:
        x = np.array([[0.] * 24, [1.] * 24])
        p = analysis.jl_projection(x, 3000, np.random.default_rng(3))
        ratio = np.sum((p[1]-p[0])**2) / 24
        self.assertAlmostEqual(float(ratio), 1, delta=.08)

    def test_jl_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            analysis.jl_projection(np.ones((4, 3)), 0, np.random.default_rng(3))

    def test_both_metric_tasks_are_seeded(self) -> None:
        for task in ("angular", "radial"):
            lhs = analysis.generate_metric_task(np.random.default_rng(4), task)
            rhs = analysis.generate_metric_task(np.random.default_rng(4), task)
            for a, b in zip(lhs, rhs):
                np.testing.assert_array_equal(a, b)

    def test_invalid_metric_task(self) -> None:
        with self.assertRaises(ValueError):
            analysis.generate_metric_task(np.random.default_rng(4), "neither")


class SummaryAndIOTests(unittest.TestCase):
    def test_summary_mean_and_mc_se(self) -> None:
        rows = [{"g": 1, "value": 2.}, {"g": 1, "value": 4.}]
        result = analysis.summarize(rows, ("g",), ("value",))
        self.assertEqual(result[0]["repeats"], 2)
        self.assertEqual(result[0]["value_mean"], 3.)
        self.assertEqual(result[0]["value_mc_se"], 1.)

    def test_summary_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([], ("g",), ("value",))

    def test_write_csv_rejects_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as dest:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(dest)/"bad.csv", [{"x": 1}, {"y": 2}])

    def test_write_csv_rejects_empty(self) -> None:
        with tempfile.TemporaryDirectory() as dest:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(dest)/"bad.csv", [])

    def test_quick_and_full_replicate_counts(self) -> None:
        self.assertEqual(analysis.Design(quick=True).repeats["classification"], 5)
        self.assertEqual(analysis.Design().repeats["classification"], 36)
        self.assertEqual(analysis.Design(quick=True).suffix, "_quick")


class GeneratedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.result = analysis.run(analysis.Design(quick=True, output_root=cls.root))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def rows(self, table: str) -> list[dict[str, str]]:
        return read_rows(self.root/"results"/f"{table}_quick.csv")

    def test_all_nine_figures_are_nonempty_pngs(self) -> None:
        for figure in analysis.FIGURE_NAMES:
            file = self.root/"figures"/f"{figure}_quick.png"
            with self.subTest(figure=figure):
                self.assertGreater(file.stat().st_size, 10000)
                self.assertEqual(file.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_all_eighteen_csvs_have_rows(self) -> None:
        for table in analysis.TABLE_NAMES:
            with self.subTest(table=table):
                self.assertGreater(len(self.rows(table)), 0)

    def test_quick_does_not_write_canonical_filenames(self) -> None:
        self.assertTrue(all(f.stem.endswith("_quick")
                            for f in (self.root/"results").glob("*.csv")))

    def test_distance_sample_count(self) -> None:
        self.assertEqual(len(self.rows("distance_samples")), 2100*len(analysis.DIMS))

    def test_distance_theoretical_cv_recorded(self) -> None:
        for row in self.rows("distance_summary"):
            self.assertAlmostEqual(float(row["theory_cv"]),
                                   analysis.gaussian_distance_moments(int(row["dimension"]))[2],
                                   places=9)

    def test_distance_concentration_synthetic_pattern(self) -> None:
        self.assertGreater(self.result["distance_p2_cv"],
                           self.result["distance_p512_cv"])

    def test_contrast_replicate_count(self) -> None:
        self.assertEqual(len(self.rows("neighbor_contrast_replicates")),
                         len(analysis.DIMS)*3)

    def test_contrast_decreases_under_this_design(self) -> None:
        self.assertGreater(self.result["contrast_p2"], self.result["contrast_p512"])

    def test_classification_paired_replicate_count(self) -> None:
        rows = self.rows("classification_replicates")
        self.assertEqual(len(rows), len(analysis.DIMS)*5)
        self.assertTrue(all(len({r["oracle_accuracy"] for r in rows
                                 if r["repeat"] == str(i)}) == 1 for i in range(5)))

    def test_high_dim_knn_loss_under_this_design(self) -> None:
        self.assertGreater(self.result["knn_p2_acc"], self.result["knn_p512_acc"])

    def test_nuisance_scaling_degradation(self) -> None:
        self.assertGreater(self.result["scaled_noise_01_accuracy"],
                           self.result["scaled_noise_20_accuracy"])

    def test_jl_distortion_improves_with_dimension(self) -> None:
        self.assertGreater(self.result["jl_k8_p95"], self.result["jl_k128_p95"])

    def test_jl_pair_count(self) -> None:
        rows = self.rows("jl_replicates")
        self.assertTrue(all(int(r["pairs"]) == 170*169//2 for r in rows))

    def test_hub_occurrence_conservation(self) -> None:
        counts = self.rows("hub_example_counts")
        for dim in (2, 512):
            self.assertEqual(sum(int(r["occurrences"]) for r in counts
                                 if int(r["dimension"]) == dim), 3500)

    def test_hubness_increase_in_this_design(self) -> None:
        self.assertGreater(self.result["hub_p512_share"],
                           self.result["hub_p2_share"])

    def test_intrinsic_population_rank_without_noise(self) -> None:
        row = next(r for r in self.rows("intrinsic_summary")
                   if float(r["noise_sd"]) == 0.)
        self.assertAlmostEqual(float(row["population_participation_rank_mean"]), 5.)

    def test_intrinsic_sample_rank_increases_with_noise(self) -> None:
        self.assertGreater(self.result["intrinsic_max_noise_effective_rank"],
                           self.result["intrinsic_no_noise_effective_rank"])

    def test_metric_tasks_have_all_three_dissimilarities(self) -> None:
        rows = self.rows("metric_summary")
        self.assertEqual(len(rows), 6)
        self.assertEqual(Counter(r["task"] for r in rows),
                         {"angular": 3, "radial": 3})

    def test_study_summary_covers_returned_values(self) -> None:
        rows = self.rows("study_summary")
        self.assertEqual({r["metric"] for r in rows}, set(self.result))
        self.assertTrue(all(math.isfinite(float(r["value"])) for r in rows))


class NotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_explains_quick_scope(self) -> None:
        notebook = nbformat.read(ROOT/"nearest_neighbor_geometry.ipynb", as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(str(cell.source) for cell in notebook.cells)
        self.assertIn("Design(quick=True", source)
        self.assertIn("synthetic", source.lower())
        self.assertIn("Bayes", source)

    @unittest.skipUnless(os.environ.get("RUN_NOTEBOOK_EXECUTION") == "1",
                         "Set RUN_NOTEBOOK_EXECUTION=1 for a fresh-kernel execution")
    def test_fresh_kernel_executes_without_errors(self) -> None:
        from nbclient import NotebookClient

        notebook = nbformat.read(ROOT/"nearest_neighbor_geometry.ipynb", as_version=4)
        # nbclient's Jupyter kernel needs local IPC sockets; the GitHub
        # runner permits them even if an interactive sandbox does not.
        client = NotebookClient(notebook, timeout=240, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        self.assertTrue(any(c.cell_type == "code" and c.get("outputs")
                            for c in notebook.cells))


if __name__ == "__main__":
    unittest.main()
