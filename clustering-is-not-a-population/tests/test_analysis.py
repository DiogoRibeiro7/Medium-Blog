"""Unit, numerical, evidence-integration, and notebook execution checks."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

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
    """Parse versioned evidence without implicit numeric-type conversion."""

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class MatrixValidationTests(unittest.TestCase):
    def test_finite_matrix_is_valid(self) -> None:
        x = np.arange(12, dtype=float).reshape(4, 3)
        np.testing.assert_array_equal(analysis.check_matrix(x), x)

    def test_integer_data_converted(self) -> None:
        self.assertEqual(analysis.check_matrix(np.ones((3, 2), dtype=int)).dtype, float)

    def test_vector_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.ones(5))

    def test_cube_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.ones((3, 2, 2)))

    def test_one_row_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.ones((1, 3)))

    def test_two_rows_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.ones((2, 3)))

    def test_empty_features_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.empty((5, 0)))

    def test_nan_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.array([[1., 2.], [2., np.nan], [3., 4.]]))

    def test_infinity_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.check_matrix(np.array([[1., 2.], [2., np.inf], [3., 4.]]))

    def test_config_seed_rejected(self) -> None:
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as directory:
            analysis.run(analysis.Design(root=Path(directory), quick=True, seed=-1))

    def test_config_root_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            analysis.run(analysis.Design(root="bad", quick=True))  # type: ignore[arg-type]

    def test_standardizing_constant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.marginal_standardize(np.ones((5, 2)))


class AlgorithmTests(unittest.TestCase):
    def test_kmeans_two_separated_groups(self) -> None:
        x = np.array([[-5., 0.], [-4., 0.], [-3., 0.],
                      [3., 0.], [4., 0.], [5., 0.]])
        y = [0, 0, 0, 1, 1, 1]
        self.assertAlmostEqual(adjusted_rand_score(y, analysis.kmeans(x, 2, 9).labels_), 1.)

    def test_kmeans_one_center_equals_mean(self) -> None:
        x = np.arange(12, dtype=float).reshape(6, 2)
        np.testing.assert_allclose(analysis.kmeans(x, 1, 3).cluster_centers_[0], x.mean(axis=0))

    def test_kmeans_inertia_nonnegative(self) -> None:
        x = np.random.default_rng(3).normal(size=(35, 4))
        self.assertGreaterEqual(analysis.kmeans(x, 3, 4).inertia_, 0.)

    def test_kmeans_inertia_matches_definition(self) -> None:
        x = np.random.default_rng(4).normal(size=(45, 3))
        fit = analysis.kmeans(x, 3, 5)
        expected = np.sum((x - fit.cluster_centers_[fit.labels_]) ** 2)
        self.assertAlmostEqual(fit.inertia_, float(expected))

    def test_kmeans_same_seed_reproducible(self) -> None:
        x = np.random.default_rng(5).normal(size=(65, 4))
        a = analysis.kmeans(x, 3, 19)
        b = analysis.kmeans(x, 3, 19)
        np.testing.assert_array_equal(a.labels_, b.labels_)

    def test_kmeans_bad_zero_k(self) -> None:
        with self.assertRaises(ValueError):
            analysis.kmeans(np.ones((6, 2)), 0, 2)

    def test_kmeans_bad_large_k(self) -> None:
        with self.assertRaises(ValueError):
            analysis.kmeans(np.ones((6, 2)), 7, 2)

    def test_gmm_probabilities_sum_to_one(self) -> None:
        x = np.random.default_rng(6).normal(size=(60, 2))
        p = analysis.gaussian_mixture(x, 2, 7).predict_proba(x)
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12)

    def test_gmm_bic_finite(self) -> None:
        x = np.random.default_rng(7).normal(size=(60, 2))
        model = analysis.gaussian_mixture(x, 2, 8)
        self.assertTrue(np.isfinite(model.bic(x)))

    def test_gmm_covariance_positive(self) -> None:
        x = np.random.default_rng(8).normal(size=(60, 2))
        model = analysis.gaussian_mixture(x, 2, 9)
        self.assertTrue(np.all(np.linalg.eigvalsh(model.covariances_) > 0))

    def test_gmm_prediction_shape(self) -> None:
        x = np.random.default_rng(9).normal(size=(60, 2))
        self.assertEqual(analysis.gaussian_mixture(x, 2, 10).predict(x).shape, (60,))

    def test_gmm_bad_k(self) -> None:
        with self.assertRaises(ValueError):
            analysis.gaussian_mixture(np.ones((6, 2)), -1, 3)

    def test_ari_invariant_to_label_permutation(self) -> None:
        a = [0, 0, 1, 1, 2, 2]
        b = [2, 2, 0, 0, 1, 1]
        self.assertAlmostEqual(adjusted_rand_score(a, b), 1.)

    def test_ari_is_chance_corrected(self) -> None:
        rng = np.random.default_rng(10)
        a = np.repeat(np.arange(3), 100)
        b = rng.permutation(a)
        self.assertLess(abs(adjusted_rand_score(a, b)), .2)

    def test_silhouette_finite(self) -> None:
        x = np.array([[-3., 0.], [-2., 0.], [-4., 0.],
                      [3., 0.], [2., 0.], [4., 0.]])
        self.assertGreater(silhouette_score(x, [0, 0, 0, 1, 1, 1]), .5)

    def test_dbscan_noise_label_is_minus_one(self) -> None:
        x = np.array([[0., 0.], [.01, 0.], [0., .01], [.01, .01], [100., 100.]])
        label = DBSCAN(eps=.05, min_samples=3).fit_predict(x)
        self.assertEqual(int(label[-1]), -1)


class GeneratorTests(unittest.TestCase):
    def test_mixture_shape(self) -> None:
        x, y = analysis.mixture_sample(np.random.default_rng(1), [5, 6],
                [np.zeros(2), np.ones(2)], [np.eye(2), np.eye(2)])
        self.assertEqual(x.shape, (11, 2))
        self.assertEqual(y.shape, (11,))

    def test_mixture_label_counts(self) -> None:
        _, y = analysis.mixture_sample(np.random.default_rng(1), [6, 9],
                [np.zeros(2), np.ones(2)], [np.eye(2), np.eye(2)])
        self.assertEqual(np.bincount(y).tolist(), [6, 9])

    def test_mixture_seed_reproducible(self) -> None:
        args = ([4, 5], [np.zeros(2), np.ones(2)], [np.eye(2), np.eye(2)])
        a, _ = analysis.mixture_sample(np.random.default_rng(8), *args)
        b, _ = analysis.mixture_sample(np.random.default_rng(8), *args)
        np.testing.assert_array_equal(a, b)

    def test_mixture_rejects_sizes(self) -> None:
        with self.assertRaises(ValueError):
            analysis.mixture_sample(np.random.default_rng(1), [1],
                                    [np.zeros(2)], [np.eye(2)])

    def test_mixture_rejects_component_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            analysis.mixture_sample(np.random.default_rng(1), [3, 4],
                                    [np.zeros(2)], [np.eye(2)])

    def test_rings_shape(self) -> None:
        x, y = analysis.rings_sample(np.random.default_rng(9), count=100)
        self.assertEqual(x.shape, (100, 2))
        self.assertEqual(np.bincount(y).tolist(), [50, 50])

    def test_rings_radius_differentiation(self) -> None:
        x, _ = analysis.rings_sample(np.random.default_rng(9), count=600)
        r = np.linalg.norm(x, axis=1)
        self.assertAlmostEqual(float(r[:300].mean()), 1., delta=.04)
        self.assertAlmostEqual(float(r[300:].mean()), 2., delta=.04)

    def test_rings_bad_odd_count(self) -> None:
        with self.assertRaises(ValueError):
            analysis.rings_sample(np.random.default_rng(1), count=99)

    def test_rings_bad_noise(self) -> None:
        with self.assertRaises(ValueError):
            analysis.rings_sample(np.random.default_rng(1), noise=0)

    def test_standardization_zero_mean(self) -> None:
        x = np.random.default_rng(9).normal(size=(100, 4))
        self.assertLess(float(np.max(np.abs(analysis.marginal_standardize(x).mean(0)))), 1e-12)

    def test_standardization_unit_sd(self) -> None:
        x = np.random.default_rng(9).normal(size=(100, 4))
        np.testing.assert_allclose(analysis.marginal_standardize(x).std(0, ddof=1), 1.)


class ConfigurationAndSummaryTests(unittest.TestCase):
    def test_study_count(self) -> None:
        self.assertEqual(len(analysis.EXPERIMENTS), 9)

    def test_figure_count(self) -> None:
        self.assertEqual(len(analysis.FIGURES), 10)

    def test_full_repetition_count(self) -> None:
        self.assertEqual(analysis.Design().repetitions["scale"], 24)

    def test_quick_repetition_count(self) -> None:
        self.assertEqual(analysis.Design(quick=True).repetitions["scale"], 3)

    def test_full_suffix_empty(self) -> None:
        self.assertEqual(analysis.Design().suffix, "")

    def test_quick_suffix(self) -> None:
        self.assertEqual(analysis.Design(quick=True).suffix, "_quick")

    def test_independent_streams(self) -> None:
        d = analysis.Design()
        self.assertNotEqual(d.stream("elbow", 0).random(),
                            d.stream("elbow", 1).random())

    def test_stream_deterministic(self) -> None:
        d = analysis.Design()
        self.assertEqual(d.stream("bic", 4).random(), d.stream("bic", 4).random())

    def test_unknown_stream_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.Design().stream("fake", 0)

    def test_negative_replicate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.Design().stream("bic", -1)

    def test_mcse_known_values(self) -> None:
        mu, se = analysis.mean_and_mcse([2., 4.])
        self.assertAlmostEqual(mu, 3.)
        self.assertAlmostEqual(se, 1.)

    def test_mcse_singleton_nan(self) -> None:
        _, se = analysis.mean_and_mcse([2.])
        self.assertTrue(math.isnan(se))

    def test_empty_mcse_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.mean_and_mcse([])

    def test_nonfinite_mcse_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.mean_and_mcse([1., np.inf])

    def test_summary_grouping(self) -> None:
        rows = [{"k": 1, "x": 2.}, {"k": 1, "x": 4.}, {"k": 2, "x": 9.}]
        summary = analysis.summarize(rows, ("k",), ("x",))
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["x_mean"], 3.)

    def test_summary_missing_groups_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([{"k": 1, "x": 2.}], (), ("x",))

    def test_write_empty_csv_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            analysis.write_csv(Path(d) / "no.csv", [])

    def test_write_ragged_csv_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            analysis.write_csv(Path(d) / "bad.csv", [{"a": 1}, {"b": 2}])

    def test_write_csv_precision(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "value.csv"
            analysis.write_csv(p, [{"x": 1 / 3}])
            self.assertAlmostEqual(float(csv_rows(p)[0]["x"]), 1 / 3, places=10)


class GeneratedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config = analysis.Design(root=cls.root, quick=True)
        cls.summary = analysis.run(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_all_eighteen_tables(self) -> None:
        self.assertEqual(len(list((self.root / "results").glob("*.csv"))), 18)

    def test_all_ten_figures(self) -> None:
        self.assertEqual(len(list((self.root / "figures").glob("*.png"))), 10)

    def test_every_figure_is_nonempty_png(self) -> None:
        for name in analysis.FIGURES:
            with self.subTest(name=name):
                p = self.root / "figures" / f"{name}_quick.png"
                self.assertGreater(p.stat().st_size, 10_000)
                self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_every_table_has_rows(self) -> None:
        for name in analysis.EXPERIMENTS:
            for kind in ("replicates", "summary"):
                with self.subTest(name=name, kind=kind):
                    p = self.root / "results" / f"{name}_{kind}_quick.csv"
                    self.assertTrue(csv_rows(p))

    def test_elbow_has_eight_choices_per_rep(self) -> None:
        rows = csv_rows(self.root / "results/elbow_replicates_quick.csv")
        self.assertEqual(len(rows), 3 * 8)

    def test_elbow_mean_decreases(self) -> None:
        values = [float(r["inertia_per_observation_mean"]) for r in self.summary["elbow"]]
        self.assertTrue(all(x > y for x, y in zip(values, values[1:], strict=False)))

    def test_scale_has_five_settings_per_rep(self) -> None:
        rows = csv_rows(self.root / "results/scale_replicates_quick.csv")
        self.assertEqual(len(rows), 3 * 5)

    def test_scale_raw_partition_reverses(self) -> None:
        s = self.summary["scale"]
        self.assertGreater(float(s[0]["raw_ari_mean"]), float(s[-1]["raw_ari_mean"]) + .3)

    def test_scale_standardization_resists_rescaling(self) -> None:
        s = self.summary["scale"]
        values = [float(r["standardized_ari_mean"]) for r in s]
        self.assertLess(max(values) - min(values), .08)

    def test_rings_has_five_eps_per_rep(self) -> None:
        rows = csv_rows(self.root / "results/rings_replicates_quick.csv")
        self.assertEqual(len(rows), 3 * 5)

    def test_rings_density_recovers_annuli(self) -> None:
        values = self.summary["rings"]
        near = next(r for r in values if float(r["eps"]) == .3)
        self.assertGreater(float(near["dbscan_ari_mean"]), .75)

    def test_rings_kmeans_does_not_recover_annuli(self) -> None:
        self.assertLess(float(self.summary["rings"][0]["kmeans_ari_mean"]), .1)

    def test_rings_parameter_sensitivity(self) -> None:
        values = self.summary["rings"]
        self.assertGreater(float(values[3]["dbscan_ari_mean"]),
                           float(values[0]["dbscan_ari_mean"]) + .5)

    def test_elliptical_gmm_beats_euclidean(self) -> None:
        row = self.summary["ellipses"][0]
        self.assertGreater(float(row["gmm_ari_mean"]),
                           float(row["kmeans_ari_mean"]) + .5)

    def test_elliptical_gmm_is_highly_aligned(self) -> None:
        self.assertGreater(float(self.summary["ellipses"][0]["gmm_ari_mean"]), .9)

    def test_dimensions_replicate_count(self) -> None:
        rows = csv_rows(self.root / "results/dimensions_replicates_quick.csv")
        self.assertEqual(len(rows), 3 * 5)

    def test_dimensions_noise_degrades_ari(self) -> None:
        s = self.summary["dimensions"]
        self.assertGreater(float(s[0]["kmeans_ari_mean"]),
                           float(s[-1]["kmeans_ari_mean"]) + .35)

    def test_silhouette_four_choices_per_rep(self) -> None:
        rows = csv_rows(self.root / "results/silhouette_replicates_quick.csv")
        self.assertEqual(len(rows), 3 * 4)

    def test_silhouette_prefers_two_for_this_generator(self) -> None:
        s = self.summary["silhouette"]
        self.assertGreater(float(s[0]["silhouette_mean"]),
                           float(s[1]["silhouette_mean"]) + .1)

    def test_silhouette_generator_has_three_components(self) -> None:
        rows = csv_rows(self.root / "results/silhouette_replicates_quick.csv")
        self.assertTrue(all(int(r["true_components"]) == 3 for r in rows))

    def test_bic_five_choices_per_rep(self) -> None:
        rows = csv_rows(self.root / "results/bic_replicates_quick.csv")
        self.assertEqual(len(rows), 3 * 5)

    def test_bic_prefers_two_for_this_generator(self) -> None:
        s = self.summary["bic"]
        self.assertEqual(min(s, key=lambda r: float(r["bic_mean"]))["k"], 2)

    def test_stability_is_high_even_for_one_gaussian(self) -> None:
        s = self.summary["stability"][0]
        self.assertGreater(float(s["bootstrap_ari_mean"]), .5)
        self.assertEqual(s["scenario"], "stability")

    def test_stability_generator_is_unimodal(self) -> None:
        rows = csv_rows(self.root / "results/stability_replicates_quick.csv")
        self.assertTrue(all(int(r["known_gaussian_components"]) == 1 for r in rows))

    def test_posterior_four_separations_per_rep(self) -> None:
        rows = csv_rows(self.root / "results/posterior_replicates_quick.csv")
        self.assertEqual(len(rows), 3 * 4)

    def test_posterior_hard_ari_improves_with_separation(self) -> None:
        s = self.summary["posterior"]
        self.assertGreater(float(s[-1]["gmm_ari_mean"]),
                           float(s[0]["gmm_ari_mean"]) + .5)

    def test_posterior_ambiguity_declines_at_large_separation(self) -> None:
        s = self.summary["posterior"]
        self.assertLess(float(s[-1]["ambiguous_fraction_mean"]),
                        float(s[1]["ambiguous_fraction_mean"]))

    def test_all_summary_means_finite(self) -> None:
        for name, rows in self.summary.items():
            for row in rows:
                for key, value in row.items():
                    if key.endswith("_mean"):
                        with self.subTest(experiment=name, key=key):
                            self.assertTrue(np.isfinite(float(value)))

    def test_quick_suffix_protects_canonical_outputs(self) -> None:
        paths = [p.name for p in (self.root / "results").glob("*.csv")]
        self.assertTrue(all(name.endswith("_quick.csv") for name in paths))

    def test_all_experiments_record_independent_repetitions(self) -> None:
        for name in analysis.EXPERIMENTS:
            rows = csv_rows(self.root / "results" / f"{name}_replicates_quick.csv")
            self.assertEqual(sorted({int(r["replicate"]) for r in rows}), [0, 1, 2])

    def test_figures_unique_names(self) -> None:
        self.assertEqual(len(set(analysis.FIGURES)), len(analysis.FIGURES))


class NotebookTests(unittest.TestCase):
    def test_notebook_builder_shape(self) -> None:
        nb = notebook_builder.notebook()
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 6)

    def test_notebook_ids_unique(self) -> None:
        cells = notebook_builder.notebook()["cells"]
        self.assertEqual(len({c["id"] for c in cells}), len(cells))

    def test_notebook_json_exists(self) -> None:
        data = json.loads((ROOT / "clustering_is_not_a_population.ipynb").read_text())
        self.assertEqual(data, notebook_builder.notebook())

    def test_notebook_valid_if_nbformat_available(self) -> None:
        if nbformat is None:
            self.skipTest("nbformat is not installed in this shell")
        nbformat.validate(nbformat.read(
            ROOT / "clustering_is_not_a_population.ipynb", as_version=4
        ))

    def test_fresh_kernel_executes(self) -> None:
        if os.getenv("RUN_NOTEBOOK_EXECUTION") != "1":
            self.skipTest("Enable with RUN_NOTEBOOK_EXECUTION=1")
        if nbformat is None or NotebookClient is None:
            self.fail("Install notebook dependencies for the fresh-kernel test")
        notebook = nbformat.read(
            ROOT / "clustering_is_not_a_population.ipynb", as_version=4
        )
        client = NotebookClient(
            notebook, timeout=300, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        result = client.execute()
        self.assertTrue(any(
            output.output_type == "display_data"
            for cell in result.cells if cell.cell_type == "code"
            for output in cell.get("outputs", [])
        ))


if __name__ == "__main__":
    unittest.main()
