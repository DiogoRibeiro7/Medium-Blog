"""Tests of mathematical claims, reproducibility, and publication artifacts."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis import (  # noqa: E402
    ALPHAS,
    FIGURE_NAMES,
    SAMPLE_SIZES,
    TABLE_NAMES,
    ExperimentConfig,
    chain_precision,
    covariance_to_correlation,
    edge_scores,
    fit_graph,
    heldout_nll,
    network_precision,
    partial_correlations,
    require_matrix,
    residualize,
    run,
    summarize_recovery,
)


class MathematicalTests(unittest.TestCase):
    def test_chain_spd(self) -> None:
        self.assertGreater(np.linalg.eigvalsh(chain_precision()).min(), 0)

    def test_chain_precision_zero(self) -> None:
        self.assertEqual(chain_precision()[0, 2], 0)

    def test_chain_partial_zero(self) -> None:
        self.assertAlmostEqual(partial_correlations(chain_precision())[0, 2], 0)

    def test_chain_marginal_nonzero(self) -> None:
        covariance = np.linalg.inv(chain_precision())
        self.assertGreater(covariance_to_correlation(covariance)[0, 2], 0.20)

    def test_partial_correlations_symmetric(self) -> None:
        r = partial_correlations(network_precision())
        assert_allclose(r, r.T)

    def test_partial_correlation_diagonal(self) -> None:
        assert_allclose(np.diag(partial_correlations(network_precision())), 1)

    def test_network_spd(self) -> None:
        self.assertGreater(np.linalg.eigvalsh(network_precision()).min(), 0)

    def test_network_edge_count(self) -> None:
        self.assertEqual(np.count_nonzero(np.triu(network_precision(), k=1)), 15)

    def test_network_rejects_other_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            network_precision(10)

    def test_zero_off_diagonal_precision_has_no_edges(self) -> None:
        scores = edge_scores(np.eye(12), network_precision())
        self.assertEqual(scores["true_positives"], 0)
        self.assertEqual(scores["false_positives"], 0)
        self.assertEqual(scores["false_negatives"], 15)

    def test_oracle_edge_recovery(self) -> None:
        scores = edge_scores(network_precision(), network_precision())
        self.assertEqual(scores["edge_f1"], 1)
        self.assertEqual(scores["structural_hamming_distance"], 0)

    def test_spurious_complete_graph_edges(self) -> None:
        omega = network_precision()
        full = omega.copy()
        full[np.triu_indices(12, 1)] = 0.15
        full[np.tril_indices(12, -1)] = 0.15
        scores = edge_scores(full, omega)
        self.assertEqual(scores["edge_recall"], 1)
        self.assertGreater(scores["false_positive_rate"], .9)

    def test_edge_scores_reject_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            edge_scores(np.eye(3), np.eye(4))

    def test_correlation_identity(self) -> None:
        assert_allclose(covariance_to_correlation(np.eye(4)), np.eye(4))

    def test_correlation_rejects_zero_variance(self) -> None:
        with self.assertRaises(ValueError):
            covariance_to_correlation(np.diag([1.0, 0.0]))

    def test_partial_rejects_non_spd(self) -> None:
        with self.assertRaises(ValueError):
            partial_correlations(np.diag([1.0, -1.0]))

    def test_partial_rejects_non_symmetric(self) -> None:
        with self.assertRaises(ValueError):
            partial_correlations(np.array([[2.0, 1.0], [0.0, 2.0]]))

    def test_nonfinite_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_matrix(np.array([[np.nan, 0.0]]))

    def test_vector_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_matrix(np.arange(4.0))

    def test_nonsquare_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_matrix(np.ones((3, 2)), square=True)

    def test_residual_orthogonality(self) -> None:
        rng = np.random.default_rng(11)
        z = rng.normal(size=300)
        y = 3 + 2 * z + rng.normal(size=300)
        res = residualize(y, z)
        self.assertAlmostEqual(float(np.mean(res)), 0, places=12)
        self.assertAlmostEqual(float(np.mean(res * z)), 0, places=12)

    def test_residual_rejects_bad_length(self) -> None:
        with self.assertRaises(ValueError):
            residualize(np.ones(5), np.ones(4))

    def test_fit_graph_spd(self) -> None:
        rng = np.random.default_rng(12)
        x = rng.normal(size=(100, 6))
        fitted = fit_graph(x, 0.1)
        self.assertGreater(np.linalg.eigvalsh(fitted).min(), 0)

    def test_fit_graph_rejects_zero_penalty(self) -> None:
        with self.assertRaises(ValueError):
            fit_graph(np.ones((5, 2)), 0)

    def test_fit_graph_rejects_constant_column(self) -> None:
        with self.assertRaises(ValueError):
            fit_graph(np.ones((5, 2)), 0.1)

    def test_heldout_nll_matches_analytic_identity(self) -> None:
        rng = np.random.default_rng(13)
        x = rng.normal(size=(200, 2))
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        expected = float(np.log(2 * np.pi) + np.mean(np.sum(z * z, axis=1)) / 2)
        self.assertAlmostEqual(heldout_nll(x, x, np.eye(2)), expected)

    def test_nll_rejects_non_spd(self) -> None:
        rng = np.random.default_rng(14)
        x = rng.normal(size=(20, 2))
        with self.assertRaises(ValueError):
            heldout_nll(x, x, np.diag([1., -1.]))

    def test_quick_config_repetitions(self) -> None:
        config = ExperimentConfig(quick=True)
        self.assertEqual((config.recovery_repeats, config.bootstrap_repeats), (8, 12))

    def test_full_config_repetitions(self) -> None:
        config = ExperimentConfig(quick=False)
        self.assertEqual((config.recovery_repeats, config.bootstrap_repeats), (70, 150))

    def test_summarize_uses_paired_design(self) -> None:
        rows: list[dict[str, float]] = []
        for n in SAMPLE_SIZES:
            for a in ALPHAS:
                for i in range(3):
                    rows.append({"n_train": float(n), "alpha": a,
                                 **{key: float(i) for key in (
                                     "edge_precision", "edge_recall", "edge_f1",
                                     "false_positive_rate", "structural_hamming_distance",
                                     "heldout_nll", "diagonal_nll")}})
        summary = summarize_recovery(rows)
        self.assertEqual(len(summary), 9)
        for row in summary:
            self.assertEqual(row["repeats"], 3)
            self.assertEqual(row["edge_f1_mean"], 1)

    def test_colliding_parents_analytic_partial(self) -> None:
        self.assertAlmostEqual(-(.9**2) / (1 + .9**2), -.4475138121546961)

    def test_nonlinear_theoretical_explained_variance(self) -> None:
        self.assertAlmostEqual(2 / (2 + .7**2), .8032128514056225)


class EvidenceTests(unittest.TestCase):
    """One quick run generates all artifacts for deterministic, focused checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="ggm-test-")
        cls.root = Path(cls.temporary.name)
        cls.metrics = run(ExperimentConfig(quick=True, output_root=cls.root))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def read_rows(self, name: str) -> list[dict[str, str]]:
        path = self.root / "results" / f"{name}_quick.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_all_figure_names_unique(self) -> None:
        self.assertEqual(len(set(FIGURE_NAMES)), len(FIGURE_NAMES))

    def test_all_table_names_unique(self) -> None:
        self.assertEqual(len(set(TABLE_NAMES)), len(TABLE_NAMES))

    def test_figure_count(self) -> None:
        self.assertEqual(len(FIGURE_NAMES), 9)

    def test_csv_count(self) -> None:
        self.assertEqual(len(TABLE_NAMES), 13)

    def test_png_files_nonempty(self) -> None:
        for name in FIGURE_NAMES:
            with self.subTest(name=name):
                path = self.root / "figures" / f"{name}_quick.png"
                self.assertGreater(path.stat().st_size, 8_000)

    def test_csv_files_nonempty(self) -> None:
        for name in TABLE_NAMES:
            with self.subTest(name=name):
                self.assertGreater(len(self.read_rows(name)), 0)

    def test_chain_population_has_no_shortcut(self) -> None:
        self.assertLess(abs(self.metrics["chain_population_partial_corr_13"]), 1e-12)

    def test_chain_marginal_association(self) -> None:
        self.assertGreater(self.metrics["chain_population_marginal_corr_13"], .2)

    def test_chain_empirical_partial_near_zero(self) -> None:
        self.assertLess(abs(self.metrics["chain_sample_partial_corr_13"]), .08)

    def test_collider_empirical_sign(self) -> None:
        self.assertLess(self.metrics["collider_sample_partial_corr"], -.35)

    def test_collider_unconditional_near_zero(self) -> None:
        self.assertLess(abs(self.metrics["collider_sample_marginal_corr"]), .08)

    def test_latent_produces_six_edges(self) -> None:
        self.assertEqual(self.metrics["latent_nonzero_observed_edges"], 6)

    def test_rank_is_at_most_n_minus_one(self) -> None:
        self.assertLessEqual(self.metrics["sample_covariance_rank"], 19)

    def test_recovery_has_nine_cells(self) -> None:
        self.assertEqual(len(self.read_rows("recovery_summary")), 9)

    def test_recovery_has_paired_alpha_repeats(self) -> None:
        self.assertEqual(len(self.read_rows("recovery_replicates")), 72)

    def test_alpha_order_and_repeats(self) -> None:
        rows = self.read_rows("recovery_replicates")
        for n in SAMPLE_SIZES:
            ids = {int(float(r["repeat"])) for r in rows
                   if int(float(r["n_train"])) == n}
            self.assertEqual(ids, set(range(8)))

    def test_recovery_metrics_bounded(self) -> None:
        for row in self.read_rows("recovery_replicates"):
            for name in ("edge_precision", "edge_recall", "edge_f1",
                         "false_positive_rate"):
                self.assertLessEqual(0, float(row[name]))
                self.assertLessEqual(float(row[name]), 1)

    def test_bootstrap_all_undirected_pairs(self) -> None:
        self.assertEqual(len(self.read_rows("edge_stability")), 66)

    def test_bootstrap_frequencies_bounded(self) -> None:
        for row in self.read_rows("edge_stability"):
            freq = float(row["bootstrap_selection_frequency"])
            self.assertGreaterEqual(freq, 0)
            self.assertLessEqual(freq, 1)

    def test_bootstrap_true_edge_count(self) -> None:
        count = sum(int(row["true_edge"]) for row in self.read_rows("edge_stability"))
        self.assertEqual(count, 15)

    def test_nonlinear_outperforms_linear(self) -> None:
        self.assertGreater(self.metrics["nonlinear_quadratic_test_r2"], .65)
        self.assertLess(self.metrics["nonlinear_linear_test_r2"], .05)

    def test_orientation_observational_equivalence(self) -> None:
        for row in self.read_rows("orientation_covariance"):
            self.assertEqual(row["forward_sem_covariance"],
                             row["reverse_sem_covariance"])

    def test_summary_matches_returned_metrics(self) -> None:
        table = {r["metric"]: float(r["value"]) for r in
                 self.read_rows("study_summary")}
        self.assertEqual(set(table), set(self.metrics))
        for key in self.metrics:
            self.assertAlmostEqual(table[key], self.metrics[key], places=8)

    def test_quick_never_generates_full_names(self) -> None:
        for name in TABLE_NAMES:
            self.assertFalse((self.root / "results" / f"{name}.csv").exists())
        for name in FIGURE_NAMES:
            self.assertFalse((self.root / "figures" / f"{name}.png").exists())

    def test_recovery_rows_finite(self) -> None:
        for row in self.read_rows("recovery_replicates"):
            self.assertTrue(all(np.isfinite(float(v)) for v in row.values()))


class NotebookTests(unittest.TestCase):
    """Execute shipped notebook in a fresh kernel, without hidden state."""

    def test_clean_notebook(self) -> None:
        import nbformat
        from nbclient import NotebookClient

        notebook_path = PROJECT_ROOT / "gaussian_graphical_models.ipynb"
        self.assertTrue(notebook_path.is_file())
        with notebook_path.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        nbformat.validate(notebook)
        executed = NotebookClient(notebook, timeout=240,
                                  kernel_name="python3",
                                  resources={"metadata": {"path": str(PROJECT_ROOT)}})
        executed.execute()


if __name__ == "__main__":
    unittest.main()
