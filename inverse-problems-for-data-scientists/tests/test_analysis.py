"""Mathematical invariants and reproduction checks for the inverse study."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    DECAY,
    FIGURE_NAMES,
    N,
    NOISE_SD,
    REFERENCE_ALPHA,
    RESULT_NAMES,
    SENSOR_BUDGET,
    SEED,
    bayesian_full,
    cosine_basis,
    greedy_a_optimal,
    normal_prior_sd,
    plot_figures,
    posterior_sensor,
    posterior_trace_sequence,
    regularization_path,
    run_study,
    signal_truth,
    smoothing_operator,
    spread_sensor_order,
    tikhonov,
    truncated_svd,
    write_results,
)


class InverseMathematicsTests(unittest.TestCase):
    """Validate formulae separately from a particular random realization."""

    def test_basis_is_orthonormal(self) -> None:
        basis = cosine_basis(20)
        np.testing.assert_allclose(basis.T @ basis, np.eye(20), atol=1e-13)

    def test_invalid_size_and_numerical_params(self) -> None:
        for value in (True, 3, 2.5, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cosine_basis(value)
        with self.assertRaises(ValueError):
            smoothing_operator(16, 0.0)
        with self.assertRaises(ValueError):
            tikhonov(np.ones(8), cosine_basis(8), np.ones(8), 0.0)
        with self.assertRaises(ValueError):
            truncated_svd(np.ones(8), cosine_basis(8), np.ones(8), 9)

    def test_forward_spectrum_matches_constructed_operator(self) -> None:
        operator, modes, spectrum = smoothing_operator(32, DECAY)
        np.testing.assert_allclose(operator, operator.T, atol=1e-12)
        np.testing.assert_allclose(operator @ modes, modes * spectrum, atol=1e-12)
        self.assertTrue(np.all(np.diff(spectrum) < 0.0))

    def test_tikhonov_obeys_normal_equations(self) -> None:
        operator, modes, spectrum = smoothing_operator(16, DECAY)
        y = signal_truth(16)
        alpha = 0.01
        answer = tikhonov(y, modes, spectrum, alpha)
        residual = (operator.T @ operator + alpha * np.eye(16)) @ answer - operator.T @ y
        self.assertLess(float(np.linalg.norm(residual)), 1e-11)

    def test_tsvd_exact_on_retained_modes(self) -> None:
        operator, modes, spectrum = smoothing_operator(24, DECAY)
        truth = signal_truth(24)
        observed = operator @ truth
        recovered = truncated_svd(observed, modes, spectrum, 10)
        np.testing.assert_allclose(modes[:, :10].T @ recovered, modes[:, :10].T @ truth, atol=1e-11)
        np.testing.assert_allclose(modes[:, 10:].T @ recovered, 0, atol=1e-11)

    def test_regularization_path_has_valid_effective_df_and_gcv(self) -> None:
        operator, modes, spectrum = smoothing_operator(16, DECAY)
        truth = signal_truth(16)
        gcv, latent_errors, df, residuals = regularization_path(
            operator @ truth + 0.01, truth, modes, spectrum, np.geomspace(1e-5, 0.1, 8)
        )
        self.assertTrue(np.all(gcv > 0))
        self.assertTrue(np.all(latent_errors >= 0))
        self.assertTrue(np.all((df > 0) & (df < 16)))
        self.assertTrue(np.all(np.diff(df) < 0))
        self.assertTrue(np.all(residuals > 0))

    def test_bayesian_posterior_variance_is_below_prior(self) -> None:
        operator, modes, spectrum = smoothing_operator(16, DECAY)
        prior = normal_prior_sd(16)
        mean, sd, variances = bayesian_full(
            operator @ signal_truth(16), modes, spectrum, NOISE_SD, prior
        )
        self.assertEqual(mean.shape, (16,))
        self.assertTrue(np.all(sd > 0))
        self.assertTrue(np.all(variances < prior**2))

    def test_sensor_update_reduces_covariance_trace(self) -> None:
        operator, modes, _ = smoothing_operator(16, DECAY)
        prior = (modes * normal_prior_sd(16)[None, :] ** 2) @ modes.T
        indices = np.array([0, 8, 4, 12], dtype=np.int64)
        _, posterior = posterior_sensor(operator, indices, prior, NOISE_SD)
        self.assertLess(float(np.trace(posterior)), float(np.trace(prior)))
        traces = posterior_trace_sequence(operator, prior, NOISE_SD, indices)
        self.assertTrue(np.all(np.diff(traces) < 0))
        self.assertAlmostEqual(float(traces[-1]), float(np.trace(posterior)), places=10)

    def test_greedy_selection_is_distinct_and_shrinks_trace(self) -> None:
        operator, modes, _ = smoothing_operator(16, DECAY)
        prior = (modes * normal_prior_sd(16)[None, :] ** 2) @ modes.T
        selected, traces = greedy_a_optimal(operator, prior, NOISE_SD, 8)
        self.assertEqual(len(np.unique(selected)), 8)
        self.assertTrue(np.all(np.diff(traces) < 0))
        with self.assertRaises(ValueError):
            greedy_a_optimal(operator, prior, NOISE_SD, 17)

    def test_spread_order_is_permutation(self) -> None:
        order = spread_sensor_order(N)
        np.testing.assert_array_equal(np.sort(order), np.arange(N))
        with self.assertRaises(ValueError):
            spread_sensor_order(18)

    def test_duplicate_sensor_indices_are_not_allowed(self) -> None:
        operator, modes, _ = smoothing_operator(16, DECAY)
        prior = (modes * normal_prior_sd(16)[None, :] ** 2) @ modes.T
        with self.assertRaises(ValueError):
            posterior_sensor(operator, np.array([0, 0]), prior, NOISE_SD)


class InverseStudyTests(unittest.TestCase):
    """Check reproducibility and interpretation of constructed evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_study(quick=True)

    def test_exact_nullspace_has_nontrivial_alternative(self) -> None:
        s = self.study
        self.assertEqual(s.sparse_indices.shape, (SENSOR_BUDGET,))
        self.assertGreater(float(np.linalg.norm(s.alternate_truth - s.truth)), 1.0)
        np.testing.assert_allclose(
            s.operator[s.sparse_indices] @ s.truth,
            s.operator[s.sparse_indices] @ s.alternate_truth,
            atol=1e-11,
        )

    def test_direct_inverse_fits_better_but_recovers_worse(self) -> None:
        s = self.study
        self.assertGreater(s.summary["condition_number"], 100_000)
        self.assertGreater(s.summary["ls_latent_rmse"], 100)
        self.assertLess(s.summary["tikhonov_latent_rmse"], 0.10)
        self.assertLess(
            float(np.linalg.norm(s.operator @ s.ls_solution - s.observed)),
            1e-8,
        )

    def test_gcv_and_oracle_choose_distinct_regularization(self) -> None:
        s = self.study
        self.assertIn(s.alpha_gcv, REFERENCE_ALPHA)
        self.assertIn(s.alpha_oracle, REFERENCE_ALPHA)
        self.assertNotEqual(s.alpha_gcv, s.alpha_oracle)
        self.assertLessEqual(s.summary["oracle_latent_rmse"], s.summary["tikhonov_latent_rmse"])

    def test_fixed_truth_coverage_not_assumed_nominal(self) -> None:
        coverage = self.study.pointwise_coverage
        self.assertTrue(np.all((coverage >= 0) & (coverage <= 1)))
        self.assertLess(float(np.min(coverage)), 0.6)
        self.assertGreater(float(np.max(coverage)), 0.95)

    def test_design_trace_and_paired_trial_count(self) -> None:
        s = self.study
        self.assertLess(s.design_trace["Greedy A-opt"][-1], s.design_trace["Clustered"][-1])
        self.assertLess(s.design_trace["Spread"][-1], s.design_trace["Clustered"][-1])
        for errors in s.sensor_trial_errors.values():
            self.assertEqual(errors[0].shape, (64,))
            self.assertEqual(errors[1].shape, (64,))

    def test_repeated_runs_and_quick_flag_types(self) -> None:
        first = self.study
        second = run_study(quick=True, seed=SEED)
        np.testing.assert_array_equal(first.observed, second.observed)
        np.testing.assert_array_equal(
            first.mc_method_errors["Tikhonov (GCV)"],
            second.mc_method_errors["Tikhonov (GCV)"],
        )
        with self.assertRaises(TypeError):
            run_study(quick="yes")  # type: ignore[arg-type]

    def test_publication_assets_and_smoke_outputs_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            figures, results = root / "figures", root / "results"
            plot_figures(self.study, figures)
            write_results(self.study, results)
            plot_figures(self.study, figures, quick=True)
            write_results(self.study, results, quick=True)
            self.assertEqual(len(list(figures.glob("*.png"))), 16)
            self.assertEqual(len(list(results.glob("*.csv"))), 20)
            for name in FIGURE_NAMES:
                self.assertTrue((figures / name).stat().st_size > 10_000)
            for name in RESULT_NAMES:
                self.assertTrue((results / name).stat().st_size > 30)
            self.assertEqual(len(self.study.result_rows), 10)

    def test_notebook_is_valid_code_json(self) -> None:
        notebook = json.loads((ROOT / "inverse_problems.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        for cell in code_cells:
            compile("".join(cell["source"]), "<notebook-cell>", "exec")

    @unittest.skipUnless(
        importlib.util.find_spec("nbclient") is not None
        and importlib.util.find_spec("nbformat") is not None,
        "notebook execution dependencies are optional locally",
    )
    def test_notebook_executes_in_fresh_kernel(self) -> None:
        """GitHub Actions installs the optional notebook runner dependencies."""

        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(ROOT / "inverse_problems.ipynb", as_version=4)
        client = NotebookClient(
            notebook, timeout=180, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()


if __name__ == "__main__":
    unittest.main()
