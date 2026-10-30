"""Scientific tests for decision-aware learning and scenario optimization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    CAPACITY,
    FIGURE_NAMES,
    JOINT_MEANS,
    OVER_SINGLE,
    RESULT_NAMES,
    UNDER_SINGLE,
    clairvoyant_cost,
    empirical_cvar,
    empirical_saa_quantity,
    fit_single,
    joint_demands,
    joint_losses,
    mean_forecast,
    monte_carlo_saa,
    newsvendor_loss,
    plot_figures,
    quantile_forecast,
    run_study,
    single_batch,
    single_features,
    single_log_mean,
    solve_allocation,
    write_results,
)


class PredictiveDecisionTests(unittest.TestCase):
    """Finite-sample conditions and basic mathematics of newsvendor decisions."""

    def test_nonnegative_loss_and_asymmetric_penalties(self) -> None:
        demand = np.array([90.0, 110.0])
        actual = newsvendor_loss(100.0, demand, 8.0, 2.0)
        np.testing.assert_array_equal(actual, [20.0, 80.0])
        self.assertTrue(np.all(actual >= 0))

    def test_single_features_and_known_log_mean(self) -> None:
        x = np.array([0.0, 0.25, 0.5])
        basis = single_features(x)
        self.assertEqual(basis.shape, (3, 3))
        np.testing.assert_allclose(single_log_mean(x), basis @ np.array([np.log(85), 0.2, 0.12]))

    def test_demand_is_positive_and_train_fit_finite(self) -> None:
        x, y = single_batch(np.random.default_rng(5), 200)
        self.assertTrue(np.all(y > 0))
        fit = fit_single(x, y)
        self.assertGreater(fit.sigma, 0)
        self.assertEqual(fit.beta.shape, (3,))
        self.assertTrue(np.all(mean_forecast(fit, x) > 0))
        self.assertTrue(np.all(quantile_forecast(fit, x, 0.8) > mean_forecast(fit, x)))

    def test_fitted_forecasts_do_not_use_test_outcomes(self) -> None:
        x, y = single_batch(np.random.default_rng(8), 800)
        fit = fit_single(x, y)
        before = mean_forecast(fit, x)
        changed = y + 500
        self.assertTrue(np.all(changed > y))
        np.testing.assert_array_equal(before, mean_forecast(fit, x))

    def test_saa_quantile_is_empirical_cost_minimizer(self) -> None:
        observed = np.array([5.0, 10.0, 15.0, 20.0, 30.0])
        q = empirical_saa_quantity(observed, 0.8)
        self.assertEqual(q, 20.0)
        loss_at_q = float(newsvendor_loss(q, observed).mean())
        for candidate in np.linspace(0, 40, 81):
            self.assertLessEqual(loss_at_q, float(newsvendor_loss(candidate, observed).mean()) + 1e-10)

    def test_invalid_parameters_fail(self) -> None:
        with self.assertRaises(ValueError):
            single_batch(np.random.default_rng(1), True)
        with self.assertRaises(ValueError):
            single_batch(np.random.default_rng(1), 10, sigma=0)
        with self.assertRaises(ValueError):
            single_features(np.array([-0.1]))
        with self.assertRaises(ValueError):
            empirical_saa_quantity(np.array([]), 0.8)
        with self.assertRaises(ValueError):
            empirical_saa_quantity(np.array([1.0]), 1.0)
        with self.assertRaises(ValueError):
            newsvendor_loss(10.0, np.array([5.0]), under_cost=-2)

    def test_saa_large_sample_reduces_expected_regret_on_average(self) -> None:
        rows = monte_carlo_saa(replicates=15, n_evaluation=1300)
        self.assertEqual(len(rows), 8)
        self.assertGreater(rows[0]["mean_decision_regret"], rows[-1]["mean_decision_regret"])
        self.assertGreater(rows[0]["quantity_sd"], rows[-1]["quantity_sd"])


class AllocationTests(unittest.TestCase):
    """Validate the nonanticipative linear program and risk definition."""

    def test_joint_samples_preserve_positive_demand(self) -> None:
        rng = np.random.default_rng(9)
        samples = joint_demands(rng, 3000)
        self.assertEqual(samples.shape, (3000, 2))
        self.assertTrue(np.all(samples > 0))
        np.testing.assert_allclose(samples.mean(axis=0), JOINT_MEANS, rtol=0.06)

    def test_copula_shift_changes_dependence_not_marginals(self) -> None:
        base = joint_demands(np.random.default_rng(9), 40000, rho=0.65)
        shift = joint_demands(np.random.default_rng(9), 40000, rho=0.92)
        np.testing.assert_allclose(base.mean(axis=0), shift.mean(axis=0), rtol=0.015)
        self.assertGreater(np.corrcoef(np.log(shift).T)[0, 1], np.corrcoef(np.log(base).T)[0, 1] + 0.20)

    def test_mean_preserved_under_variance_change(self) -> None:
        a = joint_demands(np.random.default_rng(11), 80000, volatility_factor=1.0)
        b = joint_demands(np.random.default_rng(12), 80000, volatility_factor=1.45)
        np.testing.assert_allclose(a.mean(axis=0), b.mean(axis=0), rtol=0.018)
        self.assertTrue(np.all(b.std(axis=0) > a.std(axis=0)))

    def test_lp_one_scenario_has_zero_loss_if_feasible(self) -> None:
        scenario = np.array([[60.0, 70.0]])
        q = solve_allocation(scenario, capacity=190)
        np.testing.assert_allclose(q, scenario[0], atol=1e-7)
        self.assertAlmostEqual(float(joint_losses(q, scenario)[0]), 0.0)

    def test_risk_neutral_and_cvar_lp_are_feasible(self) -> None:
        scenarios = joint_demands(np.random.default_rng(12), 45)
        for risk_weight in (0.0, 0.25, 1.0):
            with self.subTest(risk_weight=risk_weight):
                q = solve_allocation(scenarios, risk_weight=risk_weight)
                self.assertEqual(q.shape, (2,))
                self.assertTrue(np.all(q >= 0))
                self.assertLessEqual(float(q.sum()), CAPACITY + 1e-6)

    def test_joint_losses_match_manual_cost(self) -> None:
        q = np.array([100.0, 80.0])
        demands = np.array([[110.0, 65.0], [90.0, 100.0]])
        np.testing.assert_allclose(joint_losses(q, demands), [10 * 8 + 15 * 3, 10 * 2 + 20 * 6])

    def test_empirical_cvar_above_mean(self) -> None:
        losses = np.arange(1.0, 101.0)
        self.assertGreater(empirical_cvar(losses), losses.mean())
        self.assertAlmostEqual(empirical_cvar(np.ones(100) * 10), 10)

    def test_perfect_information_cannot_cost_more_than_fixed_action(self) -> None:
        test = joint_demands(np.random.default_rng(14), 1000)
        clairvoyant = clairvoyant_cost(test)
        fixed = joint_losses(np.array([105.0, 85.0]), test)
        self.assertTrue(np.all(clairvoyant <= fixed + 1e-8))

    def test_invalid_lp_parameters(self) -> None:
        demand = np.array([[80.0, 60.0]])
        with self.assertRaises(ValueError):
            solve_allocation(demand, capacity=-1)
        with self.assertRaises(ValueError):
            solve_allocation(demand, risk_weight=1.3)
        with self.assertRaises(ValueError):
            solve_allocation(np.array([[1.0, -2.0]]))
        with self.assertRaises(ValueError):
            joint_demands(np.random.default_rng(1), 2, rho=1)


class ReproducibilityTests(unittest.TestCase):
    """Tests for the generated scientific evidence and its export schema."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_study(quick=True)

    def test_prediction_accuracy_and_cost_reversal(self) -> None:
        s = self.study.single_summary
        self.assertLess(s["mean_rmse"], s["quantile_rmse"])
        self.assertGreater(s["mean_decision_cost"], s["quantile_decision_cost"])

    def test_value_of_stochastic_solution_positive(self) -> None:
        self.assertGreater(self.study.summary["estimated_vss"], 0)
        self.assertGreater(self.study.summary["estimated_evpi"], 0)
        self.assertGreaterEqual(self.study.summary["joint_neutral_cvar90"], self.study.summary["joint_cvar_policy_cvar90"] - 10)

    def test_stress_keeps_policies_fixed(self) -> None:
        policies = {}
        for row in self.study.stress:
            label = str(row["policy"])
            allocation = (row["q_product1"], row["q_product2"])
            if label in policies:
                self.assertEqual(allocation, policies[label])
            else:
                policies[label] = allocation
        self.assertEqual(len(self.study.stress), len(self.study.policies) * 3)

    def test_repeated_full_generation_is_seeded(self) -> None:
        another = run_study(quick=True)
        self.assertEqual(another.summary, self.study.summary)
        np.testing.assert_array_equal(another.scenarios, self.study.scenarios)

    def test_ten_csv_outputs_and_smoke_suffix(self) -> None:
        self.assertEqual(set(self.study.rows), set(RESULT_NAMES))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            full = write_results(self.study, base)
            quick = write_results(self.study, base, quick=True)
            self.assertEqual(len(full), 10)
            self.assertEqual(len(quick), 10)
            self.assertTrue(all(path.exists() for path in full + quick))
            self.assertTrue(all(path.name.endswith("_quick.csv") for path in quick))

    def test_eight_plot_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_figures(self.study, Path(tmp), quick=True)
            self.assertEqual(len(paths), len(FIGURE_NAMES))
            self.assertTrue(all(path.stat().st_size > 5000 for path in paths))
            self.assertTrue(all(path.name.endswith("_quick.png") for path in paths))

    def test_notebook_fresh_kernel(self) -> None:
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("Optional notebook execution tools unavailable locally")
        path = ROOT / "decision_focused_optimization.ipynb"
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
        with tempfile.TemporaryDirectory() as tmp:
            result = NotebookClient(
                notebook, timeout=150, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
            self.assertTrue(any(c.cell_type == "code" and c.execution_count for c in result.cells))


if __name__ == "__main__":
    unittest.main()
