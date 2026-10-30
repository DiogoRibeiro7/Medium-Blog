"""Mathematical, statistical, and reproducibility tests for article #8."""

from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import weibull_min

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    COMPETING_SCALES,
    COMPETING_SHAPES,
    FAILURE_COST,
    FIGURE_NAMES,
    PREVENTIVE_COST,
    RESULT_NAMES,
    TRUE_SCALE,
    TRUE_SHAPE,
    CensoredSample,
    aalen_johansen,
    censored_log_likelihood,
    competing_truth,
    conditional_failure_probability,
    expected_age_replacement_rate,
    fit_weibull,
    kaplan_meier,
    optimize_replacement_age,
    run_experiment,
    simulate_censored,
    simulate_competing,
    simulate_renewal_policy,
    step_values,
    weibull_hazard,
    weibull_mean,
    weibull_survival,
)


class WeibullMathTests(unittest.TestCase):
    """Check identities and validation independently of the plotting pipeline."""

    def test_characteristic_life_and_survival_zero(self) -> None:
        self.assertAlmostEqual(float(weibull_survival(0.0, 2.2, 1000)), 1.0)
        self.assertAlmostEqual(
            float(weibull_survival(1000.0, 2.2, 1000)), float(np.exp(-1)), places=12
        )

    def test_exponential_hazard_constant(self) -> None:
        t = np.array([100., 500., 2000.])
        np.testing.assert_allclose(weibull_hazard(t, 1., 1000.), 1/1000)
        p = conditional_failure_probability(t, 120., 1., 1000.)
        np.testing.assert_allclose(p, 1 - np.exp(-.12), atol=1e-12)

    def test_increasing_and_decreasing_failure_hazards(self) -> None:
        age = np.array([300., 900.])
        self.assertGreater(weibull_hazard(age, 2.3, 1400.)[1],
                           weibull_hazard(age, 2.3, 1400.)[0])
        self.assertLess(weibull_hazard(age, .7, 1400.)[1],
                        weibull_hazard(age, .7, 1400.)[0])

    def test_conditional_failure_equals_survival_ratio(self) -> None:
        age = np.array([0., 400., 1100.])
        value = conditional_failure_probability(age, 280., 2.3, 1400.)
        ratio = 1 - weibull_survival(age+280, 2.3, 1400.) / weibull_survival(
            age, 2.3, 1400.
        )
        np.testing.assert_allclose(value, ratio, atol=1e-12)

    def test_weibull_mean_matches_scipy(self) -> None:
        self.assertAlmostEqual(
            weibull_mean(2.3, 1400.), float(weibull_min.mean(2.3, scale=1400.))
        )

    def test_invalid_parameters_rejected(self) -> None:
        with self.assertRaises(ValueError):
            weibull_survival(-1, 2, 1000)
        with self.assertRaises(ValueError):
            weibull_survival(1, 0, 1000)
        with self.assertRaises(ValueError):
            weibull_hazard(0, 2, 1000)
        with self.assertRaises(ValueError):
            conditional_failure_probability(100, 0, 2, 1000)
        with self.assertRaises(ValueError):
            expected_age_replacement_rate(0, 2, 1000)

    def test_right_censor_sample_consistent(self) -> None:
        d = simulate_censored(200, np.random.default_rng(123))
        np.testing.assert_allclose(d.observed, np.minimum(d.latent_lifetime, d.censor_time))
        np.testing.assert_array_equal(d.event, d.latent_lifetime <= d.censor_time)
        self.assertGreater(int(np.sum(d.event)), 0)
        self.assertLess(int(np.sum(d.event)), d.observed.size)

    def test_censored_sample_invalid_event_indicator(self) -> None:
        with self.assertRaises(TypeError):
            CensoredSample(np.array([1.]), np.array([1]), np.array([1.]), np.array([2.]))

    def test_censored_likelihood_matches_scipy_when_all_fail(self) -> None:
        obs = np.array([40., 140., 330., 550.])
        e = np.ones(4, dtype=np.bool_)
        actual = censored_log_likelihood(1.7, 300., obs, e)
        expected = np.sum(weibull_min.logpdf(obs, 1.7, scale=300.))
        self.assertAlmostEqual(actual, float(expected))

    def test_likelihood_survival_contribution_for_censored_unit(self) -> None:
        t = np.array([500.])
        e = np.array([False])
        self.assertAlmostEqual(
            censored_log_likelihood(2., 1000., t, e),
            float(np.log(weibull_survival(500, 2., 1000.))),
        )

    def test_km_manual_tied_event_and_censor(self) -> None:
        obs = np.array([1., 2., 3., 3.])
        e = np.array([True, False, True, False])
        km = kaplan_meier(obs, e)
        np.testing.assert_allclose(km.times, [1., 3.])
        np.testing.assert_array_equal(km.risk_set, [4, 2])
        np.testing.assert_array_equal(km.failures, [1, 1])
        np.testing.assert_allclose(km.survival, [.75, .375])
        np.testing.assert_allclose(km.cumulative_hazard, [.25, .75])
        self.assertTrue(np.all(km.greenwood_se >= 0))

    def test_km_without_failures_is_initial_one(self) -> None:
        km = kaplan_meier(np.array([1., 2.]), np.array([False, False]))
        np.testing.assert_allclose(step_values(km.times, km.survival,
                                               np.array([0., 5.]), 1.0), [1., 1.])

    def test_step_function_before_and_at_event(self) -> None:
        output = step_values(np.array([2., 4.]), np.array([.8, .5]),
                             np.array([0., 2., 3., 4., 6.]), 1.0)
        np.testing.assert_allclose(output, [1., .8, .8, .5, .5])

    def test_weibull_censored_fit_recovers_large_sample_scale(self) -> None:
        d = simulate_censored(2200, np.random.default_rng(435))
        valid = fit_weibull(d.observed, d.event)
        naive = fit_weibull(
            d.observed[d.event], np.ones(int(np.count_nonzero(d.event)), dtype=np.bool_)
        )
        self.assertLess(abs(valid.scale-TRUE_SCALE), .12*TRUE_SCALE)
        self.assertLess(abs(valid.scale-TRUE_SCALE), abs(naive.scale-TRUE_SCALE))
        self.assertGreater(valid.log_likelihood, -1e10)

    def test_fitting_requires_five_failures(self) -> None:
        with self.assertRaises(ValueError):
            fit_weibull(np.array([1., 2., 3.]),
                        np.array([True, False, True]))


class MaintenanceTests(unittest.TestCase):
    """Verify renewal–reward identities and policy implications."""

    def test_run_to_failure_rate_matches_corrective_cost_over_mean_life(self) -> None:
        actual = float(expected_age_replacement_rate(float("inf"), TRUE_SHAPE, TRUE_SCALE))
        self.assertAlmostEqual(actual, FAILURE_COST / weibull_mean(TRUE_SHAPE, TRUE_SCALE))

    def test_exponential_memoryless_age_replacement_is_not_better(self) -> None:
        reference = float(expected_age_replacement_rate(float("inf"), 1., 1000.))
        early = expected_age_replacement_rate(np.array([100., 500., 2000.]), 1., 1000.)
        self.assertTrue(np.all(early > reference))
        self.assertTrue(np.isinf(optimize_replacement_age(1., 1000.)))

    def test_true_age_replacement_is_finite_and_better(self) -> None:
        age = optimize_replacement_age(TRUE_SHAPE, TRUE_SCALE)
        self.assertGreater(age, 300)
        self.assertLess(age, 1500)
        optimum = float(expected_age_replacement_rate(age, TRUE_SHAPE, TRUE_SCALE))
        run_to_fail = float(expected_age_replacement_rate(float("inf"), TRUE_SHAPE, TRUE_SCALE))
        self.assertLess(optimum, run_to_fail)

    def test_optimality_local_cost_check(self) -> None:
        age = optimize_replacement_age(TRUE_SHAPE, TRUE_SCALE)
        center = float(expected_age_replacement_rate(age, TRUE_SHAPE, TRUE_SCALE))
        for offset in (-50., -10., 10., 50.):
            self.assertGreater(
                float(expected_age_replacement_rate(age+offset, TRUE_SHAPE, TRUE_SCALE)),
                center,
            )

    def test_renewal_simulation_agrees_with_analytic_cost_rate(self) -> None:
        age = optimize_replacement_age(TRUE_SHAPE, TRUE_SCALE)
        simulated, failed = simulate_renewal_policy(
            age, TRUE_SHAPE, TRUE_SCALE, np.random.default_rng(48), 70000,
        )
        true = float(expected_age_replacement_rate(age, TRUE_SHAPE, TRUE_SCALE))
        self.assertLess(abs(simulated/true - 1), .03)
        self.assertGreater(failed, 0)
        self.assertLess(failed, 1)

    def test_policy_costs_have_consistent_units(self) -> None:
        rate = float(expected_age_replacement_rate(
            800., TRUE_SHAPE, TRUE_SCALE, PREVENTIVE_COST, FAILURE_COST))
        doubled = float(expected_age_replacement_rate(
            800., TRUE_SHAPE, TRUE_SCALE, 2*PREVENTIVE_COST, 2*FAILURE_COST))
        self.assertAlmostEqual(doubled, 2*rate)


class CompetingRisksTests(unittest.TestCase):
    """Check the cumulative-incidence interpretation under competing events."""

    def test_competing_generator_codes_and_first_event(self) -> None:
        d = simulate_competing(1000, np.random.default_rng(52))
        self.assertTrue(set(np.unique(d.cause)).issubset({0, 1, 2}))
        np.testing.assert_allclose(
            d.observed, np.minimum.reduce((d.latent_cause1, d.latent_cause2, d.censor_time)))

    def test_aj_partition_at_every_event_time(self) -> None:
        d = simulate_competing(400, np.random.default_rng(99))
        aj = aalen_johansen(d.observed, d.cause)
        np.testing.assert_allclose(aj.survival + aj.cause1 + aj.cause2, 1., atol=1e-12)
        self.assertTrue(np.all(np.diff(aj.cause1) >= -1e-15))
        self.assertTrue(np.all(np.diff(aj.cause2) >= -1e-15))

    def test_aj_no_censor_equals_empirical_proportions(self) -> None:
        obs = np.array([1., 2., 3., 4.])
        causes = np.array([1, 2, 1, 2], dtype=np.int64)
        aj = aalen_johansen(obs, causes)
        self.assertAlmostEqual(aj.cause1[-1], .5)
        self.assertAlmostEqual(aj.cause2[-1], .5)
        self.assertAlmostEqual(aj.survival[-1], 0.)

    def test_true_cif_sum_matches_joint_survival(self) -> None:
        grid = np.linspace(0., 2200., 1001)
        c1, c2 = competing_truth(grid)
        s1 = weibull_survival(grid, COMPETING_SHAPES[0], COMPETING_SCALES[0])
        s2 = weibull_survival(grid, COMPETING_SHAPES[1], COMPETING_SCALES[1])
        np.testing.assert_allclose(c1 + c2 + s1*s2, 1, atol=1e-5)

    def test_true_cause1_cumulative_incidence_below_net_failure_risk(self) -> None:
        grid = np.linspace(0, 2000, 300)
        c1, _ = competing_truth(grid)
        net = 1 - weibull_survival(grid, COMPETING_SHAPES[0], COMPETING_SCALES[0])
        self.assertTrue(np.all(c1 <= net + 1e-8))
        self.assertGreater(net[-1] - c1[-1], .20)


class ReproducibilityTests(unittest.TestCase):
    """Run an isolated short study; never mutate committed full-study outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.summary = run_experiment(cls.output, quick=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_eight_figures_and_ten_tables(self) -> None:
        for name in FIGURE_NAMES:
            path = self.output / "figures" / name.replace(".png", "_quick.png")
            self.assertGreater(path.stat().st_size, 5000)
        for name in RESULT_NAMES:
            path = self.output / "results" / name.replace(".csv", "_quick.csv")
            self.assertGreater(path.stat().st_size, 50)

    def test_repeated_run_same_seed_and_separate_suffix(self) -> None:
        path = self.output / "results" / "summary_quick.csv"
        initial = hashlib.sha256(path.read_bytes()).hexdigest()
        new_summary = run_experiment(self.output, quick=True)
        self.assertEqual(new_summary, self.summary)
        self.assertEqual(initial, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse((self.output / "results" / "summary.csv").exists())

    def test_summary_and_tables_have_expected_fields(self) -> None:
        with (self.output/"results"/"maintenance_policies_quick.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 5)
        self.assertIn("corrective_fraction", rows[0])
        self.assertGreater(self.summary["n_observed_failures"], 50)

    def test_censored_policy_better_than_failures_only(self) -> None:
        self.assertLess(self.summary["censored_policy_true_cost_per_hour"],
                        self.summary["failures_only_policy_true_cost_per_hour"])

    def test_cause1_net_risk_exceeds_actual_incidence(self) -> None:
        self.assertGreater(self.summary["naive_net_risk_cause1_at_1800"],
                           self.summary["true_cause1_cif_at_1800"])

    def test_notebook_clean_and_executable_in_fresh_kernel(self) -> None:
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("Optional notebook execution tools not installed locally")
        notebook = nbformat.read(ROOT/"reliability_statistics.ipynb", as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
        output = NotebookClient(
            notebook, timeout=120, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        self.assertGreater(sum(c.cell_type == "code" and c.execution_count is not None
                               for c in output.cells), 4)


if __name__ == "__main__":
    unittest.main()
