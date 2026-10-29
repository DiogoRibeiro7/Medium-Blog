"""Scientific invariants and notebook smoke tests for the queue experiments."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

# The tests run from the project directory; no repository-wide packaging is needed.
from analysis import (
    ARRIVAL_RATE,
    QueueSpec,
    expected_mg1_wait,
    expected_mmc_wait,
    kingman_wait,
    make_figures,
    one_replication,
    prediction,
    run_experiment,
    sample_interarrivals,
    sample_services,
    simulate_fcfs,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]


class QueueMathematicsTests(unittest.TestCase):
    """Checks independent of any particular Monte Carlo seed."""

    def test_analytic_models_are_numerically_correct(self) -> None:
        self.assertAlmostEqual(expected_mg1_wait(0.8, 1.0, 0.0), 2.0)
        self.assertAlmostEqual(expected_mg1_wait(0.8, 1.0, 1.0), 4.0)
        self.assertAlmostEqual(expected_mg1_wait(0.8, 1.0, 4.0), 10.0)
        self.assertAlmostEqual(expected_mmc_wait(0.8, 1.0, 2), 4.0 / 21.0)
        self.assertAlmostEqual(kingman_wait(0.8, 1.0, 4.0, 1.0), 10.0)
        self.assertAlmostEqual(expected_mg1_wait(0.8, 0.85, 1.0), 1.80625)

    def test_unstable_queues_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expected_mg1_wait(1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            expected_mmc_wait(2.0, 1.0, 2)
        with self.assertRaises(ValueError):
            QueueSpec("unstable", "unstable", "exponential", "exponential", mean_service=1.25)

    def test_fcfs_hand_calculation(self) -> None:
        arrivals = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        services = np.ones(3, dtype=np.float64)
        np.testing.assert_allclose(simulate_fcfs(arrivals, services, 1), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(simulate_fcfs(arrivals, services, 2), [0.0, 0.0, 0.0])

    def test_more_servers_cannot_increase_waits_for_same_job_trace(self) -> None:
        rng = np.random.default_rng(19)
        arrivals = np.cumsum(rng.exponential(1.0 / ARRIVAL_RATE, size=3_000))
        services = rng.exponential(1.0, size=3_000)
        single = simulate_fcfs(arrivals, services, 1)
        double = simulate_fcfs(arrivals, services, 2)
        self.assertTrue(np.all(double <= single + 1e-12))

    def test_inputs_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            simulate_fcfs(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            simulate_fcfs(np.array([1.0]), np.array([-2.0]))
        with self.assertRaises(ValueError):
            sample_services(np.random.default_rng(1), 0, 1.0, "exponential")

    def test_distribution_moments_are_close_to_design(self) -> None:
        rng = np.random.default_rng(321)
        interarrival = sample_interarrivals(rng, 100_000, 0.8, "hyperexponential")
        long_service = sample_services(rng, 100_000, 1.0, "lognormal")
        self.assertAlmostEqual(float(np.mean(interarrival)), 1.25, delta=0.04)
        self.assertAlmostEqual(float(np.var(interarrival) / np.mean(interarrival) ** 2), 4.0, delta=0.3)
        self.assertAlmostEqual(float(np.mean(long_service)), 1.0, delta=0.025)
        self.assertAlmostEqual(float(np.var(long_service) / np.mean(long_service) ** 2), 4.0, delta=1.0)

    def test_exponential_queue_agrees_with_exact_stationary_formula(self) -> None:
        spec = QueueSpec("mm1_test", "test", "exponential", "exponential")
        observations = one_replication(spec, 991, 120_000, 20_000)
        self.assertAlmostEqual(float(observations.mean()), 4.0, delta=0.55)
        theory_type, expected = prediction(spec)
        self.assertEqual(theory_type, "Pollaczek–Khinchine exact")
        self.assertAlmostEqual(expected, 4.0)


class ArtifactTests(unittest.TestCase):
    """Every generated number and plot is produced by the same code path."""

    def test_quick_experiment_and_figures(self) -> None:
        summary, tails, sensitivity = run_experiment(quick=True)
        self.assertEqual(len(summary), 6)
        self.assertEqual(len(tails), 6 * 10)
        self.assertEqual(len(sensitivity), 5 * 3)
        self.assertTrue(all(float(row["simulated_mean_wait_min"]) >= 0 for row in summary))
        self.assertTrue(all(math.isfinite(float(row["simulated_p95_wait_min"])) for row in summary))
        with tempfile.TemporaryDirectory(prefix="queue-test-") as folder:
            output = Path(folder)
            write_csv(output / "results" / "summary_quick.csv", summary)
            make_figures(summary, tails, sensitivity, output / "figures", suffix="_quick")
            with (output / "results" / "summary_quick.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 3)

    def test_notebook_schema_and_python_cells_compile(self) -> None:
        # A dependency-free check is useful even when the optional Jupyter
        # execution stack is not present in a local developer environment.
        payload = json.loads((ROOT / "queueing_theory.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        self.assertEqual(payload["metadata"]["kernelspec"]["name"], "python3")
        code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 3)
        for cell in code_cells:
            self.assertEqual(cell["execution_count"], None)
            self.assertEqual(cell["outputs"], [])
            compile("".join(cell["source"]), "queueing_theory.ipynb", "exec")

    @unittest.skipUnless(
        importlib.util.find_spec("nbformat") is not None
        and importlib.util.find_spec("nbclient") is not None,
        "install the notebook dependencies to execute it in a fresh kernel",
    )
    def test_notebook_executes_in_fresh_kernel(self) -> None:
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(ROOT / "queueing_theory.ipynb", as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        executed = client.execute()
        self.assertTrue(all(cell.get("execution_count") is not None for cell in executed.cells if cell.cell_type == "code"))


if __name__ == "__main__":
    unittest.main()
