"""Scientific invariants, finite-horizon calibration and artifact tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    AR_COEFFICIENT,
    CHANGE_INDEX,
    HORIZON,
    METHODS,
    SCALE_ALTERNATIVE,
    _first_alarm,
    _generator,
    _sample_streams,
    detector_statistics,
    make_figures,
    run_study,
    wilson_interval,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]


class MathematicsTests(unittest.TestCase):
    """Test recursive statistics and probabilistic assumptions directly."""

    def test_recursive_statistics_are_causal_and_exact(self) -> None:
        values = np.array([[0.0, 1.0, 2.0]], dtype=float)
        statistics = detector_statistics(values)
        np.testing.assert_allclose(statistics["shewhart"], values)
        np.testing.assert_allclose(statistics["cusum_mean"], [[0.0, 0.5, 2.0]])
        np.testing.assert_allclose(statistics["ewma_mean"], [[0.0, 0.2, 0.56]])
        increments = -np.log(SCALE_ALTERNATIVE) + (
            0.5 * (1.0 - SCALE_ALTERNATIVE ** -2) * values[0] ** 2
        )
        expected = []
        state = 0.0
        for inc in increments:
            state = max(0.0, state + float(inc))
            expected.append(state)
        np.testing.assert_allclose(statistics["cusum_variance"][0], expected)

    def test_first_alarm_uses_strict_threshold_and_censors(self) -> None:
        values = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(_first_alarm(values, 1.0), [1, 3])
        with self.assertRaises(ValueError):
            _first_alarm(values, float("nan"))

    def test_ar1_null_has_nearly_unit_marginal_variance(self) -> None:
        samples = _sample_streams(_generator(88), 1000, "null_ar1")
        self.assertEqual(samples.shape, (1000, HORIZON))
        self.assertLess(abs(float(samples.mean())), 0.02)
        self.assertLess(abs(float(samples.var()) - 1.0), 0.02)
        self.assertAlmostEqual(
            float(np.corrcoef(samples[:, 80], samples[:, 81])[0, 1]),
            AR_COEFFICIENT,
            delta=0.07,
        )

    def test_synthetic_shift_starts_at_declared_index(self) -> None:
        standard = _sample_streams(_generator(80), 4, "null_iid")
        shifted = _sample_streams(_generator(80), 4, "mean_1_0")
        np.testing.assert_array_equal(standard[:, :CHANGE_INDEX], shifted[:, :CHANGE_INDEX])
        np.testing.assert_allclose(
            shifted[:, CHANGE_INDEX:] - standard[:, CHANGE_INDEX:], 1.0
        )

    def test_wilson_and_validation_boundaries(self) -> None:
        low, high = wilson_interval(10, 100)
        self.assertLess(low, 0.10)
        self.assertGreater(high, 0.10)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)
        with self.assertRaises(ValueError):
            detector_statistics(np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            detector_statistics(np.array([[np.nan, 0.0]]))
        with self.assertRaises(ValueError):
            _sample_streams(_generator(2), 0, "mean_1_0")
        with self.assertRaises(TypeError):
            run_study(quick="yes")  # type: ignore[arg-type]


class ExperimentTests(unittest.TestCase):
    """Check empirical conclusions without fragile exact Monte Carlo values."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_study(quick=True)

    def test_calibration_on_independent_holdout(self) -> None:
        for row in self.study.calibration_rows:
            self.assertLess(
                abs(float(row["validated_iid_false_alarm_probability"]) - 0.10),
                0.045,
            )
            self.assertGreater(float(row["threshold"]), 0.0)

    def test_autocorrelation_breaks_mean_cusum_and_ewma_calibration(self) -> None:
        for method in ("cusum_mean", "ewma_mean"):
            iid = next(
                r for r in self.study.dependence_rows
                if r["method"] == method and r["null_process"] == "iid_gaussian"
            )
            correlated = next(
                r for r in self.study.dependence_rows
                if r["method"] == method and r["null_process"] == "ar1_gaussian_phi_0_6"
            )
            self.assertGreater(
                float(correlated["false_alarm_probability_by_200"]),
                3 * float(iid["false_alarm_probability_by_200"]),
            )

    def test_large_mean_shift_and_variance_alternative(self) -> None:
        rows = self.study.scenario_rows
        for method in METHODS:
            small = next(
                r for r in rows if r["method"] == method and r["scenario"] == "mean_0_5"
            )
            large = next(
                r for r in rows if r["method"] == method and r["scenario"] == "mean_2_0"
            )
            self.assertGreater(
                float(large["conditional_detection_50"]),
                float(small["conditional_detection_50"]),
            )
        variance_cusum = next(
            r for r in rows
            if r["method"] == "cusum_variance" and r["scenario"] == "variance_1_7"
        )
        mean_cusum = next(
            r for r in rows
            if r["method"] == "cusum_mean" and r["scenario"] == "variance_1_7"
        )
        self.assertGreater(
            float(variance_cusum["conditional_detection_50"]),
            float(mean_cusum["conditional_detection_50"]) + 0.25,
        )

    def test_delay_denominator_retains_undetected_survivors(self) -> None:
        for method in METHODS:
            delays = [r for r in self.study.delay_rows if r["method"] == method]
            self.assertEqual(len(delays), 50)
            self.assertTrue(all(int(r["survivors"]) > 0 for r in delays))
            self.assertLessEqual(
                float(delays[-1]["fraction_of_survivors_detected_by_delay"]),
                1.0,
            )
            self.assertAlmostEqual(
                float(delays[-1]["fraction_of_survivors_detected_by_delay"]),
                int(delays[-1]["detected_within_50"]) / int(delays[-1]["survivors"]),
            )

    def test_shapes_and_reproducible_seeded_outputs(self) -> None:
        study = self.study
        self.assertEqual(len(study.calibration_rows), 4)
        self.assertEqual(len(study.scenario_rows), 20)
        self.assertEqual(len(study.dependence_rows), 8)
        self.assertEqual(len(study.delay_rows), 200)
        self.assertEqual(len(study.sample_path_rows), HORIZON)
        second = run_study(quick=True)
        self.assertEqual(study.summary_rows, second.summary_rows)
        for method in METHODS:
            np.testing.assert_array_equal(
                study.calibration_maxima[method], second.calibration_maxima[method]
            )

    def test_artifacts_and_csv_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_figures(self.study, root / "figures")
            figures = sorted((root / "figures").glob("fig*.png"))
            self.assertEqual(len(figures), 8)
            self.assertTrue(all(path.stat().st_size > 10_000 for path in figures))
            write_csv(root / "results" / "summary.csv", self.study.summary_rows)
            self.assertTrue((root / "results" / "summary.csv").is_file())

    def test_notebook_json_and_code_compile(self) -> None:
        notebook = json.loads((ROOT / "sequential_change_detection.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 3)
        for cell in code_cells:
            compile("".join(cell["source"]), "sequential_change_detection.ipynb", "exec")

    def test_notebook_executes_in_fresh_kernel(self) -> None:
        try:
            import nbclient
            import nbformat
        except ImportError:
            self.skipTest("notebook dependencies are optional locally")
        notebook = nbformat.read(ROOT / "sequential_change_detection.ipynb", as_version=4)
        nbclient.NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()


if __name__ == "__main__":
    unittest.main()
