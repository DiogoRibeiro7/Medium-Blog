"""Scientific formula, design, decision, and reproducibility checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    A2_5,
    D2_5,
    D3_5,
    D4_5,
    DOE_CENTERPOINTS,
    DOE_REPLICATES,
    FALSE_ACCEPT_COST,
    GAUGE_BIAS,
    LSL,
    MASTER_SEED,
    MU_BASELINE,
    MU_SHIFT,
    N_BASELINE,
    N_OPERATORS,
    N_PARTS,
    N_REPLICATES,
    N_SUBGROUPS,
    SIGMA_PROCESS,
    SUBGROUP_SIZE,
    TARGET,
    USL,
    _doe_model,
    capability,
    crossed_anova_components,
    generator,
    inspection_classification,
    make_figures,
    normal_nonconformance,
    process_stream,
    run_study,
    simulate_factorial,
    simulate_gauge_design,
    write_csv,
    xbar_r_limits,
)

ROOT = Path(__file__).resolve().parents[1]


class MathematicsTests(unittest.TestCase):
    """Verify identities separately from one Monte Carlo realization."""

    def test_capability_normal_theory(self) -> None:
        result = capability(TARGET, 0.2)
        expected = (USL - LSL) / (6.0 * 0.2)
        self.assertAlmostEqual(result["cp"], expected)
        self.assertAlmostEqual(result["cpk"], expected)
        self.assertAlmostEqual(result["cpm"], expected)
        shifted = capability(TARGET + 0.2, 0.2)
        self.assertAlmostEqual(shifted["cp"], result["cp"])
        self.assertLess(shifted["cpk"], result["cpk"])
        self.assertLess(shifted["cpm"], result["cpm"])

    def test_normal_specification_risk_is_symmetric_about_midpoint(self) -> None:
        self.assertAlmostEqual(
            normal_nonconformance(TARGET - .3, SIGMA_PROCESS),
            normal_nonconformance(TARGET + .3, SIGMA_PROCESS),
            places=11,
        )
        self.assertGreater(
            normal_nonconformance(TARGET + .3, SIGMA_PROCESS),
            normal_nonconformance(TARGET, SIGMA_PROCESS),
        )
        with self.assertRaises(ValueError):
            normal_nonconformance(TARGET, 0.0)
        with self.assertRaises(ValueError):
            capability(TARGET, float("nan"))

    def test_xbar_r_constants_and_invalid_shapes(self) -> None:
        baseline = np.tile(np.array([9.8, 9.9, 10.0, 10.1, 10.2]), (30, 1))
        result = xbar_r_limits(baseline)
        self.assertAlmostEqual(result["xbar_center"], 10.0)
        self.assertAlmostEqual(result["range_center"], 0.4)
        self.assertAlmostEqual(result["within_sigma"], 0.4 / D2_5)
        self.assertAlmostEqual(result["xbar_ucl"], 10.0 + A2_5 * 0.4)
        self.assertAlmostEqual(result["range_lcl"], D3_5 * 0.4)
        self.assertAlmostEqual(result["range_ucl"], D4_5 * 0.4)
        with self.assertRaises(ValueError):
            xbar_r_limits(np.ones((19, SUBGROUP_SIZE)))
        with self.assertRaises(ValueError):
            xbar_r_limits(np.ones((30, SUBGROUP_SIZE + 1)))

    def test_crossed_anova_no_interaction_fixture(self) -> None:
        parts = np.array([9.0, 10.0, 11.0])[:, None, None]
        operators = np.array([-0.1, 0.1])[None, :, None]
        reps = np.zeros((1, 1, 3))
        values = parts + operators + reps
        comp = crossed_anova_components(values)
        self.assertAlmostEqual(comp["repeatability"], 0.0, places=12)
        self.assertAlmostEqual(comp["part_operator"], 0.0, places=12)
        self.assertGreater(comp["operator"], 0.0)
        self.assertGreater(comp["part"], comp["operator"])
        with self.assertRaises(ValueError):
            crossed_anova_components(np.ones((2, 3)))

    def test_inspection_errors_have_correct_denominators(self) -> None:
        truth = np.array([9.0, 9.8, 10.0, 10.9], dtype=float)
        measured = np.array([9.5, 9.0, 10.0, 11.0], dtype=float)
        result = inspection_classification(truth, measured)
        self.assertEqual(result["truly_nonconforming"], 2)
        self.assertEqual(result["truly_conforming"], 2)
        self.assertEqual(result["false_accepts"], 1)
        self.assertEqual(result["false_rejects"], 1)
        self.assertAlmostEqual(result["false_accepts_given_bad"], 0.5)
        self.assertAlmostEqual(result["false_rejects_given_good"], 0.5)
        self.assertAlmostEqual(
            result["illustrative_loss_per_10000"],
            10000 * (FALSE_ACCEPT_COST + 1.0) / 4,
        )
        with self.assertRaises(ValueError):
            inspection_classification(truth, measured, guardband=-0.01)
        with self.assertRaises(ValueError):
            inspection_classification(truth, measured[:3])

    def test_factorial_interaction_is_difference_of_simple_effects(self) -> None:
        means = {(a, b): _doe_model(a, b) for a in (-1, 1) for b in (-1, 1)}
        interaction_contrast = (
            means[(1, 1)] - means[(1, -1)]
            - means[(-1, 1)] + means[(-1, -1)]
        )
        self.assertAlmostEqual(interaction_contrast, 4.0 * 0.18)
        self.assertAlmostEqual(_doe_model(0, 0), 10.05)

    def test_invalid_generator_and_type(self) -> None:
        with self.assertRaises(ValueError):
            generator(-1)
        with self.assertRaises(ValueError):
            generator(True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            process_stream("not-a-generator")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            run_study(quick="yes")  # type: ignore[arg-type]


class StudyTests(unittest.TestCase):
    """Validate the constructed comparisons and all deliverable artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_study(quick=True)

    def test_stable_baseline_and_shifted_subgroups(self) -> None:
        study = self.study
        self.assertEqual(study.true_process.shape, (N_SUBGROUPS, SUBGROUP_SIZE))
        self.assertEqual(len(study.subgroup_rows), N_SUBGROUPS)
        difference = float(
            study.true_process[N_BASELINE:].mean()
            - study.true_process[:N_BASELINE].mean()
        )
        self.assertAlmostEqual(difference, MU_SHIFT, delta=0.08)
        baseline_signals = sum(
            int(row["xbar_out_of_control"]) for row in study.subgroup_rows[:N_BASELINE]
        )
        changed_signals = sum(
            int(row["xbar_out_of_control"]) for row in study.subgroup_rows[N_BASELINE:]
        )
        self.assertGreater(changed_signals, baseline_signals + 20)

    def test_capability_is_not_claimed_from_pooled_mixture(self) -> None:
        rows = self.study.capability_rows
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(
            float(rows[0]["cpk"]),
            capability(MU_BASELINE, SIGMA_PROCESS)["cpk"],
        )
        self.assertGreater(float(rows[0]["cpk"]), float(rows[2]["cpk"]))
        self.assertLess(float(rows[-1]["cpk"]), float(rows[2]["cpk"]))
        self.assertIn("NOT valid", str(rows[-1]["methodological_note"]))

    def test_gauge_design_and_component_identities(self) -> None:
        study = self.study
        self.assertEqual(study.gauge_design.values.shape, (N_PARTS, N_OPERATORS, N_REPLICATES))
        self.assertEqual(len(study.gauge_rows), N_PARTS * N_OPERATORS * N_REPLICATES)
        self.assertEqual(len(study.gauge_component_rows), 4)
        self.assertAlmostEqual(
            study.gauge_components["gauge_sd"] ** 2,
            sum(study.gauge_components[k] for k in ("repeatability", "operator", "part_operator")),
        )
        self.assertEqual(study.gauge_studies.shape, (70, 2))
        self.assertTrue(np.isfinite(study.gauge_studies).all())
        self.assertAlmostEqual(study.bias_estimate, GAUGE_BIAS, delta=0.05)

    def test_guardband_monotonicity_and_bias_decision(self) -> None:
        study = self.study
        self.assertEqual(len(study.inspection_rows), 2)
        self.assertEqual(len(study.guardband_rows), 11)
        self.assertLess(
            float(study.inspection_rows[1]["false_rejects_per_10000"]),
            float(study.inspection_rows[0]["false_rejects_per_10000"]),
        )
        accepts = [int(row["false_accepts"]) for row in study.guardband_rows]
        rejects = [int(row["false_rejects"]) for row in study.guardband_rows]
        self.assertEqual(accepts, sorted(accepts, reverse=True))
        self.assertEqual(rejects, sorted(rejects))

    def test_factorial_is_replicated_and_detects_interaction(self) -> None:
        study = self.study
        self.assertEqual(len(study.factorial_rows), DOE_REPLICATES * 4 + DOE_CENTERPOINTS)
        self.assertEqual(sum(int(row["centerpoint"]) for row in study.factorial_rows), DOE_CENTERPOINTS)
        self.assertEqual(len(study.factorial_setting_rows), 4)
        self.assertGreater(float(study.estimated_beta[3]), 0.09)
        self.assertLess(float(study.estimated_beta[3]), 0.32)
        self.assertTrue(np.all(study.beta_standard_errors > 0.0))

    def test_reproducibility_and_separate_quick_datasets(self) -> None:
        other = run_study(quick=True)
        np.testing.assert_array_equal(self.study.true_process, other.true_process)
        np.testing.assert_array_equal(self.study.measured_process, other.measured_process)
        np.testing.assert_array_equal(self.study.gauge_studies, other.gauge_studies)
        np.testing.assert_array_equal(self.study.estimated_beta, other.estimated_beta)
        self.assertEqual(self.study.summary_rows, other.summary_rows)

    def test_eight_figures_and_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            make_figures(self.study, directory / "figures")
            files = sorted((directory / "figures").glob("fig*.png"))
            self.assertEqual(len(files), 8)
            self.assertTrue(all(file.stat().st_size > 10_000 for file in files))
            write_csv(directory / "results" / "summary.csv", self.study.summary_rows)
            self.assertTrue((directory / "results" / "summary.csv").is_file())

    def test_notebook_json_code_compiles(self) -> None:
        notebook = json.loads((ROOT / "industrial_statistics.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(cells), 4)
        for cell in cells:
            compile("".join(cell["source"]), "industrial_statistics.ipynb", "exec")

    def test_notebook_executes_in_fresh_kernel(self) -> None:
        try:
            import nbclient
            import nbformat
        except ImportError:
            self.skipTest("notebook dependencies are optional locally")
        notebook = nbformat.read(ROOT / "industrial_statistics.ipynb", as_version=4)
        nbclient.NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()


if __name__ == "__main__":
    unittest.main()
