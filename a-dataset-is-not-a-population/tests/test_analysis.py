"""Tests for synthetic selection and population-inference experiments."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import METHODS, STUDIES, rng_for, simulate, write_evidence
from figures import FIGURE_NAMES, read_replications, render_all
from notebook_builder import NOTEBOOK, build


class SamplingStudiesTests(unittest.TestCase):
    """Protect study schemas, seeds and basic theoretical direction."""

    def test_study_streams_are_reproducible_and_independent(self) -> None:
        for study in STUDIES:
            self.assertEqual(rng_for(study, 5).random(), rng_for(study, 5).random())
            self.assertNotEqual(rng_for(study, 5).random(), rng_for(study, 6).random())

    def test_complete_replicates_and_finite_rows(self) -> None:
        for study in STUDIES:
            rows = simulate(study, 10)
            self.assertEqual(len(rows), 10 * len(METHODS[study]))
            self.assertEqual(set(row.method for row in rows), set(METHODS[study]))
            self.assertTrue(all(np.isfinite(row.estimate) and np.isfinite(row.target) for row in rows))
            self.assertEqual(rows, simulate(study, 10))

    def test_stratification_reduces_oversampling_bias(self) -> None:
        rows = simulate("strata", 200)
        means = {method: np.mean([row.estimate - row.target for row in rows if row.method == method])
                 for method in METHODS["strata"]}
        self.assertGreater(means["unweighted"], 1.0)
        self.assertLess(abs(means["poststratified"]), 0.08)
        self.assertAlmostEqual(means["ht"], means["hajek"], places=12)

    def test_oracle_weights_reduce_outcome_selection_bias(self) -> None:
        rows = simulate("nonresponse", 200)
        biases = {method: abs(np.mean([row.estimate - row.target for row in rows if row.method == method]))
                  for method in METHODS["nonresponse"]}
        self.assertLess(biases["ipw_known"], biases["respondents"])
        self.assertLess(biases["ipw_known"], biases["ipw_misspecified"])

    def test_zero_coverage_cannot_estimate_full_population(self) -> None:
        rows = simulate("coverage", 200)
        errors = np.asarray([row.estimate - row.target for row in rows])
        self.assertLess(float(errors.mean()), -0.85)
        # More repeated frame draws cannot eliminate a systematic frame gap.
        self.assertLess(abs(float(errors[100:].mean()) - float(errors[:100].mean())), 0.15)

    def test_large_volunteer_panel_is_more_biased_than_probability_sample(self) -> None:
        rows = simulate("volunteer", 200)
        by_method = {method: np.asarray([row.estimate - row.target for row in rows if row.method == method])
                     for method in METHODS["volunteer"]}
        self.assertGreater(float(by_method["opt_in"].mean()), 0.5)
        self.assertLess(abs(float(by_method["probability_sample"].mean())), 0.08)

    def test_margins_do_not_identify_joint_cell_composition(self) -> None:
        rows = simulate("calibration", 200)
        by_method = {method: np.asarray([row.estimate - row.target for row in rows if row.method == method])
                     for method in METHODS["calibration"]}
        np.testing.assert_array_equal(by_method["unweighted"], by_method["marginal_calibration"])
        self.assertLess(float(by_method["unweighted"].mean()), -1.5)
        self.assertLess(abs(float(by_method["joint_calibration"].mean())), 0.08)

    def test_raw_ipw_reduces_selection_bias_but_clipping_can_reintroduce_it(self) -> None:
        rows = simulate("weighting", 300)
        bias = {method: float(np.mean([row.estimate - row.target for row in rows if row.method == method]))
                for method in METHODS["weighting"]}
        self.assertGreater(bias["unweighted"], 0.5)
        self.assertLess(abs(bias["ipw_raw"]), 0.13)
        self.assertGreater(bias["ipw_clipped"], bias["ipw_raw"] + 0.08)

    def test_covariate_shift_changes_deployment_risk(self) -> None:
        rows = simulate("predictive", 180)
        risk = {method: float(np.mean([row.estimate for row in rows if row.method == method]))
                for method in METHODS["predictive"]}
        self.assertGreater(risk["target_test"], risk["sampled_test"] + 2.0)
        self.assertLess(risk["target_weighted_fit"], risk["target_test"] - 1.0)

    def test_composition_reweighting_cannot_fix_concept_shift(self) -> None:
        rows = simulate("drift", 180)
        bias = {method: float(np.mean([row.estimate - row.target for row in rows if row.method == method]))
                for method in METHODS["drift"]}
        self.assertLess(bias["historical"], -1.7)
        self.assertLess(abs(bias["updated_margins_stable"]), 0.08)
        self.assertLess(bias["updated_margins_shift"], -1.3)

    def test_archived_summaries_have_correct_identities(self) -> None:
        rows = simulate("srs", 8)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_evidence(rows, output, "_quick")
            with (output / "srs_replicates_quick.csv").open(newline="", encoding="utf-8") as handle:
                replicates = list(csv.DictReader(handle))
            with (output / "srs_summary_quick.csv").open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(len(replicates), 8)
            errors = np.asarray([float(row["error"]) for row in replicates])
            self.assertAlmostEqual(float(summary["bias"]), float(errors.mean()))
            self.assertAlmostEqual(float(summary["rmse"]), float(np.sqrt(np.mean(errors**2))))
            self.assertGreaterEqual(float(summary["ci_high"]), float(summary["ci_low"]))

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            simulate("missing", 10)
        with self.assertRaises(ValueError):
            simulate("srs", 0)
        with self.assertRaises(ValueError):
            rng_for("srs", -1)

    def test_nine_figures_generated_from_archived_replications(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for study in STUDIES:
                write_evidence(simulate(study, 4), data, "_quick")
            outputs = render_all(data, root / "figures", "_quick")
            self.assertEqual(len(outputs), len(STUDIES))
            self.assertEqual(len(FIGURE_NAMES), len(STUDIES))
            self.assertTrue(all(path.is_file() and path.stat().st_size > 5000 for path in outputs))
            self.assertEqual(len(read_replications(data / "drift_replicates_quick.csv", "drift")), 3)

    def test_notebook_is_deterministic_and_output_free(self) -> None:
        """The checked-in notebook must match the source builder exactly."""
        import nbformat

        expected = build()
        current = nbformat.read(NOTEBOOK, as_version=4)
        self.assertEqual(expected, current)
        self.assertTrue(all(not cell.get("outputs") for cell in current.cells if cell.cell_type == "code"))

    @unittest.skipUnless(os.environ.get("RUN_NOTEBOOK_EXECUTION") == "1", "opt-in fresh kernel")
    def test_notebook_from_external_working_directory(self) -> None:
        """A fresh kernel runs without changing archival publication files."""
        import nbformat
        from nbclient import NotebookClient

        with tempfile.TemporaryDirectory() as directory:
            before = {path.name: path.read_bytes() for path in NOTEBOOK.parent.glob("data/*.csv")}
            prior = os.environ.get("POPULATION_ARTICLE_DIR")
            os.environ["POPULATION_ARTICLE_DIR"] = str(NOTEBOOK.parent)
            try:
                client = NotebookClient(nbformat.read(NOTEBOOK, as_version=4), timeout=120,
                                        kernel_name="python3", resources={"metadata": {"path": directory}})
                executed = client.execute()
                self.assertEqual(len([cell for cell in executed.cells if cell.cell_type == "code"]), 2)
            finally:
                if prior is None:
                    os.environ.pop("POPULATION_ARTICLE_DIR", None)
                else:
                    os.environ["POPULATION_ARTICLE_DIR"] = prior
            after = {path.name: path.read_bytes() for path in NOTEBOOK.parent.glob("data/*.csv")}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

