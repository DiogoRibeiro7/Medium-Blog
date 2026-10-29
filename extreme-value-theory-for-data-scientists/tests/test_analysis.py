"""Scientific invariants, reproducibility and artifact tests for the EVT study."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    BLOCK_SIZES,
    POTFit,
    TRUE_EXCEEDANCE_PROBABILITY,
    TRUE_SCALE,
    TRUE_SHAPE,
    TRUE_THRESHOLD,
    _dependence_diagnostics,
    _dependence_samples,
    _generator,
    extremal_index_proxy,
    fit_pot,
    make_figures,
    population_quantile,
    population_survival,
    pot_quantile,
    run_study,
    sample_population,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]


class EVTMathematicsTests(unittest.TestCase):
    """Check formulas separately from any plotted conclusions."""

    def test_exact_population_quantiles_invert_survival(self) -> None:
        for probability in (0.1, 0.7, 0.89, 0.9, 0.99, 0.999, 0.9999):
            with self.subTest(probability=probability):
                quantile = population_quantile(probability)
                survival = float(population_survival(quantile))
                self.assertAlmostEqual(survival, 1.0 - probability, places=9)

    def test_threshold_splice_has_the_claimed_tail_probability(self) -> None:
        self.assertAlmostEqual(
            float(population_survival(TRUE_THRESHOLD)),
            TRUE_EXCEEDANCE_PROBABILITY,
            places=12,
        )
        sample = sample_population(_generator(99), 50_000)
        self.assertLess(
            abs(float(np.mean(sample > TRUE_THRESHOLD)) - TRUE_EXCEEDANCE_PROBABILITY),
            0.006,
        )

    def test_gpd_shape_and_scale_recover_design_with_large_sample(self) -> None:
        sample = sample_population(_generator(100), 60_000)
        fitted = fit_pot(sample)
        self.assertLess(abs(fitted.shape - TRUE_SHAPE), 0.08)
        self.assertLess(abs(fitted.scale - TRUE_SCALE), 0.40)

    def test_exponential_limit_and_probability_domain(self) -> None:
        fitted = POTFit(
            threshold=5.0,
            shape=0.0,
            scale=2.0,
            p_exceed=0.2,
            n_exceed=200,
            sample_size=1_000,
        )
        expected = 5.0 + 2.0 * np.log(0.2 / 0.02)
        self.assertAlmostEqual(pot_quantile(0.98, fitted), float(expected), places=10)
        with self.assertRaises(ValueError):
            pot_quantile(0.7, fitted)
        with self.assertRaises(ValueError):
            population_quantile(1.0)

    def test_invalid_samples_and_types_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_pot(np.full(100, np.nan))
        with self.assertRaises(ValueError):
            fit_pot(np.arange(1.0, 15.0))
        with self.assertRaises(ValueError):
            sample_population(_generator(2), 0)
        with self.assertRaises(TypeError):
            run_study(quick="yes")  # type: ignore[arg-type]

    def test_equal_marginal_laws_do_not_mean_equal_clusters(self) -> None:
        iid, dependent = _dependence_samples(30_000)
        q99 = float((1.0 - np.sqrt(0.99)) ** (-1.0 / 3.0))
        self.assertAlmostEqual(float(np.mean(iid > q99)), 0.01, delta=0.003)
        self.assertAlmostEqual(float(np.mean(dependent > q99)), 0.01, delta=0.003)
        self.assertGreater(extremal_index_proxy(iid, q99), 0.90)
        self.assertAlmostEqual(extremal_index_proxy(dependent, q99), 0.5, delta=0.08)
        rows = _dependence_diagnostics(iid, dependent)
        self.assertEqual(len(rows), 2 * len(BLOCK_SIZES))
        self.assertLess(
            float(rows[-1]["exact_block_exceedance_probability"]),
            float(rows[-2]["exact_block_exceedance_probability"]),
        )


class ArtifactsTests(unittest.TestCase):
    """Verify that the code, manuscript figures and notebook agree."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.quick_study = run_study(quick=True)

    def test_study_tables_have_expected_shapes(self) -> None:
        study = self.quick_study
        self.assertEqual(len(study.quantile_rows), 5)
        self.assertEqual(len(study.threshold_rows), 8)
        self.assertEqual(len(study.dependence_rows), 10)
        self.assertEqual(len(study.bootstrap_levels), 18)
        self.assertEqual(study.training.size, 5_000)
        self.assertGreater(study.pot.n_exceed, 300)
        self.assertTrue(np.isfinite(study.summary_rows[0]["pot_target_quantile"]))

    def test_independent_random_streams_reproduce_exactly(self) -> None:
        other = run_study(quick=True)
        self.assertTrue(np.array_equal(self.quick_study.training, other.training))
        self.assertTrue(
            np.array_equal(self.quick_study.bootstrap_levels, other.bootstrap_levels)
        )
        self.assertEqual(self.quick_study.summary_rows, other.summary_rows)

    def test_eight_full_sized_figures_and_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            make_figures(self.quick_study, path / "figures")
            files = sorted((path / "figures").glob("fig*.png"))
            self.assertEqual(len(files), 8)
            self.assertTrue(all(file.stat().st_size > 10_000 for file in files))
            write_csv(path / "results" / "summary.csv", self.quick_study.summary_rows)
            self.assertTrue((path / "results" / "summary.csv").is_file())

    def test_notebook_json_and_code_compile(self) -> None:
        with (ROOT / "extreme_value_theory.ipynb").open(encoding="utf-8") as stream:
            notebook = json.load(stream)
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 3)
        for cell in code_cells:
            compile("".join(cell["source"]), "extreme_value_theory.ipynb", "exec")

    def test_notebook_executes_in_a_fresh_kernel(self) -> None:
        try:
            import nbclient
            import nbformat
        except ImportError:
            self.skipTest("install notebook dependencies for fresh-kernel execution")
        notebook = nbformat.read(ROOT / "extreme_value_theory.ipynb", as_version=4)
        nbclient.NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()


if __name__ == "__main__":
    unittest.main()
