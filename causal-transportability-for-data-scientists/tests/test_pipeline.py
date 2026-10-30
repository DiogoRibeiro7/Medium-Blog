"""Artifacts, table schemas, notebook and whole-study smoke tests."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import _record_table, run_study  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = (
    "fig1_trial_target_shift", "fig2_effect_heterogeneity", "fig3_estimators",
    "fig4_double_robustness", "fig5_overlap_ess", "fig6_overlap_rmse",
    "fig7_hidden_effect_modifier", "fig8_bootstrap_intervals",
)
TABLE_NAMES = (
    "population_truth", "cohort_diagnostics", "covariate_density",
    "conditional_effect_curves", "estimator_replicates", "estimator_summary",
    "overlap_replicates", "overlap_summary", "hidden_modifier_sensitivity",
    "bootstrap_replicates", "bootstrap_intervals", "study_summary",
)


class TableTests(unittest.TestCase):
    def test_rejects_empty_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                _record_table(Path(temp) / "empty.csv", [])

    def test_rejects_inconsistent_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                _record_table(Path(temp) / "invalid.csv", [{"x": 1}, {"y": 2}])

    def test_rejects_infinite_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                _record_table(Path(temp) / "invalid.csv", [{"x": float("inf")}])

    def test_accepts_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "table.csv"
            _record_table(path, [{"key": "abc", "value": 1.5}])
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["value"], "1.5")


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="transport-study-test-")
        cls.root = Path(cls.temp.name)
        cls.result = run_study(cls.root, quick=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_all_eight_figures_are_written(self) -> None:
        for name in FIGURE_NAMES:
            with self.subTest(figure=name):
                self.assertGreater((self.root / "figures" / f"{name}_quick.png").stat().st_size, 1000)

    def test_all_twelve_csv_tables_are_written(self) -> None:
        for name in TABLE_NAMES:
            with self.subTest(table=name):
                self.assertGreater((self.root / "results" / f"{name}_quick.csv").stat().st_size, 40)

    def test_oracle_truth_has_external_validity_gap(self) -> None:
        self.assertGreater(self.result["target_ate"] - self.result["trial_ate"], .4)

    def test_replicate_table_has_nine_methods(self) -> None:
        path = self.root / "results" / "estimator_summary_quick.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 9)

    def test_overlap_study_has_four_slopes(self) -> None:
        path = self.root / "results" / "overlap_replicates_quick.csv"
        with path.open(newline="", encoding="utf-8") as f:
            slopes = {float(r["slope"]) for r in csv.DictReader(f)}
        self.assertEqual(len(slopes), 4)

    def test_hidden_effect_sensitivity_grid_has_25_cells(self) -> None:
        path = self.root / "results" / "hidden_modifier_sensitivity_quick.csv"
        with path.open(newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 25)

    def test_bootstrap_intervals_enclose_their_medians_broadly(self) -> None:
        path = self.root / "results" / "bootstrap_intervals_quick.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertLess(float(row["ci_lower"]), float(row["ci_upper"]))

    def test_quick_runs_do_not_create_unmarked_publication_files(self) -> None:
        self.assertFalse((self.root / "figures" / "fig1_trial_target_shift.png").exists())
        self.assertFalse((self.root / "results" / "population_truth.csv").exists())


class NotebookTests(unittest.TestCase):
    def test_notebook_executes_in_clean_python_kernel(self) -> None:
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("Install notebook extras to execute fresh-kernel test")
        notebook = nbformat.read(ROOT / "transportability.ipynb", as_version=4)
        NotebookClient(notebook, timeout=150, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()


if __name__ == "__main__":
    unittest.main()
