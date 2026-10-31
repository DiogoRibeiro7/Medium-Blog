"""Review the published article, canonical evidence, and notebook contract."""

from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "spurious_regression_cointegration.ipynb"


class ArticleAndNotebookTests(unittest.TestCase):
    """Guard the scientific manuscript and executable reader companion."""

    def test_article_is_at_least_five_thousand_words(self) -> None:
        words = (PROJECT / "article.md").read_text(encoding="utf-8").split()
        self.assertGreaterEqual(len(words), 5000)

    def test_article_has_eight_figure_links(self) -> None:
        manuscript = (PROJECT / "article.md").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count("![]("), 0)
        for number in range(1, 9):
            with self.subTest(number=number):
                self.assertIn(f"(figures/fig{number}_", manuscript)

    def test_all_eight_canonical_figures_exist(self) -> None:
        self.assertEqual(len(list((PROJECT / "figures").glob("fig*.png"))), 8)

    def test_all_fifteen_canonical_tables_exist(self) -> None:
        self.assertEqual(len(list((PROJECT / "results").glob("*.csv"))), 15)

    def test_canonical_numbers_support_article_claim(self) -> None:
        values: dict[str, float] = {}
        with (PROJECT / "results/study_summary.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                values[row["metric"]] = float(row["value"])
        self.assertGreater(values["rw_levels_false_rejection"], .75)
        self.assertAlmostEqual(values["rw_differences_false_rejection"], .05, delta=.03)
        self.assertAlmostEqual(values["iid_false_rejection"], .05, delta=.03)
        self.assertLess(values["eg_null_critical_5pct"], values["df_null_critical_5pct"])
        self.assertTrue(all(math.isfinite(value) for value in values.values()))

    def test_notebook_has_valid_schema(self) -> None:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        self.assertGreaterEqual(len(notebook.cells), 8)

    def test_notebook_contains_executable_code(self) -> None:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        self.assertGreaterEqual(len([cell for cell in notebook.cells if cell.cell_type == "code"]), 4)

    def test_notebook_executable_in_fresh_kernel(self) -> None:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT)}},
        )
        executed = client.execute()
        self.assertTrue(all(cell.execution_count is not None
                            for cell in executed.cells if cell.cell_type == "code"))


if __name__ == "__main__":
    unittest.main()
