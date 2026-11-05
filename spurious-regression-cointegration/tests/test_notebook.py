"""Review canonical evidence artifacts and the executable notebook contract."""

from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path
from typing import Final

import nbformat
from nbclient import NotebookClient


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "spurious_regression_cointegration.ipynb"


PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"

FIGURE_NAMES: Final[tuple[str, ...]] = (
    "fig1_independent_random_walks.png",
    "fig2_naive_t_distribution.png",
    "fig3_r_squared_nulls.png",
    "fig4_false_discovery_rates.png",
    "fig5_df_versus_eg_null.png",
    "fig6_genuine_cointegration.png",
    "fig7_slope_convergence.png",
    "fig8_breaks_and_instability.png",
)

TABLE_NAMES: Final[tuple[str, ...]] = (
    "break_replicates.csv",
    "break_summary.csv",
    "broken_relation_path.csv",
    "calibrated_critical_values.csv",
    "cointegrated_path.csv",
    "cointegration_diagnostics.csv",
    "illustrative_null_path.csv",
    "naive_t_histogram.csv",
    "null_regression_replicates.csv",
    "null_regression_summary.csv",
    "rolling_break_estimates.csv",
    "slope_scaling_replicates.csv",
    "slope_scaling_summary.csv",
    "study_summary.csv",
    "unit_root_null_replicates.csv",
)


class ResearchArtifactsAndNotebookTests(unittest.TestCase):
    """Guard public numerical evidence and notebook execution, not private prose."""

    def test_article_path_is_a_short_compatibility_notice(self) -> None:
        notice: str = (PROJECT / "article.md").read_text(encoding="utf-8")
        self.assertIn("research-articles-medium", notice)
        self.assertIn("private", notice.lower())
        self.assertLess(len(notice.split()), 250)

    def test_all_eight_canonical_figures_are_valid_pngs(self) -> None:
        figure_dir: Path = PROJECT / "figures"
        actual_names: set[str] = {path.name for path in figure_dir.glob("*.png")}
        self.assertEqual(actual_names, set(FIGURE_NAMES))
        for name in FIGURE_NAMES:
            with self.subTest(figure=name):
                # The PNG signature and IHDR dimensions catch incomplete artifacts.
                with (figure_dir / name).open("rb") as stream:
                    header: bytes = stream.read(24)
                self.assertEqual(len(header), 24)
                self.assertEqual(header[:8], PNG_SIGNATURE)
                self.assertEqual(header[12:16], b"IHDR")
                self.assertGreater(int.from_bytes(header[16:20], "big"), 0)
                self.assertGreater(int.from_bytes(header[20:24], "big"), 0)

    def test_all_fifteen_canonical_tables_have_records(self) -> None:
        result_dir: Path = PROJECT / "results"
        actual_names: set[str] = {path.name for path in result_dir.glob("*.csv")}
        self.assertEqual(actual_names, set(TABLE_NAMES))
        for name in TABLE_NAMES:
            with self.subTest(table=name):
                with (result_dir / name).open(newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream)
                    header: list[str] | None = next(reader, None)
                    self.assertIsNotNone(header)
                    self.assertTrue(header and all(column.strip() for column in header))
                    self.assertIsNotNone(next(reader, None))

    def test_canonical_numbers_validate_research_claims(self) -> None:
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
