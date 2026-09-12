"""Validate and execute the Medium-facing blood-pressure notebook."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_DIR / "blood-pressure-missingness.ipynb"

LEGACY_NOTEBOOK_MODULES = {
    "analysis",
    "observation_process_sensitivity",
    "gap_aware_trend_decomposition",
    "day_influence_sensitivity",
    "episode_observation_sensitivity",
    "episode_time_form_sensitivity",
    "temporal_dependence_diagnostics",
}


class BloodPressureNotebookTests(unittest.TestCase):
    """Protect notebook structure and executable reproducibility."""

    def test_notebook_is_schema_valid(self) -> None:
        """The notebook must conform to the nbformat schema."""

        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        self.assertGreaterEqual(len(notebook.cells), 10)
        self.assertTrue(all(cell.get("id") for cell in notebook.cells))

    def test_notebook_uses_canonical_package_imports(self) -> None:
        """The supported notebook must not depend on legacy root-module shims."""

        notebook = nbformat.read(NOTEBOOK, as_version=4)
        imported: set[str] = set()
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)

        self.assertTrue(LEGACY_NOTEBOOK_MODULES.isdisjoint(imported), imported)
        self.assertIn("blood_pressure_missingness", imported)
        self.assertIn("blood_pressure_missingness.analyses", imported)

    def test_notebook_executes_from_project_directory(self) -> None:
        """Execute every code cell against the committed public snapshot."""

        notebook = nbformat.read(NOTEBOOK, as_version=4)
        previous = Path.cwd()
        try:
            os.chdir(PROJECT_DIR)
            client = NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                allow_errors=False,
            )
            client.execute()
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
