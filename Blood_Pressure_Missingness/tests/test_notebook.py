"""Validate and execute the Medium-facing blood-pressure notebook."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_DIR / "blood-pressure-missingness.ipynb"


class BloodPressureNotebookTests(unittest.TestCase):
    """Protect notebook structure and executable reproducibility."""

    def test_notebook_is_schema_valid(self) -> None:
        """The notebook must conform to the nbformat schema."""

        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        self.assertGreaterEqual(len(notebook.cells), 10)
        self.assertTrue(all(cell.get("id") for cell in notebook.cells))

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
