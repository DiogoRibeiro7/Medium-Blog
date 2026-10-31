"""Validate and execute the pure-Python notebook cells without Jupyter services."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    """Check a minimal notebook contract and execute all code cells."""

    def test_valid_notebook_and_clean_execution(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "model_walkthrough.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        cells = notebook["cells"]
        self.assertGreaterEqual(len(cells), 8)
        programs: list[str] = []
        for number, cell in enumerate(cells, 1):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            source = "".join(cell["source"])
            compile(source, f"notebook-cell-{number}", "exec")
            programs.append(source)
        # A fresh process catches hidden cross-session state and preserves
        # the order of notebook cells without installing Jupyter services.
        result = subprocess.run(
            [sys.executable, "-c", "\n\n".join(programs)],
            check=False,
            text=True,
            capture_output=True,
            cwd=notebook_path.parent,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Generated 8 figures and 4 tables", result.stdout)


if __name__ == "__main__":
    unittest.main()
