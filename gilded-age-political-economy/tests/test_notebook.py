"""Run the committed notebook cells in a fresh interpreter without Jupyter."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_all_code_cells_execute_in_sequence(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "model_walkthrough.ipynb"
        raw = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(raw.get("nbformat"), 4)
        programs: list[str] = []
        for number, cell in enumerate(raw["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"notebook-cell-{number}", "exec")
            programs.append(source)
        self.assertGreaterEqual(len(programs), 4)
        process = subprocess.run(
            [sys.executable, "-c", "\n\n".join(programs)],
            cwd=notebook_path.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Generated 8 figures and 3 tables", process.stdout)


if __name__ == "__main__":
    unittest.main()
