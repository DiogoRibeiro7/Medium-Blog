"""Execute the publication notebook with a fresh Python kernel."""

from __future__ import annotations

import unittest

import nbformat
from nbclient import NotebookClient

from analysis import ROOT


class NotebookExecutionTests(unittest.TestCase):
    def test_model_walkthrough_executes(self) -> None:
        notebook = nbformat.read(ROOT / "model_walkthrough.ipynb", as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook,
            timeout=150,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        executed = client.execute()
        outputs = [output for cell in executed.cells if cell.cell_type == "code"
                   for output in cell.get("outputs", [])]
        self.assertFalse(any(output.output_type == "error" for output in outputs))
        self.assertTrue(any("0.893" in str(output.get("text", "")) for output in outputs))


if __name__ == "__main__":
    unittest.main()
