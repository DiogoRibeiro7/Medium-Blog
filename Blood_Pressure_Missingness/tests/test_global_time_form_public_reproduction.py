"""Public reproduction contract for global time-form sensitivity."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "analysis_snapshot.csv"
COMMITTED_PATH = PROJECT_DIR / "figures" / "global_time_form_sensitivity.json"
FLOAT_REL_TOL = 1e-8
FLOAT_ABS_TOL = 1e-8


def _assert_close(test: unittest.TestCase, actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        test.assertIsInstance(actual, dict, path)
        test.assertEqual(set(actual), set(expected), path)
        for key in expected:
            _assert_close(test, actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        test.assertEqual(actual, expected, path)
        return
    if isinstance(expected, int):
        test.assertIs(type(actual), int, path)
        test.assertEqual(actual, expected, path)
        return
    if isinstance(expected, float):
        test.assertTrue(
            math.isclose(
                float(actual),
                expected,
                rel_tol=FLOAT_REL_TOL,
                abs_tol=FLOAT_ABS_TOL,
            ),
            f"{path}: regenerated {actual!r} != committed {expected!r}",
        )
        return
    test.fail(f"{path}: unsupported JSON value type {type(expected).__name__}")


class GlobalTimeFormPublicReproductionTests(unittest.TestCase):
    """Require CLI reproduction from the privacy-safe public snapshot."""

    def test_cli_reproduces_committed_global_time_form_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "global_time_form_sensitivity.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "blood_pressure_missingness.analyses.global_time_form",
                    "--data",
                    str(DATA_PATH),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_DIR,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            regenerated = json.loads(output.read_text(encoding="utf-8"))

        committed = json.loads(COMMITTED_PATH.read_text(encoding="utf-8"))
        _assert_close(self, regenerated, committed, "global_time_form_sensitivity.json")


if __name__ == "__main__":
    unittest.main()
