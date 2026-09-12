"""Deterministic reproduction contract for non-secret public JSON outputs."""

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
AUDIT_PATH = PROJECT_DIR / "data" / "source_audit.json"
FIGURES_DIR = PROJECT_DIR / "figures"

FLOAT_REL_TOL = 1e-12
FLOAT_ABS_TOL = 1e-12

REPRODUCIBLE_JSON_OUTPUTS = {
    "results.json",
    "observation_process_sensitivity.json",
    "gap_aware_trend_decomposition.json",
    "day_influence_sensitivity.json",
    "episode_observation_sensitivity.json",
    "episode_time_form_sensitivity.json",
    "temporal_dependence_diagnostics.json",
}


def _run(module_name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module_name, *args],
        cwd=PROJECT_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_json_reproduced(
    test: unittest.TestCase,
    actual: Any,
    expected: Any,
    path: str = "<root>",
) -> None:
    """Compare JSON semantics exactly except for tiny floating-point roundoff."""

    if isinstance(expected, dict):
        test.assertIsInstance(actual, dict, path)
        test.assertEqual(set(actual), set(expected), path)
        for key in expected:
            _assert_json_reproduced(test, actual[key], expected[key], f"{path}.{key}")
        return

    if isinstance(expected, list):
        test.assertIsInstance(actual, list, path)
        test.assertEqual(len(actual), len(expected), path)
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_json_reproduced(
                test,
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
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
            isinstance(actual, (int, float)) and not isinstance(actual, bool),
            f"{path}: expected numeric value, got {type(actual).__name__}",
        )
        test.assertTrue(
            math.isclose(
                float(actual),
                expected,
                rel_tol=FLOAT_REL_TOL,
                abs_tol=FLOAT_ABS_TOL,
            ),
            f"{path}: regenerated {actual!r} != committed {expected!r} within "
            f"rtol={FLOAT_REL_TOL:g}, atol={FLOAT_ABS_TOL:g}",
        )
        return

    test.fail(f"{path}: unsupported JSON value type {type(expected).__name__}")


class PublicReproductionContractTests(unittest.TestCase):
    """Require committed non-secret statistical JSON to regenerate stably."""

    def test_committed_public_json_is_reproducible_from_public_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            _run(
                "blood_pressure_missingness.public_analysis",
                "--data",
                str(DATA_PATH),
                "--audit",
                str(AUDIT_PATH),
                "--output-dir",
                str(output_dir),
            )
            _run(
                "blood_pressure_missingness.analyses.observation_process",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "observation_process_sensitivity.json"),
                "--output-figure",
                str(output_dir / "observation_process_sensitivity.svg"),
            )
            _run(
                "blood_pressure_missingness.analyses.gap_aware",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "gap_aware_trend_decomposition.json"),
                "--output-figure",
                str(output_dir / "gap_aware_trend_decomposition.svg"),
            )
            _run(
                "blood_pressure_missingness.analyses.day_influence",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "day_influence_sensitivity.json"),
                "--output-figure",
                str(output_dir / "day_influence_sensitivity.svg"),
            )
            _run(
                "blood_pressure_missingness.analyses.episode_observation",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "episode_observation_sensitivity.json"),
                "--output-figure",
                str(output_dir / "episode_observation_sensitivity.svg"),
            )
            _run(
                "blood_pressure_missingness.analyses.episode_time_form",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "episode_time_form_sensitivity.json"),
                "--output-figure",
                str(output_dir / "episode_time_form_sensitivity.svg"),
            )
            _run(
                "blood_pressure_missingness.analyses.temporal_dependence",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "temporal_dependence_diagnostics.json"),
            )

            produced = {path.name for path in output_dir.glob("*.json")}
            self.assertTrue(REPRODUCIBLE_JSON_OUTPUTS <= produced)

            for name in sorted(REPRODUCIBLE_JSON_OUTPUTS):
                with self.subTest(output=name):
                    _assert_json_reproduced(
                        self,
                        _load(output_dir / name),
                        _load(FIGURES_DIR / name),
                        name,
                    )

    def test_secret_backed_temperature_outputs_are_outside_public_reproduction_gate(self) -> None:
        self.assertNotIn("temperature_covariate.json", REPRODUCIBLE_JSON_OUTPUTS)
        self.assertNotIn("temperature_covariate_sensitivity.json", REPRODUCIBLE_JSON_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
