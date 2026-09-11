"""Deterministic reproduction contract for non-secret public JSON outputs."""

from __future__ import annotations

import json
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

REPRODUCIBLE_JSON_OUTPUTS = {
    "results.json",
    "observation_process_sensitivity.json",
    "gap_aware_trend_decomposition.json",
    "day_influence_sensitivity.json",
    "episode_observation_sensitivity.json",
    "episode_time_form_sensitivity.json",
    "temporal_dependence_diagnostics.json",
}


def _run(*args: str) -> None:
    subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class PublicReproductionContractTests(unittest.TestCase):
    """Require committed non-secret statistical JSON to regenerate exactly."""

    def test_committed_public_json_is_reproducible_from_public_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            _run(
                "analysis.py",
                "--data",
                str(DATA_PATH),
                "--audit",
                str(AUDIT_PATH),
                "--output-dir",
                str(output_dir),
            )
            _run(
                "observation_process_sensitivity.py",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "observation_process_sensitivity.json"),
                "--output-figure",
                str(output_dir / "observation_process_sensitivity.svg"),
            )
            _run(
                "gap_aware_trend_decomposition.py",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "gap_aware_trend_decomposition.json"),
                "--output-figure",
                str(output_dir / "gap_aware_trend_decomposition.svg"),
            )
            _run(
                "day_influence_sensitivity.py",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "day_influence_sensitivity.json"),
                "--output-figure",
                str(output_dir / "day_influence_sensitivity.svg"),
            )
            _run(
                "episode_observation_sensitivity.py",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "episode_observation_sensitivity.json"),
                "--output-figure",
                str(output_dir / "episode_observation_sensitivity.svg"),
            )
            _run(
                "episode_time_form_sensitivity.py",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "episode_time_form_sensitivity.json"),
                "--output-figure",
                str(output_dir / "episode_time_form_sensitivity.svg"),
            )
            _run(
                "temporal_dependence_diagnostics.py",
                "--data",
                str(DATA_PATH),
                "--output-json",
                str(output_dir / "temporal_dependence_diagnostics.json"),
            )

            produced = {path.name for path in output_dir.glob("*.json")}
            self.assertTrue(REPRODUCIBLE_JSON_OUTPUTS <= produced)

            for name in sorted(REPRODUCIBLE_JSON_OUTPUTS):
                with self.subTest(output=name):
                    self.assertEqual(
                        _load(output_dir / name),
                        _load(FIGURES_DIR / name),
                        f"{name} is stale relative to the committed public snapshot",
                    )

    def test_secret_backed_temperature_outputs_are_outside_public_reproduction_gate(self) -> None:
        self.assertNotIn("temperature_covariate.json", REPRODUCIBLE_JSON_OUTPUTS)
        self.assertNotIn("temperature_covariate_sensitivity.json", REPRODUCIBLE_JSON_OUTPUTS)


if __name__ == "__main__":
    unittest.main()
