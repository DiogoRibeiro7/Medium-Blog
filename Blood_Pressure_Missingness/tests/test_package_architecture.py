"""Smoke tests for the canonical blood-pressure package architecture."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


class PackageArchitectureTests(unittest.TestCase):
    """Protect the canonical package boundary after compatibility cleanup."""

    def test_canonical_modules_import(self) -> None:
        modules = (
            "blood_pressure_missingness.public_analysis",
            "blood_pressure_missingness.data_sources.google_sheets",
            "blood_pressure_missingness.analyses.observation_process",
            "blood_pressure_missingness.analyses.gap_aware",
            "blood_pressure_missingness.analyses.day_influence",
            "blood_pressure_missingness.analyses.episode_observation",
            "blood_pressure_missingness.analyses.episode_time_form",
            "blood_pressure_missingness.analyses.temporal_dependence",
            "blood_pressure_missingness.analyses.temperature",
            "blood_pressure_missingness.analyses.temperature_sensitivity",
            "blood_pressure_missingness.validation.current_influence",
            "blood_pressure_missingness.validation.current_narrative",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_installed_package_imports_outside_project_directory(self) -> None:
        code = (
            "from blood_pressure_missingness import public_analysis; "
            "print(public_analysis.__name__)"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temporary_directory,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            completed.stdout.strip(),
            "blood_pressure_missingness.public_analysis",
        )

    def test_migrated_analyses_do_not_import_legacy_root_modules(self) -> None:
        analyses_dir = PROJECT_DIR / "blood_pressure_missingness" / "analyses"
        paths = tuple(
            analyses_dir / filename
            for filename in (
                "gap_aware.py",
                "day_influence.py",
                "episode_observation.py",
                "episode_time_form.py",
                "temporal_dependence.py",
            )
        )
        forbidden = {"analysis", "gap_aware_trend_decomposition"}

        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)

            with self.subTest(path=path.name):
                self.assertTrue(forbidden.isdisjoint(imports), imports)

    def test_legacy_compatibility_files_are_absent(self) -> None:
        legacy_paths = (
            "analysis.py",
            "observation_process_sensitivity.py",
            "gap_aware_trend_decomposition.py",
            "day_influence_sensitivity.py",
            "episode_observation_sensitivity.py",
            "episode_time_form_sensitivity.py",
            "temporal_dependence_diagnostics.py",
            "refresh_from_google_sheets.py",
            "temperature_covariate.py",
            "temperature_covariate_sensitivity.py",
            "validate_current_influence_findings.py",
            "validate_current_narrative.py",
            "blood_pressure_missingness/_compat.py",
        )
        for relative_path in legacy_paths:
            with self.subTest(path=relative_path):
                self.assertFalse((PROJECT_DIR / relative_path).exists())


if __name__ == "__main__":
    unittest.main()
