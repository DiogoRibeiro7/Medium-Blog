"""Smoke tests for the staged blood-pressure package refactor."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_legacy_module(module_name: str, path: Path) -> ModuleType:
    """Load one compatibility shim by file path without mutating ``sys.path``."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load compatibility shim: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackageArchitectureTests(unittest.TestCase):
    """Keep canonical package imports equivalent to historical module imports."""

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

    def test_public_analysis_facade_preserves_core_objects(self) -> None:
        legacy = _load_legacy_module("legacy_analysis", PROJECT_DIR / "analysis.py")
        canonical = importlib.import_module("blood_pressure_missingness.public_analysis")
        self.assertIs(canonical.DailyRecord, legacy.DailyRecord)
        self.assertIs(canonical.load_snapshot, legacy.load_snapshot)
        self.assertIs(canonical.linear_trend, legacy.linear_trend)

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

    def test_temperature_facade_preserves_private_test_helpers(self) -> None:
        legacy = _load_legacy_module(
            "legacy_temperature_covariate_sensitivity",
            PROJECT_DIR / "temperature_covariate_sensitivity.py",
        )
        canonical = importlib.import_module(
            "blood_pressure_missingness.analyses.temperature_sensitivity"
        )
        self.assertIs(canonical._fit_linear, legacy._fit_linear)
        self.assertIs(canonical._fit_quadratic, legacy._fit_quadratic)


if __name__ == "__main__":
    unittest.main()
