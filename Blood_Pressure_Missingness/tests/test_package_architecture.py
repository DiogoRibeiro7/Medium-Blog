"""Smoke tests for the staged blood-pressure package refactor."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


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

    def test_public_analysis_facade_preserves_core_objects(self) -> None:
        legacy = importlib.import_module("analysis")
        canonical = importlib.import_module("blood_pressure_missingness.public_analysis")
        self.assertIs(canonical.DailyRecord, legacy.DailyRecord)
        self.assertIs(canonical.load_snapshot, legacy.load_snapshot)
        self.assertIs(canonical.linear_trend, legacy.linear_trend)

    def test_temperature_facade_preserves_private_test_helpers(self) -> None:
        legacy = importlib.import_module("temperature_covariate_sensitivity")
        canonical = importlib.import_module(
            "blood_pressure_missingness.analyses.temperature_sensitivity"
        )
        self.assertIs(canonical._fit_linear, legacy._fit_linear)
        self.assertIs(canonical._fit_quadratic, legacy._fit_quadratic)


if __name__ == "__main__":
    unittest.main()
