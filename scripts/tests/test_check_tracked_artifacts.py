"""Regression tests for the repository-wide generated-file guard."""

from __future__ import annotations

import unittest

from scripts.check_tracked_artifacts import find_disposable_paths, is_disposable


class DisposableArtifactTests(unittest.TestCase):
    """Protect the narrow distinction between scratch and research artifacts."""

    def test_generated_artifacts_are_blocked(self) -> None:
        """Reject compiled files, temporary checkpoints, and known exports."""
        examples: tuple[str, ...] = (
            "Django-Charts/__pycache__/views.cpython-36.pyc",
            "clima_portugal/.ipynb_checkpoints/Untitled-checkpoint.ipynb",
            "project/.pytest_cache/v/cache/nodeids",
            "project/.mypy_cache/3.13/cache.json",
            "project/.ruff_cache/0.14/file",
            "project/.Rproj.user/shared/notebooks/abc",
            "just_fun/temp.csv",
            "just_fun/AlcoholCPI.tiff",
            "just_fun/DisposableIncomes.tiff",
            "project/.DS_Store",
            "project/Thumbs.db",
            "project/.Rhistory",
            "project/.coverage.12",
            "a.pyc",
            "backup.swp",
        )
        for path in examples:
            with self.subTest(path=path):
                self.assertTrue(is_disposable(path))

    def test_historical_outputs_and_inputs_are_allowed(self) -> None:
        """Do not block article assets or legitimate reproducibility records."""
        examples: tuple[str, ...] = (
            "just_fun/Untitled.ipynb",
            "island-cable-resilience/results/exp1_structure.log",
            "post-cutoff-forecasting/results/scores.csv",
            "eu-wheat-weather/docs/panel/weather.csv",
            "clima_portugal/kadabra.ipynb",
            "HeartRate.RData",
            "R_code/study_data.csv",
            "Django-Charts/__init__.py",
            "data/my__pycache__notes.txt",
            "tests/test_statistics.py",
        )
        for path in examples:
            with self.subTest(path=path):
                self.assertFalse(is_disposable(path))

    def test_windows_separator_and_duplicate_paths(self) -> None:
        """Handle local Windows paths and report each violation once."""
        paths: list[str] = [
            "project\\__pycache__\\x.pyc",
            "project\\__pycache__\\x.pyc",
            "research/data.csv",
        ]
        self.assertEqual(
            find_disposable_paths(paths),
            ["project\\__pycache__\\x.pyc"],
        )

    def test_non_string_paths_raise(self) -> None:
        """Keep invalid API usage explicit instead of silently accepting it."""
        with self.assertRaises(TypeError):
            is_disposable(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
