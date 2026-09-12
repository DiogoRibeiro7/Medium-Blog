"""Architecture tests for validation dependency direction."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_DIR / "blood_pressure_missingness"
ANALYSES_DIR = PACKAGE_DIR / "analyses"
VALIDATION_DIR = PACKAGE_DIR / "validation"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


class ValidationDependencySpineTests(unittest.TestCase):
    """Keep validation above analyses and away from historical root modules."""

    def test_validation_modules_use_package_native_dependencies(self) -> None:
        forbidden = {
            "analysis",
            "day_influence_sensitivity",
            "episode_observation_sensitivity",
            "episode_time_form_sensitivity",
            "gap_aware_trend_decomposition",
            "temporal_dependence_diagnostics",
        }
        for filename in ("current_influence.py", "current_narrative.py"):
            imports = _imports(VALIDATION_DIR / filename)
            with self.subTest(filename=filename):
                self.assertTrue(forbidden.isdisjoint(imports), imports)
                self.assertIn("blood_pressure_missingness", imports)
                self.assertIn("blood_pressure_missingness.analyses", imports)

    def test_analyses_do_not_import_validation_layer(self) -> None:
        for path in ANALYSES_DIR.glob("*.py"):
            imports = _imports(path)
            with self.subTest(filename=path.name):
                self.assertFalse(
                    any(
                        module == "blood_pressure_missingness.validation"
                        or module.startswith("blood_pressure_missingness.validation.")
                        for module in imports
                    ),
                    imports,
                )


if __name__ == "__main__":
    unittest.main()
