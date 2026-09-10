"""Architecture tests for the package-native core analysis dependency spine."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_ANALYSES = PROJECT_DIR / "blood_pressure_missingness" / "analyses"


class PackageNativeDependencySpineTests(unittest.TestCase):
    """Prevent canonical implementations from regressing to legacy flat imports."""

    def _imports(self, filename: str) -> set[str]:
        tree = ast.parse((PACKAGE_ANALYSES / filename).read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.add(node.module)
        return imports

    def test_gap_aware_does_not_import_legacy_analysis_module(self) -> None:
        imports = self._imports("gap_aware.py")
        self.assertNotIn("analysis", imports)
        self.assertIn("blood_pressure_missingness", imports)

    def test_day_influence_does_not_import_legacy_dependency_modules(self) -> None:
        imports = self._imports("day_influence.py")
        self.assertNotIn("analysis", imports)
        self.assertNotIn("gap_aware_trend_decomposition", imports)
        self.assertIn("blood_pressure_missingness", imports)
        self.assertIn("blood_pressure_missingness.analyses", imports)


if __name__ == "__main__":
    unittest.main()
