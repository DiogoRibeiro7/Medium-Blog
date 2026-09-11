"""Architecture guards for the private blood-pressure data dependency spine."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_DIR / "blood_pressure_missingness"
ANALYSES_DIR = PACKAGE_DIR / "analyses"
DATA_SOURCES_DIR = PACKAGE_DIR / "data_sources"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


class PrivateDataDependencySpineTests(unittest.TestCase):
    """Keep private ingestion below temperature analyses in the package graph."""

    def test_temperature_uses_canonical_google_sheets_source(self) -> None:
        imports = _imports(ANALYSES_DIR / "temperature.py")
        self.assertNotIn("refresh_from_google_sheets", imports)
        self.assertIn("blood_pressure_missingness.data_sources", imports)

    def test_temperature_sensitivity_uses_canonical_dependencies(self) -> None:
        imports = _imports(ANALYSES_DIR / "temperature_sensitivity.py")
        self.assertNotIn("refresh_from_google_sheets", imports)
        self.assertNotIn("temperature_covariate", imports)
        self.assertIn("blood_pressure_missingness.data_sources", imports)
        self.assertIn("blood_pressure_missingness.analyses", imports)

    def test_google_sheets_source_does_not_import_upward(self) -> None:
        imports = _imports(DATA_SOURCES_DIR / "google_sheets.py")
        forbidden_prefixes = (
            "blood_pressure_missingness.analyses",
            "blood_pressure_missingness.validation",
        )
        upward = {
            module
            for module in imports
            if module.startswith(forbidden_prefixes)
        }
        self.assertEqual(set(), upward)


if __name__ == "__main__":
    unittest.main()
