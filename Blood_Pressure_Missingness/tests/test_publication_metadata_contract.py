"""Contract for scoped publication and citation metadata."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_DIR / "pyproject.toml"
CITATION = PROJECT_DIR / "CITATION.cff"
SNAPSHOT_NOTE = PROJECT_DIR / "PUBLICATION_SNAPSHOT.md"


class PublicationMetadataContractTests(unittest.TestCase):
    """Keep version, licence, and publication-snapshot semantics coherent."""

    def test_citation_version_matches_package_version(self) -> None:
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        citation = CITATION.read_text(encoding="utf-8")
        package_match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        citation_match = re.search(r"^version: ([^\s]+)$", citation, re.MULTILINE)
        self.assertIsNotNone(package_match)
        self.assertIsNotNone(citation_match)
        self.assertEqual(package_match.group(1), citation_match.group(1))

    def test_citation_is_scoped(self) -> None:
        citation = CITATION.read_text(encoding="utf-8")
        self.assertIn(
            "https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/Blood_Pressure_Missingness",
            citation,
        )
        self.assertIn("license: Apache-2.0", citation)

    def test_snapshot_note_rejects_independent_release_lifecycle(self) -> None:
        note = SNAPSHOT_NOTE.read_text(encoding="utf-8")
        self.assertIn("does **not** have an independent GitHub release lifecycle", note)
        self.assertIn("No subdirectory-specific GitHub release or tag is required", note)
        self.assertNotIn("blood-pressure-missingness-v", note)


if __name__ == "__main__":
    unittest.main()
