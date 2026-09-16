"""Contract for scoped publication and citation metadata."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_DIR / "pyproject.toml"
CITATION = PROJECT_DIR / "CITATION.cff"
RELEASE_NOTE = PROJECT_DIR / "PUBLICATION_RELEASE.md"


class PublicationMetadataContractTests(unittest.TestCase):
    """Keep version, licence, and release namespace coherent."""

    def test_citation_version_matches_package_version(self) -> None:
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        citation = CITATION.read_text(encoding="utf-8")

        package_match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        citation_match = re.search(r"^version: ([^\s]+)$", citation, re.MULTILINE)
        self.assertIsNotNone(package_match)
        self.assertIsNotNone(citation_match)
        self.assertEqual(package_match.group(1), citation_match.group(1))

    def test_citation_is_scoped_and_does_not_invent_archival_metadata(self) -> None:
        citation = CITATION.read_text(encoding="utf-8")

        self.assertIn(
            "https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/Blood_Pressure_Missingness",
            citation,
        )
        self.assertIn("license: Apache-2.0", citation)
        self.assertNotRegex(citation, r"(?m)^doi:")
        self.assertNotRegex(citation, r"(?m)^date-released:")

    def test_release_namespace_uses_case_study_version(self) -> None:
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        release_note = RELEASE_NOTE.read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(version)
        self.assertIn(
            f"blood-pressure-missingness-v{version.group(1)}",
            release_note,
        )
        self.assertIn("Do not pre-populate", release_note)


if __name__ == "__main__":
    unittest.main()
