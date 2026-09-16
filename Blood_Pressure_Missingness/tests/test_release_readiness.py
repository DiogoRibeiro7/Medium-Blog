"""Tests for the executable release-readiness gate."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from blood_pressure_missingness.release_readiness import validate_release_readiness

PROJECT_DIR = Path(__file__).resolve().parent.parent


class ReleaseReadinessTests(unittest.TestCase):
    def _copy_project(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name) / "project"
        shutil.copytree(PROJECT_DIR, root)
        return directory, root

    def test_current_tree_is_release_ready(self) -> None:
        self.assertEqual(validate_release_readiness(PROJECT_DIR), [])

    def test_stale_manifest_is_rejected(self) -> None:
        directory, root = self._copy_project()
        self.addCleanup(directory.cleanup)
        manifest_path = root / "RELEASE_MANIFEST.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["artifact_count"] += 1
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        errors = validate_release_readiness(root)
        self.assertTrue(any("does not match" in error for error in errors))

    def test_citation_version_mismatch_is_rejected(self) -> None:
        directory, root = self._copy_project()
        self.addCleanup(directory.cleanup)
        citation = root / "CITATION.cff"
        citation.write_text(
            citation.read_text(encoding="utf-8").replace("version: 0.1.0", "version: 9.9.9"),
            encoding="utf-8",
        )

        errors = validate_release_readiness(root)
        self.assertTrue(any("does not match package version" in error for error in errors))

    def test_unpinned_archival_requirement_is_rejected(self) -> None:
        directory, root = self._copy_project()
        self.addCleanup(directory.cleanup)
        lock = root / "requirements-publication-lock.txt"
        lock.write_text(lock.read_text(encoding="utf-8") + "numpy>=1.0\n", encoding="utf-8")

        errors = validate_release_readiness(root)
        self.assertTrue(any("non-exact requirements" in error for error in errors))

    def test_premature_doi_is_rejected_but_can_be_explicitly_allowed(self) -> None:
        directory, root = self._copy_project()
        self.addCleanup(directory.cleanup)
        citation = root / "CITATION.cff"
        citation.write_text(
            citation.read_text(encoding="utf-8") + "doi: 10.1234/example\n",
            encoding="utf-8",
        )

        errors = validate_release_readiness(root)
        self.assertTrue(any("DOI" in error for error in errors))

        allowed_errors = validate_release_readiness(root, allow_archival_metadata=True)
        self.assertFalse(any("DOI" in error for error in allowed_errors))


if __name__ == "__main__":
    unittest.main()
