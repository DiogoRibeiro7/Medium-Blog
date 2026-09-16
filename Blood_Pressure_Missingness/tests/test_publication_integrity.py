"""Tests for the executable publication-snapshot integrity gate."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from blood_pressure_missingness.publication_integrity import validate_publication_integrity

PROJECT_DIR = Path(__file__).resolve().parent.parent


class PublicationIntegrityTests(unittest.TestCase):
    def _copy_project(self) -> Path:
        temp_dir = Path(tempfile.mkdtemp())
        target = temp_dir / "Blood_Pressure_Missingness"
        shutil.copytree(PROJECT_DIR, target)
        self.addCleanup(shutil.rmtree, temp_dir)
        return target

    def test_current_tree_is_integrity_clean(self) -> None:
        self.assertEqual(validate_publication_integrity(PROJECT_DIR), [])

    def test_stale_manifest_is_rejected(self) -> None:
        root = self._copy_project()
        manifest_path = root / "PUBLICATION_MANIFEST.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["artifact_count"] += 1
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        errors = validate_publication_integrity(root)
        self.assertTrue(any("does not match" in error for error in errors))

    def test_citation_version_mismatch_is_rejected(self) -> None:
        root = self._copy_project()
        citation = root / "CITATION.cff"
        citation.write_text(
            citation.read_text(encoding="utf-8").replace("version: 0.1.0", "version: 9.9.9"),
            encoding="utf-8",
        )
        errors = validate_publication_integrity(root)
        self.assertTrue(any("does not match package version" in error for error in errors))

    def test_unpinned_archival_dependency_is_rejected(self) -> None:
        root = self._copy_project()
        lock = root / "requirements-publication-lock.txt"
        lock.write_text(lock.read_text(encoding="utf-8") + "\nnumpy>=2\n", encoding="utf-8")
        errors = validate_publication_integrity(root)
        self.assertTrue(any("non-exact requirements" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
