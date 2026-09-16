"""Contract for the committed public publication snapshot manifest."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from blood_pressure_missingness.publication_manifest import build_publication_manifest

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "PUBLICATION_MANIFEST.json"


class PublicationManifestContractTests(unittest.TestCase):
    def test_publication_manifest_matches_current_public_artifacts(self) -> None:
        committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = build_publication_manifest(PROJECT_DIR)
        self.assertEqual(committed, expected)
        self.assertEqual(committed["snapshot_kind"], "publication")
        self.assertEqual(committed["artifact_count"], len(committed["artifacts"]))
        self.assertNotIn("release_tag", committed)


if __name__ == "__main__":
    unittest.main()
