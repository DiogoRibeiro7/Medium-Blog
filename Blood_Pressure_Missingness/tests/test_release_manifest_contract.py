"""Contract for the committed public release integrity manifest."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from blood_pressure_missingness.release_manifest import build_release_manifest

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_DIR / "RELEASE_MANIFEST.json"


class ReleaseManifestContractTests(unittest.TestCase):
    """Require the committed manifest to match current public artifacts exactly."""

    def test_release_manifest_matches_current_public_artifacts(self) -> None:
        expected = build_release_manifest(PROJECT_DIR)
        committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        if committed != expected:
            self.fail(
                "RELEASE_MANIFEST.json is stale. Expected generated payload:\n"
                + json.dumps(expected, indent=2, sort_keys=False)
            )

        self.assertEqual(committed["hash_algorithm"], "sha256")
        self.assertEqual(committed["artifact_count"], len(committed["artifacts"]))
        self.assertEqual(
            committed["release_tag"],
            f"blood-pressure-missingness-v{committed['version']}",
        )


if __name__ == "__main__":
    unittest.main()
