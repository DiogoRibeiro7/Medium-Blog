"""Contract for atomic refresh of the public release manifest."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "refresh-blood-pressure-analysis.yml"


class RefreshReleaseManifestContractTests(unittest.TestCase):
    """Keep release-manifest regeneration inside the refresh transaction."""

    def test_refresh_regenerates_manifest_before_contract_validation(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        generate = workflow.index("- name: Regenerate public release manifest")
        validate = workflow.index("- name: Validate generated public artifact contracts")
        self.assertLess(generate, validate)
        self.assertIn(
            "python -m blood_pressure_missingness.release_manifest",
            workflow,
        )
        self.assertIn("--output RELEASE_MANIFEST.json", workflow)

    def test_manifest_participates_in_change_detection_and_staging(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        manifest_path = "Blood_Pressure_Missingness/RELEASE_MANIFEST.json"
        self.assertGreaterEqual(workflow.count(manifest_path), 2)

        detect_start = workflow.index("- name: Detect public-output changes")
        create_pr_start = workflow.index("- name: Create or update refresh pull request")
        detect_block = workflow[detect_start:create_pr_start]
        self.assertIn(manifest_path, detect_block)

        staging_block = workflow[create_pr_start:]
        self.assertIn(manifest_path, staging_block)


if __name__ == "__main__":
    unittest.main()
