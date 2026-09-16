"""Static contract for the guarded case-study release workflow."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_DIR.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-blood-pressure-analysis.yml"
TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test-blood-pressure-analysis.yml"


class ReleaseWorkflowContractTests(unittest.TestCase):
    def test_release_workflow_is_manual_main_scoped_and_guarded(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("ref: main", workflow)
        self.assertIn("contents: write", workflow)
        self.assertIn("python-version: '3.13.15'", workflow)
        self.assertIn("requirements-publication-lock.txt", workflow)
        self.assertIn("release_readiness --root .", workflow)
        self.assertIn("REQUESTED_VERSION", workflow)
        self.assertIn("pyproject.toml", workflow)
        self.assertIn('target_sha="$(git rev-parse HEAD)"', workflow)
        self.assertIn("gh release view", workflow)
        self.assertIn("git show-ref --verify --quiet", workflow)
        self.assertIn("gh release create", workflow)
        self.assertIn('--target "$TARGET_SHA"', workflow)
        self.assertIn("blood-pressure-missingness-v", workflow)

    def test_release_notes_are_generated_without_shell_heredoc_backticks(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertNotIn("cat > release-notes.md <<EOF", workflow)
        self.assertIn('Path("release-notes.md").write_text', workflow)
        self.assertIn("observational sensitivity analyses, not causal or clinical claims", workflow)
        self.assertIn("does not include raw health rows", workflow)

    def test_release_workflow_changes_trigger_project_ci(self) -> None:
        workflow = TEST_WORKFLOW.read_text(encoding="utf-8")
        marker = ".github/workflows/release-blood-pressure-analysis.yml"
        self.assertGreaterEqual(workflow.count(marker), 2)


if __name__ == "__main__":
    unittest.main()
