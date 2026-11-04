"""Verify that Medium derivatives do not modify the frozen scientific claim."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("post_election_medium", ROOT / "prepare_medium_publication.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load post-election Medium preparation module")
publication = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publication
SPEC.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    """Enforce reproducible image bytes and unambiguous publication status."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = publication.load_figure_manifest()
        cls.source = publication.SOURCE.read_text(encoding="utf-8")

    def test_publication_copy_is_current(self) -> None:
        prepared = publication.build_medium_copy(self.source, self.manifest)
        current = publication.OUTPUT.read_text(encoding="utf-8")
        self.assertEqual(prepared, current)

    def test_eight_images_link_to_a_fixed_commit(self) -> None:
        prepared = publication.build_medium_copy(self.source, self.manifest)
        prefix = f"https://raw.githubusercontent.com/{publication.REPOSITORY}/{self.manifest.commit}/"
        self.assertEqual(prepared.count(prefix), 8)
        self.assertNotIn("](figures/", prepared)

    def test_all_images_are_pinned_by_checksum(self) -> None:
        self.assertEqual(len(self.manifest.sha256), 8)
        publication.build_medium_copy(self.source, self.manifest)

    def test_stale_image_manifest_is_rejected(self) -> None:
        modified = dict(self.manifest.sha256)
        first = next(iter(modified))
        modified[first] = "0" * 64
        with self.assertRaises(ValueError):
            publication.build_medium_copy(self.source, replace(self.manifest, sha256=modified))

    def test_removed_uncalled_disclaimer_is_rejected(self) -> None:
        invalid = self.source.replace("68 House races remained uncalled", "all House races are final")
        with self.assertRaises(ValueError):
            publication.validate_manuscript(invalid)

    def test_raw_markdown_table_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            publication.validate_manuscript(self.source + "\n| unsupported | table |\n")

    def test_absolute_companion_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            publication.resolve_companion("/etc/passwd")

    def test_source_status_and_publication_link_remain_explicit(self) -> None:
        prepared = publication.build_medium_copy(self.source, self.manifest)
        self.assertIn("Research draft, 4 November 2026", prepared)
        self.assertIn("not certified election results", prepared)
        self.assertIn("https://github.com/DiogoRibeiro7/Medium-Blog/blob/", prepared)


if __name__ == "__main__":
    unittest.main()
