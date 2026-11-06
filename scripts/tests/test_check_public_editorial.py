"""Tests for public editorial manuscript migration guard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_public_editorial import (
    find_oversized_editorial_files,
    is_editorial_path,
)


class EditorialGuardTests(unittest.TestCase):
    """Keep public research documentation while excluding full Medium prose."""

    def test_detects_conventional_manuscript_names(self) -> None:
        for path in (
            "study/article.md",
            "study/publication/medium_ready.md",
            r"study\publication\article.template.md",
        ):
            with self.subTest(path=path):
                self.assertTrue(is_editorial_path(path))

    def test_allows_research_docs_and_source_code(self) -> None:
        for path in (
            "study/README.md",
            "study/METHODS.md",
            "study/publication/AUDIT.md",
            "study/prepare_medium_publication.py",
            "study/figures/figure.png",
        ):
            with self.subTest(path=path):
                self.assertFalse(is_editorial_path(path))

    def test_flags_full_prose_but_keeps_short_notices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "study" / "article.md"
            full.parent.mkdir()
            full.write_text("Long form " * 400, encoding="utf-8")
            notice = root / "other" / "publication" / "medium_ready.md"
            notice.parent.mkdir(parents=True)
            notice.write_text("Private manuscript archived.\n", encoding="utf-8")
            results = find_oversized_editorial_files(
                ["study/article.md", "other/publication/medium_ready.md"], root
            )
            self.assertEqual(results, ["study/article.md"])

    def test_rejects_invalid_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                find_oversized_editorial_files([], Path(directory), max_bytes=-1)
            with self.assertRaises(TypeError):
                is_editorial_path(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
