"""Migration guards for manuscript-dependent public audit scripts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manuscript_sources import read_manuscript


class ManuscriptSourceTests(unittest.TestCase):
    """A compatibility notice must never be silently audited as a manuscript."""

    def test_reads_complete_private_manuscript(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "part1"
            path.mkdir()
            article = "# Example\n\n" + "A" * 1100 + "\n\n## References\n\nAuthor (2020)."
            (path / "article.md").write_text(article, encoding="utf-8")
            self.assertEqual(read_manuscript(Path(name), 1), article)

    def test_reads_series_layout(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "part-01-regression-wrong-effect"
            path.mkdir()
            article = "# Example\n\n" + "A" * 1100 + "\n\n## References\n\nAuthor (2020)."
            (path / "article.md").write_text(article, encoding="utf-8")
            self.assertEqual(read_manuscript(Path(name), 1), article)

    def test_rejects_public_compatibility_notice(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "part1"
            path.mkdir()
            (path / "article.md").write_text(
                "# Example\nThe full editorial manuscript is now maintained in private.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "public compatibility notice"):
                read_manuscript(Path(name), 1)

    def test_rejects_short_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "part1"
            path.mkdir()
            (path / "article.md").write_text("# Example\n## References\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a complete"):
                read_manuscript(Path(name), 1)

    def test_invalid_part(self) -> None:
        for value in (0, 21, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_manuscript(Path("."), value)

    def test_missing_manuscript(self) -> None:
        with tempfile.TemporaryDirectory() as name, self.assertRaises(FileNotFoundError):
            read_manuscript(Path(name), 3)


if __name__ == "__main__":
    unittest.main()
