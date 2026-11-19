"""Tests for the README project list builder (run without PyYAML)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_readme import (
    END,
    START,
    ProjectsError,
    clean_title,
    link,
    normalise,
    readme_title,
    render,
    splice,
)

MODEL = {
    "sections": [
        {"id": "series-x", "kind": "series", "title": "Series"},
        {"id": "themes", "kind": "themes", "title": "Themes"},
        {"id": "archive", "kind": "archive", "title": "Archive"},
    ],
    "groups": [
        {"id": "s", "section": "series-x", "title": "A series", "about": "About it."},
        {"id": "t", "section": "themes", "title": "A theme"},
        {"id": "old", "section": "archive", "title": "Old"},
    ],
    "folders": {
        "part-two": {"group": "s", "part": 2},
        "part-one": {"group": "s", "part": 1},
        "extra": "s",
        "study": {"group": "t", "title": "Given title"},
        "Old Notebook": "old",
    },
}
FOLDERS = ["extra", "new-one", "Old Notebook", "part-one", "part-two", "study"]


class TitleTests(unittest.TestCase):
    def test_strips_companion_and_article_number_noise(self) -> None:
        self.assertEqual(clean_title("Article #28 — A Cluster Is Not a Population"), "A Cluster Is Not a Population")
        self.assertEqual(
            clean_title("A Dataset Is Not a Population — public research companion"), "A Dataset Is Not a Population"
        )
        self.assertEqual(
            clean_title("From Statute to Enforcement: Reproducible companion"), "From Statute to Enforcement"
        )
        self.assertEqual(clean_title("Measuring the Pandemic Recovery — Article 26"), "Measuring the Pandemic Recovery")
        self.assertEqual(clean_title("Routing on real roads"), "Routing on real roads")

    def test_reads_first_heading_of_readme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "README.md").write_text("Intro\n\n# The study\n\n## Data\n", encoding="utf-8")
            self.assertEqual(readme_title(folder), "The study")
            self.assertIsNone(readme_title(folder / "missing"))


class ModelTests(unittest.TestCase):
    def test_reports_errors_and_unsorted_folders(self) -> None:
        data = {
            **MODEL,
            "folders": {**MODEL["folders"], "gone": "t", "bad": "nope", "odd": {"group": "t", "part": "x"}},
        }
        _, errors, unsorted = normalise(data, FOLDERS)
        joined = " ".join(errors)
        self.assertIn("'gone' is listed but there is no such folder", joined)
        self.assertIn("'bad': unknown group", joined)
        self.assertIn("'odd': part must be a whole number", joined)
        self.assertEqual(unsorted, ["new-one"])

    def test_rejects_unknown_section_kind(self) -> None:
        data = {**MODEL, "sections": MODEL["sections"] + [{"id": "z", "kind": "misc", "title": "Z"}]}
        _, errors, _ = normalise(data, FOLDERS)
        self.assertTrue(any("kind must be one of" in e for e in errors))


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        model, errors, self.unsorted = normalise(MODEL, FOLDERS)
        self.assertEqual(errors, [])
        titles = {"part-one": "One", "part-two": "Two", "extra": None, "study": "From README", "new-one": "New"}
        self.block = render(model, FOLDERS, titles, self.unsorted)

    def test_series_follow_part_order_then_name(self) -> None:
        one, two, extra = (self.block.index(f"[`{n}/`]") for n in ("part-one", "part-two", "extra"))
        self.assertLess(one, two)
        self.assertLess(two, extra)

    def test_title_override_and_archive_line(self) -> None:
        self.assertIn("— Given title", self.block)
        self.assertNotIn("From README", self.block)
        self.assertIn("[`Old Notebook/`](Old%20Notebook/)", self.block)
        self.assertNotIn("- [`Old Notebook/`]", self.block)

    def test_unsorted_folder_is_listed(self) -> None:
        self.assertIn("### Not yet sorted", self.block)
        self.assertIn("- [`new-one/`](new-one/) — New", self.block)
        self.assertIn("1 not yet sorted", self.block)

    def test_splice_replaces_only_the_marked_block(self) -> None:
        readme = f"head\n{START}\nold\n{END}\ntail\n"
        self.assertEqual(splice(readme, f"{START}\nnew\n{END}"), f"head\n{START}\nnew\n{END}\ntail\n")
        with self.assertRaises(ProjectsError):
            splice("no markers", "x")

    def test_link_encodes_spaces(self) -> None:
        self.assertEqual(link("Mongo DB"), "[`Mongo DB/`](Mongo%20DB/)")


if __name__ == "__main__":
    unittest.main()
