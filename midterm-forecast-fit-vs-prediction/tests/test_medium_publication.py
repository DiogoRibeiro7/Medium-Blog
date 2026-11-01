"""Editorial and technical release gates for the Medium publication copy."""

from __future__ import annotations

import unittest

from prepare_medium_publication import (
    INLINE_MATH,
    OUTPUT,
    SOURCE,
    build_release_markdown,
    check_release,
    extract_equations,
)


class MediumPublicationTests(unittest.TestCase):
    """Check publication readiness without contacting Medium or changing data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.release = OUTPUT.read_text(encoding="utf-8")

    def test_committed_publication_matches_scientific_article(self) -> None:
        self.assertEqual(build_release_markdown(self.source), self.release)

    def test_full_release_check(self) -> None:
        words, figures, equations = check_release(self.source, self.release)
        self.assertGreaterEqual(words, 8000)
        self.assertEqual((figures, equations), (13, 15))

    def test_inline_latex_is_replaced(self) -> None:
        self.assertGreater(len(INLINE_MATH.findall(self.source)), 50)
        self.assertIsNone(INLINE_MATH.search(self.release))
        self.assertIn("R² = 0.8934", self.release)

    def test_equations_have_publishable_images(self) -> None:
        self.assertEqual(len(extract_equations(self.source)), 15)
        self.assertEqual(self.release.count("![Equation "), 15)

    def test_unknown_math_requires_editorial_review(self) -> None:
        manuscript = self.source.replace("\\(R^2\\)", "\\(\\unknown_expression\\)", 1)
        with self.assertRaisesRegex(ValueError, "Unknown inline expression"):
            build_release_markdown(manuscript)

    def test_figure_loss_cannot_pass(self) -> None:
        manuscript = self.source.replace("![Historical fitted versus actual", "[Historical fitted versus actual", 1)
        with self.assertRaisesRegex(ValueError, "thirteen original evidence figures"):
            build_release_markdown(manuscript)

    def test_draft_status_cannot_pass(self) -> None:
        manuscript = self.source.replace("final pre-election manuscript", "draft for author review")
        with self.assertRaisesRegex(ValueError, "editorial sign-off"):
            build_release_markdown(manuscript)

    def test_publication_does_not_claim_an_election_result(self) -> None:
        self.assertIn("not an observed election result", self.source)
        self.assertIn("No 2026 outcome is used", self.release)
        self.assertIn("31 October 2026", self.release)
        self.assertIn("1 November 2026", self.release)


if __name__ == "__main__":
    unittest.main()
