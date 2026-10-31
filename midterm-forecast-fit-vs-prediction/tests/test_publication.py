"""Editorial completeness, internal-link integrity, and data/claim contracts."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from analysis import ROOT, compute_evidence
from figures import FIGURE_FILES


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.article = (ROOT / "article.md").read_text(encoding="utf-8")

    def test_article_meets_long_form_standard(self) -> None:
        self.assertGreaterEqual(len(self.article.split()), 8000)
        self.assertIn("Reproducibility and references", self.article)
        self.assertIn("causation", self.article.lower())
        self.assertIn("1986", self.article)

    def test_all_thirteen_figures_are_embedded_once_with_alt_text(self) -> None:
        references = re.findall(r"!\[([^\]]+)\]\((figures/[^)]+)\)", self.article)
        self.assertEqual(len(references), len(FIGURE_FILES))
        self.assertEqual({Path(path).name for _, path in references}, set(FIGURE_FILES))
        self.assertTrue(all(len(alt.strip()) > 40 for alt, _ in references))

    def test_all_local_markdown_links_resolve(self) -> None:
        targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", self.article)
        for target in targets:
            if target.startswith(("https://", "http://", "#")):
                continue
            with self.subTest(link=target):
                self.assertTrue((ROOT / target).is_file())

    def test_article_claims_match_canonical_computation(self) -> None:
        evidence = compute_evidence(draws=200)
        model = evidence.metrics
        self.assertIn(f"{model['forecast']:.2f}", self.article)
        self.assertIn(f"{model['in_sample_r2']:.4f}", self.article)
        self.assertIn(f"{model['forward_rmse']:.2f}", self.article)
        self.assertIn(f"{model['loocv_rmse']:.2f}", self.article)
        self.assertIn(f"{model['forward_generic_only_rmse']:.2f}", self.article)
        self.assertIn(f"{model['forward_seats_only_rmse']:.2f}", self.article)
        self.assertIn(f"{model['forecast_generic_only']:.2f}".replace("-", "−"),
                      self.article)
        self.assertIn(f"{model['alternative_forward_rmse']:.2f}", self.article)
        self.assertIn(f"{model['alternative_forward_generic_only_rmse']:.2f}",
                      self.article)

    def test_sources_and_index_are_complete(self) -> None:
        provenance = (ROOT / "data" / "PROVENANCE.md").read_text(encoding="utf-8")
        for domain in ("news.gallup.com", "realclearpolitics.com", "presidency.ucsb.edu"):
            self.assertIn(domain, provenance)
        index = (ROOT.parent / "ARTICLES.md").read_text(encoding="utf-8")
        self.assertRegex(
            index,
            r"(?m)^\|\s*\d+\s*\|\s*\[The Forecast Is Not the Fit: "
            r"Rebuilding an 89% Midterm Election Model\]"
            r"\(midterm-forecast-fit-vs-prediction/article\.md\)",
        )


if __name__ == "__main__":
    unittest.main()
