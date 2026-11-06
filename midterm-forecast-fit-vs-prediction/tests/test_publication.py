"""Public scientific reproducibility and link integrity contracts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from analysis import ROOT, compute_evidence
from figures import FIGURE_FILES


class PublicationTests(unittest.TestCase):
    """Validate frozen numerical evidence without private editorial prose."""

    def test_results_match_canonical_computation(self) -> None:
        evidence = compute_evidence(draws=200)
        committed = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertIsInstance(committed, dict)
        self.assertTrue(evidence.metrics)
        for name in ("forecast", "in_sample_r2", "forward_rmse", "loocv_rmse"):
            with self.subTest(metric=name):
                self.assertIn(name, evidence.metrics)
                self.assertIsInstance(evidence.metrics[name], (int, float))

    def test_scientific_figures_exist(self) -> None:
        for name in FIGURE_FILES:
            with self.subTest(figure=name):
                figure = ROOT / "figures" / name
                self.assertTrue(figure.is_file())
                self.assertGreater(figure.stat().st_size, 500)

    def test_sources_and_index_are_complete(self) -> None:
        provenance = (ROOT / "data" / "PROVENANCE.md").read_text(encoding="utf-8")
        for domain in ("news.gallup.com", "realclearpolitics.com", "presidency.ucsb.edu"):
            self.assertIn(domain, provenance)
        index = (ROOT.parent / "ARTICLES.md").read_text(encoding="utf-8")
        self.assertIn("midterm-forecast-fit-vs-prediction/README.md", index)
        self.assertTrue((ROOT / "PRE_ELECTION_REGISTRATION.md").is_file())


if __name__ == "__main__":
    unittest.main()
