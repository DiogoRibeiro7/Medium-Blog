"""Public scientific reproducibility and source-provenance contracts."""

from __future__ import annotations

import json
import math
import unittest

from analysis import ROOT, compute_evidence
from figures import FIGURE_FILES
from prepare_medium_publication import TABLE_FIGURES, check_scientific_assets


class PublicationTests(unittest.TestCase):
    """Validate the frozen evidence independently of private Medium prose."""

    def test_numerical_results_match_canonical_computation(self) -> None:
        model: dict[str, object] = compute_evidence(draws=200).metrics
        committed: dict[str, object] = json.loads(
            (ROOT / "results" / "summary.json").read_text(encoding="utf-8")
        )
        stable_metrics: tuple[str, ...] = (
            "forecast",
            "in_sample_r2",
            "forward_rmse",
            "loocv_rmse",
            "forward_generic_only_rmse",
            "forward_seats_only_rmse",
            "forecast_generic_only",
            "alternative_forward_rmse",
            "alternative_forward_generic_only_rmse",
        )
        for name in stable_metrics:
            with self.subTest(metric=name):
                self.assertIn(name, model)
                self.assertIn(name, committed)
                self.assertTrue(
                    math.isclose(
                        float(model[name]),
                        float(committed[name]),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ),
                    f"{name}: regenerated={model[name]}, committed={committed[name]}",
                )

        registration: dict[str, object] = json.loads(
            (ROOT / "results" / "forecast_lock.json").read_text(encoding="utf-8")
        )
        locked: dict[str, object] = registration["primary_forecast"]
        for frozen_key, metric_key in (
            ("predicted_seat_change", "forecast"),
            ("in_sample_r_squared", "in_sample_r2"),
            ("forward_rmse_house_seats", "forward_rmse"),
        ):
            with self.subTest(locked=frozen_key):
                self.assertTrue(
                    math.isclose(
                        float(locked[frozen_key]),
                        float(model[metric_key]),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                )

    def test_all_scientific_publication_pngs_exist(self) -> None:
        self.assertEqual(len(FIGURE_FILES), 13)
        self.assertEqual(len(TABLE_FIGURES), 4)
        self.assertEqual(check_scientific_assets(ROOT), (17, 15))

    def test_public_editorial_notices_are_not_manuscripts(self) -> None:
        for manuscript in (
            ROOT / "article.md",
            ROOT / "publication" / "medium_ready.md",
        ):
            with self.subTest(notice=manuscript.name):
                notice: str = manuscript.read_text(encoding="utf-8")
                self.assertLess(len(notice.split()), 150)
                self.assertIn("private", notice)
                self.assertIn("research-articles-medium", notice)

    def test_sources_and_index_are_complete(self) -> None:
        provenance: str = (ROOT / "data" / "PROVENANCE.md").read_text(encoding="utf-8")
        for domain in ("news.gallup.com", "realclearpolitics.com", "presidency.ucsb.edu"):
            self.assertIn(domain, provenance)
        index: str = (ROOT.parent / "ARTICLES.md").read_text(encoding="utf-8")
        self.assertIn("midterm-forecast-fit-vs-prediction/README.md", index)
        self.assertTrue((ROOT / "PRE_ELECTION_REGISTRATION.md").is_file())


if __name__ == "__main__":
    unittest.main()
