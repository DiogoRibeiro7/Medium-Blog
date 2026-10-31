"""Regression and evidence-classification tests for article #20."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    FIGURE_NAMES,
    age_poverty,
    benefit_schedule,
    build_all,
    historical_events,
    medicare_enrollment,
    national_poverty,
    poverty_classification,
    poverty_gap,
    racial_poverty,
    realized_voter_access,
)


class HistoricalEvidenceTests(unittest.TestCase):
    def test_national_series_exact_and_continuous(self) -> None:
        d = national_poverty()
        self.assertEqual(d["year"].astype(int).tolist(), list(range(1959, 1976)))
        np.testing.assert_allclose(
            d["poverty_rate_percent"],
            [22.4, 22.2, 21.9, 21, 19.5, 19, 17.3, 14.7, 14.2,
             12.8, 12.1, 12.6, 12.5, 11.9, 11.1, 11.2, 12.3],
        )

    def test_race_gap_preserves_source_missing_years(self) -> None:
        d = racial_poverty()
        self.assertEqual(d["white_poverty_rate_percent"][0], 18.1)
        self.assertEqual(d["black_poverty_rate_percent"][0], 55.1)
        self.assertTrue(np.isnan(d["black_poverty_rate_percent"][1:7]).all())
        self.assertEqual(d["black_poverty_rate_percent"][7], 41.8)
        gap = poverty_gap(d["white_poverty_rate_percent"], d["black_poverty_rate_percent"])
        self.assertAlmostEqual(gap[0], 37.0)
        self.assertTrue(np.isnan(gap[1:7]).all())
        self.assertAlmostEqual(gap[-1], 21.6)

    def test_age_table_missing_not_zero(self) -> None:
        d = age_poverty()
        self.assertEqual(d["under18_poverty_rate_percent"][0], 27.3)
        self.assertTrue(np.isnan(d["age65plus_poverty_rate_percent"][1:7]).all())
        self.assertTrue(np.isnan(d["age18_to64_poverty_rate_percent"][1:7]).all())
        self.assertEqual(d["age65plus_poverty_rate_percent"][7], 28.5)
        self.assertEqual(d["age65plus_poverty_rate_percent"][-1], 15.3)

    def test_medicare_counts_add_up_and_disability_starts_1973(self) -> None:
        d = medicare_enrollment()
        self.assertEqual(d["year"].astype(int).tolist(), list(range(1966, 1976)))
        self.assertTrue(np.isnan(d["disabled_hospital_insurance_enrollees"][:7]).all())
        self.assertEqual(d["aged_hospital_insurance_enrollees"][0], 19082454)
        self.assertEqual(d["disabled_hospital_insurance_enrollees"][7], 1730538)
        np.testing.assert_allclose(
            d["total_hospital_insurance_enrollees"][7:],
            d["aged_hospital_insurance_enrollees"][7:]
            + d["disabled_hospital_insurance_enrollees"][7:],
        )
        np.testing.assert_allclose(
            d["total_hospital_insurance_enrollees"][:7],
            d["aged_hospital_insurance_enrollees"][:7],
        )

    def test_events_are_ordered_and_sourced(self) -> None:
        events = historical_events()
        self.assertEqual(len(events), 10)
        self.assertEqual(events[0][0].isoformat(), "1963-04-16")
        self.assertEqual(events[-1][0].isoformat(), "1972-03-24")
        self.assertTrue(all(url.startswith("https://") for _, _, url in events))


class HypotheticalMechanismTests(unittest.TestCase):
    def test_registration_access_formula(self) -> None:
        self.assertEqual(realized_voter_access(100000, .85, 0), 85000)
        self.assertEqual(realized_voter_access(100000, .85, 1), 0)
        self.assertAlmostEqual(realized_voter_access(100000, .85, .4), 51000)
        with self.assertRaises(TypeError):
            realized_voter_access(True, .8, .2)
        with self.assertRaises(ValueError):
            realized_voter_access(100000, .8, 1.1)
        with self.assertRaises(ValueError):
            realized_voter_access(100000, math.nan, .3)

    def test_cliff_is_discontinuous_and_taper_continuous(self) -> None:
        earnings = np.array([0, 18000, 18001, 19000, 60000], dtype=float)
        np.testing.assert_allclose(
            benefit_schedule(earnings, cliff=True), [9000, 9000, 0, 0, 0],
        )
        np.testing.assert_allclose(
            benefit_schedule(earnings, cliff=False), [9000, 9000, 8999.65, 8650, 0],
        )
        self.assertGreater((18001 + benefit_schedule(np.array([18001.]))[0]),
                           (18000 + benefit_schedule(np.array([18000.]))[0]))
        self.assertLess((18001 + benefit_schedule(np.array([18001.]), cliff=True)[0]),
                        (18000 + benefit_schedule(np.array([18000.]), cliff=True)[0]))

    def test_invalid_benefit_inputs(self) -> None:
        with self.assertRaises(ValueError):
            benefit_schedule(np.array([-2]))
        with self.assertRaises(ValueError):
            benefit_schedule(np.array([math.nan]))
        with self.assertRaises(ValueError):
            benefit_schedule(np.array([100]), withdrawal_rate=1.1)
        with self.assertRaises(ValueError):
            benefit_schedule(np.array([100]), maximum_benefit=-1)

    def test_noncash_not_in_cash_classification(self) -> None:
        values = np.array([0., 6999., 7000., 12000.])
        cash, augmented = poverty_classification(18000, values, 25000)
        self.assertEqual(cash.tolist(), [True] * 4)
        self.assertEqual(augmented.tolist(), [True, True, False, False])
        with self.assertRaises(ValueError):
            poverty_classification(-1, values, 25000)
        with self.assertRaises(ValueError):
            poverty_classification(1, np.array([-1]), 25000)


class PublicationContractTests(unittest.TestCase):
    def test_medium_article_long_form_and_figure_links(self) -> None:
        root = Path(__file__).resolve().parents[1]
        article = (root / "article.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(article.split()), 8000)
        for number, name in enumerate(FIGURE_NAMES, 1):
            self.assertEqual(article.count(f"](figures/{name})"), 1)
            self.assertIn(f"Figure {number} —", article)
        self.assertIn("## Sources and methodological notes", article)

    def test_notebook_reuses_single_calculation_module(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "model_walkthrough.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code = "".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("from analysis import", code)
        self.assertIn("build_all(", code)
        identifiers = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_eight_images_and_typed_result_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = build_all(root)
            self.assertEqual(len(outputs), 8)
            for path in outputs:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10000)
            with (root / "results" / "source_based_summary.csv").open(newline="") as f:
                historical = list(csv.DictReader(f))
            self.assertEqual(len(historical), 17 + 17 * 2 + 17 * 3 + 10 * 3)
            self.assertEqual(sum(r["evidence_class"] == "missing" for r in historical), 25)
            self.assertFalse(any(r["evidence_class"] == "hypothetical" for r in historical))
            with (root / "results" / "hypothetical_voter_access.csv").open(newline="") as f:
                model = list(csv.DictReader(f))
            self.assertEqual(len(model), 303)
            self.assertTrue(all(r["evidence_class"] == "hypothetical" for r in model))


if __name__ == "__main__":
    unittest.main()
