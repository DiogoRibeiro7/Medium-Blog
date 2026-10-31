"""Publication and quantitative contracts for the stagflation article."""

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
    annual_average_inflation,
    build_all,
    fiscal_data,
    gini_data,
    illustrative_real_wage,
    macro_data,
    racial_unemployment_gap,
    union_data,
)


class HistoricalSourceTests(unittest.TestCase):
    def test_macro_years_and_missing_black_rate(self) -> None:
        d = macro_data()
        self.assertEqual(d["year"].astype(int).tolist(), list(range(1969, 1991)))
        self.assertTrue(
            np.isnan(d["black_unemployment_annual_average_percent"][:3]).all()
        )
        self.assertTrue(
            np.isfinite(d["black_unemployment_annual_average_percent"][3:]).all()
        )
        self.assertAlmostEqual(d["cpi_u_annual_average_index"][0], 36.683333)
        self.assertAlmostEqual(d["unemployment_annual_average_percent"][13], 9.708333)
        self.assertAlmostEqual(d["fed_funds_annual_average_percent"][12], 16.378333)

    def test_inflation_derived_from_annual_cpi_levels(self) -> None:
        d = macro_data()
        inflation = annual_average_inflation(d["cpi_u_annual_average_index"])
        self.assertEqual(inflation.shape, (21,))
        self.assertAlmostEqual(inflation[0], 5.838, delta=.005)
        self.assertAlmostEqual(inflation[10], 13.549, delta=.005)
        self.assertAlmostEqual(inflation[13], 3.212, delta=.005)
        with self.assertRaises(ValueError):
            annual_average_inflation(np.array([1.0]))
        with self.assertRaises(ValueError):
            annual_average_inflation(np.array([1.0, 0.0]))

    def test_black_white_rates_and_gap_units(self) -> None:
        d = macro_data()
        b = d["black_unemployment_annual_average_percent"]
        w = d["white_unemployment_annual_average_percent"]
        gap = racial_unemployment_gap(b, w)
        self.assertTrue(np.isnan(gap[:3]).all())
        self.assertAlmostEqual(gap[14], 11.125)  # Year 1983, percentage points.
        self.assertAlmostEqual(gap[-1], 6.583333, places=5)
        with self.assertRaises(ValueError):
            racial_unemployment_gap(np.array([1.]), np.array([2., 3.]))
        with self.assertRaises(ValueError):
            racial_unemployment_gap(np.array([101.]), np.array([2.]))

    def test_census_household_gini(self) -> None:
        d = gini_data()
        self.assertEqual(d["year"].astype(int).tolist(), list(range(1967, 1991)))
        self.assertEqual(d["household_money_income_gini"][0], .397)
        self.assertEqual(d["household_money_income_gini"][13], .403)
        self.assertEqual(d["household_money_income_gini"][-1], .428)

    def test_union_data_are_comparable_only_from_1983(self) -> None:
        d = union_data()
        self.assertEqual(d["year"].astype(int).tolist(), list(range(1983, 1991)))
        self.assertEqual(d["union_membership_rate_percent"][0], 20.1)
        self.assertEqual(d["union_membership_rate_percent"][-1], 16.0)
        self.assertEqual(d["represented_by_unions_rate_percent"][0], 23.3)
        self.assertEqual(d["represented_by_unions_rate_percent"][-1], 18.2)
        share = 100 * d["union_members_thousands"] / d["employed_wage_salary_workers_thousands"]
        np.testing.assert_allclose(share, d["union_membership_rate_percent"], atol=.05)

    def test_fiscal_is_fiscal_not_calendar(self) -> None:
        d = fiscal_data()
        self.assertEqual(d["fiscal_year"].astype(int).tolist(), list(range(1970, 1991)))
        self.assertAlmostEqual(d["unified_surplus_deficit_percent_of_fiscal_gdp"][10],
                               -2.64444)
        self.assertAlmostEqual(d["unified_surplus_deficit_percent_of_fiscal_gdp"][13],
                               -5.87675)
        self.assertAlmostEqual(d["unified_surplus_deficit_percent_of_fiscal_gdp"][-1],
                               -3.74717)


class IllustrationTests(unittest.TestCase):
    def test_indexation_rule_exact_values(self) -> None:
        periods = np.arange(21)
        low = illustrative_real_wage(periods, inflation_indexation=0)
        partial = illustrative_real_wage(periods, inflation_indexation=.5)
        full = illustrative_real_wage(periods, inflation_indexation=1)
        self.assertEqual(low[0], 100)
        self.assertAlmostEqual(low[20], 100 * (1.03 / 1.08) ** 20)
        self.assertLess(low[-1], partial[-1])
        self.assertLess(partial[-1], full[-1])
        np.testing.assert_allclose(full, np.full(21, 100.0))

    def test_invalid_model_parameters_are_rejected(self) -> None:
        for periods in (np.array([]), np.array([-1.]), np.array([1.5]),
                        np.array([math.nan])):
            with self.subTest(periods=repr(periods)):
                with self.assertRaises(ValueError):
                    illustrative_real_wage(periods)
        with self.assertRaises(ValueError):
            illustrative_real_wage(np.array([0, 1]), inflation_indexation=1.1)
        with self.assertRaises(ValueError):
            illustrative_real_wage(np.array([0, 1]), annual_inflation=math.inf)
        with self.assertRaises(TypeError):
            illustrative_real_wage(np.array([0, 1]), baseline_nominal_growth=True)


class PublicationTests(unittest.TestCase):
    def test_article_has_all_figures_and_minimum_length(self) -> None:
        root = Path(__file__).resolve().parents[1]
        article = (root / "article.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(article.split()), 8000)
        for number, name in enumerate(FIGURE_NAMES, 1):
            self.assertEqual(article.count(f"](figures/{name})"), 1)
            self.assertIn(f"Figure {number} —", article)
        self.assertIn("## Sources and reproducibility notes", article)

    def test_notebook_reuses_main_code(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "model_walkthrough.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        source = "".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("from analysis import", source)
        self.assertIn("build_all(", source)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_reproduced_images_and_machine_readable_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = build_all(root)
            self.assertEqual(len(outputs), 8)
            for path in outputs:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10_000)
            expected = [
                ("source_based_summary.csv", 187, {"historical_descriptive", "missing"}),
                ("descriptive_derived_metrics.csv", 40, {"derived_descriptive"}),
                ("hypothetical_real_wage_paths.csv", 63, {"hypothetical"}),
            ]
            for filename, length, classes in expected:
                with (root / "results" / filename).open(newline="") as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual(len(records), length)
                self.assertEqual({r["evidence_class"] for r in records}, classes)
                if filename == "source_based_summary.csv":
                    self.assertEqual(sum(r["evidence_class"] == "missing"
                                         for r in records), 3)


if __name__ == "__main__":
    unittest.main()
