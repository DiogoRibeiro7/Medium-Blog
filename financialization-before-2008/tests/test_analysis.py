"""Regression, source, measurement and publication checks for article #23."""

from __future__ import annotations

import ast
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from sys import path as sys_path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path.insert(0, str(ROOT))

import analysis  # noqa: E402
import prepare_data  # noqa: E402


class HistoricalSourceTests(unittest.TestCase):
    """Reject incomplete, mislabeled or silently modified historical series."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = analysis.load_panel()

    def test_panel_period_and_schema(self) -> None:
        self.assertEqual(len(self.panel), 18)
        self.assertEqual([int(r["year"]) for r in self.panel], list(range(1990, 2008)))
        self.assertEqual(set(self.panel[0]), set(prepare_data.PANEL_FIELDS))

    def test_source_snapshot_reproduces_checked_in_panel(self) -> None:
        expected = prepare_data.serialize_panel(prepare_data.build_panel())
        actual = (ROOT / "data" / "annual_panel_1990_2007.csv").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)

    def test_expected_historical_anchor_values(self) -> None:
        first, last = self.panel[0], self.panel[-1]
        self.assertAlmostEqual(first["manufacturing_employment_millions"], 17.694667, places=5)
        self.assertAlmostEqual(last["manufacturing_employment_millions"], 13.871417, places=5)
        self.assertAlmostEqual(first["union_membership_rate_pct"], 16.0)
        self.assertAlmostEqual(last["union_membership_rate_pct"], 12.1)
        self.assertAlmostEqual(first["household_income_gini"], 0.428)
        self.assertAlmostEqual(last["household_income_gini"], 0.463)

    def test_trade_balance_accounting_identity(self) -> None:
        for row in self.panel:
            self.assertAlmostEqual(
                row["exports_gdp_pct"] - row["imports_gdp_pct"],
                row["net_exports_gdp_pct"],
                delta=0.00001,
            )

    def test_debt_uses_q4_stocks_and_q4_gdp(self) -> None:
        stocks = prepare_data.read_rows(ROOT / "data" / "raw" / "quarterly,_end_of_period.csv")
        gdp = prepare_data.read_rows(ROOT / "data" / "raw" / "quarterly.csv")
        for row in self.panel:
            year = int(row["year"])
            amount = prepare_data.fourth_quarter(stocks, "CMDEBT", year)
            q4_gdp = prepare_data.fourth_quarter(gdp, "GDP", year)
            self.assertAlmostEqual(row["debt_to_gdp_pct"], 0.1 * amount / q4_gdp, places=5)

    def test_total_debt_exceeds_mortgage_subset(self) -> None:
        for row in self.panel:
            self.assertGreater(row["debt_to_gdp_pct"], row["mortgage_debt_to_gdp_pct"])
            self.assertTrue(0 < row["mortgage_share_of_debt_pct"] < 100)

    def test_finite_and_plausible_measurements(self) -> None:
        for row in self.panel:
            self.assertTrue(all(math.isfinite(v) for v in row.values()))
            self.assertTrue(0.0 < row["household_income_gini"] < 1.0)
            self.assertTrue(0.0 < row["manufacturing_payroll_share_pct"] < 50.0)
            self.assertTrue(0.0 < row["union_membership_rate_pct"] < 100.0)

    def test_missing_quarter_rejected_not_imputed(self) -> None:
        rows = prepare_data.read_rows(ROOT / "data" / "raw" / "quarterly.csv")
        restricted = [r for r in rows if r["observation_date"] != "1990-04-01"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            prepare_data.annual_mean(restricted, "GDP", 1990, "quarterly")

    def test_missing_values_are_errors(self) -> None:
        for invalid in ("", ".", "NaN", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare_data.numeric(invalid, label="test")

    def test_metadata_identifies_each_original_series(self) -> None:
        metadata = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
        used = {
            "MANEMP", "PAYEMS", "FEDFUNDS", "GS10", "GDP", "NETEXP",
            "IMPGS", "EXPGS", "A939RX0Q048SBEA", "USSTHPI", "CMDEBT",
            "HHMSDODNS", "MEHOINUSA646N", "MEHOINUSA672N", "GINIALLRH",
        }
        self.assertEqual(set(metadata["fred_series"]), used)
        self.assertEqual(metadata["snapshot_date_utc"], "2026-10-31")
        self.assertTrue(all(v["url"].startswith("https://fred.stlouisfed.org/series/") for v in metadata["fred_series"].values()))


class AnalysisModelTests(unittest.TestCase):
    """Check indexing, descriptive lags and explicitly hypothetical leverage."""

    def test_normalization_and_invalid_inputs(self) -> None:
        self.assertTrue(np.allclose(analysis.base_index([50.0, 75.0]), [100.0, 150.0]))
        for values in ([0.0, 1.0], [1.0, -2.0], [1.0, float("nan")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.base_index(values)

    def test_initial_ltv_thresholds(self) -> None:
        shock = np.array([0.1], dtype=float)
        self.assertAlmostEqual(float(analysis.price_shock_equity(0.95, shock)[0]), -0.05)
        self.assertAlmostEqual(float(analysis.price_shock_equity(0.80, shock)[0]), 0.10)
        self.assertAlmostEqual(float(analysis.percentage_equity_lost(0.80, shock)[0]), 50.0)
        self.assertAlmostEqual(float(analysis.price_shock_equity(0.60, np.array([0.4]))[0]), 0.0)

    def test_invalid_hypothetical_scenarios(self) -> None:
        for ltv in (-1.0, 0.0, 1.0, float("nan")):
            with self.subTest(ltv=ltv), self.assertRaises(ValueError):
                analysis.price_shock_equity(ltv, np.array([0.1]))
        for shock in (-0.1, 1.0, float("inf")):
            with self.subTest(shock=shock), self.assertRaises(ValueError):
                analysis.price_shock_equity(0.8, np.array([shock]))

    def test_lead_lag_correlation_no_mistaken_sample_growth(self) -> None:
        rows = analysis.lead_lag_correlations(analysis.load_panel())
        self.assertEqual([x["lag_years"] for x in rows], [-2, -1, 0, 1, 2])
        self.assertEqual([x["paired_years"] for x in rows], [15, 16, 17, 16, 15])
        self.assertTrue(all(-1 <= x["pearson_r_descriptive"] <= 1 for x in rows))

    def test_eight_figures_three_results_with_alternate_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            analysis.run(target)
            figures = sorted((target / "figures").glob("figure_*.png"))
            tables = sorted((target / "results").glob("*.csv"))
            self.assertEqual([p.name for p in figures], sorted(analysis.FIGURE_NAMES))
            self.assertEqual(len(tables), 3)
            self.assertTrue(all(p.stat().st_size > 15_000 for p in figures))
            self.assertTrue(all(p.stat().st_size > 100 for p in tables))


class PublicationTests(unittest.TestCase):
    """Keep the long-form manuscript and notebook connected to their sources."""

    def test_article_minimum_eight_thousand_words(self) -> None:
        text = (ROOT / "article.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(text.split()), 8_000)
        self.assertNotIn("TODO", text)

    def test_figures_referenced_and_hypothetical_labeled(self) -> None:
        text = (ROOT / "article.md").read_text(encoding="utf-8")
        for name in analysis.FIGURE_NAMES:
            self.assertIn(f"figures/{name}", text)
        self.assertIn("HYPOTHETICAL", text)
        self.assertIn("1993", text)

    def test_primary_source_links(self) -> None:
        text = (ROOT / "article.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("https://"), 18)
        for publisher in ("fred.stlouisfed.org", "www.bls.gov", "www.census.gov", "www.federalreserve.gov", "www.govinfo.gov", "www.nber.org"):
            self.assertIn(publisher, text)

    def test_notebook_code_cells_compile_without_embedded_outputs(self) -> None:
        notebook = json.loads((ROOT / "model_walkthrough.ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        for cell in code_cells:
            source = "".join(cell["source"])
            ast.parse(source)
            self.assertEqual(cell.get("outputs"), [])
            self.assertIsNone(cell.get("execution_count"))


if __name__ == "__main__":
    unittest.main()
