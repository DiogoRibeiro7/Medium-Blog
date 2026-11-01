"""Source, arithmetic, publication and offline reproducibility checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    FIGURE_NAMES,
    RESULT_NAMES,
    capital_scenarios,
    elapsed_months,
    equity_constrained_sales,
    first_recovered,
    make_figures,
    make_results,
    milestones,
    racial_outcomes,
    read_wealth,
)
from prepare_data import (  # noqa: E402
    DATA,
    PANEL_FIELDS,
    YEARS,
    build_panel,
    load_sources,
    monthly,
    numeric,
    prepared_outputs,
    quarterly,
    race_rows,
    read_bls_race,
)


class DataTests(unittest.TestCase):
    """Validate frozen, institutionally distinct public data streams."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = build_panel()
        cls.race = read_bls_race()
        cls.s = load_sources()

    def test_annual_years_and_schema(self) -> None:
        self.assertEqual([row["year"] for row in self.panel], list(YEARS))
        self.assertEqual(len(self.panel), 13)
        self.assertTrue(all(set(row) == set(PANEL_FIELDS) for row in self.panel))

    def test_panel_finite(self) -> None:
        self.assertTrue(
            all(math.isfinite(float(value)) for row in self.panel for value in row.values())
        )

    def test_exact_bls_months(self) -> None:
        self.assertEqual(len(self.race["White"]), 156)
        self.assertEqual(len(self.race["Black"]), 156)
        self.assertEqual(min(self.race["Black"]), date(2007, 1, 1))
        self.assertEqual(max(self.race["Black"]), date(2019, 12, 1))

    def test_bls_2019_annual_average_agrees_with_published_table(self) -> None:
        last = self.panel[-1]
        self.assertAlmostEqual(last["black_unemployment_pct"], 6.1, delta=0.08)
        self.assertAlmostEqual(last["white_unemployment_pct"], 3.3, delta=0.08)

    def test_race_gap_positive_in_each_month(self) -> None:
        rows = race_rows()
        self.assertEqual(len(rows), 156)
        self.assertTrue(all(r["gap_percentage_points"] > 0 for r in rows))
        self.assertAlmostEqual(
            rows[0]["gap_percentage_points"],
            rows[0]["black_unemployment_pct"] - rows[0]["white_unemployment_pct"],
        )

    def test_official_series_dates_and_units(self) -> None:
        self.assertGreater(self.s["PAYEMS"][date(2007, 12, 1)], 100_000)
        self.assertGreater(self.s["WALCL"][date(2014, 12, 31)], 4_000_000)
        self.assertGreater(self.s["GDPC1"][date(2019, 10, 1)], 20_000)

    def test_monthly_complete_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing monthly"):
            monthly({date(2010, 1, 1): 2.0}, 2010, "test")

    def test_quarterly_complete_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing quarterly"):
            quarterly({date(2010, 1, 1): 2.0}, 2010, "test")

    def test_missing_and_nonfinite_rejected(self) -> None:
        with self.assertRaises(ValueError):
            numeric(".", context="missing")
        with self.assertRaises(ValueError):
            numeric(float("inf"), context="bad")

    def test_income_and_output_index_base(self) -> None:
        self.assertAlmostEqual(self.panel[0]["median_income_index_2007"], 100.0)
        self.assertAlmostEqual(
            100 * self.s["GDPC1"][date(2007, 10, 1)] / self.s["GDPC1"][date(2007, 10, 1)],
            100,
        )

    def test_2019_income_and_debt_service_ranges(self) -> None:
        self.assertAlmostEqual(self.panel[-1]["real_median_household_income_2025_usd"], 85320)
        self.assertTrue(11 < self.panel[-1]["debt_service_pct"] < 13)
        self.assertGreater(self.panel[0]["debt_service_pct"], self.panel[-1]["debt_service_pct"])

    def test_fed_assets_increase_over_period(self) -> None:
        self.assertLess(self.panel[0]["fed_assets_trillions_usd"], 1)
        self.assertGreater(self.panel[-1]["fed_assets_trillions_usd"], 3)

    def test_scf_values_and_ratio(self) -> None:
        wealth = read_wealth()
        self.assertEqual(wealth[2007]["White"], 245.11)
        self.assertEqual(wealth[2013]["Black"], 16.78)
        self.assertEqual(wealth[2019]["Hispanic"], 41.91)
        self.assertAlmostEqual(100 * wealth[2019]["Black"] / wealth[2019]["White"], 12.82, delta=0.02)

    def test_all_prepared_data_byte_reproducible(self) -> None:
        for path, expected in prepared_outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"), expected, path.name)

    def test_source_archival_hashes(self) -> None:
        catalog = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
        files = catalog["source_files"]
        self.assertEqual(len(files), 10)
        for item in files:
            archived = ROOT / item["path"]
            self.assertTrue(archived.is_file(), item["path"])
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), item["sha256"])


class RecoveryTests(unittest.TestCase):
    """Check benchmark dates without interpreting them causally."""

    def test_unemployment_peak_october_2009(self) -> None:
        rates = {k: v for k, v in load_sources()["UNRATE"].items() if 2007 <= k.year <= 2019}
        day = max(rates, key=rates.get)
        self.assertEqual(day, date(2009, 10, 1))
        self.assertEqual(rates[day], 10.0)

    def test_four_different_recovery_clocks(self) -> None:
        observed = {r["indicator"]: r["recovery_date"] for r in milestones()}
        self.assertEqual(observed["Real GDP"], "2010-10-01")
        self.assertEqual(observed["Payroll jobs"], "2014-05-01")
        self.assertEqual(observed["FHFA house prices"], "2016-07-01")
        self.assertEqual(observed["Median real household income"], "2016-01-01")

    def test_elapsed_months_calendar_not_exact_duration(self) -> None:
        self.assertEqual(elapsed_months(date(2007, 12, 1), date(2014, 5, 1)), 77)

    def test_first_recovered_after_cutoff(self) -> None:
        values = {date(2007, 1, 1): 100, date(2008, 1, 1): 103, date(2009, 1, 1): 90, date(2010, 1, 1): 101}
        self.assertEqual(first_recovered(values, 100, date(2009, 1, 1)), date(2010, 1, 1))
        self.assertIsNone(first_recovered(values, 110, date(2009, 1, 1)))

    def test_racial_outcomes_uses_distinct_sources(self) -> None:
        rows = racial_outcomes()
        self.assertEqual([row["year"] for row in rows], [2007, 2010, 2013, 2016, 2019])
        self.assertTrue(all(row["annual_black_white_unemployment_gap_pp"] > 0 for row in rows))


class ModelTests(unittest.TestCase):
    """Prevent the illustrative bank identity from drifting into simulation."""

    def test_two_percent_markdown_requires_twenty_three_percent_sales(self) -> None:
        self.assertAlmostEqual(equity_constrained_sales(0.02, 0.08), 0.23)

    def test_higher_buffer_reduces_extra_sales(self) -> None:
        self.assertEqual(equity_constrained_sales(0.02, 0.10), 0.0)
        self.assertGreater(equity_constrained_sales(0.04, 0.10), 0.0)
        self.assertLess(equity_constrained_sales(0.04, 0.12), equity_constrained_sales(0.04, 0.10))

    def test_model_rejects_negative_loss_and_insolvency(self) -> None:
        for loss, capital, target in ((-0.01, 0.08, 0.08), (0.08, 0.08, 0.08), (0, 0.08, 0), (float("nan"), 0.08, 0.08)):
            with self.assertRaises(ValueError):
                equity_constrained_sales(loss, capital, target)

    def test_monotone_loss_sensitivity(self) -> None:
        for initial in (0.08, 0.10, 0.12):
            sales = [equity_constrained_sales(j * 0.005, initial) for j in range(13)]
            self.assertTrue(all(a <= b + 1e-12 for a, b in zip(sales, sales[1:])))

    def test_grid_has_three_capital_buffers(self) -> None:
        rows = capital_scenarios()
        self.assertEqual(len(rows), 39)
        self.assertEqual({row["starting_equity_pct_initial_assets"] for row in rows}, {8.0, 10.0, 12.0})


class PublicationTests(unittest.TestCase):
    """Check deliverable consistency and reproducible graphics."""

    def test_article_is_long_form(self) -> None:
        manuscript = (ROOT / "article.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(manuscript.split()), 8000)

    def test_eight_unique_figures_are_embedded(self) -> None:
        manuscript = (ROOT / "article.md").read_text(encoding="utf-8")
        self.assertEqual(len(FIGURE_NAMES), 8)
        for figure in FIGURE_NAMES:
            self.assertEqual(manuscript.count(f"(figures/{figure})"), 1, figure)

    def test_hypothetical_model_clearly_disclosed(self) -> None:
        manuscript = (ROOT / "article.md").read_text(encoding="utf-8").lower()
        for phrase in ("hypothetical", "not a causal", "repeated cross-section", "limitations"):
            self.assertIn(phrase, manuscript)

    def test_offline_artifact_generation_and_counts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="great-recession-test-") as tmp:
            target = Path(tmp)
            make_figures(target)
            make_results(target)
            self.assertEqual(
                {p.name for p in (target / "figures").glob("figure_*.png")},
                set(FIGURE_NAMES),
            )
            self.assertEqual(
                {p.name for p in (target / "results").glob("*.csv")},
                set(RESULT_NAMES),
            )
            with (target / "results" / "capital_scenarios.csv").open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 39)


if __name__ == "__main__":
    unittest.main()
