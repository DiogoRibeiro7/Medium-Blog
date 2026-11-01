"""Reproducibility, model, source integrity, and editorial contract tests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import struct
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    FIGURE_NAMES,
    RESULT_NAMES,
    composition_scenarios,
    endpoint_comparisons,
    first_recovered,
    month_distance,
    recovery_milestones,
    render_csv,
    result_outputs,
    wage_selection_mean,
)
from prepare_data import (  # noqa: E402
    BLS_IDS,
    MONTHLY_FIELDS,
    PANEL_FIELDS,
    YEARS,
    average_observed,
    build_panel,
    monthly_rows,
    number,
    observed,
    prepared_outputs,
    read_bls,
    read_fred,
    read_fred_sources,
    serialize,
)


class SourceIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
        cls.bls = read_bls()
        cls.fred = read_fred_sources()

    def test_01_seven_frozen_sources_are_listed(self) -> None:
        self.assertEqual(len(self.manifest["sources"]), 7)

    def test_02_source_paths_are_unique_and_local(self) -> None:
        entries = [item["file"] for item in self.manifest["sources"]]
        self.assertEqual(len(entries), len(set(entries)))
        self.assertTrue(all(path.startswith("data/raw/") for path in entries))
        self.assertTrue(all(".." not in Path(path).parts for path in entries))

    def test_03_source_checksums_match_exact_archived_bytes(self) -> None:
        for item in self.manifest["sources"]:
            with self.subTest(file=item["file"]):
                self.assertEqual(hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest(), item["sha256"])

    def test_04_seven_expected_bls_series(self) -> None:
        self.assertEqual(set(self.bls), set(BLS_IDS))

    def test_05_bls_series_have_96_calendar_slots(self) -> None:
        for name, values in self.bls.items():
            with self.subTest(series=name):
                self.assertEqual(len(values), 96)
                self.assertEqual(min(values), date(2018, 1, 1))
                self.assertEqual(max(values), date(2025, 12, 1))

    def test_06_missing_values_match_only_shutdown_month(self) -> None:
        for name, values in self.bls.items():
            with self.subTest(series=name):
                missing = [day for day, value in values.items() if value is None]
                self.assertEqual(missing, [] if name == "CES0500000003" else [date(2025, 10, 1)])

    def test_07_fred_expected_series_present(self) -> None:
        self.assertEqual(set(self.fred), {"PAYEMS", "GDPC1", "FEDFUNDS", "WALCL", "USSTHPI", "MEHOINUSA672N", "GINIALLRH"})

    def test_08_fred_endpoints_cover_2025(self) -> None:
        for name in self.fred:
            with self.subTest(series=name):
                self.assertTrue(any(day.year == 2025 for day in self.fred[name]))

    def test_09_raw_fred_reader_rejects_incorrect_column(self) -> None:
        with self.assertRaises(ValueError):
            read_fred(ROOT / "data" / "raw" / "fred_GDPC1.csv", "NOT_A_REAL_SERIES")

    def test_10_nonfinite_and_missing_values_are_rejected(self) -> None:
        for raw in (None, "", ".", "NA", "NaN", "-", float("inf"), float("-inf")):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    number(raw, "test value")

    def test_11_observed_rejects_missing_baseline(self) -> None:
        with self.assertRaises(ValueError):
            observed({date(2025, 10, 1): None}, date(2025, 10, 1), "missing CPI")


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.month = monthly_rows()
        cls.lookup = {str(row["observation_date"]): row for row in cls.month}
        cls.annual = {int(row["year"]): row for row in build_panel()}

    def test_12_monthly_panel_has_84_rows(self) -> None:
        self.assertEqual(len(self.month), 84)

    def test_13_monthly_dates_are_unique_sorted(self) -> None:
        days = [str(row["observation_date"]) for row in self.month]
        self.assertEqual(days, sorted(set(days)))

    def test_14_monthly_schema_fixed(self) -> None:
        self.assertEqual(set(self.month[0]), set(MONTHLY_FIELDS))
        self.assertEqual(len(MONTHLY_FIELDS), 11)

    def test_15_december_2019_price_and_wage_baselines_are_100(self) -> None:
        december = self.lookup["2019-12-01"]
        for name in ("cpi_sa_index_dec2019", "shelter_sa_index_dec2019", "nominal_earnings_index_dec2019", "real_earnings_index_dec2019"):
            self.assertAlmostEqual(float(december[name]), 100, places=10)

    def test_16_april_2020_unemployment_revised_vintage(self) -> None:
        april = self.lookup["2020-04-01"]
        self.assertAlmostEqual(float(april["total_unemployment_pct"]), 14.8)
        self.assertAlmostEqual(float(april["black_unemployment_pct"]), 16.9)
        self.assertAlmostEqual(float(april["white_unemployment_pct"]), 14.2)

    def test_17_june_2022_annual_inflation_rounds_to_9point1(self) -> None:
        self.assertAlmostEqual(float(self.lookup["2022-06-01"]["cpi_nsa_12m_inflation_pct"]), 9.1, delta=0.05)

    def test_18_december_2025_higher_price_level(self) -> None:
        self.assertAlmostEqual(float(self.lookup["2025-12-01"]["cpi_sa_index_dec2019"]), 126.060782, places=5)

    def test_19_december_2025_annual_inflation_below_three(self) -> None:
        rate = float(self.lookup["2025-12-01"]["cpi_nsa_12m_inflation_pct"])
        self.assertGreater(rate, 2.5)
        self.assertLess(rate, 3)

    def test_20_shutdown_month_keeps_only_nominal_wages(self) -> None:
        october = self.lookup["2025-10-01"]
        self.assertAlmostEqual(float(october["nominal_hourly_earnings_usd"]), 36.85)
        self.assertAlmostEqual(float(october["nominal_earnings_index_dec2019"]), 129.8449606, delta=0.00001)
        for name in ("total_unemployment_pct", "white_unemployment_pct", "black_unemployment_pct", "black_white_gap_pp", "cpi_sa_index_dec2019", "cpi_nsa_12m_inflation_pct", "shelter_sa_index_dec2019", "real_earnings_index_dec2019"):
            self.assertIsNone(october[name])

    def test_21_annual_panel_has_seven_years(self) -> None:
        self.assertEqual(set(self.annual), set(YEARS))

    def test_22_annual_schema_fixed(self) -> None:
        self.assertEqual(set(self.annual[2025]), set(PANEL_FIELDS))
        self.assertEqual(len(PANEL_FIELDS), 21)

    def test_23_annual_2020_unemployment_matches_bls(self) -> None:
        self.assertAlmostEqual(float(self.annual[2020]["unemployment_pct"]), 8.1, places=3)

    def test_24_2025_available_month_mean_uses_eleven_cps_values(self) -> None:
        values = [r["total_unemployment_pct"] for r in self.month if str(r["observation_date"]).startswith("2025")]
        self.assertEqual(sum(v is not None for v in values), 11)
        self.assertAlmostEqual(float(self.annual[2025]["unemployment_pct"]), average_observed(values, 2025, "rate"))

    def test_25_average_observed_requires_complete_prior_year(self) -> None:
        with self.assertRaises(ValueError):
            average_observed([3.1] * 11 + [None], 2024, "unemployment")

    def test_26_household_income_consistent_2025_dollars(self) -> None:
        self.assertEqual(float(self.annual[2019]["real_median_household_income_2025_usd"]), 85320)
        self.assertEqual(float(self.annual[2025]["real_median_household_income_2025_usd"]), 87460)

    def test_27_annual_household_gini_is_bounded(self) -> None:
        self.assertTrue(all(0.4 < float(r["household_income_gini"]) < 0.55 for r in self.annual.values()))

    def test_28_prepared_csv_snapshots_are_deterministic(self) -> None:
        for path, expected in prepared_outputs().items():
            with self.subTest(path=path):
                self.assertEqual(path.read_text(encoding="utf-8"), expected)

    def test_29_shutdown_gap_serializes_as_blank_not_zero(self) -> None:
        raw = serialize([self.lookup["2025-10-01"]], MONTHLY_FIELDS)
        record = list(csv.DictReader(io.StringIO(raw)))[0]
        self.assertEqual(record["total_unemployment_pct"], "")
        self.assertEqual(record["real_earnings_index_dec2019"], "")
        self.assertNotEqual(record["nominal_hourly_earnings_usd"], "")


class ResultsAndModelTests(unittest.TestCase):
    def test_30_milestone_count_and_names(self) -> None:
        rows = recovery_milestones()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["indicator"] for r in rows}, {"Real GDP", "Payroll employment", "Real median household income"})

    def test_31_recovery_clocks_are_distinct(self) -> None:
        events = {r["indicator"]: r["first_at_or_above_date"] for r in recovery_milestones()}
        self.assertEqual(events["Real GDP"], "2021-01-01")
        self.assertEqual(events["Payroll employment"], "2022-06-01")
        self.assertEqual(events["Real median household income"], "2025-01-01")

    def test_32_month_distance_exact(self) -> None:
        self.assertEqual(month_distance(date(2020, 2, 1), date(2022, 6, 1)), 28)

    def test_33_first_recovered_handles_absence(self) -> None:
        self.assertIsNone(first_recovered({date(2021, 1, 1): 99}, 100, date(2020, 1, 1)))

    def test_34_first_recovered_rejects_invalid_baseline(self) -> None:
        with self.assertRaises(ValueError):
            first_recovered({}, -1, date(2020, 1, 1))

    def test_35_endpoint_comparison_count(self) -> None:
        self.assertEqual(len(endpoint_comparisons()), 8)

    def test_36_endpoint_inflation_and_real_wages(self) -> None:
        lookup = {r["indicator"]: float(r["percent_change"]) for r in endpoint_comparisons()}
        self.assertAlmostEqual(lookup["CPI-U all-items"], 26.060782, places=4)
        self.assertAlmostEqual(lookup["Average hourly earnings, CPI-adjusted"], 3.477047, places=4)
        self.assertAlmostEqual(lookup["Real median household money income"], 2.508204, places=4)

    def test_37_model_baseline_pay(self) -> None:
        self.assertAlmostEqual(wage_selection_mean(15, 35, 0), 25)

    def test_38_model_half_low_wage_layoffs(self) -> None:
        self.assertAlmostEqual(wage_selection_mean(15, 35, 0.5), 85 / 3)

    def test_39_model_all_low_wage_layoffs(self) -> None:
        self.assertAlmostEqual(wage_selection_mean(15, 35, 1), 35)

    def test_40_model_monotone_selection(self) -> None:
        averages = [wage_selection_mean(15, 35, p / 20) for p in range(21)]
        self.assertTrue(all(right >= left for left, right in zip(averages, averages[1:])))

    def test_41_model_supports_different_initial_shares(self) -> None:
        self.assertAlmostEqual(wage_selection_mean(10, 40, 0, 0.25), 32.5)

    def test_42_model_rejects_invalid_parameters(self) -> None:
        inputs = ((15, 35, -0.1, 0.5), (15, 35, 1.1, 0.5), (35, 15, 0.5, 0.5), (15, 35, 0.5, 0), (15, 35, 0.5, 1), (15, 35, float("nan"), 0.5))
        for args in inputs:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    wage_selection_mean(*args)

    def test_43_composition_sensitivity_grid_has_eleven_rows(self) -> None:
        rows = composition_scenarios()
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[0]["remaining_employment_pct_initial"], 100)
        self.assertEqual(rows[-1]["remaining_employment_pct_initial"], 50)

    def test_44_model_individual_wage_growth_always_zero(self) -> None:
        self.assertTrue(all(r["individual_wage_growth_pct"] == 0 for r in composition_scenarios()))

    def test_45_results_match_committed_csv_snapshots(self) -> None:
        expected = result_outputs()
        self.assertEqual({path.name for path in expected}, set(RESULT_NAMES))
        for path, value in expected.items():
            with self.subTest(result=path.name):
                self.assertEqual(path.read_text(encoding="utf-8"), value)

    def test_46_render_csv_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            render_csv([])


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.article = (ROOT / "article.md").read_text(encoding="utf-8")

    def test_47_article_exceeds_eight_thousand_words(self) -> None:
        self.assertGreaterEqual(len(self.article.split()), 8000)

    def test_48_all_eight_figures_embedded_once(self) -> None:
        self.assertEqual(len(re.findall(r"!\[[^\]]+\]\(figures/[^\)]+\)", self.article)), 8)
        self.assertEqual(len(FIGURE_NAMES), len(set(FIGURE_NAMES)))
        for filename in FIGURE_NAMES:
            self.assertEqual(self.article.count(f"(figures/{filename})"), 1)

    def test_49_figures_are_nontrivial_pngs(self) -> None:
        for filename in FIGURE_NAMES:
            path = ROOT / "figures" / filename
            with self.subTest(figure=filename):
                data = path.read_bytes()
                self.assertGreater(len(data), 20000)
                self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
                width, height = struct.unpack(">II", data[16:24])
                self.assertGreaterEqual(width, 1200)
                self.assertGreaterEqual(height, 600)

    def test_50_article_numbers_primary_sources(self) -> None:
        for n in range(1, 35):
            self.assertRegex(self.article, rf"(?m)^{n}\.\s+\*\*")

    def test_51_article_labels_hypothetical_model(self) -> None:
        self.assertIn("HYPOTHETICAL", self.article.upper())
        self.assertIn("not a claim that exactly half", self.article.lower())

    def test_52_article_discusses_shutdown_and_revisions(self) -> None:
        self.assertIn("October 2025", self.article)
        self.assertIn("SPM", self.article)
        self.assertIn("2026", self.article)

    def test_53_notebook_is_structurally_valid_json(self) -> None:
        nb = json.loads((ROOT / "model_walkthrough.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(sum(cell["cell_type"] == "code" for cell in nb["cells"]), 5)

    def test_54_pinned_python_dependencies(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("numpy==2.3.5", requirements)
        self.assertIn("matplotlib==3.10.8", requirements)

    def test_55_readme_warns_about_evidence_limits(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("October 2025", readme)
        self.assertIn("HYPOTHETICAL", readme)
        self.assertIn("SHA-256", readme)


if __name__ == "__main__":
    unittest.main()
