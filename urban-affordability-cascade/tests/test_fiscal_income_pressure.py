"""Regression and boundary checks for non-household fiscal-income proxies."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from analysis import MUNICIPALITIES, load_panel
from extract_hicp import load_hicp
from fiscal_income_pressure import (
    AREAS_M2,
    FIGURES,
    RESULTS,
    YEARS,
    _pair,
    annual_table,
    endpoint_table,
    relative_proxy_growth_percent,
    run,
    summary_table,
    synthetic_rent_income_percent,
)


class FiscalIncomePressureTests(unittest.TestCase):
    """Check the identity, interpretation and reproducibility of the proxy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = load_panel()
        cls.names = sorted(MUNICIPALITIES)
        cls.annual = annual_table(cls.panel, cls.names)
        cls.endpoints = endpoint_table(cls.annual)

    def test_window_and_exact_grid(self) -> None:
        self.assertEqual(YEARS, (2020, 2021, 2022, 2023, 2024))
        self.assertEqual(AREAS_M2, (50, 70, 90))
        self.assertEqual(len(self.annual), 85)
        self.assertEqual({(r["municipality"], r["year"]) for r in self.annual},
                         {(name, year) for name in self.names for year in YEARS})

    def test_rent_income_arithmetic_porto(self) -> None:
        reference = synthetic_rent_income_percent(8.97, 13761, 70)
        ending = synthetic_rent_income_percent(13.21, 15523, 70)
        self.assertAlmostEqual(reference, 54.754742, places=5)
        self.assertAlmostEqual(ending, 71.483605, places=5)
        porto = next(row for row in self.endpoints if row["municipality"] == "Porto")
        self.assertAlmostEqual(float(porto["synthetic_70m2_rent_over_gross_fiscal_income_2020_pct"]),
                               reference, places=5)
        self.assertAlmostEqual(float(porto["synthetic_70m2_rent_over_gross_fiscal_income_2024_pct"]),
                               ending, places=5)

    def test_floor_area_scales_level_but_not_growth(self) -> None:
        for area in AREAS_M2:
            self.assertAlmostEqual(synthetic_rent_income_percent(10, 20000, area), 0.6 * area)
        baseline = synthetic_rent_income_percent(8.97, 13761, 50)
        final = synthetic_rent_income_percent(13.21, 15523, 50)
        direct_growth = relative_proxy_growth_percent(8.97, 13.21, 13761, 15523)
        self.assertAlmostEqual(100 * (final / baseline - 1), direct_growth)

    def test_common_hicp_deflation_cancels_exactly(self) -> None:
        hicp = load_hicp()
        deflated = relative_proxy_growth_percent(
            8.97 / hicp[2020], 13.21 / hicp[2024],
            13761 / hicp[2020], 15523 / hicp[2024],
        )
        self.assertAlmostEqual(deflated, relative_proxy_growth_percent(8.97, 13.21, 13761, 15523))

    def test_log_rent_income_growth_decomposition(self) -> None:
        rent_ratio = 13.21 / 8.97
        income_ratio = 15523 / 13761
        relative = relative_proxy_growth_percent(8.97, 13.21, 13761, 15523)
        self.assertAlmostEqual(math.log1p(relative / 100), math.log(rent_ratio) - math.log(income_ratio))

    def test_input_type_and_finiteness_guards(self) -> None:
        for invalid in (0, -1, float("nan"), math.inf, -math.inf, True, "70"):
            with self.subTest(value=invalid):
                for position in range(3):
                    args = [8.0, 15000.0, 70.0]
                    args[position] = invalid
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        synthetic_rent_income_percent(*args)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            relative_proxy_growth_percent(8, 10, 0, 16000)

    def test_reject_unsupported_year_and_partial_municipality_list(self) -> None:
        for unsupported_year in (2019, 2025):
            with self.subTest(year=unsupported_year):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    _pair(self.panel, "Porto", unsupported_year)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            annual_table(self.panel, self.names[:-1])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            annual_table(self.panel, self.names[:-1] + [self.names[0]])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            annual_table(self.panel, self.names + ["Braga"])

    def test_annual_table_rejects_missing_or_nonpositive_income(self) -> None:
        broken = dict(self.panel)
        broken["Porto", 2022] = replace(self.panel["Porto", 2022], income=None)
        with self.assertRaisesRegex(ValueError, "Missing"):
            annual_table(broken, self.names)
        broken["Porto", 2022] = replace(self.panel["Porto", 2022], rent_m2=0)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            annual_table(broken, self.names)

    def test_all_17_municipalities_worsened_in_matched_period(self) -> None:
        self.assertEqual(len(self.endpoints), 17)
        self.assertTrue(all(float(r["synthetic_proxy_relative_growth_pct"]) > 0
                            for r in self.endpoints))
        porto = next(row for row in self.endpoints if row["municipality"] == "Porto")
        self.assertAlmostEqual(float(porto["synthetic_proxy_relative_growth_pct"]), 30.55, delta=0.01)
        self.assertEqual(max(self.endpoints, key=lambda r: float(r["synthetic_proxy_relative_growth_pct"])), porto)

    def test_zero_growth_baseline_and_2024_unweighted_median(self) -> None:
        for row in self.annual:
            if row["year"] == 2020:
                self.assertEqual(float(row["synthetic_rent_income_proxy_growth_from_2020_pct"]), 0)
        summaries = summary_table(self.annual)
        self.assertEqual(len(summaries), 5)
        latest = summaries[-1]
        self.assertEqual(latest["municipalities_proxy_worse_than_2020"], 17)
        self.assertEqual(latest["municipalities_proxy_better_than_2020"], 0)
        self.assertAlmostEqual(float(latest["median_municipal_synthetic_proxy_relative_growth_pct"]),
                               15.168, delta=0.002)
        self.assertAlmostEqual(float(latest["median_municipal_new_rent_growth_from_2020_pct"]),
                               47.586, delta=0.002)
        self.assertAlmostEqual(float(latest["median_municipal_fiscal_income_growth_from_2020_pct"]),
                               29.897, delta=0.002)

    def test_median_of_ratios_is_not_quotient_of_medians(self) -> None:
        latest = summary_table(self.annual)[-1]
        rent_median = float(latest["median_municipal_new_rent_growth_from_2020_pct"])
        income_median = float(latest["median_municipal_fiscal_income_growth_from_2020_pct"])
        median_of_ratios = float(latest["median_municipal_synthetic_proxy_relative_growth_pct"])
        shortcut = 100 * ((1 + rent_median / 100) / (1 + income_median / 100) - 1)
        self.assertGreater(abs(median_of_ratios - shortcut), 0.2)

    def test_malformed_endpoint_grid_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            endpoint_table(self.annual[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            endpoint_table(self.annual + [self.annual[0]])

    def test_output_files_and_provenance_do_not_claim_tenant_burden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run(output)
            self.assertEqual(report["municipality_years"], 85)
            self.assertEqual(report["municipalities"], 17)
            self.assertIn("not_household_burden", str(report["evidence_class"]))
            self.assertIn("Ratio of two municipal medians", str(report["limits"]))
            self.assertIn("2025", str(report["limits"]))
            for figure in FIGURES:
                self.assertGreater((output / "figures" / figure).stat().st_size, 10000)
            for result in RESULTS:
                self.assertGreater((output / "results" / result).stat().st_size, 100)
            with (output / "results" / RESULTS[0]).open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 85)
            with (output / "results" / RESULTS[1]).open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 17)
            with (output / "results" / RESULTS[2]).open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)
            saved = json.loads((output / "results" / RESULTS[3]).read_text(encoding="utf-8"))
            self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()
