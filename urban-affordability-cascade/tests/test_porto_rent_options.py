"""Boundary and provenance tests for Porto-relative new-contract rent options."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from analysis import load_panel
from extract_hicp import SOURCE_SHA256, load_hicp
from porto_rent_options import (
    AREAS_M2,
    BASE_YEAR,
    COMPARATORS,
    END_YEAR,
    FIGURES,
    RESULTS,
    YEARS,
    _hicp,
    _rent,
    annual_table,
    endpoint_table,
    rent_option,
    run,
    summary_table,
)


class PortoRentOptionsTests(unittest.TestCase):
    """Guard against conflating hypothetical savings with real relocation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = load_panel()
        cls.hicp = load_hicp()
        cls.annual = annual_table(cls.panel, cls.hicp)
        cls.endpoints = endpoint_table(cls.annual)
        cls.summary = summary_table(cls.annual)

    def test_exact_comparator_grid_and_no_2019_rents(self) -> None:
        self.assertEqual((BASE_YEAR, END_YEAR), (2020, 2025))
        self.assertEqual(YEARS, (2020, 2021, 2022, 2023, 2024, 2025))
        self.assertEqual(len(COMPARATORS), 16)
        self.assertNotIn("Porto", COMPARATORS)
        self.assertEqual(len(self.annual), 96)
        self.assertEqual(
            {(str(r["alternative_municipality"]), int(r["year"])) for r in self.annual},
            {(name, year) for name in COMPARATORS for year in YEARS},
        )

    def test_porto_matosinhos_reference_values(self) -> None:
        old = next(r for r in self.annual if r["alternative_municipality"] == "Matosinhos" and r["year"] == 2020)
        final = next(r for r in self.annual if r["alternative_municipality"] == "Matosinhos" and r["year"] == 2025)
        self.assertAlmostEqual(float(old["porto_new_rent_median_eur_m2_month"]), 8.97)
        self.assertAlmostEqual(float(old["alternative_new_rent_median_eur_m2_month"]), 7.66)
        self.assertAlmostEqual(float(old["synthetic_70m2_nominal_monthly_saving_eur"]), 91.7)
        self.assertAlmostEqual(float(final["porto_new_rent_median_eur_m2_month"]), 14.13)
        self.assertAlmostEqual(float(final["alternative_new_rent_median_eur_m2_month"]), 12.12)
        self.assertAlmostEqual(float(final["synthetic_70m2_nominal_monthly_saving_eur"]), 140.7)
        self.assertAlmostEqual(float(final["synthetic_70m2_real_2020_eur_monthly_saving"]), 116.73879, places=5)

    def test_floor_area_scales_euro_gap_not_relative_discount(self) -> None:
        baseline = rent_option(14.13, 12.12, 100, 82.97, area_m2=70)
        for area in AREAS_M2:
            discount, nominal, real = rent_option(14.13, 12.12, 100, 82.97, area_m2=area)
            self.assertAlmostEqual(discount, baseline[0])
            self.assertAlmostEqual(nominal, baseline[1] * area / 70)
            self.assertAlmostEqual(real, baseline[2] * area / 70)
        self.assertEqual(AREAS_M2, (50, 70, 90))

    def test_hicp_reference_scaling_and_discount_cancellation(self) -> None:
        original = rent_option(14.13, 12.12, 100, 82.97, area_m2=70)
        rescaled = rent_option(14.13 / 100, 12.12 / 100, 100, 82.97, area_m2=70)
        self.assertAlmostEqual(original[0], rescaled[0])
        self.assertAlmostEqual(original[2], original[1] * 82.97 / 100)

    def test_more_expensive_comparators_are_not_clipped(self) -> None:
        discount, nominal, real = rent_option(8, 10, 90, 80, area_m2=50)
        self.assertLess(discount, 0)
        self.assertAlmostEqual(nominal, -100)
        self.assertAlmostEqual(real, -100 * 80 / 90)
        zero = rent_option(8, 8, 90, 80, area_m2=50)
        self.assertEqual(zero, (0, 0, 0))

    def test_numeric_guards_include_bool_nan_infinite_and_zero(self) -> None:
        for bad in (0, -1, float("nan"), float("inf"), -math.inf, True, "1"):
            for index in range(5):
                values = [10.0, 8.0, 100.0, 83.0, 70.0]
                values[index] = bad
                with self.subTest(bad=bad, argument=index):
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        rent_option(*values[:4], area_m2=values[4])

    def test_reject_invalid_rent_and_hicp_years(self) -> None:
        for unsupported in (2019, 2026):
            with self.subTest(year=unsupported):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    _rent(self.panel, "Porto", unsupported)
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    _hicp(self.hicp, unsupported)

    def test_missing_or_invalid_national_hicp_fails_closed(self) -> None:
        missing = dict(self.hicp)
        del missing[2024]
        with self.assertRaisesRegex(ValueError, "Missing"):
            annual_table(self.panel, missing)
        invalid = dict(self.hicp)
        invalid[2025] = float("nan")
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            annual_table(self.panel, invalid)

    def test_missing_or_invalid_new_contract_rent_fails_closed(self) -> None:
        broken = dict(self.panel)
        broken["Matosinhos", 2024] = replace(self.panel["Matosinhos", 2024], rent_m2=None)
        with self.assertRaisesRegex(ValueError, "Missing"):
            annual_table(broken, self.hicp)
        broken["Matosinhos", 2024] = replace(self.panel["Matosinhos", 2024], rent_m2=0)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            annual_table(broken, self.hicp)

    def test_2025_all_16_cheaper_and_13_percent_discounts_narrowed(self) -> None:
        self.assertEqual(len(self.endpoints), 16)
        self.assertEqual(sum(float(r["relative_rent_discount_2025_pct"]) > 0 for r in self.endpoints), 16)
        self.assertEqual(sum(float(r["relative_discount_change_percentage_points"]) < 0
                             for r in self.endpoints), 13)
        self.assertEqual(sum(float(r["synthetic_70m2_real_saving_change_2020_to_2025_eur_month"]) > 0
                             for r in self.endpoints), 16)
        self.assertTrue(all(float(r["synthetic_70m2_real_saving_2025_in_2020_eur_month"]) > 0
                            for r in self.endpoints))

    def test_annual_discount_pattern_is_not_monotonic(self) -> None:
        medians = {int(r["year"]): float(r["median_relative_discount_pct"]) for r in self.summary}
        self.assertGreater(medians[2023], medians[2020])
        self.assertLess(medians[2025], medians[2020])
        self.assertEqual([r["alternative_municipalities_cheaper_than_porto"] for r in self.summary], [16] * 6)

    def test_exact_unweighted_medians_and_inflation_convention(self) -> None:
        old, final = self.summary[0], self.summary[-1]
        self.assertAlmostEqual(float(old["median_relative_discount_pct"]), 48.216276, places=5)
        self.assertAlmostEqual(float(final["median_relative_discount_pct"]), 46.602972, places=5)
        self.assertAlmostEqual(float(old["median_synthetic_70m2_real_2020_eur_monthly_saving"]), 302.75)
        self.assertAlmostEqual(float(final["median_synthetic_70m2_nominal_monthly_saving_eur"]), 460.95)
        self.assertAlmostEqual(float(final["median_synthetic_70m2_real_2020_eur_monthly_saving"]),
                               382.450215, places=5)
        self.assertEqual(final["alternative_municipalities_with_smaller_discount_than_2020"], 13)
        self.assertEqual(final["alternative_municipalities_with_higher_real_savings_than_2020"], 16)

    def test_duplicate_and_missing_rows_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicated"):
            endpoint_table(self.annual[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicated"):
            summary_table(self.annual + [self.annual[0]])

    def test_generated_outputs_and_machine_readable_limits(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            report = run(output)
            self.assertEqual(report["municipality_year_comparisons"], 96)
            self.assertEqual(report["hicp_source_response_sha256"], SOURCE_SHA256)
            self.assertIn("not matched apartments", str(report["limits"]))
            self.assertIn("no actual transport cost", str(report["limits"]))
            for name in FIGURES:
                self.assertGreater((output / "figures" / name).stat().st_size, 10000)
            for name in RESULTS:
                self.assertGreater((output / "results" / name).stat().st_size, 100)
            for name, count in zip(RESULTS[:3], (96, 16, 6)):
                with (output / "results" / name).open(newline="", encoding="utf-8") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            self.assertEqual(
                json.loads((output / "results" / RESULTS[-1]).read_text(encoding="utf-8")),
                report,
            )


if __name__ == "__main__":
    unittest.main()
