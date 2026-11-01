"""Official HICP source checks and nominal-vs-deflated housing diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import load_panel  # noqa: E402
from commuting_network import load_weights  # noqa: E402
from extract_hicp import (  # noqa: E402
    EXPECTED_YEARS, OUTPUT, PROVENANCE, SOURCE, SOURCE_SHA256,
    load_hicp, parse_hicp_json, run as run_extract,
)
from real_housing_pressure import (  # noqa: E402
    FIGURES, RESULTS, annual_table, endpoint_table, first_passage_table,
    nominal_and_deflated_growth, real_first_passage, run,
)


class EurostatHicpTests(unittest.TestCase):
    """Do not silently substitute CPI vintages, bases, flags or other places."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(SOURCE.read_text(encoding="utf-8"))

    def test_exact_official_source_and_reference(self) -> None:
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)
        values = load_hicp()
        self.assertEqual(set(values), set(EXPECTED_YEARS))
        self.assertEqual(values[2019], 83.07)
        self.assertEqual(values[2020], 82.97)
        self.assertEqual(values[2025], 100)
        self.assertEqual(self.document["updated"], "2026-09-17T11:00:00+0200")

    def test_wrong_dimension_codes_and_misaligned_years_rejected(self) -> None:
        corrupt = deepcopy(self.document)
        corrupt["dimension"]["geo"]["category"]["index"] = {"ES": 0}
        with self.assertRaisesRegex(ValueError, "geo"):
            parse_hicp_json(corrupt)
        corrupt = deepcopy(self.document)
        corrupt["dimension"]["time"]["category"]["index"]["2024"] = 4
        with self.assertRaisesRegex(ValueError, "time"):
            parse_hicp_json(corrupt)
        corrupt = deepcopy(self.document)
        corrupt["dimension"]["unit"]["category"]["index"] = {"RCH_A_AVG": 0}
        with self.assertRaisesRegex(ValueError, "unit"):
            parse_hicp_json(corrupt)

    def test_missing_flagged_or_invalid_observation_rejected(self) -> None:
        corrupt = deepcopy(self.document)
        del corrupt["value"]["4"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            parse_hicp_json(corrupt)
        corrupt = deepcopy(self.document)
        corrupt["status"] = {"3": "e"}
        with self.assertRaisesRegex(ValueError, "Flagged"):
            parse_hicp_json(corrupt)
        for invalid in (None, False, float("nan"), -1, 0):
            corrupt = deepcopy(self.document)
            corrupt["value"]["3"] = invalid
            with self.assertRaises(ValueError):
                parse_hicp_json(corrupt)

    def test_changed_json_source_checksum_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            different = Path(directory) / "changed.json"
            different.write_bytes(SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_hicp(different)

    def test_deterministic_derived_csv_and_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            record = run_extract(folder)
            self.assertEqual(record["source_response_sha256"], SOURCE_SHA256)
            self.assertEqual(record["dataset"], "prc_hicp_ainr")
            file = folder / "data" / OUTPUT
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),
                             record["derived_csv_sha256"])
            with file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["year"]) for row in rows], list(EXPECTED_YEARS))
            self.assertEqual(float(rows[-1]["portugal_hicp_annual_average_2025_100"]), 100)
            self.assertEqual(json.loads((folder / "data" / PROVENANCE).read_text()), record)


class InflationAdjustedHousingTests(unittest.TestCase):
    """Derived consumer-basket deflation, not affordability or moving flows."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names, cls.network = load_weights()
        cls.panel = load_panel()
        cls.hicp = load_hicp()

    def test_exact_ratio_identity_and_base_year_zero(self) -> None:
        nominal, real, inflation = nominal_and_deflated_growth(
            self.panel, self.hicp, municipality="Porto", field="price_m2",
            base_year=2019, target_year=2025,
        )
        self.assertAlmostEqual(nominal, 100 * (3347 / 1856 - 1))
        self.assertAlmostEqual(inflation, 100 * (100 / 83.07 - 1))
        self.assertAlmostEqual(real, 100 * ((1 + nominal / 100) / (1 + inflation / 100) - 1))
        self.assertEqual(nominal_and_deflated_growth(
            self.panel, self.hicp, municipality="Porto", field="rent_m2",
            base_year=2020, target_year=2020), (0, 0, 0))

    def test_no_2019_rent_imputation_or_invalid_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing"):
            nominal_and_deflated_growth(self.panel, self.hicp,
                                        municipality="Porto", field="rent_m2",
                                        base_year=2019, target_year=2025)
        changed = dict(self.hicp)
        changed[2025] = 0
        with self.assertRaisesRegex(ValueError, "consumer price index"):
            nominal_and_deflated_growth(self.panel, changed,
                                        municipality="Porto", field="price_m2",
                                        base_year=2019, target_year=2025)
        del changed[2025]
        with self.assertRaisesRegex(ValueError, "Missing exact HICP"):
            nominal_and_deflated_growth(self.panel, changed,
                                        municipality="Porto", field="price_m2",
                                        base_year=2019, target_year=2025)

    def test_deflated_rent_crossings_are_not_all_2023(self) -> None:
        rows = first_passage_table(self.panel, self.names, self.hicp)
        self.assertEqual(len(rows), 17)
        nominal = [r["new_contract_rent_plus_25_nominal_year"] for r in rows]
        adjusted = [r["new_contract_rent_plus_25_hicp_deflated_year"] for r in rows]
        self.assertEqual(set(nominal), {2023})
        self.assertEqual({y: adjusted.count(y) for y in (2023, 2024, 2025)},
                         {2023: 1, 2024: 10, 2025: 6})
        porto = next(r for r in rows if r["municipality"] == "Porto")
        self.assertEqual(porto["new_contract_rent_plus_25_hicp_deflated_year"], 2025)
        self.assertEqual(porto["sold_price_plus_50_hicp_deflated_year"], "")

    def test_real_first_passage_censoring_and_threshold_guards(self) -> None:
        self.assertIsNone(real_first_passage(self.panel, self.hicp,
                          municipality="Porto", field="price_m2", base_year=2019, threshold_pct=50))
        self.assertEqual(real_first_passage(self.panel, self.hicp,
                         municipality="Porto", field="rent_m2", base_year=2020, threshold_pct=25), 2025)
        for invalid in (-1, 0, float("nan"), math.inf):
            with self.assertRaisesRegex(ValueError, "positive"):
                real_first_passage(self.panel, self.hicp,
                                   municipality="Porto", field="rent_m2",
                                   base_year=2020, threshold_pct=invalid)

    def test_median_is_across_municipalities_not_pooled_transactions(self) -> None:
        years = annual_table(self.panel, self.names, self.hicp)
        self.assertEqual(len(years), 13)
        last_rent = next(row for row in years if row["outcome"] == "new_contract_rent"
                         and row["year"] == 2025)
        self.assertAlmostEqual(float(last_rent["median_municipality_nominal_growth_pct"]), 60.479042)
        self.assertAlmostEqual(float(last_rent["median_municipality_hicp_deflated_growth_pct"]), 33.149461)
        self.assertEqual(last_rent["municipalities"], 17)
        sales = next(row for row in years if row["outcome"] == "sold_price" and row["year"] == 2025)
        self.assertAlmostEqual(float(sales["median_municipality_hicp_deflated_growth_pct"]), 68.609649)
        self.assertEqual(len(endpoint_table(self.panel, self.names, self.hicp)), 34)

    def test_all_2025_real_50_rent_thresholds_except_one_censored(self) -> None:
        rows = first_passage_table(self.panel, self.names, self.hicp)
        self.assertEqual(sum(r["new_contract_rent_plus_50_hicp_deflated_year"] != "" for r in rows), 1)
        self.assertEqual(sum(r["sold_price_plus_100_hicp_deflated_year"] != "" for r in rows), 0)

    def test_reproducible_figures_tables_and_evidence_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report = run(folder)
            self.assertEqual(report["municipality_count"], 17)
            self.assertEqual(report["hicp_input_source_sha256"], SOURCE_SHA256)
            self.assertIn("not a municipal cost-of-living", str(report["limits"]))
            for filename in FIGURES:
                self.assertGreater((folder / "figures" / filename).stat().st_size, 5000)
            for filename in RESULTS:
                self.assertGreater((folder / "results" / filename).stat().st_size, 100)
            with (folder / "results" / RESULTS[1]).open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 17)
            saved = json.loads((folder / "results" / RESULTS[3]).read_text(encoding="utf-8"))
            self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()
