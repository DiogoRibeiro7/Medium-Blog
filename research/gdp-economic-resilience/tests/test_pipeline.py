"""Offline tests of the accounting, missingness and source integrity rules."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import pipeline as p


def encode(value: object) -> bytes:
    """Build a synthetic World Bank response."""
    return json.dumps(value).encode("utf-8")


class TestAccounting(unittest.TestCase):
    def test_complete_aggregate(self) -> None:
        self.assertEqual(p.complete_sum([1.0, 2.0]), 3.0)
        self.assertIsNone(p.complete_sum([1.0, None]))
        self.assertIsNone(p.complete_sum([]))

    def test_ratio_and_growth(self) -> None:
        self.assertEqual(p.ratio(100.0, 20.0), 5.0)
        self.assertIsNone(p.ratio(1.0, 0.0))
        self.assertIsNone(p.ratio(None, 1.0))
        self.assertAlmostEqual(p.growth(100.0, 90.0), -10.0)
        self.assertIsNone(p.growth(-1.0, 10.0))

    def test_concentration_keeps_original_denominator(self) -> None:
        total, hhi, effective = p.concentration([30.0, 30.0, 30.0])
        self.assertEqual(total, 90.0)
        self.assertAlmostEqual(hhi, 1 / 3)
        self.assertAlmostEqual(effective, 3.0)
        self.assertEqual(p.concentration([40.0, None, 40.0]), (None, None, None))
        self.assertEqual(p.concentration([20.0, -1.0, 60.0]), (79.0, None, None))

    def test_manufacturing_is_not_double_counted(self) -> None:
        data = {(code, "PRT", 2020): value for code, value in (
            (p.AGR, 10.0), (p.IND, 30.0), (p.MAN, 20.0), (p.SRV, 50.0)
        )}
        row = next(r for r in p.sector_rows(p.Snapshot(data))
                   if r["country"] == "PRT" and r["year"] == 2020)
        self.assertEqual(row["nonmanufacturing_industry_pct_gdp"], 10.0)
        self.assertEqual(row["sector_sum_pct_gdp"], 90.0)
        self.assertEqual(row["unallocated_pct_gdp"], 10.0)
        self.assertAlmostEqual(row["hhi_four_sectors"], (100 + 100 + 400 + 2500) / 8100)

    def test_missing_manufacturing_does_not_erase_three_sector_result(self) -> None:
        data = {(p.AGR, "PRT", 2020): 10.0, (p.IND, "PRT", 2020): 20.0,
                (p.SRV, "PRT", 2020): 60.0}
        row = next(r for r in p.sector_rows(p.Snapshot(data))
                   if r["country"] == "PRT" and r["year"] == 2020)
        self.assertIsNotNone(row["hhi_three_broad_sectors"])
        self.assertIsNone(row["hhi_four_sectors"])

    def test_fixed_eu_membership_and_no_double_counting(self) -> None:
        data = {(code, country, 2025): 1.0 for code in (p.GDP, p.PPP, p.POP)
                for country in p.COUNTRIES}
        rows = p.gdp_rankings(p.Snapshot(data))
        group = [r for r in rows if r["group"] == "fixed_eu27_comparators"]
        self.assertEqual(len(group), 7)
        self.assertEqual(sum(r["members"] for r in group), 33)
        eu = next(r for r in group if r["entity"] == "EU27")
        self.assertEqual(eu["nominal_usd"], 27.0)
        self.assertEqual(eu["nominal_usd_rank"], 1)
        del data[p.GDP, "PRT", 2025]
        eu = next(r for r in p.gdp_rankings(p.Snapshot(data)) if r["entity"] == "EU27")
        self.assertIsNone(eu["nominal_usd"])
        self.assertIsNone(eu["nominal_usd_rank"])

    def test_matched_endpoint_growth(self) -> None:
        wdi = p.Snapshot({(p.REAL, "PRT", 1996): 100.0, (p.REAL, "PRT", 2020): 240.0,
                          (p.POP, "PRT", 1996): 10.0, (p.POP, "PRT", 2020): 20.0})
        wealth = p.Snapshot({(p.WEALTH, "PRT", 1996): 20.0, (p.WEALTH, "PRT", 2020): 10.0})
        rows = p.wealth_rows(wdi, wealth)
        row = next(r for r in rows if r["country"] == "PRT" and r["start"] == 1996 and r["end"] == 2020)
        self.assertAlmostEqual(row["real_gdp_pc_growth_pct"], 20.0)
        self.assertAlmostEqual(row["real_wealth_pc_growth_pct"], -50.0)
        self.assertTrue(row["gdp_up_wealth_down"])
        self.assertAlmostEqual(row["growth_gap_percentage_points"], 70.0)
        excluded = next(r for r in rows if r["country"] == "CYP")
        self.assertFalse(excluded["comparable"])
        self.assertIsNone(excluded["gdp_up_wealth_down"])


class TestSourceIntegrity(unittest.TestCase):
    def fixture(self) -> dict[str, bytes]:
        """One series, two country-years represented and two absent."""
        manifest = {"source_id": 2, "start": 2020, "end": 2021,
                    "countries": ["PRT", "USA"],
                    "coverage": {p.GDP: {"records": 2, "observed": 1,
                                         "explicit_nulls": 1, "absent_cells": 2}}}
        rows = [{"countryiso3code": "PRT", "date": str(year),
                 "indicator": {"id": p.GDP}, "value": value}
                for year, value in ((2020, 100.0), (2021, None))]
        header = {"sourceid": "2", "page": 1, "pages": 1, "total": 2}
        return {"raw/acquisition-manifest.json": encode(manifest),
                f"raw/{p.GDP}.metadata.json": encode([{}, [{"id": p.GDP, "source": {"id": "2"}}]]),
                f"raw/{p.GDP}.page-1.json": encode([header, rows])}

    def decode(self, files: dict[str, bytes]) -> p.Snapshot:
        """Isolate the parser from the real 33-country source contract."""
        with patch.object(p, "COUNTRIES", ("PRT", "USA")), patch.dict(
            p.SOURCES, {p.WDI: ("2", 2020, 2021, (p.GDP,))}
        ):
            return p.decode_rows(files, p.WDI)

    def test_byte_integrity(self) -> None:
        raw = b"[]"
        item = {"path": "raw/test.json", "bytes": len(raw), "sha256": p.checksum(raw)}
        p.verify_bytes(raw, item)
        with self.assertRaises(ValueError):
            p.verify_bytes(b"{}", item)

    def test_explicit_null_and_absent(self) -> None:
        snapshot = self.decode(self.fixture())
        self.assertEqual(snapshot.value(p.GDP, "PRT", 2020), 100.0)
        self.assertIn((p.GDP, "PRT", 2021), snapshot.data)
        self.assertNotIn((p.GDP, "USA", 2021), snapshot.data)

    def test_duplicate_country_year(self) -> None:
        files = self.fixture()
        name = f"raw/{p.GDP}.page-1.json"
        header, rows = json.loads(files[name])
        rows[1] = rows[0]
        files[name] = encode([header, rows])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.decode(files)

    def test_wrong_source(self) -> None:
        files = self.fixture()
        name = f"raw/{p.GDP}.page-1.json"
        header, rows = json.loads(files[name])
        header["sourceid"] = "59"
        files[name] = encode([header, rows])
        with self.assertRaisesRegex(ValueError, "page metadata"):
            self.decode(files)

    def test_coverage_mismatch(self) -> None:
        files = self.fixture()
        name = f"raw/{p.GDP}.page-1.json"
        header, rows = json.loads(files[name])
        rows[1]["value"] = 50
        files[name] = encode([header, rows])
        with self.assertRaisesRegex(ValueError, "counts differ"):
            self.decode(files)


if __name__ == "__main__":
    unittest.main()
