"""Tests for the rental-market analysis: helpers, snapshot integrity and the pinned results."""

import csv
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import rental as RT

HERE = Path(__file__).parent
DATA = HERE / "data"


class HelperTests(unittest.TestCase):
    def test_shares_and_band_median(self) -> None:
        self.assertEqual(RT.shares({"a": 1.0, "b": 3.0}, 4.0), {"a": 25.0, "b": 75.0})
        with self.assertRaises(ValueError):
            RT.shares({"a": 1.0}, 0.0)
        labels = {"1": "low", "2": "mid", "3": "high"}
        self.assertEqual(RT.band_median({"1": 10.0, "2": 5.0, "3": 1.0}, ["1", "2", "3"], labels), "low")
        self.assertEqual(RT.band_median({"1": 4.0, "2": 2.0, "3": 5.0}, ["1", "2", "3"], labels), "mid")
        with self.assertRaises(ValueError):
            RT.band_median({"1": 0.0}, ["1"], labels)

    def test_band_bounds_follow_the_published_labels(self) -> None:
        self.assertEqual(RT.band_lower_eur("Menos de 20 euros"), 0.0)
        self.assertEqual(RT.band_upper_eur("Menos de 20 euros"), 20)
        self.assertEqual(RT.band_upper_eur("De 75 a 99,99 euros"), 100)
        self.assertEqual(RT.band_lower_eur("De 100 a 149,99 euros"), 100.0)
        self.assertEqual(RT.band_upper_eur("De 100 a 149,99 euros"), 150)
        self.assertEqual(RT.band_lower_eur("1000 euros ou mais"), 1000.0)
        self.assertEqual(RT.band_upper_eur("1000 euros ou mais"), float("inf"))

    def test_round_series(self) -> None:
        self.assertEqual(RT.round_series({2021: 1.26, 2020: 3.0}), {"2020": 3.0, "2021": 1.3})


class SnapshotTests(unittest.TestCase):
    """Every Eurostat snapshot matches its retrieval record; an edited copy does not."""

    def test_records_match_and_tampering_is_caught(self) -> None:
        docs, records = RT.load_eurostat(DATA, RT.SNAPSHOT_DATE)
        self.assertEqual(set(docs), set(RT.EUROSTAT))
        for tag, record in records.items():
            with self.subTest(tag=tag):
                self.assertTrue(record["url"].startswith("https://ec.europa.eu/eurostat/api/"))
        tmp = Path(tempfile.mkdtemp())
        for tag in RT.EUROSTAT:
            for suffix in (".json", ".meta.json"):
                shutil.copy(DATA / f"eurostat_{tag}_{RT.SNAPSHOT_DATE}{suffix}", tmp)
        path = tmp / f"eurostat_ilc_lvho07c_PT_{RT.SNAPSHOT_DATE}.json"
        path.write_bytes(path.read_bytes().replace(b'"value"', b'"valuE"'))
        with self.assertRaises(ValueError):
            RT.load_eurostat(tmp, RT.SNAPSHOT_DATE)


class PinnedResultTests(unittest.TestCase):
    """Checks on the pinned INE and Eurostat data and the outputs built from them."""

    results: dict[str, Any]
    rows: list[dict[str, str]]

    @classmethod
    def setUpClass(cls) -> None:
        missing = [code for code in RT.INE if not (DATA / f"ine_{code}_{RT.SNAPSHOT_DATE}.jsonl.gz").exists()]
        if missing:
            raise AssertionError(f"pinned INE containers missing: {missing}")
        cls.results = json.loads((HERE / "results_rental.json").read_text(encoding="utf-8"))
        with (HERE / "rental_panel.csv").open(newline="", encoding="utf-8") as fh:
            cls.rows = list(csv.DictReader(fh))

    def test_windows_and_geography(self) -> None:
        nl = self.results["new_leases"]
        self.assertEqual(nl["window_rents"], [2020, 2025])
        self.assertEqual(nl["window_with_income"], [2020, 2024])
        self.assertEqual(nl["nuts3"]["regions"], 26)
        levels = {r["level"] for r in self.rows}
        self.assertEqual(levels, {"country", "NUTS2", "NUTS3", "municipality"})

    def test_panel_ratio_and_quartiles(self) -> None:
        for row in self.rows:
            if row["m2_per_month_of_income"]:
                expected = float(row["income_gross_per_household"]) / 12 / float(row["new_lease_rent_m2"])
                self.assertTrue(math.isclose(float(row["m2_per_month_of_income"]), expected, abs_tol=0.006))
            if row["new_lease_rent_m2_q1"] and row["new_lease_rent_m2"] and row["new_lease_rent_m2_q3"]:
                self.assertLessEqual(float(row["new_lease_rent_m2_q1"]), float(row["new_lease_rent_m2"]))
                self.assertLessEqual(float(row["new_lease_rent_m2"]), float(row["new_lease_rent_m2_q3"]))
            if row["level"] != "municipality":
                self.assertTrue(row["new_lease_rent_m2"])

    def test_overburden_by_tenure_reproduces_the_total(self) -> None:
        for year, cell in self.results["overburden"]["total_implied_from_tenure_shares"].items():
            with self.subTest(year=year):
                self.assertLess(abs(cell["published"] - cell["implied_from_tenure"]), 0.35)
        for window, parts in self.results["overburden"]["shift_share"].items():
            with self.subTest(window=window):
                self.assertAlmostEqual(parts["within_groups"] + parts["composition"], parts["change"], delta=0.02)

    def test_census_tenancies_are_internally_consistent(self) -> None:
        for code, entry in self.results["tenure"]["census2021_tenancies"].items():
            with self.subTest(region=code):
                self.assertAlmostEqual(sum(entry["contract_period_pct"].values()), 100, delta=0.3)
                support = entry["support_pct"]
                self.assertAlmostEqual(support["with_support"] + support["without_support"], 100, delta=0.2)

    def test_measures_are_kept_apart_and_documented(self) -> None:
        d = self.results["definitions"]
        self.assertIn("not the rent any household pays", d["m2_per_month_of_income"])
        self.assertIn("not a rent level", d["hicp_actual_rentals"])
        self.assertIn("No official statistics of advertised rents", d["asking_rents"])
        self.assertIn("never spliced", self.results["earlier_window_2017_2019"]["note"])


if __name__ == "__main__":
    unittest.main()
