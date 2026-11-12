"""Contracts for fetch_forbes_history.py, fetch_wdi.py and part3_within_country.py."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_forbes_history as ffh  # noqa: E402
import fetch_wdi  # noqa: E402
import part3_within_country as p3  # noqa: E402


def _people(n: int, prefix: str, year: int) -> list[dict]:
    return [{"uri": f"{prefix}{i}", "countryOfCitizenship": "China" if i % 2 else "India", "country": "India",
             "finalWorth": 1000.0 + i, "year": year} for i in range(n)]


class ForbesHistoryTests(unittest.TestCase):
    def test_aggregates_without_records_and_detects_overlap(self) -> None:
        def fake(url: str, **kw: object) -> bytes:
            year = int(re.search(r"/billionaires/(\d{4})/", url).group(1))
            if year == 2011:
                raise RuntimeError("timeout")
            people = _people(6, "a", year) if year == 2013 else _people(4, "b" if year == 2010 else "a", year)
            return json.dumps({"personList": {"personsLists": people}}).encode()

        with tempfile.TemporaryDirectory() as tmp:
            m = ffh.fetch(2010, 2013, Path(tmp), fetcher=fake, pause=0)
            counts = pd.read_csv(Path(tmp) / "forbes_history_counts.csv.gz")
            self.assertEqual(m["latest_year"], 2013)
            self.assertIn("error", m["years"]["2011"])
            self.assertEqual(m["years"]["2012"]["overlap_with_latest"], 1.0)
            self.assertEqual(m["years"]["2010"]["overlap_with_latest"], 0.0)
            self.assertEqual(set(counts.columns), {"year", "field", "country_label", "n", "worth_musd"})
            self.assertNotIn("uri", " ".join(counts.astype(str).to_numpy().ravel()))
            row = counts[(counts.year == 2013) & (counts.field == "citizenship") & (counts.country_label == "China")]
            self.assertEqual(int(row.n.iloc[0]), 3)


class WdiTests(unittest.TestCase):
    def test_pages_and_archived_optional_series(self) -> None:
        def fake(url: str, **kw: object) -> bytes:
            ind = url.split("/indicator/")[1].split("?")[0]
            page = int(re.search(r"[?&]page=(\d+)", url).group(1))
            if ind == "CM.MKT.LCAP.GD.ZS":
                return json.dumps([{"message": [{"id": "175", "value": "archived"}]}]).encode()
            rows = [{"indicator": {"id": ind}, "countryiso3code": "PRT", "date": str(2009 + page), "value": 1.0},
                    {"indicator": {"id": ind}, "countryiso3code": "ESP", "date": "2010", "value": None}]
            return json.dumps([{"pages": 2}, rows]).encode()

        with tempfile.TemporaryDirectory() as tmp:
            m = fetch_wdi.fetch(Path(tmp), fetcher=fake)
            self.assertEqual(m["n_rows"], 12)  # six required series x two pages, null values dropped
            self.assertIn("CM.MKT.LCAP.GD.ZS", m["skipped"])

    def test_required_series_must_exist(self) -> None:
        def fake(url: str, **kw: object) -> bytes:
            return json.dumps([{"message": [{"id": "175", "value": "archived"}]}]).encode()

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(fetch_wdi.IndicatorMissing):
            fetch_wdi.fetch(Path(tmp), fetcher=fake)


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inp = p3.load_inputs()
        cls.panel, cls.cit, cls.w = p3.build_panel(cls.inp)

    def test_every_forbes_label_is_mapped(self) -> None:
        labels = set(self.inp["forbes"].query("year >= 2010").country_label)
        self.assertFalse(labels - set(p3.FORBES_ISO3))

    def test_forbes_totals_are_complete_from_2010(self) -> None:
        totals = self.cit.groupby("year").n.sum()
        self.assertEqual(int(totals[2025]), 3028)
        self.assertTrue((totals.loc[2010:2025].diff().dropna().abs() < 800).all())

    def test_density_defined_for_zero_counts(self) -> None:
        self.assertAlmostEqual(float(p3.density(0, 1e7)), np.log10(0.5), 12)
        self.assertFalse(self.panel.B.isna().any())

    def test_panel_shape(self) -> None:
        self.assertEqual(len(self.panel), 767)
        self.assertEqual(self.panel.country_code.nunique(), 74)
        self.assertTrue(self.panel.year.between(2010, 2024).all())
        self.assertTrue((self.panel.reporting_level == "national").all())

    def test_hash_is_enforced(self) -> None:
        original = p3.DATA
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("forbes_history_manifest.json", "forbes_history_counts.csv.gz"):
                (tmp / name).write_bytes((original / name).read_bytes())
            m = json.loads((tmp / "forbes_history_manifest.json").read_text())
            m["counts_sha256"] = "0" * 64
            (tmp / "forbes_history_manifest.json").write_text(json.dumps(m))
            p3.DATA = tmp
            try:
                with self.assertRaises(ValueError):
                    p3._checked("forbes_history_manifest.json", {"forbes_history_counts.csv.gz": "counts_sha256"})
            finally:
                p3.DATA = original


if __name__ == "__main__":
    unittest.main()
