"""Contracts for fetch_forbes_tenure.py: tenure, previous change, rounding, no identifiers."""

from __future__ import annotations

import gzip
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_forbes_flows as fff  # noqa: E402
import fetch_forbes_tenure as fft  # noqa: E402


def rec(uri: str, worth: float, country: str = "X") -> dict:
    return {"uri": uri, "naturalId": f"faces/{uri}", "personName": uri.replace("-", " "), "finalWorth": worth,
            "countryOfCitizenship": country, "selfMade": True, "age": 60}


def lists(spec: dict[int, list[dict]]) -> dict[int, list[fff.Person]]:
    return {y: [fff.Person(r, y) for r in rs] for y, rs in spec.items()}


class TenureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(fft.aggregate(lists({
            2010: [rec("a", 1000), rec("b", 2000), rec("c", 1500)],
            2011: [rec("a", 1300), rec("b", 2000), rec("d", 1050)],
            2012: [rec("a", 1000), rec("c", 1234), rec("d", 1100)],
            2013: [rec("c", 1300)],
        })))

    def test_tenure_labels(self) -> None:
        s = self.rows[self.rows.table == "stock"]
        t2012 = s[s.year == 2012].groupby("tenure").n.sum().to_dict()
        self.assertEqual(t2012, {"incumbent": 2, "returning": 1})        # a, d incumbents; c back from 2010
        t2011 = s[s.year == 2011].groupby("tenure").n.sum().to_dict()
        self.assertEqual(t2011, {"incumbent": 2, "new": 1})
        self.assertEqual(set(s[s.year == 2010].tenure), {"unknown"})

    def test_previous_change_and_outcomes(self) -> None:
        s = self.rows[(self.rows.table == "stock") & (self.rows.year == 2011) & (self.rows.tenure == "incumbent")]
        self.assertEqual(dict(zip(s.band_lo_musd, s.prev)), {1300: "rose >0.1", 2000: "flat"})
        lc = self.rows[(self.rows.table == "logchange") & (self.rows.year == 2011)]
        self.assertEqual(int(lc.n.sum()), 2)                                # a and d stay to 2012
        ex = self.rows[(self.rows.table == "exit") & (self.rows.year == 2011)]
        self.assertEqual(list(ex.band_lo_musd), [2000])                     # b leaves
        self.assertTrue(self.rows[(self.rows.table == "stock") & (self.rows.year == 2013)].empty)

    def test_rounding(self) -> None:
        r = self.rows[(self.rows.table == "rounding") & (self.rows.year == 2012)]
        self.assertEqual(r.groupby("key").n.sum().to_dict(), {"m100": 2, "other": 1})


class FetchTests(unittest.TestCase):
    def test_outputs_hold_no_identifiers(self) -> None:
        payload = {2010: [rec("alpha-one", 1050)], 2011: [rec("alpha-one", 1200)], 2012: [rec("beta-two", 1400)]}

        def fake(url: str, **kw: object) -> bytes:
            year = int(re.search(r"/billionaires/(\d{4})/", url).group(1))
            return json.dumps({"personList": {"personsLists": [dict(r, year=year) for r in payload[year]]}}).encode()

        with tempfile.TemporaryDirectory() as tmp:
            m = fft.fetch(2010, 2012, Path(tmp), fetcher=fake, pause=0)
            self.assertEqual(m["n_years_ok"], 3)
            text = gzip.decompress((Path(tmp) / "forbes_tenure.csv.gz").read_bytes()).decode()
            text += (Path(tmp) / "forbes_tenure_manifest.json").read_text()
            for token in ("alpha", "beta"):
                self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
