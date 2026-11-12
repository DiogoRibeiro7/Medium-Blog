"""Contracts for fetch_forbes_flows.py: linking rules, flows and no person-level output."""

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


def rec(uri: str, worth: float, country: str, *, natural: str | None = None, name: str | None = None,
        birth: int | None = None, self_made: bool | None = True, age: int | None = 60) -> dict:
    return {"uri": uri, "naturalId": natural or f"faces/{uri}", "personName": name or uri.replace("-", " "),
            "birthDate": birth, "finalWorth": worth, "countryOfCitizenship": country, "selfMade": self_made,
            "age": age}


def people(records: list[dict], year: int) -> list[fff.Person]:
    return [fff.Person(r, year) for r in records]


class LinkTests(unittest.TestCase):
    def test_rules_in_order_and_one_to_one(self) -> None:
        before = people([rec("a", 1500, "X"), rec("b", 2000, "X", natural="faces/b"),
                         rec("c", 3000, "X", natural="n-c", name="Carl Old", birth=1),
                         rec("d", 1100, "X")], 2010)
        after = people([rec("a", 1600, "X"), rec("b-new", 2100, "X", natural="faces/b"),
                        rec("c-new", 2900, "X", natural="n-c2", name="Carl Old & family", birth=1)], 2011)
        links, rules = fff.link(before, after)
        self.assertEqual(links, {0: 0, 1: 1, 2: 2})
        self.assertEqual(rules, {"uri": 1, "natural": 1, "name": 1})

    def test_ambiguous_keys_do_not_link(self) -> None:
        before = people([rec("a", 1500, "X", natural="dup"), rec("b", 1500, "X", natural="dup")], 2010)
        after = people([rec("z", 1500, "X", natural="dup")], 2011)
        links, _ = fff.link(before, after)
        self.assertEqual(links, {})


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lists = {
            2010: people([rec("a", 1050, "X"), rec("b", 5000, "X"), rec("c", 1200, "Y"),
                          rec("e", 1300, "X")], 2010),
            2011: people([rec("a", 1100, "X"), rec("b", 4000, "Y"), rec("d", 1400, "X", self_made=False)], 2011),
            2012: people([rec("a", 1000, "X"), rec("c", 1250, "Y"), rec("d", 1500, "X")], 2012),
        }
        self.tables = fff.aggregate(self.lists)
        self.flows = pd.DataFrame(self.tables["flows"])

    def test_flows_add_up_to_count_changes(self) -> None:
        bands = pd.DataFrame(self.tables["bands"]).groupby(["year", "country_label"])["n"].sum()
        for year in (2010, 2011):
            f = self.flows[self.flows.year == year]
            for country in ("X", "Y"):
                g = f[f.country_label == country].groupby("flow")["n"].sum()
                change = (g.get("entry", 0) - g.get("exit", 0) + g.get("switch_in", 0) - g.get("switch_out", 0))
                self.assertEqual(change, bands.get((year + 1, country), 0) - bands.get((year, country), 0))

    def test_flow_bands_and_returns(self) -> None:
        f = self.flows.set_index(["year", "country_label", "flow", "band_lo_musd"])
        self.assertEqual(f.loc[(2010, "X", "switch_out", 5000), "n"], 1)
        self.assertEqual(f.loc[(2010, "Y", "switch_in", 4000), "n"], 1)
        self.assertEqual(f.loc[(2010, "Y", "exit", 1200), "n"], 1)
        self.assertEqual(f.loc[(2010, "Y", "exit", 1200), "n_linked_elsewhere"], 1)   # c returns in 2012
        self.assertEqual(f.loc[(2010, "X", "exit", 1300), "n_linked_elsewhere"], 0)   # e never returns
        self.assertEqual(f.loc[(2011, "Y", "entry", 1200), "n_linked_elsewhere"], 1)  # c was on 2010
        self.assertEqual(f.loc[(2010, "X", "entry", 1400), "n_linked_elsewhere"], 0)

    def test_global_tables(self) -> None:
        g = pd.DataFrame(self.tables["global"])
        lc = g[(g.table == "logchange") & (g.year == 2010)]
        self.assertEqual(lc.n.sum(), 2)  # a and b stayed on the list
        self.assertEqual(set(g[g.table == "entry_selfmade"].key), {"no", "yes"})
        self.assertEqual(g[(g.table == "exit_age") & (g.year == 2010)].n.sum(), 2)


class FetchTests(unittest.TestCase):
    def test_outputs_hold_no_identifiers(self) -> None:
        payload = {2010: [rec("alpha-one", 1050, "X", name="Alpha One"), rec("beta-two", 2500, "Y")],
                   2011: [rec("alpha-one", 1200, "X", name="Alpha One")]}

        def fake(url: str, **kw: object) -> bytes:
            year = int(re.search(r"/billionaires/(\d{4})/", url).group(1))
            return json.dumps({"personList": {"personsLists": [dict(r, year=year) for r in payload[year]]}}).encode()

        with tempfile.TemporaryDirectory() as tmp:
            manifest = fff.fetch(2010, 2011, Path(tmp), fetcher=fake, pause=0)
            self.assertEqual(manifest["n_years_ok"], 2)
            self.assertEqual(manifest["pairs"]["2010-2011"]["flows"], {"stay": 1, "exit": 1})
            text = "".join(gzip.decompress(p.read_bytes()).decode() for p in Path(tmp).glob("*.csv.gz"))
            text += (Path(tmp) / "forbes_flows_manifest.json").read_text()
            for token in ("alpha", "beta", "Alpha One"):
                self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
