"""Contracts for fetch_forbes_persistence.py: history, volatility classes, Part 5 totals, no identifiers."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_forbes_flows as fff  # noqa: E402
import fetch_forbes_persistence as ffp  # noqa: E402
import fetch_forbes_tenure as fft  # noqa: E402


def rec(uri: str, worth: float, age: int | None = 60, category: str | None = "Technology") -> dict:
    r = {"uri": uri, "naturalId": f"faces/{uri}", "personName": uri.replace("-", " "), "finalWorth": worth,
         "countryOfCitizenship": "X", "selfMade": True, "age": age}
    if category is not None:
        r["category"] = category
    return r


def build(spec: dict[int, list[dict]]) -> tuple[dict[int, list[fff.Person]], dict[int, list[str]]]:
    return ({y: [fff.Person(r, y) for r in rs] for y, rs in spec.items()},
            {y: [ffp.industry(r) for r in rs] for y, rs in spec.items()})


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        # a: 1000 -> 2000 -> 1000 -> 1000 -> 1250 (c1 = +0.097, earlier changes +0.301, -0.301, 0)
        # b: enters 2013 and leaves after 2014; c: on every list, flat
        self.lists, self.labels = build({
            2010: [rec("a", 1000), rec("c", 5000, age=85)],
            2011: [rec("a", 2000), rec("c", 5000, age=86)],
            2012: [rec("a", 1000), rec("c", 5000, age=87)],
            2013: [rec("a", 1000), rec("b", 1100, age=None, category=None), rec("c", 5000, age=88)],
            2014: [rec("a", 1250), rec("b", 1050, age=None, category=None), rec("c", 5000, age=89)],
            2015: [rec("a", 1250), rec("c", 5000, age=90)],
        })
        self.rows = pd.DataFrame(ffp.aggregate(self.lists, self.labels))

    def cell(self, year: int, band: int) -> pd.Series:
        s = self.rows[(self.rows.table == "stock") & (self.rows.year == year) & (self.rows.band_lo_musd == band)]
        self.assertEqual(len(s), 1)
        return s.iloc[0]

    def test_history_length_is_capped_and_censored(self) -> None:
        self.assertEqual(self.cell(2014, 1200)["hist"], 4)        # a: four consecutive earlier changes
        self.assertEqual(self.cell(2012, 1000)["hist"], 2)        # a in 2012: only two lists before
        self.assertEqual(self.cell(2014, 1000)["hist"], 1)        # b: one earlier change

    def test_volatility_excludes_latest_change(self) -> None:
        a = self.cell(2014, 1200)
        earlier = [math.log10(2), -math.log10(2), 0.0]
        self.assertAlmostEqual(a.ms_sum, sum(c * c for c in earlier) / 3, places=6)
        self.assertEqual(a.vol, ffp.vol_class(math.sqrt(sum(c * c for c in earlier) / 3)))
        self.assertEqual(a.vol, 220)
        self.assertEqual(a.trend, "flat")
        self.assertEqual(a.prev, "rose 0.03-0.1")
        self.assertAlmostEqual(a.c1sq_sum, math.log10(1.25) ** 2, places=8)

    def test_short_history_has_no_volatility_class(self) -> None:
        b = self.cell(2014, 1000)
        self.assertEqual((b.vol, b.trend, b.ms_sum), (-1, "none", 0.0))
        self.assertEqual(b.age, "unknown")

    def test_exit_and_logchange(self) -> None:
        ex = self.rows[(self.rows.table == "exit")]
        self.assertEqual(list(zip(ex.year, ex.band_lo_musd)), [(2014, 1000)])
        lc = self.rows[(self.rows.table == "logchange") & (self.rows.year == 2014)]
        self.assertEqual(sorted(lc.key.astype(int)), [0, 0])     # a and c flat to 2015
        self.assertTrue(self.rows[self.rows.year == 2015].empty)  # no outcome for the last list

    def test_industry_tables(self) -> None:
        ind = self.rows[(self.rows.table == "industry_exit")]
        self.assertEqual(list(ind.key), ["unknown"])
        stock = self.rows[(self.rows.table == "industry_stock") & (self.rows.year == 2014)]
        self.assertEqual(stock.groupby("key").n.sum().to_dict(), {"Technology": 2, "unknown": 1})

    def test_age_groups(self) -> None:
        self.assertEqual(self.cell(2014, 5000).age, "80+")
        self.assertEqual([ffp.age_group(a) for a in (None, 59, 60, 79, 80)], ["unknown", "<60", "60-79", "60-79", "80+"])


class Part5TotalsTests(unittest.TestCase):
    """Summed over the new dimensions, the tables must equal Part 5's tenure tables."""

    def test_random_lists_reproduce_part5(self) -> None:
        rng = random.Random(7)
        pool = [f"p{k}" for k in range(120)]
        worth = {p: rng.choice([1000, 1200, 1500, 2600, 4000, 9000]) for p in pool}
        spec = {}
        for year in range(2010, 2017):
            people = []
            for p in pool:
                if rng.random() < 0.8:
                    worth[p] = max(100, round(worth[p] * 10 ** rng.gauss(0, 0.12), -2))
                    people.append(rec(p, worth[p], age=rng.choice([None, 45, 70, 90])))
            spec[year] = people
        lists, labels = build(spec)
        new = pd.DataFrame(ffp.aggregate(lists, labels))
        old = pd.DataFrame(fft.aggregate(lists))
        for table in ("stock", "exit", "logchange"):
            a = new[new.table == table].groupby(["year", "band_lo_musd", "tenure", "prev", "key"]).n.sum()
            b = old[old.table == table].groupby(["year", "band_lo_musd", "tenure", "prev", "key"]).n.sum()
            pd.testing.assert_series_equal(a.sort_index(), b.sort_index(), check_names=False)
        a = new[new.table == "industry_stock"].groupby(["year", "band_lo_musd"]).n.sum()
        b = new[new.table == "stock"].groupby(["year", "band_lo_musd"]).n.sum()
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_no_identifiers_written(self) -> None:
        lists, labels = build({2010: [rec("secret-name", 1000)], 2011: [rec("secret-name", 1100)],
                               2012: [rec("secret-name", 1300)]})
        text = pd.DataFrame(ffp.aggregate(lists, labels)).to_csv()
        self.assertNotIn("secret", text)
        self.assertEqual(list(pd.DataFrame(ffp.aggregate(lists, labels)).columns), ffp.COLUMNS)


if __name__ == "__main__":
    unittest.main()
