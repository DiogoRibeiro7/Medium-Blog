"""Contracts for fetch_forbes_arrivals.py: family, company and placebo links, self-exclusion, Part 4 totals."""

from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_forbes_arrivals as ffa  # noqa: E402
import fetch_forbes_flows as fff  # noqa: E402


def rec(uri: str, worth: float, last: str, source: str = "Acme Widgets", country: str = "United States",
        age: int | None = 60, self_made: bool | None = True, family: list | None = None, name: str | None = None) -> dict:
    r = {"uri": uri, "naturalId": f"faces/{uri}", "personName": name or f"{uri.title()} {last}", "lastName": last,
         "finalWorth": worth, "countryOfCitizenship": country, "selfMade": self_made, "age": age, "source": source}
    if family is not None:
        r["familyList"] = family
    return r


def build(spec: dict[int, list[dict]]):
    return ({y: [fff.Person(r, y) for r in rs] for y, rs in spec.items()},
            {y: [ffa.Extra(r) for r in rs] for y, rs in spec.items()})


def flags(rows: pd.DataFrame, table: str, year: int, band: int | None = None) -> list[int]:
    s = rows[(rows.table == table) & (rows.year == year)]
    if band is not None:
        s = s[s.band_lo_musd == band]
    return sorted(int(f) for f, n in zip(s["links"], s.n) for _ in range(n))


def other(k: int, worth: float = 8000) -> dict:
    return rec(f"bg{k}", worth, f"Zz{k}", source=f"Company {k}", country="Japan")


class HeirTests(unittest.TestCase):
    def setUp(self) -> None:
        base = [other(k) for k in range(3)]
        spec = {
            2010: base + [rec("pat", 9000, "Walton", age=88, self_made=True)],
            2011: base + [rec("pat", 9000, "Walton", age=89, self_made=True)],
            # 2011 -> 2012: the patriarch leaves; two heirs enter with the same surname, country and company
            2012: base + [rec("ann", 4000, "Walton", age=55, self_made=False),
                          rec("bob", 4000, "Walton", age=52, self_made=False)],
            2013: base + [rec("ann", 4100, "Walton"), rec("bob", 4100, "Walton")],
        }
        self.rows = pd.DataFrame(ffa.aggregate(*build(spec))[0])

    def test_heirs_are_linked_to_the_departure(self) -> None:
        heirs = flags(self.rows, "entry", 2011)
        self.assertEqual(len(heirs), 2)
        for f in heirs:
            self.assertTrue(f & 1 and f & 16)        # strict and loose family member left at this transition
            self.assertFalse(f & 16384)              # not the same person relisted
        ex = flags(self.rows, "exit", 2011)
        self.assertEqual(len(ex), 1)
        self.assertTrue(ex[0] & 1 and ex[0] & 8)
        self.assertFalse(ex[0] & 1024)               # never returns

    def test_co_entry_between_heirs(self) -> None:
        self.assertTrue(all(f & 128 for f in flags(self.rows, "entry", 2011)))

    def test_stock_family_prevalence(self) -> None:
        st = self.rows[(self.rows.table == "stock") & (self.rows.year == 2012)]
        self.assertEqual(int(st[st["links"].astype(int) & 1 > 0].n.sum()), 2)


class DivisionAndCompanyTests(unittest.TestCase):
    def setUp(self) -> None:
        base = [other(k) for k in range(3)]
        spec = {
            2010: base + [rec("mom", 12000, "Persson", source="H M", country="Sweden"),
                          rec("cofounder", 5000, "Smith", source="Rocket Labs Inc")],
            # mother's fortune falls by more than 0.1 while a son enters: a division among the living
            2011: base + [rec("mom", 6000, "Persson", source="H&M", country="Sweden"),
                          rec("son", 3000, "Persson", source="H M", country="Sweden", self_made=False),
                          rec("cofounder", 9000, "Smith", source="Rocket Labs"),
                          rec("partner", 2500, "Jones", source="Rocket Labs"),
                          rec("estate", 1600, "Garcia", source="real estate", country="Spain")],
            2012: base + [rec("mom", 6000, "Persson", source="H&M", country="Sweden"),
                          rec("son", 3000, "Persson", source="H M", country="Sweden", self_made=False),
                          rec("cofounder", 9000, "Smith", source="Rocket Labs"),
                          rec("partner", 2500, "Jones", source="Rocket Labs"),
                          rec("estate", 1600, "Garcia", source="real estate", country="Spain")],
        }
        self.rows = pd.DataFrame(ffa.aggregate(*build(spec))[0])
        e = self.rows[(self.rows.table == "entry") & (self.rows.year == 2010)]
        self.by_band = {int(b): int(f) for b, f in zip(e.band_lo_musd, e["links"])}

    def test_division_among_the_living(self) -> None:
        f = self.by_band[3000]
        self.assertTrue(f & 4 and f & 8)
        self.assertFalse(f & 1)

    def test_revaluation_of_a_company(self) -> None:
        f = self.by_band[2500]
        self.assertTrue(f & 256 and f & 4096)     # co-owner on the list rose by more than 0.1
        self.assertFalse(f & 4)                   # but no family link

    def test_generic_label_gives_no_company_link(self) -> None:
        f = self.by_band[1500]
        self.assertTrue(f & 8192)
        self.assertFalse(f & (128 | 256 | 4096 | 1 | 4))


class PlaceboAndSelfTests(unittest.TestCase):
    def test_placebo_window_and_self_exclusion(self) -> None:
        base = [other(k) for k in range(3)]
        uncle = rec("uncle", 5000, "Tata", source="Tata Group", country="India", age=70)
        niece = rec("niece", 2000, "Tata", source="Tata Group", country="India", self_made=False)
        spec = {y: list(base) for y in range(2010, 2017)}
        for y in range(2010, 2014):
            spec[y].append(uncle)                  # uncle on 2010-2013, leaves at the 2013 -> 2014 transition
        for y in range(2012, 2017):
            spec[y].append(niece)                  # niece enters at 2011 -> 2012 and stays
        rows = pd.DataFrame(ffa.aggregate(*build(spec))[0])
        f = flags(rows, "entry", 2011)[0]
        self.assertTrue(f & 2048 and f & 512)      # uncle leaves two transitions later: placebo, not heir
        self.assertFalse(f & 1 or f & 2)
        self.assertTrue(f & 4)                     # he is an incumbent family member when she enters

    def test_relisted_person_is_not_their_own_heir(self) -> None:
        base = [other(k) for k in range(3)]
        a = rec("old-uri", 3000, "Ambani", source="Reliance", country="India", name="Anil Ambani")
        b = rec("new-uri", 3000, "Ambani", source="Reliance", country="India", name="Anil Ambani")
        b["naturalId"] = "faces/other"
        a["birthDate"], b["birthDate"] = -315619200000, -283996800000   # a corrected birth date breaks every key
        rows = pd.DataFrame(ffa.aggregate(*build({2010: base + [a], 2011: base + [b]}))[0])
        f = flags(rows, "entry", 2010)[0]
        self.assertTrue(f & 16384)
        self.assertFalse(f & (1 | 16))
        self.assertTrue(flags(rows, "exit", 2010)[0] & 4096)


class Part4TotalsTests(unittest.TestCase):
    def test_entries_and_exits_match_the_flow_linking(self) -> None:
        rng = random.Random(4)
        pool = [(f"p{k}", rng.choice(["Li", "Smith", "Garcia", "Kim", f"Q{k}"]), rng.choice(["United States", "China"]))
                for k in range(150)]
        worth = {u: rng.choice([1000, 1300, 2000, 4000]) for u, _, _ in pool}
        spec = {}
        for y in range(2010, 2016):
            spec[y] = []
            for u, last, country in pool:
                if rng.random() < 0.8:
                    worth[u] = max(1000, round(worth[u] * 10 ** rng.gauss(0, 0.1), -2))
                    spec[y].append(rec(u, worth[u], last, source=rng.choice(["real estate", f"Firm {u}"]),
                                       country=country))
        lists, extras = build(spec)
        rows = pd.DataFrame(ffa.aggregate(lists, extras)[0])
        for y in range(2010, 2015):
            m, _ = fff.link(lists[y], lists[y + 1])
            want_entry = Counter(fff.band(p.worth, fff.FLOW_EDGES) for j, p in enumerate(lists[y + 1])
                                 if j not in set(m.values()))
            want_exit = Counter(fff.band(p.worth, fff.FLOW_EDGES) for i, p in enumerate(lists[y]) if i not in m)
            for table, want in (("entry", want_entry), ("exit", want_exit)):
                got = rows[(rows.table == table) & (rows.year == y)].groupby("band_lo_musd").n.sum().to_dict()
                self.assertEqual({int(k): v for k, v in got.items()}, dict(want))

    def test_no_identifiers_or_companies_written(self) -> None:
        base = [other(k) for k in range(3)]
        lists, extras = build({2010: base + [rec("secret", 3000, "Hidden", source="Secret Corp")],
                               2011: base + [rec("secret2", 3000, "Hidden", source="Secret Corp")]})
        text = pd.DataFrame(ffa.aggregate(lists, extras)[0]).to_csv()
        self.assertNotIn("ecret", text)
        self.assertNotIn("idden", text)


class HelperTests(unittest.TestCase):
    def test_labels_and_surnames(self) -> None:
        self.assertEqual(ffa.company_label({"organization": "Walmart Inc.", "source": "Walmart"}), "walmart")
        self.assertEqual(ffa.company_label({"source": "H&M"}), "h m")
        self.assertEqual(ffa.surname_tokens({"lastName": "Bettencourt Meyers"}), frozenset({"bettencourt", "meyers"}))
        self.assertEqual(ffa.surname_tokens({"lastName": "van der Berg"}), frozenset({"berg"}))
        self.assertEqual(ffa.surname_tokens({"personName": "Gopichand Hinduja & family"}), frozenset({"hinduja"}))
        self.assertEqual(ffa.surname_tokens({"lastName": "Li"}), frozenset({"li"}))
        self.assertEqual(ffa.region("Hong Kong"), "China")
        self.assertEqual(ffa.region("Portugal"), "Europe")
        self.assertEqual(ffa.age_group(85), "85+")

    def test_family_list_strings(self) -> None:
        r = {"familyList": [{"name": "Ana Silva", "relation": "daughter"}, "bob-silva"]}
        self.assertIn("ana silva", ffa.family_strings(r))
        self.assertIn("bob-silva", ffa.family_strings(r))


if __name__ == "__main__":
    unittest.main()
