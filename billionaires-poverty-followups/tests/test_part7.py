"""Contracts for part7_arrivals.py: categories, rule masks, the Part 4 check and the scaled placebo."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_forbes_arrivals as ffa  # noqa: E402
import part7_arrivals as p7  # noqa: E402


def bit(rule: str, k: int) -> int:
    return 1 << (4 * p7.RULES.index(rule) + k)


class ConstantsTests(unittest.TestCase):
    def test_bits_agree_with_the_fetcher(self) -> None:
        self.assertEqual(p7.RULES, ffa.RULES)
        for name in ("E_WINDOW", "E_CO_ENTRY", "E_CO_RISE", "E_CO_INC", "E_GENERIC", "E_RELISTED",
                     "X_WINDOW", "X_NEXT", "X_RETURNS", "X_RELISTED"):
            self.assertEqual(getattr(p7, name), getattr(ffa, name))

    def test_mask_unions_rules(self) -> None:
        self.assertEqual(p7.mask(("label", "rare"), p7.HEIR), bit("label", 0) | bit("rare", 0))


class ClassifyTests(unittest.TestCase):
    def test_precedence(self) -> None:
        heir = bit("rare", p7.HEIR)
        self.assertEqual(p7.classify(p7.E_RELISTED | heir, "returning", "no"), "relisted")
        self.assertEqual(p7.classify(heir, "returning", "no"), "returning")
        self.assertEqual(p7.classify(heir | p7.E_CO_RISE, "new", "no"), "heir")
        self.assertEqual(p7.classify(bit("label", p7.FELL) | bit("label", p7.INC), "new", "no"), "division")
        self.assertEqual(p7.classify(bit("label", p7.INC), "new", "yes"), "family")
        self.assertEqual(p7.classify(p7.E_CO_ENTRY, "new", "yes"), "company")
        self.assertEqual(p7.classify(0, "new", "no"), "inherited")
        self.assertEqual(p7.classify(0, "new", "yes"), "selfmade")

    def test_rule_choice_matters(self) -> None:
        loose_only = bit("loose", p7.HEIR)
        self.assertEqual(p7.classify(loose_only, "new", "no"), "inherited")          # the main rule ignores it
        self.assertEqual(p7.classify(loose_only, "new", "no", ("loose",)), "heir")


def frame(rows: list[tuple]) -> pd.DataFrame:
    cols = ["year", "table", "band_lo_musd", "tenure", "self_made", "age", "region", "links", "n"]
    return pd.DataFrame(rows, columns=cols)


class Part4CheckTests(unittest.TestCase):
    def test_unknown_citizenship_is_reported_not_compared(self) -> None:
        arr = frame([(2010, "entry", 1400, "new", "yes", "<50", "Europe", 0, 15),
                     (2010, "entry", 1400, "new", "yes", "<50", "unknown", 0, 1),
                     (2010, "exit", 1000, "exit", "yes", "<50", "Other", 0, 3)])
        flows = pd.DataFrame({"year": [2010, 2010], "country_label": ["X", "Y"], "flow": ["entry", "exit"],
                              "band_lo_musd": [1400, 1000], "n": [15, 3], "n_linked_elsewhere": [0, 0]})
        out = p7.check_against_part4(arr, flows, set())
        self.assertEqual(out["entry"]["people"], 16)
        self.assertEqual(out["entry"]["unknown_citizenship"], 1)
        flows.loc[0, "n"] = 14
        with self.assertRaises(ValueError):
            p7.check_against_part4(arr, flows, set())


class PlaceboTests(unittest.TestCase):
    def test_placebo_is_scaled_by_window_size(self) -> None:
        heir, plc = bit("label", p7.HEIR), bit("label", p7.PLACEBO)
        rows = [(2015, "entry", 2000, "new", "no", "<50", "Europe", p7.E_WINDOW | heir, 10),
                (2015, "entry", 2000, "new", "no", "<50", "Europe", p7.E_WINDOW | plc, 10),
                (2015, "entry", 2000, "new", "yes", "<50", "Europe", p7.E_WINDOW, 80)]
        # departures: 40 in the two observed transitions (2014, 2015), 20 in the placebo ones (2017, 2018)
        rows += [(y, "exit", 1000, "exit", "yes", "<50", "Europe", 0, n) for y, n in ((2014, 20), (2015, 20), (2017, 10),
                                                                                         (2018, 10))]
        arr = frame(rows)
        out = p7.placebo(p7.entries(arr), arr)["label"]["far"]
        self.assertAlmostEqual(out["observed"], 0.10)
        self.assertAlmostEqual(out["placebo_scaled"], 0.20)      # 10% of entrants, doubled for twice the departures
        self.assertAlmostEqual(out["excess"], -0.10)
        self.assertAlmostEqual(out["not_self_made_linked"], 1.0)


if __name__ == "__main__":
    unittest.main()
