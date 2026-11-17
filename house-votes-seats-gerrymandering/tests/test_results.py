"""Checks that the derived data reproduce official totals and the article's headline numbers.

Run from the folder: python -m unittest discover -s tests -v
(needs data/derived and results, i.e. the pipeline in the README run once).
"""
import json
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]

# Party division at each general election (Office of the Historian, U.S. House).
OFFICIAL = {
    1972: (242, 192), 1974: (291, 144), 1976: (292, 143), 1978: (277, 158), 1980: (243, 192),
    1982: (269, 166), 1984: (253, 182), 1986: (258, 177), 1988: (260, 175), 1990: (267, 167),
    1992: (258, 176), 1994: (204, 230), 1996: (207, 226), 1998: (211, 223), 2000: (212, 221),
    2002: (205, 229), 2004: (202, 232), 2006: (233, 202), 2008: (257, 178), 2010: (193, 242),
    2012: (201, 234), 2014: (188, 247), 2016: (194, 241), 2018: (235, 200), 2020: (222, 213),
    2022: (213, 222), 2024: (215, 220),
}


class TestDistricts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = pd.read_csv(HERE / "data" / "derived" / "house_districts_raw_1972_2024.csv")

    def test_435_districts_every_year(self):
        self.assertTrue((self.d.groupby("year").size() == 435).all())

    def test_party_division_matches_official(self):
        for year, (dem, rep) in OFFICIAL.items():
            w = self.d.loc[self.d.year == year, "winner"]
            self.assertEqual(((w == "D").sum(), (w == "R").sum()), (dem, rep), msg=str(year))


class TestHeadlines(unittest.TestCase):
    def test_post_states_reproduce(self):
        p = pd.read_csv(HERE / "results" / "post_check_2026.csv").set_index("state_po")
        self.assertEqual(int(p.loc["TX", "dem_seats"]), 14)
        self.assertAlmostEqual(p.loc["TX", "vote_pct"], 52.9, places=1)
        for st in ["GA", "TN", "OK", "SC", "LA", "UT", "TX"]:
            self.assertLessEqual(abs(p.loc[st, "vote_pct"] - p.loc[st, "post_vote"]), 0.6, msg=st)
            self.assertLessEqual(abs(p.loc[st, "seat_pct"] - p.loc[st, "post_seat"]), 0.6, msg=st)

    def test_cube_law(self):
        s = json.loads((HERE / "results" / "summary_votes_seats.json").read_text())
        self.assertAlmostEqual(s["rho_pooled"], 3.0, delta=0.05)

    def test_event_study_has_no_pretrend_and_a_jump(self):
        e = pd.read_csv(HERE / "results" / "event_study_coefs.csv").set_index("k_rel")
        self.assertLess(e.loc[-2, "lo"], 0)
        self.assertGreater(e.loc[-2, "hi"], 0)
        self.assertGreater(e.loc[0, "lo"], 0)


if __name__ == "__main__":
    unittest.main()
