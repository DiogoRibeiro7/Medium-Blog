"""Checks for Part 8: the extracts are complete, weeks and holidays are assigned as the article says, the three
datings line up, and the error measures and the bootstrap behave."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part8"))
import hub as H  # noqa: E402


class Extracts(unittest.TestCase):
    def test_forecasts(self):
        f = H.forecasts()
        self.assertEqual(len(f), 60266)
        self.assertEqual(f.model.nunique(), 45)
        self.assertEqual(set(f.location), {"GM", *H.STATES})
        self.assertEqual(set(f.horizon), {1.0, 2.0, 3.0, 4.0})
        self.assertTrue((f.week.dt.dayofweek == 5).all())  # every target week ends on a Saturday

    def test_rki_and_holidays(self):
        r = pd.read_csv(H.DATA / "rki_dates.csv.gz")
        self.assertEqual(int(r.cases.sum()), 39209067)
        self.assertEqual(sorted(r.state.unique()), list(range(1, 17)))
        hol = H.holidays()
        self.assertEqual(hol["DE-BE"][pd.Timestamp("2021-03-08").date()], "Frauentag")
        self.assertNotIn(pd.Timestamp("2021-03-08").date(), hol["DE"])


class Weeks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hol = H.holidays()
        cls.d = H.daily()
        cls.w = H.week_types(H.weekly(cls.d), cls.hol)

    def test_weekly_sums(self):
        week = pd.Timestamp("2021-04-10")
        days = self.d[(self.d.location == "GM") & self.d.date.between(week - pd.Timedelta(days=6), week)]
        row = self.w[(self.w.location == "GM") & (self.w.week == week)].iloc[0]
        for col in ("publication", "reporting", "reference"):
            self.assertEqual(row[col], days[col].sum())

    def test_week_types(self):
        row = lambda loc, wk: self.w[(self.w.location == loc) & (self.w.week == pd.Timestamp(wk))].iloc[0]
        self.assertTrue(row("GM", "2021-04-10").national)        # Easter Monday
        self.assertFalse(row("GM", "2021-04-17").national)
        self.assertTrue(row("GM16", "2021-03-13").regional)      # Women's Day, Berlin
        self.assertFalse(row("GM01", "2021-03-13").regional)
        self.assertEqual(H.neighbours(pd.Timestamp("2021-04-10"), {pd.Timestamp("2021-04-03"), pd.Timestamp("2021-04-10")}),
                         (pd.Timestamp("2021-03-27"), pd.Timestamp("2021-04-17")))

    def test_onset_dips_least(self):
        hr = H.holiday_ratios(self.d, "GM", [pd.Timestamp("2021-04-05")]).iloc[0]
        self.assertLess(hr.publication, hr.onset)
        self.assertLess(hr.reporting, hr.onset)


class Measures(unittest.TestCase):
    def test_score_and_boot(self):
        f = pd.DataFrame({"location": ["GM"], "week": [pd.Timestamp("2021-04-10")], "median": [100.0],
                          "q0100": [80.0], "q0900": [120.0]})
        w = pd.DataFrame({"location": ["GM"], "week": [pd.Timestamp("2021-04-10")], "publication": [50.0],
                          "reporting": [100.0], "reference": [200.0]})
        s = H.score(f, w).iloc[0]
        self.assertAlmostEqual(s.e_publication, np.log(2))
        self.assertAlmostEqual(s.e_reporting, 0.0)
        self.assertEqual((s.c80_publication, s.c80_reporting), (0.0, 1.0))
        v = pd.Series(np.arange(20, dtype=float))
        g = pd.Series(np.repeat(np.arange(5), 4))
        self.assertEqual(H.cluster_boot(v, g, lambda z: z.mean(), n=200), H.cluster_boot(v, g, lambda z: z.mean(), n=200))


if __name__ == "__main__":
    unittest.main()
