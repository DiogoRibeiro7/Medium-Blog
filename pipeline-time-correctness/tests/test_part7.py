"""Checks for Part 7: the Johns Hopkins extracts are complete, a file is called unchanged only when the count
is the same as the day before as the pipeline had it, days are classified by the right calendar, and the RKI
and Johns Hopkins counts line up as the article says."""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part7"))
import jhu as J  # noqa: E402


class Extracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.us = J.us_states()
        cls.c = J.countries()

    def test_complete(self):
        self.assertEqual(self.us.date.nunique(), 1062)
        self.assertEqual(self.us.state.nunique(), 51)
        self.assertEqual(len(self.us), 1062 * 51)
        self.assertEqual((self.us.date.min().date(), self.us.date.max().date()),
                         (dt.date(2020, 4, 12), dt.date(2023, 3, 9)))
        days = pd.date_range(self.c.date.min(), self.c.date.max())
        self.assertEqual(self.c.date.nunique(), len(days))
        self.assertTrue(self.us["first"].notna().all())

    def test_unchanged_means_equal_to_the_day_before_at_landing(self):
        row = lambda d, s: self.us[(self.us.date == d) & (self.us.state == s)].iloc[0]
        flat = row("2020-11-26", "Connecticut")  # Thanksgiving 2020
        self.assertEqual(flat.flat, 1.0)
        self.assertEqual(flat["first"], flat.prev_at_landing)
        moved = row("2020-11-26", "California")
        self.assertEqual(moved.flat, 0.0)
        self.assertNotEqual(moved["first"], moved.prev_at_landing)
        self.assertTrue(pd.isna(row("2020-04-12", "Alabama").flat))  # first file: nothing to compare with


class Calendars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hol = J.holidays()
        cls.ud = J.us_day_types(J.us_states(), hol)
        cls.hol = hol

    def day(self, d, s):
        return self.ud[(self.ud.date == d) & (self.ud.state == s)].day.iloc[0]

    def test_classification(self):
        self.assertEqual(self.day("2020-11-26", "Texas"), "federal and state holiday")
        self.assertEqual(self.day("2021-03-31", "California"), "state holiday only")  # Cesar Chavez Day
        self.assertEqual(self.day("2020-10-12", "California"), "federal holiday, state open")  # Columbus Day
        self.assertEqual(self.day("2020-10-12", "New York"), "federal and state holiday")
        self.assertEqual(self.day("2021-04-19", "Massachusetts"), "state holiday only")  # Patriots' Day
        self.assertEqual(self.day("2021-04-20", "Massachusetts"), "day after a holiday")
        self.assertEqual(self.day("2020-12-24", "New York"), "Christmas Eve or New Year's Eve")
        self.assertEqual(self.day("2021-06-08", "Ohio"), "ordinary")

    def test_daily_regime_is_weekdays_only(self):
        dr = J.us_daily_regime(self.ud)
        self.assertTrue((dr.wd < 5).all())
        self.assertTrue((dr.base <= J.THRESHOLD).all())


class Sources(unittest.TestCase):
    def test_rki_files_after_berlin_holidays(self):
        inc = J.rki_increments()
        self.assertEqual(inc[pd.Timestamp("2022-03-09")], 217189)
        self.assertEqual(inc[pd.Timestamp("2023-03-09")], 9742)
        self.assertEqual(inc[pd.Timestamp("2024-03-09")], 0)
        self.assertEqual(inc[pd.Timestamp("2025-05-09")], 0)

    def test_germany_lines_up_with_the_next_rki_file(self):
        c = J.countries()
        de = J.germany(c)
        jinc = (de["first"] - de.prev_in_first).astype(float)
        inc = J.rki_increments()
        corr = {}
        for k in (-1, 0, 1, 2):
            x = pd.concat([jinc, inc.shift(-k)], axis=1, sort=True).dropna()
            x = x[(x.index >= "2021-04-02") & (x.index <= "2023-03-09")]
            corr[k] = x.corr().iloc[0, 1]
        self.assertEqual(max(corr, key=corr.get), 1)

    def test_timestamps_move_on_unchanged_rows(self):
        us = J.us_states()
        lu = us[(us.flat == 1) & us.last_update.notna() & us.last_update_prev.notna()]
        self.assertGreater(lu.last_update.ne(lu.last_update_prev).mean(), 0.99)


if __name__ == "__main__":
    unittest.main()
