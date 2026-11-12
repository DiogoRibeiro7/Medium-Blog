"""Checks for Part 5: the three changes to the Part 4 pipeline leave Part 4 untouched when they are
off, find the right data cut, place the holidays correctly, compute the last-week fallback as stated,
and keep every run reproducible."""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part4"))
import pipeline as P  # noqa: E402


def days(a: dt.date, b: dt.date):
    while a <= b:
        yield a
        a += dt.timedelta(days=1)


class Defaults(unittest.TestCase):
    def test_all_off_is_the_part4_pipeline(self):
        self.assertEqual(P.variant_sql(), (P.SCHEMA_SQL, "", P.PROFILE_SQL, P.REPORT_SQL))
        self.assertEqual(P.Pipeline().code_version, P.CODE_VERSION)
        self.assertEqual(P.CODE_VERSION, "9adf5e8d8533")

    def test_easter_and_holidays(self):
        self.assertEqual([P.easter(y) for y in (2021, 2022, 2023, 2024)],
                         [dt.date(2021, 4, 4), dt.date(2022, 4, 17), dt.date(2023, 4, 9), dt.date(2024, 3, 31)])
        self.assertEqual(P.german_holidays(2023, 2023),
                         [dt.date(2023, 1, 1), dt.date(2023, 4, 7), dt.date(2023, 4, 10), dt.date(2023, 5, 1),
                          dt.date(2023, 5, 18), dt.date(2023, 5, 29), dt.date(2023, 10, 3), dt.date(2023, 12, 25),
                          dt.date(2023, 12, 26)])
        con = P.Pipeline(holidays=True).con
        got = con.execute("select core.daytype(date '2023-04-10'), core.daytype(date '2023-04-12'), "
                          "core.daytype(date '2023-05-18')").fetchone()
        self.assertEqual(got, (7, 3, 7))


class AllThree(unittest.TestCase):
    """A short replay of August 2023 with the content check, the holidays and the last-week fallback."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.incoming = P.write_incoming(cls.tmp / "in", dt.date(2023, 5, 15), dt.date(2023, 9, 5))
        cls.pipe = P.replay(dt.date(2023, 8, 14), dt.date(2023, 8, 31), cls.incoming,
                            content_check=True, holidays=True, fallback="last_week")
        cls.rep = cls.pipe.report().set_index("run_date")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_data_cut(self):
        cut = lambda d: pd.Timestamp(self.rep.data_cut[pd.Timestamp(d)]).date()  # noqa: E731
        self.assertEqual(cut("2023-08-24"), dt.date(2023, 8, 24))   # an ordinary Thursday
        self.assertEqual(cut("2023-08-28"), dt.date(2023, 8, 24))   # four files with nothing new
        self.assertEqual(cut("2023-08-21"), dt.date(2023, 8, 19))   # an ordinary Monday: Saturday's file
        self.assertEqual(cut("2023-08-29"), dt.date(2023, 8, 29))

    def test_every_run_reproduces_exactly(self):
        for d in days(dt.date(2023, 8, 14), dt.date(2023, 8, 31)):
            pr, rr = self.pipe.reproduce(d)
            ps, rs = self.pipe.stored(d)
            self.assertTrue(P.same(pr, ps), d)
            self.assertTrue(P.same(rr, rs), d)

    def test_last_week_fallback_arithmetic(self):
        d = dt.date(2023, 8, 28)
        profile, row = self.pipe.stored(d)
        row = row.iloc[0]
        pub = pd.Timestamp(row.as_of_publication).date()
        cut = pd.Timestamp(row.data_cut).date()
        known = self.pipe.con.execute(
            "select reporting_date, cases from core.known_at(?::timestamp) where publication_date = ?",
            [P.run_ts(d), pub]).df().set_index("reporting_date")["cases"]
        known.index = [pd.Timestamp(i).date() for i in known.index]

        def share(day):
            age = min(max((cut - day).days, 0), 28)
            return profile[(profile.weekday == day.isoweekday()) & (profile.age == age)].share.iloc[0]

        total = 0.0
        for age in range(1, 8):
            day = pub - dt.timedelta(days=age)
            s = share(day)
            if s >= 0.2:
                total += known.get(day, 0) / s
            else:                       # no holiday in this window, so the same weekday a week earlier
                back = day - dt.timedelta(days=7)
                while (cut - back).days < 7:
                    back -= dt.timedelta(days=7)
                total += known.get(back, 0) / share(back)
        self.assertAlmostEqual(row.nowcast_7d, total, places=6)


class Holidays(unittest.TestCase):
    def test_easter_2023(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            inc = P.write_incoming(tmp / "in", dt.date(2023, 2, 1), dt.date(2023, 4, 15))
            pipe = P.replay(dt.date(2023, 4, 6), dt.date(2023, 4, 12), inc, content_check=True, holidays=True)
            rep = pipe.report().set_index("run_date")
            # Good Friday and Easter Monday: the files of 8 and 11 April brought nothing new, nor did the
            # weekend files, so the data cut on Tuesday 11 April is Friday's file
            self.assertEqual(pd.Timestamp(rep.data_cut[pd.Timestamp('2023-04-11')]).date(), dt.date(2023, 4, 7))
            self.assertEqual(pd.Timestamp(rep.data_cut[pd.Timestamp('2023-04-12')]).date(), dt.date(2023, 4, 12))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
