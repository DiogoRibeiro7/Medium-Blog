"""Checks for Part 6: the Part 5 versions are unchanged, the new files join the old ones without a gap
or a difference, and the graded check finds short files, lets later files make them up, and keeps every
run reproducible."""
from __future__ import annotations

import datetime as dt
import itertools
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part4"))
import pipeline as P  # noqa: E402

PART5_CODES = ["9adf5e8d8533", "4b25818af673", "68736d72ccbe", "364faca191d7",
               "f9a7b1b6aa1d", "fa2957699aad", "9b1d9244bd1e", "55f281d88110"]


class Frozen(unittest.TestCase):
    def test_part5_versions_unchanged(self):
        got = [P.Pipeline(content_check=c, holidays=h, fallback=f).code_version
               for c, h, f in itertools.product((False, True), (False, True), ("level", "last_week"))]
        self.assertEqual(got, PART5_CODES)

    def test_calendar_extends(self):
        self.assertEqual((P.easter(2025), P.easter(2026)), (dt.date(2025, 4, 20), dt.date(2026, 4, 5)))
        self.assertIn(dt.date(2026, 5, 25), P.german_holidays(2026, 2026))      # Whit Monday 2026


class NewFiles(unittest.TestCase):
    def test_new_files_join_the_old(self):
        s3 = pd.read_csv(P.PART3_DATA / "rki_state.csv.gz", parse_dates=["publication", "date"])
        s6 = pd.read_csv(P.PART6_DATA / "rki_state_2024_2026.csv.gz", parse_dates=["publication", "date"])
        pubs = pd.DatetimeIndex(sorted(s6.publication.unique()))
        missing = {pd.Timestamp("2024-02-14"), pd.Timestamp("2026-05-26")}   # no file in the archive on these days
        self.assertEqual(list(pubs), [d for d in pd.date_range("2024-01-01", "2026-11-10") if d not in missing])
        a = s3[s3.publication == "2024-01-01"].sort_values(["state", "date"]).reset_index(drop=True)
        b = s6[s6.publication == "2024-01-01"].sort_values(["state", "date"]).reset_index(drop=True)
        self.assertTrue(a.equals(b))
        tmp = Path(tempfile.mkdtemp())
        try:
            inc = P.write_incoming(tmp / "in", dt.date(2023, 12, 20), dt.date(2024, 1, 10))
            self.assertEqual([i.publication_date for i in inc],
                             [d.date() for d in pd.date_range("2023-12-20", "2024-01-10")])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Graded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        inc = P.write_incoming(cls.tmp / "in", dt.date(2023, 6, 1), dt.date(2023, 9, 3))
        cls.pipe = P.replay(dt.date(2023, 8, 20), dt.date(2023, 8, 31), inc, holidays=True, fallback="last_week",
                            graded=0.5, holiday_years=(2020, 2027))
        cls.rep = cls.pipe.report().set_index("run_date")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_short_files_and_repayment(self):
        miss = self.rep.missing_cases
        self.assertEqual(miss[pd.Timestamp("2023-08-24")], 0)        # nothing short yet
        self.assertGreater(miss[pd.Timestamp("2023-08-28")], 0)      # Friday and Saturday files brought nothing
        self.assertEqual(miss[pd.Timestamp("2023-08-29")], 0)        # Tuesday's file made it up
        sql = self.pipe.report_sql.split("past_runs as (")[0].rstrip().rstrip(",") + "\nselect q, r, w from gw order by q"
        gw = self.pipe.con.execute(sql, {"as_of": P.run_ts(dt.date(2023, 8, 28))}).df().set_index("q")
        self.assertEqual(gw.loc[pd.Timestamp("2023-08-25"), "r"], 0)
        self.assertEqual(gw.loc[pd.Timestamp("2023-08-25"), "w"], 0)
        self.assertEqual(gw.loc[pd.Timestamp("2023-08-22"), "w"], 1)

    def test_every_run_reproduces_exactly(self):
        day = dt.date(2023, 8, 20)
        while day <= dt.date(2023, 8, 31):
            pr, rr = self.pipe.reproduce(day)
            ps, rs = self.pipe.stored(day)
            self.assertTrue(P.same(pr, ps) and P.same(rr, rs), day)
            day += dt.timedelta(days=1)


if __name__ == "__main__":
    unittest.main()
