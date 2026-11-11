"""Checks for Part 4: the pipeline lands each file once, never lets later files into earlier runs,
recomputes any run exactly, handles late and republished files, and computes what it says."""
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

FIRST_FILE, LAST_FILE = dt.date(2021, 4, 2), dt.date(2021, 8, 15)
FIRST_RUN, LAST_RUN = dt.date(2021, 6, 1), dt.date(2021, 7, 15)
LATE_DAY = dt.date(2021, 7, 1)


def days(a: dt.date, b: dt.date):
    while a <= b:
        yield a
        a += dt.timedelta(days=1)


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.incoming = P.write_incoming(cls.tmp / "in", FIRST_FILE, LAST_FILE, {LATE_DAY: 2})
        cls.pipe = P.replay(FIRST_RUN, LAST_RUN, cls.incoming)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_run_reproduces_exactly(self):
        for d in days(FIRST_RUN, LAST_RUN):
            pr, rr = self.pipe.reproduce(d)
            ps, rs = self.pipe.stored(d)
            self.assertTrue(P.same(pr, ps), d)
            self.assertTrue(P.same(rr, rs), d)

    def test_a_second_replay_gives_the_same_mart(self):
        other = P.replay(FIRST_RUN, LAST_RUN, self.incoming)
        self.assertTrue(P.same(other.report(), self.pipe.report()))

    def test_the_late_file_makes_its_day_stale_and_stays_out_of_it(self):
        row = self.pipe.stored(LATE_DAY)[1].iloc[0]
        self.assertTrue(bool(row.stale))
        self.assertEqual(pd.Timestamp(row.as_of_publication).date(), LATE_DAY - dt.timedelta(days=1))
        loaded = self.pipe.con.execute("select loaded_at from landing.files where publication_date = ?",
                                       [LATE_DAY]).fetchone()[0]
        self.assertEqual(loaded, P.run_ts(LATE_DAY + dt.timedelta(days=2)))

    def test_landing_the_same_file_again_is_a_no_op(self):
        item = self.incoming[10]
        n = self.pipe.con.execute("select count(*) from landing.cases").fetchone()[0]
        self.assertFalse(self.pipe.land(item, dt.datetime(2030, 1, 1)))
        self.assertEqual(self.pipe.con.execute("select count(*) from landing.cases").fetchone()[0], n)

    def test_report_arithmetic(self):
        d = dt.date(2021, 7, 10)
        profile, row = self.pipe.stored(d)
        row = row.iloc[0]
        pub = pd.Timestamp(row.as_of_publication).date()
        known = self.pipe.con.execute(
            "select reporting_date, cases from core.known_at(?::timestamp) where publication_date = ?",
            [P.run_ts(d), pub]).df().set_index("reporting_date")["cases"]
        total, now = 0, 0.0
        for age in range(1, 8):
            day = pub - dt.timedelta(days=age)
            k = float(known.get(pd.Timestamp(day), 0))
            pr = profile[(profile.weekday == day.isoweekday()) & (profile.age == age)].iloc[0]
            total += k
            now += k / pr.share if pr.share >= 0.2 else pr.mean_full
        self.assertEqual(int(row.published_7d), int(total))
        self.assertAlmostEqual(row.nowcast_7d, now, places=6)

    def test_profile_is_a_share(self):
        prof = self.pipe.con.execute("select * from core.delay_profile").df()
        self.assertTrue((prof[prof.age == 28].share.round(12) == 1.0).all())
        self.assertTrue((prof.share >= 0).all())


class LaterFiles(unittest.TestCase):
    """A future file cannot change an earlier run; a republished file is a new version."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.incoming = P.write_incoming(self.tmp / "in", FIRST_FILE, dt.date(2021, 7, 10))
        self.pipe = P.replay(dt.date(2021, 6, 1), dt.date(2021, 7, 10), self.incoming)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_future_file_does_not_leak(self):
        before = self.pipe.stored(dt.date(2021, 7, 5))
        fake = self.tmp / "rki_2021-07-11.csv"
        pd.DataFrame({"state": [5], "reporting_date": ["2021-07-04"], "cases": [10_000_000]}).to_csv(fake, index=False)
        self.pipe.land(P.Incoming(fake, dt.date(2021, 7, 11), dt.datetime(2021, 7, 11, 7)), dt.datetime(2021, 7, 11, 8))
        after = self.pipe.reproduce(dt.date(2021, 7, 5))
        self.assertTrue(P.same(before[0], after[0]))
        self.assertTrue(P.same(before[1], after[1]))

    def test_republished_file_is_a_new_version(self):
        item = [i for i in self.incoming if i.publication_date == dt.date(2021, 7, 10)][0]
        df = pd.read_csv(item.path)
        df.loc[0, "cases"] += 1000
        df.to_csv(item.path, index=False)
        republished = P.Incoming(item.path, item.publication_date, dt.datetime(2021, 7, 11, 7))
        landed = self.pipe.load_incoming(self.incoming + [republished], dt.datetime(2021, 7, 11, 8))
        self.assertEqual(landed, [item.path.name])
        versions = self.pipe.con.execute("select count(*) from landing.files where file_name = ?",
                                         [item.path.name]).fetchone()[0]
        self.assertEqual(versions, 2)
        q = "select sum(cases) from core.known_at(?::timestamp) where publication_date = date '2021-07-10'"
        old = self.pipe.con.execute(q, [dt.datetime(2021, 7, 10, 8)]).fetchone()[0]
        new = self.pipe.con.execute(q, [dt.datetime(2021, 7, 11, 8)]).fetchone()[0]
        self.assertEqual(new - old, 1000)
        pr, rr = self.pipe.reproduce(dt.date(2021, 7, 10))
        ps, rs = self.pipe.stored(dt.date(2021, 7, 10))
        self.assertTrue(P.same(rr, rs))


class TrainingRows(unittest.TestCase):
    def test_feature_at_28_days_matches_a_direct_lookup(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            inc = P.write_incoming(tmp / "in", FIRST_FILE, LAST_FILE)
            pipe = P.Pipeline()
            pipe.load_incoming(inc, dt.datetime(2021, 8, 16, 8))
            tr = pipe.training_rows(dt.datetime(2021, 8, 16, 8), dt.date(2021, 6, 1))
            row = tr[(pd.to_datetime(tr.run_date) == pd.Timestamp("2021-06-07")) & (tr.state == 9)].iloc[0]
            total = 0
            for age in range(1, 8):
                day = dt.date(2021, 6, 7) - dt.timedelta(days=age)
                pub = day + dt.timedelta(days=28)
                total += pipe.con.execute(
                    "select coalesce(sum(cases), 0) from landing.cases where publication_date = ? and state = 9 "
                    "and reporting_date = ?", [pub, day]).fetchone()[0]
            self.assertEqual(int(row.x_final), int(total))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
