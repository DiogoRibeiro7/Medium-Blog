"""Checks for Part 9: the producers and their calendars are assembled as the article says, no prediction uses
a later day, the change-point tools find what they should in data with a known answer, and the announced
changes line up as reported."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part9"))
import learned as L  # noqa: E402


class Producers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = L.frame()

    def test_extract(self):
        g = pd.read_csv(L.DATA / "jhu_global_rows.csv.gz", keep_default_na=False, na_values=[""])
        self.assertEqual(len(g), 98647)
        self.assertEqual(g.date.nunique(), 1081)

    def test_groups_and_names(self):
        n = self.df.groupby("group").producer.nunique()
        self.assertEqual((n["US states"], n["provinces"], n["countries"]), (51, 22, 193))
        self.assertFalse(self.df.duplicated(["producer", "date"]).any())
        # the US state and the country of the same name are two producers
        self.assertEqual(set(self.df[self.df.producer == "Georgia"].group), {"US states"})
        self.assertEqual(set(self.df[self.df.producer == "Georgia (country)"].group), {"countries"})

    def test_calendars(self):
        d = self.df.set_index(["producer", "date"])
        self.assertEqual(d.loc[("Ohio", pd.Timestamp("2021-07-05")), "kind"], "holiday")  # Independence Day observed
        self.assertEqual(d.loc[("Ohio", pd.Timestamp("2021-07-06")), "kind"], "after")
        # Saudi Arabia's legal weekend is Friday and Saturday
        self.assertEqual(d.loc[("Saudi Arabia", pd.Timestamp("2021-03-05")), "kind"], "weekend")
        self.assertEqual(d.loc[("Saudi Arabia", pd.Timestamp("2021-03-07")), "kind"], "working")
        self.assertEqual(d.loc[("Quebec", pd.Timestamp("2021-06-24")), "kind"], "holiday")  # Saint-Jean-Baptiste
        self.assertEqual(d.loc[("Ontario", pd.Timestamp("2021-06-24")), "kind"], "working")

    def test_unchanged_and_eligible(self):
        d = self.df.set_index(["producer", "date"])
        r = d.loc[("Spain", pd.Timestamp("2020-07-04"))]
        self.assertEqual(r.flat, 1.0)
        self.assertEqual(r["first"], r.prev)
        e = self.df[self.df.eligible]
        self.assertTrue((e.count28 >= L.MIN_COUNT).all())
        self.assertTrue(e.flat.notna().all())


class NoLookAhead(unittest.TestCase):
    """A prediction for a date must not change when everything after that date is removed."""

    @classmethod
    def setUpClass(cls):
        cls.df = L.frame()
        cls.cut = pd.Timestamp("2021-10-15")
        cls.early = cls.df[cls.df.date <= cls.cut].copy()

    def test_weekday_rates_and_published(self):
        full, _, _ = L.weekday_rates(self.df)
        part, _, _ = L.weekday_rates(self.early)
        idx = self.early.index
        np.testing.assert_allclose(full[idx].to_numpy(), part[idx].to_numpy())
        np.testing.assert_allclose(L.published(self.df)[idx].to_numpy(), L.published(self.early)[idx].to_numpy())

    def test_holiday_effects(self):
        full = L.predictions(self.df)
        part = L.predictions(self.early)
        idx = part.index
        np.testing.assert_allclose(full.loc[idx, "learned"].to_numpy(), part["learned"].to_numpy())

    def test_change_points(self):
        sub = self.df[self.df.producer.isin(["Florida", "Spain", "Ontario"])]
        early = sub[sub.date <= self.cut]
        _, pooled, _ = L.weekday_rates(self.df)
        a = L.bocpd_joint(sub, 1 / 365, pooled)
        b = L.bocpd_joint(early, 1 / 365, pooled)
        np.testing.assert_allclose(a[b.index].to_numpy(), b.to_numpy())
        a = L.bocpd_rates(sub, 1 / 52, pooled)
        b = L.bocpd_rates(early, 1 / 52, pooled)
        np.testing.assert_allclose(a[b.index].to_numpy(), b.to_numpy())


class Tools(unittest.TestCase):
    def test_segment_finds_a_step(self):
        rng = np.random.default_rng(1)
        y = np.r_[rng.random(60) < 0.03, rng.random(60) < 0.95].astype(float)
        segs = L.segment(np.arange(120) * 7, y)
        self.assertEqual(len(segs), 2)
        self.assertLessEqual(abs(segs[1][0] - 60), 2)

    def test_segment_respects_minimum_length(self):
        y = np.zeros(100)
        y[50:52] = 1  # two odd weeks are not a regime
        self.assertEqual(len(L.segment(np.arange(100) * 7, y)), 1)

    def test_joint_change_points_learn_faster_than_one_day_at_a_time(self):
        dates = pd.date_range("2021-01-04", periods=280)
        y = (dates >= "2021-06-07").astype(float)  # every day of the week stops at once
        df = pd.DataFrame({"producer": "x", "group": "countries", "date": dates, "wd": dates.dayofweek,
                           "kind": "working", "eligible": True, "flat": y})
        pooled = pd.Series(0.2, index=df.index)
        joint = L.bocpd_joint(df, 1 / 365, pooled)
        single = L.bocpd_rates(df, 1 / 52, pooled)
        after = df.index[(df.date >= "2021-06-07") & (df.date < "2021-06-21")]
        self.assertGreater(joint[after].mean(), single[after].mean())
        self.assertGreater(joint[df.index[-1]], 0.9)

    def test_cluster_boot(self):
        v = pd.Series(np.r_[np.zeros(50), np.ones(50)])
        g = pd.Series(np.repeat(np.arange(10), 10))
        lo, hi = L.cluster_boot(v, g, n=500)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


class Announced(unittest.TestCase):
    def test_offsets(self):
        df = L.frame()
        _, _, ch = L.changes(df)
        doc = L.documented(ch)
        self.assertEqual(len(doc), 15)
        self.assertTrue(doc.detected.notna().all())
        self.assertTrue(doc.offset.between(-1, 3).all())
        self.assertEqual(int((doc.offset == 0).sum()), 9)


if __name__ == "__main__":
    unittest.main()
