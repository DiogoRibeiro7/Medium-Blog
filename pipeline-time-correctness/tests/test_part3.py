"""Checks for Part 3: the extracted vintage tables are internally consistent, the as-of and
first-release SQL agree with plain pandas, and the chain ladder behaves as its algebra says."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part3"))
import realdata as r  # noqa: E402


class RkiVintages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.melde = r.load_vintages("melde")
        cls.final = r.final_melde()

    def test_signs_of_flags(self):
        self.assertTrue((self.melde.new.fillna(0) >= 0).all().all())
        self.assertTrue((self.melde.removed.fillna(0) <= 0).all().all())
        self.assertTrue((self.melde.V.fillna(0) >= 0).all().all())

    def test_nothing_is_known_on_the_reporting_date_itself(self):
        # the files hold what had reached the RKI by 00:00, so a date is first counted the next day
        s = self.melde.at_age(0).loc["2021-04-12":"2023-08-31"]
        self.assertEqual(float(s.sum()), 0.0)

    def test_counts_after_120_days_are_close_to_the_last_file(self):
        a120 = self.melde.at_age(r.MAX_AGE).loc["2021-04-12":"2023-08-31"]
        ratio = self.final.reindex(a120.index).sum() / a120.sum()
        self.assertAlmostEqual(ratio, 1.0, delta=0.005)

    def test_seven_day_sum_uses_the_seven_dates_before(self):
        sd = r.seven_day_sums(self.melde, self.final)
        p = pd.Timestamp("2022-03-01")
        dates = pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))
        self.assertEqual(sd.at[p, "published"], self.melde.V.loc[p, dates].fillna(0).sum())
        self.assertEqual(sd.at[p, "final"], self.final.reindex(dates).sum())

    def test_snapshot_of_last_day_matches_recent_vintage(self):
        snap = r.load_snapshots()
        last = r.RKI_LAST
        d = last - pd.Timedelta(days=30)
        self.assertEqual(snap.at[d, last], self.melde.V.at[last, d])


class HubTriangle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tri = r.load_hub_triangle()

    def test_final_is_the_sum_of_all_changes(self):
        np.testing.assert_allclose(self.tri.cum[:, -1] + self.tri.inc[:, -1], self.tri.final)

    def test_known_far_after_is_the_80_day_value(self):
        t = self.tri.index("2022-01-15")
        self.assertEqual(self.tri.known7(t, t + 500), self.tri.final7(t, "80"))

    def test_point_nowcast_falls_back_to_the_mean(self):
        q = r.load_hub_nowcasts()
        rki = q[q.model == "RKI-weekly_report"]
        self.assertTrue(rki["point"].notna().all())
        self.assertTrue((rki["point"] == rki["mean"]).all())


class ChainLadder(unittest.TestCase):
    def test_exact_on_a_deterministic_triangle(self):
        p = np.array([0.2, 0.3, 0.25, 0.15, 0.1])
        C = np.outer(np.full(60, 1000.0), p).cumsum(axis=1)
        T = 50
        Cm = C[:T + 1].copy()
        for t in range(T - 4, T + 1):
            Cm[t, T - t + 1:] = np.nan
        est = r.chain_ladder_daily(Cm, T, 14, 4)
        np.testing.assert_allclose(est[T - 4:T + 1], np.full(5, 1000.0), rtol=1e-12)


class Gdp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = r.gdp_connection()

    def test_as_of_sql_matches_pandas(self):
        lv = r.gdp_levels("EA")
        as_of = pd.Timestamp("2015-03-01")
        vintage = max(c for c in lv.columns if c <= as_of)
        sql = self.con.execute(r.AS_OF_SQL, {"id": "EA", "as_of": as_of.date()}).df().set_index("time")["value"]
        expected = lv[vintage].dropna()
        np.testing.assert_allclose(sql.loc[expected.index].to_numpy(), expected.to_numpy())

    def test_first_release_sql_matches_pandas(self):
        lv = r.gdp_levels("JP")
        sql = self.con.execute(r.FIRST_RELEASE_SQL, {"id": "JP"}).df().set_index("time")["first_value"]
        for t in [pd.Timestamp("2005-01-01"), pd.Timestamp("2011-01-01"), pd.Timestamp("2020-04-01")]:
            self.assertAlmostEqual(sql.loc[t], lv.loc[t].dropna().iloc[0])

    def test_releases_start_with_the_first_vintage_containing_the_quarter(self):
        rel = r.releases(r.gdp_growth("US"))
        t = pd.Timestamp("2010-01-01")
        g = r.gdp_growth("US").loc[t].dropna()
        self.assertEqual(rel[t][0], g.iloc[0])
        self.assertEqual(rel[t][-1], g.iloc[-1])

    def test_recession_quarters(self):
        s = pd.Series([1, -1, -1, 1, -1, 1, -2, -3, -1], index=pd.period_range("2000Q1", periods=9, freq="Q"))
        self.assertEqual(len(r.recession_quarters(s)), 5)


if __name__ == "__main__":
    unittest.main()
