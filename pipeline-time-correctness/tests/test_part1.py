"""Checks for Part 1: published numbers reproduced, and invariants of each construction."""
from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part1"))
import history as h  # noqa: E402


class Insurance(unittest.TestCase):
    def test_reproduces_mack_1993(self):
        tri = h.taylor_ashe()
        _, ult = h.chain_ladder_sql(tri)
        self.assertEqual(round(ult["reserve"].sum()), 18_680_856)  # Mack (1993), Table 2: 18,681 (1000s)
        se, se_total = h.mack_standard_errors(h.triangle_matrix(tri))
        self.assertEqual(round(se_total), 2_447_095)
        self.assertEqual(round(100 * se_total / ult["reserve"].sum()), 13)  # Table 3: 13%
        published = [80, 26, 19, 27, 29, 26, 22, 23, 29]  # Table 3, accident years 2-10
        ours = [round(100 * s / r) for s, r in zip(se[1:], ult["reserve"].to_numpy()[1:])]
        self.assertEqual(ours, published)

    def test_sql_matches_matrix_chain_ladder(self):
        tri = h.taylor_ashe()
        factors, ult = h.chain_ladder_sql(tri)
        c = h.triangle_matrix(tri)
        n = c.shape[0]
        f = [np.nansum(c[: n - 1 - k, k + 1]) / np.nansum(c[: n - 1 - k, k]) for k in range(n - 1)]
        np.testing.assert_allclose(factors["f"].to_numpy(), f, rtol=1e-12)


class Epidemiology(unittest.TestCase):
    def setUp(self):
        self.cases = h.hus_cases()

    def test_data(self):
        self.assertEqual(len(self.cases), 630)
        self.assertTrue((self.cases["delay"].between(0, h.MAX_DELAY)).all())

    def test_known_by_end_is_final(self):
        end = self.cases["reported"].max()
        known = h.known_by(self.cases, end)
        final = h.final_counts(self.cases).reindex(known.index, fill_value=0)
        pd.testing.assert_series_equal(known, final, check_names=False, check_freq=False)

    def test_nowcast_never_below_known_and_mature_days_complete(self):
        as_of = pd.Timestamp("2011-06-05")
        nc = h.nowcast_chain_ladder(self.cases, as_of)
        self.assertTrue((nc["nowcast"] >= nc["known"] - 1e-9).all())
        old = nc.index <= as_of - pd.Timedelta(days=h.MAX_DELAY)
        np.testing.assert_allclose(nc.loc[old, "completeness"], 1.0)


class TemporalDatabase(unittest.TestCase):
    def test_exactly_one_row_per_point_of_the_plane(self):
        con = h.bitemporal_connection()
        for v in range(0, 100, 3):
            for k in range(0, 120, 3):
                valid, known = date(2026, 1, 1) + timedelta(days=v), date(2025, 12, 20) + timedelta(days=k)
                n = con.execute("select count(*) from price where valid_from <= $v and $v < valid_to "
                                "and recorded_from <= $k and $k < recorded_to", {"v": valid, "k": known}).fetchone()[0]
                self.assertEqual(n, 1, (valid, known))

    def test_restatement(self):
        con = h.bitemporal_connection()
        inv = h.invoices(con, today=date(2026, 4, 1))
        self.assertAlmostEqual(float(inv["restatement"].astype(float).sum()), 105.0)
        self.assertEqual(h.price_as_of(con, date(2026, 2, 20), date(2026, 2, 20)), 10.0)
        self.assertEqual(h.price_as_of(con, date(2026, 2, 20), date(2026, 4, 1)), 11.5)


class Warehouse(unittest.TestCase):
    def test_type1_moves_revenue_between_regions_only(self):
        wh = h.simulate_warehouse()
        rev = h.regional_revenue_2024(wh)
        totals = rev.sum()
        self.assertAlmostEqual(totals["type2"], totals["type1_run_2025_01_05"], places=6)
        self.assertAlmostEqual(totals["type2"], totals["type1_run_2026_01_05"], places=6)

    def test_type2_history_has_no_gaps_or_overlaps(self):
        wh = h.simulate_warehouse()
        for _, g in wh.customers.sort_values("valid_from").groupby("customer_id"):
            self.assertTrue((g["valid_from"].to_numpy()[1:] == g["valid_to"].to_numpy()[:-1]).all())


class Streaming(unittest.TestCase):
    def test_accumulation_modes(self):
        panes, dropped = h.stream()
        self.assertEqual(dropped, [(9, 100, 380)])
        accepted = h.true_window_sums([e for e in h.EVENTS if e not in dropped])
        discarding = panes.groupby("window")["discarding"].sum().to_dict()
        self.assertEqual(discarding, accepted)
        retracting = (panes.groupby("window")["accumulating"].sum()
                      + panes.groupby("window")["retraction"].sum()).to_dict()
        self.assertEqual({k: int(v) for k, v in retracting.items()}, accepted)
        accumulating = panes.groupby("window")["accumulating"].sum().to_dict()
        self.assertEqual(accumulating[0], accepted[0] + 5)  # the first pane is counted twice

    def test_processing_time_windows_keep_the_total(self):
        self.assertEqual(sum(h.processing_time_sums().values()), sum(v for v, _, _ in h.EVENTS))


if __name__ == "__main__":
    unittest.main()
