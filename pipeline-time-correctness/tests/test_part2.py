"""Checks for Part 2: each mechanism against a case where the answer is known in closed form,
and the SQL against an independent implementation of the same rule."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part2"))
import mechanism as m  # noqa: E402


class Delays(unittest.TestCase):
    def test_pmf_is_a_distribution(self):
        for d in m.SCENARIOS + (m.BACKLOG,):
            p = d.pmf()
            self.assertTrue(np.all(p >= 0))
            self.assertAlmostEqual(p.sum(), 1.0, places=12)
            self.assertTrue(np.all(np.diff(d.F()) >= 0))

    def test_median_is_where_the_cdf_is_one_half(self):
        for d in m.SCENARIOS:
            self.assertAlmostEqual(float(d.cdf(d.median)[0]), 0.5, places=12)


class Truncation(unittest.TestCase):
    def test_no_growth_no_delay_means_no_change(self):
        instant = m.Delay("instant", 1e-9, 0.01)
        self.assertAlmostEqual(m.apparent_weekly_change(instant, 0.0), 0.0, places=9)
        self.assertAlmostEqual(m.apparent_weekly_change(instant, 0.1), math.exp(0.7) - 1, places=9)

    def test_flat_growth_rate_is_a_root(self):
        for d in m.SCENARIOS:
            g = m.growth_that_looks_flat(d)
            self.assertAlmostEqual(m.apparent_weekly_change(d, g), 0.0, places=9)

    def test_monte_carlo_matches_expectation(self):
        for d in m.SCENARIOS:
            r = m.truncation_bias_against_noise(d)
            expected = 7000 * m.window_shortfall(d, 7)
            self.assertLess(abs(r["mean_shortfall"] - expected), 4 * r["sd_final"] / math.sqrt(2000) + 5)


class Watermarks(unittest.TestCase):
    def test_on_time_share_rises_with_slack(self):
        for d in m.SCENARIOS:
            vals = [m.on_time_completeness(d, s) for s in (0, 1, 3, 7, 14)]
            self.assertTrue(all(a < b for a, b in zip(vals, vals[1:])))

    def test_slack_for_inverts_the_curve(self):
        s = m.slack_for(m.MEDIUM, 0.9)
        self.assertAlmostEqual(m.on_time_completeness(m.MEDIUM, s), 0.9, places=6)


class Nowcast(unittest.TestCase):
    def test_nowcast_is_exact_when_factors_are_exact(self):
        # A deterministic triangle with the same proportions every day: the chain ladder recovers it.
        p = m.MEDIUM.pmf()
        n = np.outer(np.full(120, 1000.0), p)
        C = n.cumsum(axis=1)
        ages = np.arange(15)
        np.testing.assert_allclose(m.nowcast_by_age(C, 100, ages), np.full(15, 1000.0), rtol=1e-12)

    def test_nowcast_beats_raw_with_stable_delays(self):
        waits = np.arange(15)
        r = m.compare_policies(m.simulate_reporting(), range(100, 400), waits)
        self.assertTrue(np.all(r.nowcast < r.raw))


class HighWaterMark(unittest.TestCase):
    def test_vectorized_rule_matches_a_loop(self):
        rng = np.random.default_rng(0)
        stamp = rng.integers(0, 10_000, 400)
        visible = stamp + rng.integers(0, 3_000, 400)
        interval = 700
        loaded = np.zeros(400, bool)
        mark = np.iinfo(np.int64).min
        run = 0
        while run <= visible.max() + interval:
            new = (visible <= run) & (visible > run - interval) & (stamp > mark)
            loaded |= new
            if new.any():
                mark = max(mark, stamp[new].max())
            run += interval
        np.testing.assert_array_equal(m.hwm_loaded(stamp, visible, interval), loaded)

    def test_sql_matches_vectorized(self):
        shop = m.simulate_shop()
        o = shop.orders
        np_rows = int(m.hwm_loaded(o.event_ts.to_numpy(), o.arrival_ts.to_numpy(), m.DAY, m.RUN_HOUR * m.HOUR).sum())
        self.assertEqual(m.hwm_daily_sql(o), np_rows)

    def test_arrival_time_mark_loses_nothing(self):
        o = m.simulate_shop().orders
        arr = o.arrival_ts.to_numpy()
        self.assertTrue(m.hwm_loaded(arr, arr, m.HOUR).all())


class ShopPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = m.run_shop(m.simulate_shop())

    def test_upsert_and_log_end_at_the_truth(self):
        t = self.pipeline.totals.set_index("pattern")
        self.assertEqual(t.loc["upsert", "revenue"], t.loc["truth", "revenue"])
        self.assertEqual(t.loc["version_log", "revenue"], t.loc["truth", "revenue"])
        self.assertGreater(t.loc["insert_only", "revenue"], t.loc["truth", "revenue"])

    def test_only_the_log_reproduces_what_was_sent(self):
        sent = self.pipeline.sent.set_index(["pattern", "day"])
        regen = self.pipeline.regenerated.set_index(["pattern", "day"])
        self.assertTrue((sent.loc["version_log"] == regen.loc["version_log"]).all().all())
        self.assertFalse((sent.loc["upsert"] == regen.loc["upsert"]).all().all())

    def test_replace_backfill_is_idempotent_and_append_doubles(self):
        daily = self.pipeline.daily.set_index("event_day")
        truth = self.pipeline.truth.set_index("day")
        self.assertEqual(daily.index.tolist(), list(range(m.SHOP_DAYS)))  # one row per day
        appended = self.pipeline.daily_appended.set_index("event_day")
        days = list(range(0, m.BACKFILL_DAY - 1))
        ratio = appended.orders.loc[days].sum() / daily.orders.loc[days].sum()
        self.assertGreater(ratio, 1.9)
        # after the backfill, the recomputed days are at least as complete as when sent
        frozen = self.pipeline.frozen.set_index("event_day")
        self.assertTrue((daily.orders.loc[days] >= frozen.orders.loc[days]).all())
        self.assertTrue((daily.orders.loc[days] <= truth.orders.loc[days]).all())


class Features(unittest.TestCase):
    def test_as_of_training_is_calibrated(self):
        c = m.calibration(m.count_features(seed=5), m.count_features(seed=7))
        self.assertAlmostEqual(c["trained_as_of"], 1.0, delta=0.02)
        self.assertLess(c["trained_on_final"], 0.8)

    def test_auc_of_a_perfect_and_a_constant_score(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(m.auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(m.auc(y, np.zeros(4)), 0.5)

    def test_point_in_time_join_uses_the_plan_on_the_snapshot_day(self):
        customers, plans = m.simulate_plan_history(n=2000, seed=11)
        type1, as_of = m.churn_features(customers, plans)
        merged = customers.merge(plans, on="customer_id")
        valid = merged[(merged.valid_from <= merged.snapshot) & (merged.snapshot < merged.valid_to)]
        expected = (valid.sort_values("customer_id").plan == "basic").astype(int).to_numpy()
        np.testing.assert_array_equal(as_of.basic.to_numpy(), expected)
        current = plans[plans.valid_to == 999_999].sort_values("customer_id")
        np.testing.assert_array_equal(type1.basic.to_numpy(), (current.plan == "basic").astype(int).to_numpy())

    def test_leaky_join_looks_better_offline_and_is_worse_in_production(self):
        r = m.leakage_experiment()
        self.assertGreater(r["type1_offline_auc"], r["as_of_offline_auc"])
        self.assertLess(r["type1_production_auc"], r["as_of_production_auc"])


if __name__ == "__main__":
    unittest.main()
