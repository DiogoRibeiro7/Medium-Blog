"""The readers, the return rules and the window statistics, on cases with a known answer."""

import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import horizon_core as C  # noqa: E402

HEADER = "DATAFLOW,REF_AREA,FREQ,MEASURE,TIME_PERIOD,OBS_VALUE\r\n"


def sdmx(rows):
    return "﻿" + HEADER + "".join(f"X,{a},{f},M,{t},{v}\r\n" for a, f, t, v in rows)


class Readers(unittest.TestCase):
    def test_sdmx_skips_empty_values_and_a_byte_order_mark(self):
        t = C.read_sdmx(sdmx([("AAA", "M", "2000-12", "1.5"), ("AAA", "M", "2001-12", "")]))
        self.assertEqual(t, {("AAA", "M"): {"2000-12": 1.5}})

    def test_cpi_prefers_december_and_falls_back_to_november(self):
        cpi = C.read_sdmx(
            sdmx(
                [
                    ("AAA", "M", "2000-11", "99"),
                    ("AAA", "M", "2001-11", "101"),
                    ("AAA", "M", "2001-12", "102"),
                    ("AAA", "Q", "2001-Q4", "101.5"),
                ]
            )
        )
        levels, source = C.cpi_year_end(cpi, "AAA")
        self.assertEqual(source, "monthly")
        self.assertEqual(levels, {2000: 99.0, 2001: 102.0})

    def test_cpi_uses_the_fourth_quarter_when_quarters_are_longer(self):
        cpi = C.read_sdmx(
            sdmx(
                [
                    ("BBB", "Q", "1999-Q4", "90"),
                    ("BBB", "Q", "2000-Q4", "95"),
                    ("BBB", "Q", "2000-Q3", "94"),
                    ("BBB", "M", "2000-12", "96"),
                ]
            )
        )
        levels, source = C.cpi_year_end(cpi, "BBB")
        self.assertEqual(source, "quarterly")
        self.assertEqual(levels, {1999: 90.0, 2000: 95.0})

    def test_longest_run(self):
        self.assertEqual(C.longest_run([1990, 1991, 1993, 1994, 1995, 1997]), [1993, 1994, 1995])


class Bonds(unittest.TestCase):
    def test_a_bond_at_its_own_yield_is_priced_at_par(self):
        for y in (0.05, 0.0, -0.005):
            self.assertAlmostEqual(C.par_bond_price(y, y, 9), 1.0)

    def test_an_unchanged_yield_returns_the_coupon(self):
        self.assertAlmostEqual(C.bond_return(0.04, 0.04), 0.04)

    def test_a_rise_in_yield_costs_about_duration_times_the_change(self):
        r = C.bond_return(0.04, 0.05)
        price_change = r - 0.04
        self.assertLess(price_change, -0.06)
        self.assertGreater(price_change, -0.08)


class Markets(unittest.TestCase):
    def test_flat_prices_and_flat_consumer_prices_return_the_dividend_yield(self):
        share = C.read_sdmx(sdmx([("AAA", "M", f"{y}-12", "100") for y in range(2000, 2011)]))
        irlt = C.read_sdmx(sdmx([("AAA", "M", f"{y}-12", "5") for y in range(2000, 2011)]))
        cpi = C.read_sdmx(sdmx([("AAA", "M", f"{y}-12", "100") for y in range(2000, 2011)]))
        m = C.oecd_market("AAA", share, irlt, cpi, 0.03)
        self.assertAlmostEqual(m.stocks[2010] / m.stocks[2000], 1.03**10)
        self.assertAlmostEqual(m.bonds[2010] / m.bonds[2000], 1.05**10)

    def test_inflation_is_taken_out(self):
        share = C.read_sdmx(sdmx([("AAA", "M", f"{y}-12", str(100 * 1.02 ** (y - 2000))) for y in range(2000, 2006)]))
        cpi = C.read_sdmx(sdmx([("AAA", "M", f"{y}-12", str(100 * 1.02 ** (y - 2000))) for y in range(2000, 2006)]))
        m = C.oecd_market("AAA", share, {}, cpi, 0.0)
        self.assertAlmostEqual(m.stocks[2005] / m.stocks[2000], 1.0)
        self.assertEqual(m.bonds, {})

    def test_shiller_adds_a_twelfth_of_the_annual_dividend_each_month(self):
        rows = [
            {"month": f"{y}-{mo:02d}", "price": 100.0, "dividend": 12.0, "cpi": 10.0, "long_rate": 4.0}
            for y in (2000, 2001)
            for mo in range(1, 13)
        ]
        us = C.shiller_market(rows)
        self.assertAlmostEqual(us.stocks[2001] / us.stocks[2000], 1.01**12)
        self.assertAlmostEqual(us.bonds[2001] / us.bonds[2000], 1.04)
        self.assertAlmostEqual(C.shiller_dividend_yield(rows, 2000, 2001), 0.12)

    def test_equal_weight_mix_averages_returns_and_waits_for_enough_markets(self):
        a = {2000: 1.0, 2001: 1.10, 2002: 1.21}
        b = {2000: 1.0, 2001: 0.90, 2002: 0.81}
        c = {2001: 1.0, 2002: 1.0}
        mix = C.equal_weight_mix([a, b, c], minimum=3)
        self.assertEqual(sorted(mix), [2001, 2002])
        self.assertAlmostEqual(mix[2002], 1.0)
        mix2 = C.equal_weight_mix([a, b], minimum=2)
        self.assertAlmostEqual(mix2[2001], 1.0)


class Windows(unittest.TestCase):
    def setUp(self):
        self.level = {2000: 1.0, 2001: 0.8, 2002: 0.9, 2003: 1.2, 2004: 1.1}

    def test_windows_and_their_multiples(self):
        ws = C.windows(self.level, "AAA", 2)
        self.assertEqual([w.start for w in ws], [2000, 2001, 2002])
        self.assertAlmostEqual(ws[0].multiple, 0.9)
        self.assertAlmostEqual(ws[1].multiple, 1.5)

    def test_relative_windows_divide_by_bonds(self):
        m = C.Market("AAA", {2000: 1.0, 2001: 1.2}, {2000: 1.0, 2001: 1.5}, "monthly")
        (w,) = C.relative_windows(m, 1)
        self.assertAlmostEqual(w.multiple, 0.8)

    def test_episodes_are_runs_of_losing_starts_per_market(self):
        ws = [
            C.Window("A", 2000, 1, 0.9),
            C.Window("A", 2001, 1, 0.8),
            C.Window("A", 2002, 1, 1.1),
            C.Window("A", 2003, 1, 0.7),
            C.Window("B", 2001, 1, 0.9),
        ]
        self.assertEqual(C.episodes(ws), [("A", 2000, 2001), ("A", 2003, 2003), ("B", 2001, 2001)])

    def test_summary(self):
        ws = [C.Window("A", 2000 + k, 1, m) for k, m in enumerate([0.8, 0.9, 1.2, 1.5])]
        s = C.summarise(ws, 1)
        self.assertEqual(s.n, 4)
        self.assertAlmostEqual(s.p_loss, 0.5)
        self.assertAlmostEqual(s.median_loss, 0.15)
        self.assertAlmostEqual(s.expected_shortfall, 0.3 / 4)
        self.assertEqual(s.worst.start, 2000)
        self.assertEqual(s.episodes, 1)

    def test_quantile_matches_numpy(self):
        rng = random.Random(1)
        v = [rng.random() for _ in range(37)]
        for q in (0.05, 0.25, 0.5, 0.95):
            self.assertAlmostEqual(C.quantile(v, q), float(np.quantile(v, q)))


class VarianceRatio(unittest.TestCase):
    def test_alternating_returns_revert(self):
        logs = [0.1, -0.1] * 30
        self.assertLess(C.variance_ratio(logs, 2), 0.05)

    def test_trending_returns_persist(self):
        logs = [math.sin(k / 10) for k in range(200)]
        self.assertGreater(C.variance_ratio(logs, 5), 4.0)

    def test_levels_from_logs_round_trip(self):
        level = {2000: 1.0, 2001: 1.1, 2002: 0.99}
        back = C.level_from_logs(C.log_returns(level), 2000)
        for y in level:
            self.assertAlmostEqual(back[y], level[y])


if __name__ == "__main__":
    unittest.main()
