"""The replay on cases with a known answer."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lump_core as lc  # noqa: E402

START = lc.month("2000-01")


def steady(growth, n=60, first=START):
    return {first + k: (1 + growth) ** k for k in range(n)}


def flat_cash(index, rate):
    return dict.fromkeys(index, rate)


class Months(unittest.TestCase):
    def test_month_round_trip(self):
        self.assertEqual(lc.label(lc.month("1987-10")), "1987-10")
        self.assertEqual(lc.month("2000-01-01"), lc.month("2000-01"))


class Replay(unittest.TestCase):
    def test_flat_prices_and_no_interest_give_the_same_value(self):
        idx = steady(0.0)
        lump, spread = lc.replay(idx, flat_cash(idx, 0.0), min(idx), 12)
        self.assertAlmostEqual(lump, 1.0)
        self.assertAlmostEqual(spread, 1.0)

    def test_rising_prices_favour_the_lump_sum(self):
        idx = steady(0.01)
        lump, spread = lc.replay(idx, flat_cash(idx, 0.0), min(idx), 12)
        self.assertAlmostEqual(lump, 1.01**12)
        expected = math.fsum(1.01 ** (12 - k) for k in range(12)) / 12
        self.assertAlmostEqual(spread, expected)
        self.assertGreater(lump, spread)

    def test_falling_prices_favour_spreading(self):
        idx = steady(-0.01)
        lump, spread = lc.replay(idx, flat_cash(idx, 0.0), min(idx), 12)
        self.assertLess(lump, spread)

    def test_interest_on_waiting_cash_is_invested_in_the_last_instalment(self):
        idx = steady(0.0)
        _, spread = lc.replay(idx, flat_cash(idx, 12.0), min(idx), 3)
        # 1/3 invested at once; 2/3 earn 1% for a month, then 1/3 invested; the rest earns another 1% and is invested
        left = (2 / 3) * 1.01
        left = (left - 1 / 3) * 1.01
        self.assertAlmostEqual(spread, 1 / 3 + 1 / 3 + left)

    def test_one_instalment_is_the_lump_sum(self):
        idx = steady(0.02)
        lump, spread = lc.replay(idx, flat_cash(idx, 5.0), min(idx), 1)
        self.assertAlmostEqual(lump, spread)

    def test_start_needs_prices_and_cash(self):
        idx = steady(0.0, n=10)
        cash = flat_cash(idx, 1.0)
        self.assertTrue(lc.can_start(idx, cash, min(idx), 9))
        self.assertFalse(lc.can_start(idx, cash, min(idx), 10))


class Indices(unittest.TestCase):
    def test_dividend_yield_adds_to_the_price_return(self):
        prices = steady(0.0, n=13)
        tr = lc.total_return(prices, 0.12)
        self.assertAlmostEqual(tr[max(tr)], 1.01**12)

    def test_gaps_are_refused(self):
        prices = {lc.month("2000-01"): 1.0, lc.month("2000-03"): 1.0}
        with self.assertRaises(ValueError):
            lc.total_return(prices)

    def test_shiller_total_return_adds_monthly_dividends(self):
        rows = {lc.month("2000-01"): (100.0, 12.0), lc.month("2000-02"): (100.0, 12.0)}
        price, tr = lc.shiller_indices(rows)
        self.assertAlmostEqual(tr[lc.month("2000-02")], 1.01)
        self.assertEqual(price[lc.month("2000-02")], 100.0)


class Outcomes(unittest.TestCase):
    def test_all_time_high_and_fall_flags(self):
        first = lc.month("1990-01")
        values = [1.0] * lc.HISTORY + [2.0, 1.7, 1.0] + [1.0] * 15
        prices = {first + k: v for k, v in enumerate(values)}
        d = lc.Data(prices={"AUS": prices}, cash={"AUS": flat_cash(prices, 0.0)}, cash_source={})
        outs = {o.start - first: o for o in lc.outcomes(d, n=3, countries=("AUS",))}
        self.assertTrue(outs[lc.HISTORY].at_high)
        self.assertFalse(outs[lc.HISTORY + 1].after_fall)  # 15% below the high
        self.assertTrue(outs[lc.HISTORY + 2].after_fall)  # 50% below the high
        self.assertFalse(outs[0].at_high)  # not enough history yet

    def test_quantile_is_linear_interpolation(self):
        self.assertAlmostEqual(lc.quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)


if __name__ == "__main__":
    unittest.main()
