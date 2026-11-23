"""The portfolio rules, the survival counts and the number reader, on synthetic snapshots with known answers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_snapshots as B  # noqa: E402
import crypto_core as C  # noqa: E402


def coin(key, rank, cap, price):
    return C.Coin(
        key=key, name=key.upper(), symbol=key[:3].upper(), rank=rank, market_cap=cap, price=price, supply=cap / price
    )


def snap(date, rows):
    return C.Snapshot(date, [coin(k, i + 1, cap, p) for i, (k, cap, p) in enumerate(rows)])


# A start with four coins; a year later "dead" has left the list and "new" has entered it.
START = snap("2020-01-05", [("btc", 600.0, 10.0), ("eth", 200.0, 2.0), ("dead", 150.0, 1.0), ("mid", 50.0, 5.0)])
END = snap("2021-01-03", [("btc", 1200.0, 20.0), ("new", 500.0, 1.0), ("eth", 100.0, 1.0), ("mid", 20.0, 2.0)])


class Holding(unittest.TestCase):
    def test_a_listed_coin_moves_with_its_price(self):
        h = C.hold(START.by_key["btc"], END)
        self.assertTrue(h.listed_at_end)
        self.assertAlmostEqual(h.multiple_high, 2.0)
        self.assertAlmostEqual(h.multiple_low, 2.0)

    def test_a_dropped_coin_is_worth_at_most_the_cutoff_over_its_starting_cap(self):
        h = C.hold(START.by_key["dead"], END)
        self.assertFalse(h.listed_at_end)
        self.assertAlmostEqual(h.multiple_high, 20.0 / 150.0)
        self.assertEqual(h.multiple_low, 0.0)


class Portfolios(unittest.TestCase):
    def test_cap_weighted_point_in_time(self):
        p = C.point_in_time(START, END, 3, "cap")
        w = (600 / 950, 200 / 950, 150 / 950)
        self.assertAlmostEqual(p.high, w[0] * 2.0 + w[1] * 0.5 + w[2] * (20 / 150))
        self.assertAlmostEqual(p.low, w[0] * 2.0 + w[1] * 0.5)
        self.assertEqual(p.dropped, 1)
        self.assertAlmostEqual(p.dropped_weight, w[2])

    def test_equal_weight(self):
        p = C.point_in_time(START, END, 4, "equal")
        self.assertAlmostEqual(p.low, (2.0 + 0.5 + 0.0 + 0.4) / 4)

    def test_survivors_take_only_todays_leaders_that_already_existed(self):
        latest = END
        s = C.survivors(START, END, latest, 3, "equal")
        self.assertEqual(s.members, 2)  # btc and eth; "new" did not exist at the start
        self.assertAlmostEqual(s.high, (2.0 + 0.5) / 2)
        self.assertEqual(s.dropped, 0)

    def test_gap_per_year(self):
        self.assertAlmostEqual(C.gap_per_year(4.0, 1.0, 2.0), 1.0)
        self.assertAlmostEqual(C.annualised(0.25, 2.0), -0.5)
        self.assertEqual(C.annualised(0.0, 2.0), -1.0)

    def test_years_between(self):
        self.assertAlmostEqual(C.years_between("2020-01-05", "2021-01-03"), 1 - 2 / 365.25)


class Survival(unittest.TestCase):
    def setUp(self):
        s1 = snap("2020-01-05", [("a", 9.0, 1.0), ("b", 8.0, 1.0), ("c", 7.0, 1.0)])
        s2 = snap("2021-01-03", [("a", 9.0, 1.0), ("d", 8.0, 1.0), ("b", 7.0, 1.0)])
        s3 = snap("2022-01-02", [("a", 9.0, 1.0), ("d", 8.0, 1.0), ("e", 7.0, 1.0)])
        self.snaps = [s1, s2, s3]

    def test_entry_durations_in_the_top_two(self):
        d = C.entry_durations(self.snaps, 2)
        self.assertEqual(d["a"], ("2020-01-05", 2, True))
        self.assertEqual(d["b"], ("2020-01-05", 0, False))
        self.assertEqual(d["d"], ("2021-01-03", 1, True))

    def test_kaplan_meier(self):
        km = C.kaplan_meier([(2, True), (0, False), (1, True)])
        # step 1: all three at risk, b leaves; step 2: only a is still observed (d is censored after step 1)
        self.assertEqual((km[0][0], km[0][2], km[0][3]), (1, 3, 1))
        self.assertAlmostEqual(km[0][1], 2 / 3)
        self.assertEqual((km[1][0], km[1][2], km[1][3]), (2, 1, 0))
        self.assertAlmostEqual(km[1][1], 2 / 3)
        self.assertEqual(len(km), 2)

    def test_turnover(self):
        t = C.turnover(self.snaps, 2)
        self.assertEqual(t[0], ("2020-01-05", "2021-01-03", 0.5))
        self.assertEqual(t[1], ("2020-01-05", "2022-01-02", 0.5))
        self.assertEqual(t[2], ("2021-01-03", "2022-01-02", 1.0))


class Numbers(unittest.TestCase):
    def test_displayed_numbers(self):
        self.assertEqual(C.number("$1,234.56"), 1234.56)
        self.assertEqual(C.number("18,123,456 BTC"), 18123456.0)
        self.assertEqual(C.number("$1.2B"), 1.2e9)
        self.assertAlmostEqual(C.number("$0.0₄123"), 0.0000123)
        self.assertTrue(C.number("--") != C.number("--"))


HTML = """<html><body><table>
<tr><th>Rank</th><th>Name</th><th>Symbol</th><th>Market Cap</th><th>Price</th><th>Circulating Supply</th></tr>
<tr><td>1</td><td><a href="/currencies/bitcoin/">Bitcoin</a></td><td>BTC</td><td>$10,000,000</td><td>$1,000.00</td>
<td>10,000 BTC</td></tr>
<tr><td>2</td><td><a href="/currencies/litecoin/">Litecoin</a></td><td>LTC</td><td>$500,000</td><td>$5.00</td>
<td>100,000 LTC</td></tr>
</table></body></html>"""


class Readers(unittest.TestCase):
    def test_saved_table_with_slugs(self):
        t = B.Table()
        t.feed(HTML)
        header = t.rows[0][0]
        coins = B.from_cells(header, t.rows)
        self.assertEqual([c.key for c in coins], ["bitcoin", "litecoin"])
        self.assertEqual(coins[1].market_cap, 500000.0)
        self.assertEqual(coins[0].supply, 10000.0)

    def test_embedded_json(self):
        rows = [
            {
                "cmcRank": 1,
                "slug": "bitcoin",
                "name": "Bitcoin",
                "symbol": "BTC",
                "circulatingSupply": 10.0,
                "quotes": [{"marketCap": 100.0, "price": 10.0}],
            },
        ]
        page = '<script id="__NEXT_DATA__">' + __import__("json").dumps({"props": {"x": rows}}) + "</script>"
        (c,) = B.from_json(page)
        self.assertEqual((c.key, c.rank, c.market_cap, c.price), ("bitcoin", 1, 100.0, 10.0))

    def test_copied_csv_uses_name_and_symbol(self):
        coins = B.from_cells(
            ["#", "Name", "Symbol", "Market Cap", "Price", "Circulating Supply"],
            [(["1", "Bitcoin", "BTC", "$1.2B", "$100", "12,000,000 BTC"], None)],
        )
        self.assertEqual(coins[0].key, "Bitcoin|BTC")
        self.assertEqual(coins[0].market_cap, 1.2e9)


if __name__ == "__main__":
    unittest.main()
