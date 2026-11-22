"""Tests of the readers and the replay on cases with a known answer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rentbuy_core as C  # noqa: E402


def flat_area(name="Flat", price=100_000.0, rent=6_000.0, tax=0.01, years=range(1980, 2031)) -> C.Area:
    idx = pd.Index(list(years), name="year")
    return C.Area(name, pd.Series(price, index=idx), pd.Series(rent, index=idx), tax, 1.5)


def market(rate=0.06, ret=0.0, years=range(1980, 2031)) -> C.Market:
    idx = pd.Index(list(years), name="year")
    r = pd.DataFrame({"stocks": ret, "bills": ret, "bonds": ret}, index=idx)
    return C.Market(pd.Series(rate, index=idx), r, pd.Series(100.0, index=idx))


class Readers(unittest.TestCase):
    def test_fred_semiannual_monthly_and_annual(self):
        semi = "observation_date,X\n" + "".join(f"{y}-01-01,100\n{y}-07-01,110\n" for y in (2018, 2019, 2020)) + "2021-01-01,120\n"
        self.assertEqual(C.annual_fred(semi).to_dict(), {2018: 105.0, 2019: 105.0, 2020: 105.0})
        annual = "observation_date,X\n2019-01-01,90\n2020-01-01,100\n"
        self.assertEqual(C.annual_fred(annual).to_dict(), {2019: 90.0, 2020: 100.0})
        months = [f"2020-{m:02d}-01,{m}" for m in range(1, 13)] + [f"2021-{m:02d}-01,1" for m in range(1, 12) if m != 10]
        got = C.annual_fred("observation_date,X\n" + "\n".join(months) + "\n2021-10-01,\n")
        self.assertAlmostEqual(got[2020], 6.5)
        self.assertNotIn(2021, got.index)  # only 10 months

    def test_dbnomics(self):
        semi = '{"series":{"docs":[{"period":["2020-S1","2020-S2","2021-S1"],"value":[1,3,5]}]}}'
        self.assertEqual(C.annual_dbnomics(semi).to_dict(), {2020: 2.0})
        periods = ",".join(f'"2020-{m:02d}"' for m in range(1, 13)) + ',"2021-01"'
        monthly = '{"series":{"docs":[{"period":[' + periods + '],"value":[' + ",".join(["2"] * 13) + "]}]}}"
        self.assertEqual(C.annual_dbnomics(monthly).to_dict(), {2020: 2.0})

    def test_damodaran_block(self):
        sheet = pd.DataFrame(
            [
                ["note", None, None, None, None],
                ["Year", "S&P", "Small", "Bill", "Bond"],
                [2000, 0.1, 0.2, 0.03, 0.05],
                [2001, -0.1, 0.0, 0.02, 0.04],
                [None, None, None, None, None],
                ["Average", 1, 1, 1, 1],
            ]
        )
        r = C.damodaran_returns(sheet)
        self.assertEqual(list(r.index), [2000, 2001])
        self.assertAlmostEqual(r.loc[2001, "bonds"], 0.04)
        self.assertAlmostEqual(r.loc[2000, "bills"], 0.03)


class Replay(unittest.TestCase):
    def test_payment_pays_off_the_loan(self):
        pmt = C.payment(100_000, 0.06, 360)
        bal = 100_000.0
        for _ in range(360):
            bal -= pmt - bal * 0.06 / 12
        self.assertAlmostEqual(bal, 0.0, places=4)

    def test_identities_with_flat_prices_and_no_returns(self):
        a = C.Assumptions(insurance=0, maintenance=0, buy_cost=0.0, sell_cost=0.0, refinance=False)
        o = C.replay(flat_area(rent=4_000.0, tax=0.0), market(), 2000, 10, a)
        # nothing grows, so the owner's equity is the down payment plus the principal repaid
        self.assertAlmostEqual(o.owner_equity, o.down_payment + o.principal_paid, places=4)
        self.assertAlmostEqual(o.renter_base, 20_000.0)
        # the renter's invested differences are the owner's cash costs minus rent
        self.assertAlmostEqual(
            o.renter_side + 0.0,
            10 * (12 * C.payment(80_000, 0.06, 360) - 4_000),
            places=4,
        )

    def test_breakeven_and_categories(self):
        a = C.Assumptions(refinance=False)
        o = C.replay(flat_area(), market(ret=0.05), 2000, 10, a)
        s = o.breakeven_s()
        if np.isfinite(s):
            self.assertAlmostEqual(o.owner(s), o.renter(s), places=4)
        self.assertIn(
            o.category(),
            {
                "renting wins either way",
                "saving decides",
                "owning wins either way",
                "saving favours the owner",
            },
        )

    def test_refinance_when_rates_fall_a_point(self):
        m = market()
        rate = m.rate.copy()
        rate.loc[2003:] = 0.045
        m = C.Market(rate, m.returns, m.cpi)
        on = C.replay(flat_area(), m, 2000, 10, C.Assumptions())
        off = C.replay(flat_area(), m, 2000, 10, C.Assumptions(refinance=False))
        self.assertEqual(on.refinances, 1)
        self.assertLess(on.interest_paid, off.interest_paid)

    def test_prop13_tax_follows_purchase_price(self):
        idx = pd.Index(range(1980, 2031), name="year")
        rising = pd.Series(100_000 * 1.10 ** (idx - 2000), index=idx)
        ca = C.Area("Los Angeles", rising, pd.Series(6_000.0, index=idx), 0.01, 1.5)
        other = C.Area("Denver", rising, pd.Series(6_000.0, index=idx), 0.01, 1.5)
        a = C.Assumptions(refinance=False)
        self.assertGreater(C.replay(ca, market(), 2000, 10, a).renter_side, 0)
        # with prices rising 10% a year, a 2%-capped tax base leaves the Californian owner cheaper
        self.assertLess(
            C.replay(ca, market(), 2000, 10, a).renter_side,
            C.replay(other, market(), 2000, 10, a).renter_side,
        )

    def test_cohorts_respect_data_and_start_year(self):
        a = C.Assumptions()
        out = C.cohorts(flat_area(years=range(1980, 2026)), market(years=range(1980, 2026)), 10, a)
        self.assertEqual(out[0].year, 1987)
        self.assertEqual(out[-1].year, 2015)


if __name__ == "__main__":
    unittest.main()
