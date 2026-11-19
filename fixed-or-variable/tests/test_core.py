"""The methods on cases with a known answer."""

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mortgage_core as mc  # noqa: E402


def months(first_year, last_year):
    return [f"{y}-{m:02d}" for y in range(first_year, last_year + 1) for m in range(1, 13)]


def toy_market(euribor, spot=None, par10=None, first_year=1999, last_year=2026):
    """A market with constant or given rates. euribor and par10 are numbers or functions of the month."""
    idx = months(first_year, last_year)

    def series(v):
        return pd.Series([v(m) if callable(v) else v for m in idx], index=idx, dtype=float)

    e = series(euribor)
    curves = {}
    for bonds in ("HIST", "CGB_EA_AAA", "CGB_EA"):
        for h in range(1, 31):
            curves[f"SPOT_RT|{bonds}|Y{h}"] = series(spot if spot is not None else euribor)
            curves[f"PAR|{bonds}|Y{h}"] = series(par10 if par10 is not None else euribor)
    prices = {
        "HICP": series(100.0),
        "COMP_EMP": pd.Series({str(y): 1000.0 * 1.02 ** (y - first_year) for y in range(first_year - 1, last_year)}),
    }
    euribor_items = {"IRT_M3|": e, "IRT_M6|": e, "IRT_M12|": e}
    market = mc.Market(euribor_items, curves, prices)
    market.months = [m for m in idx if mc.FIRST <= m <= mc.LAST]
    return market


class Loan(unittest.TestCase):
    def test_payment_matches_the_annuity_formula(self):
        self.assertAlmostEqual(mc.payment(100_000, 3.0, 360), 421.6040, places=3)
        self.assertAlmostEqual(mc.payment(120_000, 0.0, 360), 120_000 / 360)

    def test_a_constant_rate_repays_the_loan_exactly(self):
        s = mc.replay(lambda k: 4.0, 12, 360)
        self.assertAlmostEqual(s.balance.iloc[-1], 0.0, places=6)
        self.assertLess(s.payment.max() - s.payment.min(), 1e-8)
        self.assertAlmostEqual(math.fsum(s.interest), 360 * s.payment.iloc[0] - 100_000, places=6)

    def test_a_reset_recomputes_the_payment_on_the_remaining_balance(self):
        s = mc.replay(lambda k: 2.0 if k < 12 else 5.0, 12, 24)
        balance_12 = s.balance.iloc[11]
        self.assertAlmostEqual(s.payment.iloc[12], mc.payment(balance_12, 5.0, 348), places=8)
        self.assertAlmostEqual(s.payment.iloc[0], mc.payment(100_000, 2.0, 360), places=8)


class Cohorts(unittest.TestCase):
    def test_equal_rates_cost_the_same(self):
        m = toy_market(3.0)
        r = mc.cohort(mc.Loans(m), "2005-01")
        self.assertAlmostEqual(r["extra_fixed"], 0.0, places=6)
        self.assertAlmostEqual(r["rise"], 0.0, places=12)
        self.assertTrue(r["complete"])

    def test_floors_and_margins(self):
        m = toy_market(-0.5, par10=-0.3)
        loans = mc.Loans(m)
        self.assertEqual(loans.variable("2020-01", 1).rate.iloc[0], mc.MARGIN)
        self.assertEqual(loans.fixed("2020-01", 1).rate.iloc[0], mc.MARGIN)
        free = mc.Loans(m, floor=False)
        self.assertAlmostEqual(free.variable("2020-01", 1).rate.iloc[0], mc.MARGIN - 0.5)

    def test_a_rate_rise_shows_up_once_a_year_and_favours_fixing(self):
        m = toy_market(lambda mo: 1.0 if mo < "2010-07" else 4.0, par10=2.0)
        r = mc.cohort(mc.Loans(m), "2010-01")
        v = mc.Loans(m).variable("2010-01", 24)
        self.assertEqual(v.rate.iloc[11], 2.0)  # set in January 2010
        self.assertEqual(v.rate.iloc[12], 5.0)  # set in January 2011
        self.assertGreater(r["rise"], 0.3)
        self.assertLess(r["extra_fixed"], 0)

    def test_payment_to_income_follows_income(self):
        m = toy_market(3.0)
        r = mc.cohort(mc.Loans(m), "2005-07", income=m.income())
        # constant payment, income up 2% a year: the share falls from 30%
        self.assertAlmostEqual(r["share_var_max"], mc.START_SHARE, places=12)
        self.assertLess(r["share_fix_end"], mc.START_SHARE)


class Expectations(unittest.TestCase):
    def test_flat_curve_implies_flat_forwards(self):
        m = toy_market(3.0, spot=2.5)
        fwd = mc.forward_path(m, "2010-01")
        self.assertEqual(fwd[0], 3.0)
        for f in fwd[1:]:
            self.assertAlmostEqual(f, 3.0, places=12)  # 2.5 forward plus the 0.5 Euribor gap

    def test_rising_curve_implies_rising_forwards(self):
        m = toy_market(2.0)
        for h in range(1, 31):
            for b in ("HIST", "CGB_EA_AAA"):
                m.curves[f"SPOT_RT|{b}|Y{h}"] = m.curves[f"SPOT_RT|{b}|Y{h}"] * 0 + (1.0 + 0.1 * h)
        fwd = mc.forward_path(m, "2010-01")
        # continuous compounding: forward between h and h+1 = s(h+1)(h+1) - s(h)h = 1 + 0.1(2h+1)
        for h in range(1, 10):
            self.assertAlmostEqual(fwd[h], 1.0 + 0.1 * (2 * h + 1) + (2.0 - 1.1), places=12)

    def test_no_errors_when_rates_turn_out_as_implied(self):
        m = toy_market(3.0)
        e = mc.forecast_errors(m)
        self.assertEqual(e.index[0], "1999-01")
        self.assertAlmostEqual(float(e.abs().to_numpy().max()), 0.0, places=12)

    def test_scenarios_on_a_flat_world(self):
        m = toy_market(3.0)
        s, fwd, fixed = mc.new_loan_scenarios(m, start="2026-10")
        self.assertAlmostEqual(fixed, 3.0 + mc.MARGIN)
        self.assertAlmostEqual(float(s.extra_fixed.abs().max()), 0.0, places=6)
        self.assertAlmostEqual(float(s.rise.abs().max()), 0.0, places=12)


class Files(unittest.TestCase):
    def test_jsonstat_and_wide_csv(self):
        d = {
            "id": ["freq", "int_rt", "geo", "time"],
            "size": [1, 2, 1, 2],
            "dimension": {
                "freq": {"category": {"index": {"M": 0}}},
                "int_rt": {"category": {"index": {"IRT_M3": 0, "IRT_M12": 1}}},
                "geo": {"category": {"index": {"EA": 0}}},
                "time": {"category": {"index": {"2020-01": 0, "2020-02": 1}}},
            },
            "value": {"0": 1.0, "3": 4.0},
        }
        rows = sorted(mc.jsonstat_rows(d, ["int_rt"]))
        self.assertEqual(rows, [("EA", "IRT_M12", "2020-02", 4.0), ("EA", "IRT_M3", "2020-01", 1.0)])
        text = mc.wide_csv(rows)
        self.assertEqual(text.splitlines()[0], "geo,item,2020-01,2020-02")
        self.assertEqual(text, mc.wide_csv(list(reversed(rows))))
        with self.assertRaises(ValueError):
            mc.wide_csv(rows + [("EA", "IRT_M3", "2020-01", 9.0)])


if __name__ == "__main__":
    unittest.main()
