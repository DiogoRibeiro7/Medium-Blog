"""Checks of the definitions on cases with a known answer."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_data  # noqa: E402
import heat_core as hc  # noqa: E402
import heat_numbers  # noqa: E402


class HalfYears(unittest.TestCase):
    def test_round_trip(self):
        for text in ("2007-S2", "2022-S1", "2026-S1"):
            self.assertEqual(hc.label(hc.half(text)), text)

    def test_order(self):
        self.assertEqual(hc.half("2023-S1") - hc.half("2022-S2"), 1)


class BreakEven(unittest.TestCase):
    def test_breakeven_spf(self):
        self.assertAlmostEqual(hc.breakeven_spf(3.0, 0.9), 2.7)

    def test_saving_is_zero_at_breakeven(self):
        # Electricity three times gas, boiler at 100%: a heat pump with SPF 3 costs the same.
        self.assertAlmostEqual(hc.saving(0.30, 0.10, 3.0, eta=1.0, heat=10_000), 0.0)

    def test_saving_value(self):
        # Gas 0.10 at 90%: 1,111.11 a year; electricity 0.25 at SPF 2.5: 1,000 a year.
        self.assertAlmostEqual(hc.saving(0.25, 0.10, 2.5, eta=0.9, heat=10_000), 10_000 * (0.10 / 0.9 - 0.10))

    def test_cheaper_matches_the_sign_of_the_saving(self):
        for p_e, p_g in ((0.30, 0.10), (0.20, 0.10), (0.35, 0.08), (0.12, 0.11)):
            p = hc.Point("XX", 0, p_e, p_g)
            for spf in (2.5, 2.8, 3.4, 4.0):
                self.assertEqual(p.cheaper(spf), p.saving(spf) > 0)


class Statistics(unittest.TestCase):
    def test_quantile_type_7(self):
        self.assertAlmostEqual(hc.quantile([1, 2, 3, 4], 0.25), 1.75)
        self.assertAlmostEqual(hc.quantile([4, 1, 3, 2], 0.5), 2.5)

    def test_share(self):
        self.assertAlmostEqual(hc.share([True, False, True, True]), 0.75)


class Panel(unittest.TestCase):
    def setUp(self):
        a, b = hc.half("2007-S1"), hc.half("2007-S2")
        c, d = hc.half("2008-S1"), hc.half("2026-S2")
        self.e = {("AT", hc.ELECTRICITY_BAND, "I_TAX"): {a: (0.2, ""), b: (0.2, ""), c: (0.3, "e"), d: (0.3, "")}}
        self.g = {("AT", hc.GAS_BAND, "I_TAX"): {a: (0.1, ""), b: (0.1, ""), d: (0.1, "")}}

    def test_points_need_both_prices_inside_the_period(self):
        pts = hc.points(self.e, self.g, countries=("AT",))
        self.assertEqual([hc.label(p.half) for p in pts], ["2007-S2"])
        self.assertAlmostEqual(pts[0].ratio, 2.0)

    def test_lifetime(self):
        s = {h: hc.Point("AT", h, 0.3, 0.1) for h in (10, 11)}
        one = hc.Point("AT", 10, 0.3, 0.1).saving(3.4)
        self.assertAlmostEqual(hc.lifetime(s, 10, 2, 3.4), one)
        self.assertIsNone(hc.lifetime(s, 10, 3, 3.4))

    def test_latest_year_averages_the_last_two_halves(self):
        s = {"AT": {20: hc.Point("AT", 20, 0.2, 0.1), 21: hc.Point("AT", 21, 0.4, 0.1)}}
        last, p = heat_numbers.latest_year(s)["AT"]
        self.assertEqual(last, 21)
        self.assertAlmostEqual(p.ratio, 3.0)


class Snapshot(unittest.TestCase):
    def test_tidy_keeps_the_columns_and_sorts(self):
        text = (
            "DATAFLOW,LAST UPDATE,freq,siec,nrg_cons,unit,tax,currency,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG,CONF_STATUS\n"
            "X,1,S,G3000,GJ20-199,KWH,I_TAX,EUR,BE,2008-S1,0.07,,\n"
            "X,1,S,G3000,GJ20-199,KWH,I_TAX,EUR,AT,2008-S1,0.06,e,\n"
            "X,1,S,G3000,GJ20-199,KWH,I_TAX,EUR,AT,2008-S2,,,\n"
        )
        out = fetch_data.tidy("gas", text).splitlines()
        self.assertEqual(out[0], "country,half,band,tax,value,flag")
        self.assertEqual(out[1:], ["AT,2008-S1,GJ20-199,I_TAX,0.06,e", "BE,2008-S1,GJ20-199,I_TAX,0.07,"])


if __name__ == "__main__":
    unittest.main()
