"""Tests of the Portuguese reference pricing rule and the price conversions (no downloaded data)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import erp  # noqa: E402


class TestPortugueseRule(unittest.TestCase):
    def test_y_factor_is_continuous_and_bounded(self):
        self.assertEqual(erp.y_factor(1.0), 1.0)
        self.assertAlmostEqual(erp.y_factor(1 / 3), 1.03, places=9)
        self.assertAlmostEqual(erp.y_factor(1 / 3 - 1e-12), 1.05, places=9)   # the jump Infarmed's form has
        self.assertAlmostEqual(erp.y_factor(3.0), 0.97, places=9)
        self.assertEqual(erp.y_factor(3.0001), 0.95)
        self.assertAlmostEqual(erp.y_factor(2.0), 0.985, places=9)
        self.assertAlmostEqual(erp.y_factor(0.5), 1.015, places=9)

    def test_adjust(self):
        # Same presentation: the reference price passes through.
        self.assertEqual(erp.adjust(63.62, 5, 5, 60, 60), 63.62)
        # Twice the pack: double, less 1.5 percent.
        self.assertEqual(erp.adjust(10.00, 5, 5, 56, 28), round(20.00 * 0.985, 2))
        # Half the strength: half the price.
        self.assertEqual(erp.adjust(10.00, 2.5, 5, 28, 28), 5.00)

    def test_cap(self):
        self.assertEqual(erp.pt_cap([10.0, 20.0, None, 30.0]), 20.0)
        self.assertEqual(erp.pt_cap([10.0, 20.0, 30.0, 40.0], "min"), 10.0)
        self.assertTrue(np.isnan(erp.pt_cap([None, np.nan])))

    def test_retail_tiers_and_inverse(self):
        # The top of each tier, in retail prices, is the limit Infarmed's form uses to invert them.
        tops = [round(a * top + b, 2) for top, a, b in erp.PT_TIERS[:-1]]
        self.assertEqual(tops, [6.68, 9.97, 14.10, 26.96, 64.58])
        # The fixed part rises from tier to tier, so the retail price jumps at each limit.
        self.assertGreater(erp.pt_pvp(5.01) - erp.pt_pvp(5.00), 1.0)
        for pva in (0.5, 4.99, 5.0, 6.5, 9.0, 15.0, 33.3, 50.0, 63.62, 812.4):
            self.assertAlmostEqual(erp.pt_pva(erp.pt_pvp(pva)), pva, delta=0.005)


class TestReferenceCountries(unittest.TestCase):
    def test_spain(self):
        self.assertAlmostEqual(erp.es_pva(15.61083), 10.0, places=6)
        self.assertAlmostEqual(erp.es_pva(600.0), 600 / 1.04 - 55.91, places=9)
        self.assertAlmostEqual(erp.es_pva(200.0), 200 / 1.04 - 45.91, places=9)
        # Above each limit the fixed amount taken off is 5 euros larger (the margins have fixed parts).
        for limit in (260.9464, 578.1464):
            self.assertAlmostEqual(erp.es_pva(limit) - erp.es_pva(limit + 1e-9), 5.0, delta=1e-6)

    def test_italy(self):
        self.assertAlmostEqual(erp.it_pva(105.0), 105.0 / 1.6504, places=9)
        self.assertAlmostEqual(erp.it_pva(10.0, "C"), 6.091, places=9)

    def test_france_round_trip(self):
        pfht = np.round(np.linspace(0.5, 4000, 997), 2)
        back = erp.fr_pfht(erp.fr_ppttc(pfht))
        self.assertTrue(np.allclose(back, pfht, atol=0.006))
        # 6.93 percent wholesale and 10 percent pharmacy margin on the first 1.91 euros, then 7 percent.
        self.assertAlmostEqual(float(erp.fr_ppttc(10.0)), round((10 + 0.693 + 0.191 + 0.07 * 8.09) * 1.021, 2))


class TestNames(unittest.TestCase):
    def test_stem(self):
        for a, b in (("DAPAGLIFLOZINE PROPANEDIOL MONOHYDRATÉE", "Dapagliflozin propanediolo"),
                     ("empagliflozine", "EMPAGLIFLOZINA"), ("lévothyroxine sodique", "LEVOTIROXINA SODICA"),
                     ("Sacubitril/Valsartan", "valsartan + sacubitril"), ("ticagrélor", "Ticagrelor"),
                     ("apixabán", "APIXABAN")):
            self.assertEqual(erp.stem(a), erp.stem(b), (a, b))

    def test_strengths(self):
        self.assertEqual(erp.strengths("ENTRESTO 49 mg/51 mg, comprimé pelliculé"), (49.0, 51.0))
        self.assertEqual(erp.strengths("SACUBITRIL+VALSARTAN 49+51MG 56 UNITA' USO ORALE"), (49.0, 51.0))
        self.assertEqual(erp.strengths("APIXABAN 2,5MG 10 UNITA' USO ORALE"), (2.5,))
        self.assertEqual(erp.strengths("SYNJARDY*56x1 cpr rivestite 12,5 mg + 1.000 mg"), (12.5, 1000.0))
        self.assertEqual(erp.strengths("LEVOTHYROX 100 microgrammes, comprimé sécable"), (0.1,))
        self.assertEqual(erp.strengths("ELIQUIS 5 MG COMPRIMIDOS RECUBIERTOS CON PELICULA, 60 comprimidos"), (5.0,))


class TestReferencePrice(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.t = pd.DataFrame({
            "brand": ["eliquis", "eliquis", "eliquis", "entresto", "other"],
            "strengths": [(5.0,), (5.0,), (2.5,), (49.0, 51.0), (5.0,)],
            "units": [56, 168, 60, 56, 60],
            "pva": [52.23, 156.71, 55.97, 106.96, 1.0],
            "label": ["E5x56", "E5x168", "E2.5x60", "N56", "O"],
        })

    def test_identical_and_nearest_pack(self):
        self.assertEqual(erp.reference_price(self.t, "eliquis", (5.0,), 56), (52.23, "identical", "E5x56"))
        adj, kind, label = erp.reference_price(self.t, "eliquis", (5.0,), 60)
        self.assertEqual((kind, label), ("same strength", "E5x56"))
        self.assertEqual(adj, erp.adjust(52.23, 5, 5, 60, 56))

    def test_other_strength_and_misses(self):
        adj, kind, label = erp.reference_price(self.t, "eliquis", (10.0,), 56)
        self.assertEqual((kind, label), ("other strength", "E5x56"))
        self.assertEqual(adj, round(52.23 * 2, 2))
        self.assertIsNone(erp.reference_price(self.t, "entresto", (24.0, 26.0), 28))
        self.assertIsNone(erp.reference_price(self.t, "jardiance", (10.0,), 30))


if __name__ == "__main__":
    unittest.main()
