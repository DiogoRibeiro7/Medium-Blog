"""Tests of the shared drug-price tools, on small hand-made tables (no downloaded data)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import drugprices as dp  # noqa: E402


class TestNames(unittest.TestCase):
    def test_us_names(self):
        self.assertEqual(dp.molecule_us("Sacubitril/Valsartan"), "sacubitril + valsartan")
        self.assertEqual(dp.molecule_us("Apixaban"), "apixaban")
        self.assertEqual(dp.molecule_us("Dapagliflozin Propanediol"), "dapagliflozin")
        self.assertEqual(dp.molecule_us("Sitagliptin Phos/Metformin HCl"), "metformin + sitagliptin")
        self.assertEqual(dp.molecule_us("Cabozantinib S-Malate"), "cabozantinib")
        self.assertEqual(dp.molecule_us("Acetaminophen With Codeine"), "codein + paracetamol")
        self.assertEqual(dp.molecule_us("Tafamidis Meglumine"), "tafamidis")

    def test_french_names(self):
        self.assertEqual(dp.substance_fr("CHLORHYDRATE DE METFORMINE"), "metformin")
        self.assertEqual(dp.substance_fr("DAPAGLIFLOZINE PROPANEDIOL MONOHYDRATÉE"), "dapagliflozin")
        self.assertEqual(dp.substance_fr("BÉSILATE D'AMLODIPINE"), "amlodipin")
        self.assertEqual(dp.substance_fr("ATORVASTATINE CALCIQUE TRIHYDRATÉE"), "atorvastatin")

    def test_both_sides_meet(self):
        pairs = [("Dapagliflozin Propanediol", "DAPAGLIFLOZINE PROPANEDIOL MONOHYDRATÉE"),
                 ("Amlodipine Besylate", "BÉSILATE D'AMLODIPINE"),
                 ("Metformin HCl", "CHLORHYDRATE DE METFORMINE")]
        for us, fr in pairs:
            self.assertEqual(dp.molecule_us(us), dp.substance_fr(fr))

    def test_harmonise_truncated_names(self):
        us = pd.DataFrame({"molecule": ["bictegrav + emtricit + tenofov ala", "tenofovir", "apixaban"],
                           "segment": ["originator"] * 3, "q_us": [10.0, 5.0, 2.0], "v_us": [100.0, 5.0, 20.0]})
        fr = ["bictegravir + emtricitabin + tenofovir alafenamid", "tenofovir alafenamid",
              "tenofovir disoproxil", "apixaban"]
        h = dp.harmonise(us, fr).set_index("molecule")
        self.assertIn("bictegravir + emtricitabin + tenofovir alafenamid", h.index)
        # "tenofovir" begins two French names of a different length: it is left alone.
        self.assertIn("tenofovir", h.index)
        self.assertAlmostEqual(h.loc["apixaban", "p_us"], 10.0)


class TestFrenchLabels(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(dp.french_number("24,34"), 24.34)
        self.assertEqual(dp.french_number("1.466,29"), 1466.29)
        self.assertEqual(dp.french_number("1.466.29"), 1466.29)
        self.assertTrue(np.isnan(dp.french_number("")))

    def test_units_per_box(self):
        self.assertEqual(dp.units_per_box("plaquette(s) PVC PVDC aluminium de 30 comprimé(s)"), 30)
        self.assertEqual(dp.units_per_box("3 plaquette(s) PVC-Aluminium de 28 comprimé(s)"), 84)
        self.assertEqual(dp.units_per_box("plaquette(s) de 2 x 14 comprimés"), 28)
        self.assertEqual(dp.units_per_box("flacon(s) de 60 gélule(s)"), 60)
        self.assertTrue(np.isnan(dp.units_per_box("flacon de 100 ml")))


def toy_basket(p_us, p_fr, q_us, q_fr):
    m = pd.DataFrame({"p_us": p_us, "p_fr_usd": p_fr, "q_us": q_us, "q_fr": q_fr})
    m["ratio"] = m["p_us"] / m["p_fr_usd"]
    return m


class TestIndices(unittest.TestCase):
    def test_equal_prices_give_one(self):
        m = toy_basket([1.0, 5.0, 0.1], [1.0, 5.0, 0.1], [3.0, 1.0, 9.0], [1.0, 4.0, 2.0])
        for f in (dp.laspeyres, dp.paasche, dp.fisher, dp.jevons):
            self.assertAlmostEqual(f(m), 1.0)

    def test_proportional_prices(self):
        m = toy_basket([3.0, 15.0, 0.3], [1.0, 5.0, 0.1], [3.0, 1.0, 9.0], [1.0, 4.0, 2.0])
        for f in (dp.laspeyres, dp.paasche, dp.fisher, dp.jevons):
            self.assertAlmostEqual(f(m), 3.0)

    def test_weights_matter(self):
        # a cheap item bought mostly in the US and an expensive one bought mostly in France
        m = toy_basket([1.0, 100.0], [2.0, 10.0], [100.0, 1.0], [1.0, 100.0])
        L, P = dp.laspeyres(m), dp.paasche(m)
        self.assertAlmostEqual(L, (100 * 1 + 1 * 100) / (100 * 2 + 1 * 10))
        self.assertAlmostEqual(P, (1 * 1 + 100 * 100) / (1 * 2 + 100 * 10))
        self.assertTrue(min(L, P) <= dp.fisher(m) <= max(L, P))
        self.assertAlmostEqual(dp.jevons(m), np.sqrt(0.5 * 10))

    def test_bootstrap_is_reproducible_and_brackets(self):
        rng = np.random.default_rng(1)
        m = toy_basket(rng.lognormal(1, 1, 40), rng.lognormal(0, 1, 40), rng.uniform(1, 9, 40), rng.uniform(1, 9, 40))
        a = dp.bootstrap(m, np.random.default_rng(7), reps=300)
        b = dp.bootstrap(m, np.random.default_rng(7), reps=300)
        self.assertEqual(a, b)
        self.assertLess(a["L"][0], dp.laspeyres(m))
        self.assertGreater(a["L"][1], dp.laspeyres(m))


class TestBaskets(unittest.TestCase):
    def setUp(self):
        self.us = pd.DataFrame({
            "molecule": ["alpha", "beta", "beta", "gamma", "delta", "delta"],
            "segment": ["originator", "originator", "generic", "originator", "originator", "generic"],
            "q_us": [100.0, 1.0, 99.0, 50.0, 2.0, 98.0],
            "v_us": [1000.0, 50.0, 99.0, 500.0, 40.0, 98.0],
        })
        self.us["p_us"] = self.us["v_us"] / self.us["q_us"]
        self.fr = pd.DataFrame({
            "molecule": ["alpha", "beta", "beta", "gamma", "gamma", "delta"],
            "segment": ["originator", "originator", "generic", "originator", "generic", "originator"],
            "q_fr": [10.0, 40.0, 60.0, 20.0, 80.0, 30.0],
            "v_fr": [20.0, 40.0, 30.0, 40.0, 40.0, 60.0],
        })
        self.fr["p_fr"] = self.fr["v_fr"] / self.fr["q_fr"]

    def test_baskets_and_molecule_prices(self):
        m = dp.molecule_baskets(self.us, self.fr, usd_per_eur=1.0).set_index("molecule")
        self.assertEqual(m.loc["alpha", "basket"], "single-source")
        self.assertEqual(m.loc["beta", "basket"], "multi-source")
        self.assertEqual(m.loc["gamma", "basket"], "mixed")
        self.assertEqual(m.loc["gamma", "mixed_side"], "generic in France only")
        self.assertEqual(m.loc["delta", "mixed_side"], "generic in US only")
        # a molecule's price adds up its brand and generic cells
        self.assertAlmostEqual(m.loc["beta", "p_us"], (50 + 99) / 100)
        self.assertAlmostEqual(m.loc["beta", "p_fr"], (40 + 30) / 100)
        self.assertAlmostEqual(m.loc["beta", "gen_us"], 0.99)
        self.assertAlmostEqual(m.loc["alpha", "ratio"], 10 / 2)

    def test_exchange_rate_scales_ratio(self):
        a = dp.molecule_baskets(self.us, self.fr, usd_per_eur=1.0).set_index("molecule")
        b = dp.molecule_baskets(self.us, self.fr, usd_per_eur=1.25).set_index("molecule")
        np.testing.assert_allclose(b["ratio"], a["ratio"] / 1.25)

    def test_mixing_endpoints(self):
        m = dp.molecule_baskets(self.us, self.fr, usd_per_eur=1.0)
        sm = m[m["basket"].isin(["single-source", "multi-source"])]
        for w, f in (("us", dp.laspeyres), ("fr", dp.paasche)):
            end_multi = dp.mixed_index(sm, 1.0, "multi-source", "single-source", col="basket", weights=w)
            end_single = dp.mixed_index(sm, 0.0, "multi-source", "single-source", col="basket", weights=w)
            self.assertAlmostEqual(end_multi, f(sm[sm["basket"] == "multi-source"]))
            self.assertAlmostEqual(end_single, f(sm[sm["basket"] == "single-source"]))


if __name__ == "__main__":
    unittest.main()
