"""Tests for the affordability panel: parsing, geography matching, comparisons and the pinned INE data."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import affordability as F
import regional as R
from test_regional import _response, _write

HERE = Path(__file__).parent
DATA = HERE / "data"

# Portugal, NUTS 1, NUTS 2, NUTS 3, two municipalities and a parish, in INE's hierarchical order.
GEO = [("PT", "Portugal", 1), ("1", "Continente", 2), ("1X", "Region X", 3), ("1X0", "Region X", 4),
       ("1X00001", "Alpha", 5), ("1X0000101", "Alpha parish", 6), ("1X00002", "Beta", 5)]


def _meta(code: str, periods: list[tuple[str, str]], extra: dict[str, list[tuple[str, str]]]) -> list[dict[str, Any]]:
    cats: dict[str, Any] = {f"Dim_Num1_{c}": [{"dim_num": "1", "categ_cod": c, "categ_dsg": lab, "categ_ord": str(i),
                                               "categ_nivel": "1"}] for i, (c, lab) in enumerate(periods)}
    for i, (c, name, lvl) in enumerate(GEO):
        cats[f"Dim_Num2_{c}"] = [{"dim_num": "2", "categ_cod": c, "categ_dsg": name, "categ_ord": str(i), "categ_nivel": str(lvl)}]
    for n, items in extra.items():
        for i, (c, lab) in enumerate(items):
            cats[f"Dim_Num{n}_{c}"] = [{"dim_num": n, "categ_cod": c, "categ_dsg": lab, "categ_ord": str(i), "categ_nivel": "1"}]
    desc = [{"dim_num": "1"}, {"dim_num": "2"}] + [{"dim_num": n} for n in extra]
    return [{"IndicadorCod": code, "IndicadorNome": "test", "UnidadeMedida": "€", "Dimensoes": {"Descricao_Dim": desc,
                                                                                                "Categoria_Dim": [cats]}}]


class SeriesParsingTests(unittest.TestCase):
    """Containers are decoded strictly: fourth quarters only, no duplicates, gaps kept as gaps."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.extra = {"3": [("H.1", "Total"), ("H.1.1", "Novos"), ("H.1.2", "Existentes")]}
        self.periods = [("S5A20194", "4.º Trimestre de 2019"), ("S5A20204", "4.º Trimestre de 2020")]

    def _container(self, data: dict[str, list[dict[str, str]]], ok: bool = True) -> Path:
        payload = [{"IndicadorCod": "0000009", "Sucesso": {"Verdadeiro" if ok else "Falso": [{"Msg": "x"}]}, "Dados": data}]
        lines = [_response("metadata", _meta("0000009", self.periods, self.extra)), _response("data", payload)]
        header = {"kind": "header", "indicator": "0000009", "name": "t", "filters": {}, "periods": [], "responses": 2}
        return _write(self.dir / "c.jsonl.gz", header, lines)

    @staticmethod
    def _rows(value: str = "100") -> list[dict[str, str]]:
        # INE's data rows drop the dots of the metadata codes (H12 for H.1.2).
        return [{"geocod": c, "geodsg": n, "dim_3": d, "valor": value} for c, n, _l in GEO for d in ("H1", "H11", "H12")]

    def test_keeps_fourth_quarters_and_unavailable_cells(self) -> None:
        rows = self._rows()
        rows[-1] = {"geocod": "1X00002", "geodsg": "Beta", "dim_3": "H12", "sinal_conv": "x", "sinal_conv_desc": "Valor confidencial"}
        s = F.parse_series(self._container({"4.º Trimestre de 2019": rows, "4.º Trimestre de 2020": self._rows("110")}))
        self.assertEqual(s.years(), [2019, 2020])
        self.assertEqual(s.category("3", "Existentes"), "H.1.2")
        self.assertEqual(s.get({"3": "H.1"})[("1X00001", 2020)], 110.0)
        self.assertNotIn(("1X00002", 2019), s.get({"3": "H.1.2"}))
        self.assertEqual(s.missing({"3": "H.1.2"}), {("1X00002", 2019): "Valor confidencial"})

    def test_rejects_other_quarters_failures_duplicates_and_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            F.parse_series(self._container({"1.º Trimestre de 2020": self._rows()}))
        with self.assertRaises(ValueError):
            F.parse_series(self._container({"4.º Trimestre de 2019": self._rows()}, ok=False))
        with self.assertRaises(ValueError):
            F.parse_series(self._container({"4.º Trimestre de 2019": self._rows() + self._rows()[:1]}))
        with self.assertRaises(ValueError):
            F.parse_series(self._container({"4.º Trimestre de 2019": self._rows("0")}))
        with self.assertRaises(ValueError):
            F.parse_series(self._container({"4.º Trimestre de 2019": self._rows() + [{"geocod": "9Z9", "dim_3": "H.1", "valor": "1"}]}))
        with self.assertRaises(ValueError):
            F.parse_series(self._container({"4.º Trimestre de 2019": [{"geocod": "PT", "dim_3": "H9", "valor": "1"}]}))

    def test_a_selection_must_name_every_extra_dimension(self) -> None:
        s = F.parse_series(self._container({"4.º Trimestre de 2019": self._rows()}))
        with self.assertRaises(ValueError):
            s.get()
        with self.assertRaises(ValueError):
            s.get({"3": "H.9"})
        with self.assertRaises(ValueError):
            s.category("3", "Usados")

    def test_nuts2024_keys_match_municipalities_by_dico_and_drop_other_levels(self) -> None:
        s = F.parse_series(self._container({"4.º Trimestre de 2019": self._rows()}))
        geo = R.Geography(municipality={"0001": "1X00001", "0002": "1X00002"}, nuts3={"0001": "1X0", "0002": "1X0"},
                          names={}, members3={"1X0": ["0001", "0002"]}, members2={"1X": ["0001", "0002"]},
                          nuts2_of3={"1X0": "1X"})
        values, missing = F.on_nuts2024(s, geo, {"3": "H.1"})
        self.assertEqual(sorted(r for r, _y in values), ["1X", "1X0", "1X00001", "1X00002", "PT"])
        self.assertEqual(missing, {})
        renamed = R.Geography(municipality={"0001": "9Y00001", "0002": "9Y00002"}, nuts3={"0001": "1X0", "0002": "1X0"},
                              names={}, members3={"1X0": ["0001", "0002"]}, members2={"1X": ["0001", "0002"]},
                              nuts2_of3={"1X0": "1X"})
        self.assertIn(("9Y00001", 2019), F.on_nuts2024(s, renamed, {"3": "H.1"})[0])
        other = R.Geography(municipality={"0001": "1X00001"}, nuts3={"0001": "1X0"}, names={}, members3={"1X0": ["0001"]},
                            members2={"1X": ["0001"]}, nuts2_of3={"1X0": "1X"})
        with self.assertRaises(ValueError):
            F.on_nuts2024(s, other, {"3": "H.1"})


class ComparisonTests(unittest.TestCase):
    def test_compare_regions_counts_and_correlates(self) -> None:
        price = {(r, y): v for r, a, b in (("A", 100, 150), ("B", 100, 110), ("C", 100, 130), ("D", 100, 90))
                 for y, v in ((2019, a), (2024, b))}
        income = {(r, y): v for r, a, b in (("A", 100, 120), ("B", 100, 120), ("C", 100, 125), ("D", 100, 100))
                  for y, v in ((2019, a), (2024, b))}
        out = F.compare_regions(price, income, ["A", "B", "C", "D", "E"], 2019, 2024)
        self.assertEqual((out["regions"], out["regions_without_both_years"], out["price_outpaced_income"]), (4, 1, 2))
        self.assertEqual(out["max_gap"]["region"], "A")
        self.assertEqual(out["min_gap"]["region"], "D")
        self.assertEqual(F.compare_regions(price, income, ["A", "B"], 2019, 2024), {"regions": 2})

    def test_method_overlap(self) -> None:
        old = {("o1", 2019): 100.0, ("o2", 2019): 200.0, ("o1", 2020): 100.0}
        new = {("n1", 2019): 100.0, ("n2", 2019): 220.0}
        out = F.method_overlap(old, new, [("o1", "n1"), ("o2", "n2")], [2019, 2020])
        self.assertEqual((out["cells"], out["identical"], out["share_within_5pct"]), (2, 1, 0.5))
        self.assertEqual(out["largest"], {"old": "o2", "new": "n2", "year": 2019, "diff_pct": 10.0})


PINNED = sorted(DATA.glob(f"ine_*_{F.SNAPSHOT_DATE}.jsonl.gz"))


class PinnedDataTests(unittest.TestCase):
    """Checks on the pinned price and income snapshots and the panel built from them."""

    src: dict[str, F.Series]
    results: dict[str, Any]
    rows: list[dict[str, str]]

    @classmethod
    def setUpClass(cls) -> None:
        missing = [code for code in F.CODES if not (DATA / f"ine_{code}_{F.SNAPSHOT_DATE}.jsonl.gz").exists()]
        if missing:
            raise AssertionError(f"pinned INE containers missing: {missing}")
        cls.src = F.load(DATA, F.SNAPSHOT_DATE)
        cls.results = json.loads((HERE / "results_affordability.json").read_text(encoding="utf-8"))
        with (HERE / "affordability_panel.csv").open(newline="", encoding="utf-8") as fh:
            cls.rows = list(csv.DictReader(fh))

    def test_sources_cover_the_documented_periods(self) -> None:
        self.assertEqual(self.src["price"].years(), list(range(2019, 2026)))
        self.assertEqual(sorted(self.src["price"].dims["3"]), ["H.1", "H.1.1", "H.1.2"])
        self.assertLessEqual(self.src["income"].years()[0], 2019)
        self.assertEqual(self.src["income"].years()[-1], 2024)
        self.assertEqual(self.src["price_2018"].years()[-1], 2020)

    def test_geography_and_singleton_regions(self) -> None:
        geo = self.results["geography"]
        self.assertEqual((geo["municipalities"], geo["nuts3"], geo["nuts2"]), (308, 26, 9))
        pairs = {(p["nuts2"], p["nuts3"]) for p in geo["nuts2_equal_to_one_nuts3"]}
        self.assertIn(("1A", "1A0"), pairs)
        self.assertIn(("1B", "1B0"), pairs)

    def test_panel_ratio_is_income_over_price_and_never_filled(self) -> None:
        for row in self.rows:
            price, income, m2 = row["price_m2_total"], row["income_gross_per_household"], row["m2_per_income_year"]
            if m2:
                self.assertTrue(price and income)
                self.assertTrue(math.isclose(float(m2), float(income) / float(price), abs_tol=0.006))
            else:
                self.assertFalse(price and income)
            if not price:
                self.assertEqual(row["level"], "municipality")

    def test_regional_cells_are_complete_and_headline_reproduces(self) -> None:
        regional = [r for r in self.rows if r["level"] != "municipality" and 2019 <= int(r["year"]) <= 2024]
        self.assertEqual(len(regional), 36 * 6)
        self.assertTrue(all(r["m2_per_income_year"] for r in regional))
        pt = {int(r["year"]): r for r in self.rows if r["region_code"] == "PT"}
        gap = 100 * (math.log(float(pt[2024]["price_m2_total"]) / float(pt[2019]["price_m2_total"]))
                     - math.log(float(pt[2024]["income_gross_per_household"]) / float(pt[2019]["income_gross_per_household"])))
        self.assertAlmostEqual(self.results["regions"]["PT"]["growth_2019_2024"]["gap_log_points"], gap, places=1)

    def test_index_is_documented_as_relative(self) -> None:
        text = self.results["definitions"]["m2_per_income_year"] + self.results["definitions"]["price_income_index"]
        self.assertIn("not the years needed to buy", text)
        self.assertIn("not a level", text)


if __name__ == "__main__":
    unittest.main()
