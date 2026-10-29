"""Tests for the regional panel: container integrity, parsing, the NUTS mapping and the pinned INE data."""

import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import regional as R

DATA = Path(__file__).parent / "data"


def _response(kind: str, payload: Any, url: str = "https://www.ine.pt/x") -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False)
    raw = body.encode("utf-8")
    return {"kind": kind, "url": url, "retrieved_at_utc": "2026-10-29T00:00:00+00:00", "http_status": 200,
            "encoding": "utf-8", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "body": body}


def _meta(code: str, geo: list[tuple[str, str, int]]) -> list[dict[str, Any]]:
    cats = {f"Dim_Num2_{c}": [{"dim_num": "2", "cat_id": c, "categ_cod": c, "categ_dsg": n,
                               "categ_ord": str(i), "categ_nivel": str(lvl)}] for i, (c, n, lvl) in enumerate(geo)}
    return [{"IndicadorCod": code, "IndicadorNome": "test", "Dimensoes": {"Categoria_Dim": [cats]}}]


def _data(code: str, year: int, rows: list[dict[str, str]], ok: bool = True) -> list[dict[str, Any]]:
    return [{"IndicadorCod": code, "Sucesso": {"Verdadeiro" if ok else "Falso": [{"Msg": "x"}]},
             "Dados": {str(year): rows}}]


def _write(path: Path, header: dict[str, Any], lines: list[dict[str, Any]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for line in (header, *lines):
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


GEO = [("PT", "Portugal", 1), ("1", "Continente", 2), ("11", "Norte", 3), ("111", "Old A", 4),
       ("1110001", "Alpha", 5), ("1110002", "Beta", 5), ("112", "Old B", 4), ("1120003", "Gamma", 5)]


class ContainerTests(unittest.TestCase):
    """The container refuses edited, truncated or inconsistent files."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        rows = [{"geocod": c, "geodsg": n, "dim_3": "T", "valor": v} for (c, n, _), v in
                zip(GEO, ["60", "60", "60", "40", "30", "10", "20", "20"], strict=True)]
        rows.append({"geocod": "1120003", "geodsg": "Gamma", "dim_3": "1", "valor": "5"})
        self.rows = rows
        self.header = {"kind": "header", "indicator": "0000001", "name": "t", "filters": {"Dim3": "T"},
                       "years": [2020], "responses": 2}

    def _path(self, rows: list[dict[str, str]], ok: bool = True) -> Path:
        lines = [_response("metadata", _meta("0000001", GEO)), _response("data", _data("0000001", 2020, rows, ok))]
        return _write(self.dir / "c.jsonl.gz", self.header, lines)

    def test_parses_values_and_keeps_unavailable_cells_out(self) -> None:
        rows = [r for r in self.rows if r.get("dim_3") == "T"]
        rows[5] = {"geocod": "1110002", "geodsg": "Beta", "dim_3": "T", "sinal_conv": "x", "sinal_conv_desc": "Dado não disponível"}
        source = R.parse_source(self._path(rows))
        self.assertEqual(source.values[("1110001", 2020)], 30.0)
        self.assertNotIn(("1110002", 2020), source.values)
        self.assertEqual(source.unavailable[("1110002", 2020)], "Dado não disponível")

    def test_rejects_rows_outside_the_requested_filter(self) -> None:
        with self.assertRaises(ValueError):
            R.parse_source(self._path(self.rows))

    def test_rejects_failure_and_duplicates(self) -> None:
        rows = [r for r in self.rows if r.get("dim_3") == "T"]
        with self.assertRaises(ValueError):
            R.parse_source(self._path(rows, ok=False))
        with self.assertRaises(ValueError):
            R.parse_source(self._path(rows + [rows[0]]))

    def test_detects_an_edited_body_and_a_missing_response(self) -> None:
        rows = [r for r in self.rows if r.get("dim_3") == "T"]
        path = self._path(rows)
        lines = [json.loads(x) for x in gzip.open(path, "rt", encoding="utf-8")]
        lines[2]["body"] = lines[2]["body"].replace('"30"', '"31"')
        with self.assertRaises(ValueError):
            R.read_container(_write(self.dir / "edited.jsonl.gz", lines[0], lines[1:]))
        with self.assertRaises(ValueError):
            R.read_container(_write(self.dir / "short.jsonl.gz", lines[0], lines[1:2]))


class CorrespondenceTests(unittest.TestCase):
    """Old regions are linked to NUTS 2024 regions through shared municipalities."""

    def test_split_region_forms_one_group_and_totals_reconcile(self) -> None:
        # Old A = {Alpha, Beta} is split into new X = {Alpha} and Y = {Beta}; Old B = new Z.
        source = R.Source(code="0000001", name="t", title="t", last_update="", note="",
                          values={("111", 2020): 40.0, ("1110001", 2020): 30.0, ("1110002", 2020): 10.0,
                                  ("112", 2020): 20.0, ("1120003", 2020): 20.0, ("PT", 2020): 60.0},
                          geo_names={c: n for c, n, _ in GEO}, geo_levels={c: lvl for c, _, lvl in GEO},
                          geo_order=[c for c, _, _ in GEO], years=[2020])
        geo = R.Geography(municipality={"0001": "1X00001", "0002": "1Y00002", "0003": "1Z00003"},
                          nuts3={"0001": "1X0", "0002": "1Y0", "0003": "1Z0"}, names={},
                          members3={"1X0": ["0001"], "1Y0": ["0002"], "1Z0": ["0003"]},
                          members2={"1X": ["0001"], "1Y": ["0002"], "1Z": ["0003"]},
                          nuts2_of3={"1X0": "1X", "1Y0": "1Y", "1Z0": "1Z"})
        groups = R.correspondence(source, geo)
        self.assertIn((["111"], ["1X0", "1Y0"], ["0001", "0002"]), groups)
        self.assertIn((["112"], ["1Z0"], ["0003"]), groups)
        report = R.check_published_totals(source, geo)
        self.assertEqual(report["region_mismatches"], [])
        self.assertEqual(report["changed_groups"], [{"source_regions": ["111"], "nuts2024_regions": ["1X0", "1Y0"]}])
        source.values[("111", 2020)] = 41.0
        self.assertEqual(len(R.check_published_totals(source, geo)["region_mismatches"]), 1)

    def test_flags_follow_ine_notes(self) -> None:
        self.assertIn("completions_underreported_amadora", R.flags_for([R.AMADORA], 2015))
        self.assertNotIn("completions_underreported_amadora", R.flags_for([R.AMADORA], 2018))
        self.assertIn("construction_underreported_lisboa_seia", R.flags_for([R.SEIA], 2003))
        self.assertIn("completions_estimated", R.flags_for(["0001"], 2025))
        self.assertIn("population_administrative_basis", R.flags_for(["0001"], 2021))


PINNED = sorted(DATA.glob(f"ine_*_{R.SNAPSHOT_DATE}.jsonl.gz"))


class PinnedDataTests(unittest.TestCase):
    """Checks on the pinned INE snapshots and the panel built from them."""

    src: dict[str, R.Source]
    geo: R.Geography

    @classmethod
    def setUpClass(cls) -> None:
        if len(PINNED) != 6:
            raise AssertionError(f"expected 6 pinned INE containers, found {len(PINNED)}")
        cls.src = R.load_sources(DATA, R.SNAPSHOT_DATE)
        cls.geo = R.nuts2024_geography(cls.src["pop24"])

    def test_geography_shape_and_lisbon_union(self) -> None:
        self.assertEqual((len(self.geo.municipality), len(self.geo.members3), len(self.geo.members2)), (308, 26, 9))
        self.assertEqual(sorted(self.geo.regions()[R.LISBON_UNION]), sorted(self.geo.members3["1A0"] + self.geo.members3["1B0"]))
        self.assertIn(R.AMADORA, self.geo.members3["1A0"])
        self.assertIn(R.LISBOA, self.geo.members3["1A0"])

    def test_every_source_reconciles_with_its_own_published_totals(self) -> None:
        for label, source in self.src.items():
            with self.subTest(source=label):
                R.check_dico_universe(source, self.geo)
                report = R.check_published_totals(source, self.geo)
                self.assertEqual(report["region_mismatches"], [])
                self.assertEqual(report["national_mismatches"], [])

    def test_nuts2013_lisbon_maps_to_the_two_new_regions(self) -> None:
        groups = R.correspondence(self.src["pop13"], self.geo)
        news = [n for _o, n, _m in groups if "1A0" in n]
        self.assertEqual(news, [["1A0", "1B0"]])

    def test_panel_file_has_no_value_where_a_member_is_missing(self) -> None:
        panel = Path(__file__).parent / "regional_panel.csv"
        with panel.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({r["level"] for r in rows}, {"NUTS3", "NUTS2", "NUTS2 union", "country"})
        for row in rows:
            if row["completed_per_1000_residents"]:
                self.assertTrue(row["completed_dwellings"] and row["population"])


if __name__ == "__main__":
    unittest.main()
