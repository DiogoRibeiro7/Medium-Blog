"""Offline, synthetic contract tests for real 2021 Census category capture."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from extract_commuting import municipality_codes_from_panel
from ine_residential_categories_probe import CaptureResponse, municipality_candidate_url
from ine_residential_municipal_capture import (
    CAPTURE_PROTOCOL,
    CATEGORY_NAMES,
    SEX_NAMES,
    capture_amp,
    parse_municipality,
    validate_metadata,
)


def encode(value: Any) -> bytes:
    """Encode a synthetic INE-shaped fixture without representing real data."""
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def fake_metadata(names: dict[str, str]) -> bytes:
    """Produce a minimal four-dimension source metadata specimen."""
    categories: dict[str, list[dict[str, str]]] = {}
    for dim, pairs in (
        (1, {"S7A2021": "2021"}),
        (2, names),
        (3, SEX_NAMES),
        (4, CATEGORY_NAMES),
    ):
        for code, label in pairs.items():
            categories[f"Dim_Num{dim}_{code}"] = [
                {"dim_num": str(dim), "categ_cod": code, "categ_dsg": label}
            ]
    return encode([{
        "IndicadorCod": "0011631",
        "PrimeiroPeriodo": "2011",
        "UltimoPeriodo": "2021",
        "UnidadeMedida": "Número (N.º)",
        "Sucesso": {"Verdadeiro": [{"Msg": "OK"}]},
        "Dimensoes": {
            "Descricao_Dim": [{"dim_num": str(i)} for i in (1, 2, 3, 4)],
            "Categoria_Dim": [categories],
        },
    }])


def fake_municipality(code: str, names: dict[str, str]) -> bytes:
    """Build internally consistent but entirely artificial Census counts."""
    both = {"1": 10, "2": 100, "3": 30, "31": 12,
            "32": 8, "33": 6, "34": 4, "T": 140}
    rows: list[dict[str, str]] = []
    for sex, label in SEX_NAMES.items():
        for category, name in CATEGORY_NAMES.items():
            rows.append({
                "geocod": code,
                "geodsg": names[code],
                "dim_3": sex,
                "dim_3_t": label,
                "dim_4": category,
                "dim_4_t": name,
                "valor": str(both[category] if sex == "T" else both[category] // 2),
            })
    return encode([{
        "IndicadorCod": "0011631",
        "Sucesso": {"Verdadeiro": [{"Msg": "OK"}]},
        "Dados": {"2021": rows},
    }])


class INEResidentialMunicipalCaptureTests(unittest.TestCase):
    """Require all official category accounting and no inferred OD source."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names = municipality_codes_from_panel()
        cls.metadata = fake_metadata(cls.names)

    def test_metadata_matches_exact_amp_names_and_categories(self) -> None:
        report = validate_metadata(self.metadata, self.names)
        self.assertEqual(report["validated_amp_municipalities"], 17)
        self.assertEqual(report["residential_category_dimension"], 4)
        self.assertEqual(report["metadata_sha256"], hashlib.sha256(self.metadata).hexdigest())

        altered = json.loads(self.metadata)
        altered[0]["Dimensoes"]["Categoria_Dim"][0]["Dim_Num4_33"][0]["categ_dsg"] = "Other"
        with self.assertRaisesRegex(ValueError, "category 4:33"):
            validate_metadata(encode(altered), self.names)

    def test_municipal_category_counts_reconcile_with_both_sexes(self) -> None:
        raw = fake_municipality("1312", self.names)
        counts = parse_municipality(raw, "1312", self.names)
        self.assertEqual(counts["T"], 140)
        self.assertEqual(counts["33"], 6)
        self.assertEqual(counts["3"], 30)

    def test_missing_rows_suppression_wrong_filter_and_bad_sums_fail_closed(self) -> None:
        source = json.loads(fake_municipality("1312", self.names))
        changes: list[tuple[str, str, Any]] = [
            ("wrong filter", "geocod", "1317"),
            ("missing count", "valor", ""),
            ("suppressed", "valor", "x"),
            ("wrong label", "dim_4_t", "Origin municipality 1317"),
        ]
        for label, field, value in changes:
            edited = copy.deepcopy(source)
            edited[0]["Dados"]["2021"][0][field] = value
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_municipality(encode(edited), "1312", self.names)

        edited = copy.deepcopy(source)
        edited[0]["Dados"]["2021"].pop()
        with self.assertRaisesRegex(ValueError, "24 sex/category"):
            parse_municipality(encode(edited), "1312", self.names)

        edited = copy.deepcopy(source)
        target = next(row for row in edited[0]["Dados"]["2021"]
                      if row["dim_3"] == "T" and row["dim_4"] == "33")
        target["valor"] = "7"
        with self.assertRaisesRegex(ValueError, "male/female"):
            parse_municipality(encode(edited), "1312", self.names)

        edited = copy.deepcopy(source)
        for row in edited[0]["Dados"]["2021"]:
            if row["dim_4"] == "33":
                row["valor"] = str(int(row["valor"]) + (1 if row["dim_3"] != "T" else 2))
        with self.assertRaisesRegex(ValueError, "subgroup totals"):
            parse_municipality(encode(edited), "1312", self.names)

    def test_bounded_two_municipality_capture_records_real_scope(self) -> None:
        def fake_get(url: str) -> CaptureResponse:
            code = url.split("Dim2=")[1].split("&", 1)[0]
            return CaptureResponse(fake_municipality(code, self.names), url, "application/json")

        with tempfile.TemporaryDirectory() as folder:
            report = capture_amp(
                Path(folder), metadata_bytes=self.metadata,
                start_index=15, count=2, fetcher=fake_get,
            )
            self.assertEqual(report["requested_codes"], ["1317", "1318"])
            self.assertEqual(report["protocol"], CAPTURE_PROTOCOL)
            self.assertEqual(report["validated_count"], 2)
            self.assertFalse(report["full_residential_origin_destination_matrix_observed"])
            for item in report["items"]:
                self.assertEqual(item["status"], "validated_2021_migration_categories")
                self.assertEqual(item["both_sexes_counts"]["33"], 6)
                self.assertEqual(item["source_url"], municipality_candidate_url(item["code"]))
                raw = (Path(folder) / item["filename"]).read_bytes()
                self.assertEqual(item["source_sha256"], hashlib.sha256(raw).hexdigest())

    def test_one_failed_request_is_not_a_zero_or_a_success(self) -> None:
        def fake_get(url: str) -> CaptureResponse:
            code = url.split("Dim2=")[1].split("&", 1)[0]
            if code == "1318":
                raise TimeoutError("Synthetic timeout")
            return CaptureResponse(fake_municipality(code, self.names), url, "application/json")

        with tempfile.TemporaryDirectory() as folder:
            report = capture_amp(
                Path(folder), metadata_bytes=self.metadata,
                start_index=15, count=2, fetcher=fake_get,
            )
        self.assertEqual(report["validated_count"], 1)
        self.assertEqual(report["items"][1]["status"], "failed")
        self.assertFalse(report["items"][1]["captured_raw_unverified"])
        self.assertNotIn("both_sexes_counts", report["items"][1])


if __name__ == "__main__":
    unittest.main()
