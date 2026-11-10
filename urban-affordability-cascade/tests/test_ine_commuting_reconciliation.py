"""Synthetic-format and pinned-source tests for the INE commuting audit.

No test claims to have retrieved or authenticated a live INE API response.
Synthetic JSON only checks the explicitly documented schema assumptions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_commuting import municipality_codes_from_panel
from ine_commuting_reconciliation import (
    INDICATOR_ID,
    PERIOD,
    SOURCE_GEOGRAPHY,
    _source_url_is_exact_indicator,
    compare_pairs,
    parse_ine_matrix,
    read_code_map,
    read_frozen_pairs,
    read_receipt,
    run,
)

API_URL = (
    "https://www.ine.pt/ine/json_indicador/pindica.jsp?"
    "op=2&varcd=0012340&Dim1=S7A2021&lang=PT"
)


class SyntheticINECommutingTests(unittest.TestCase):
    """Exercise strict input and reconciliation rules on labelled mock data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.codes = municipality_codes_from_panel()
        cls.frozen = read_frozen_pairs()
        cls.mapping = {f"INE-{code}": code for code in cls.codes}

    @classmethod
    def build_payload(cls, *, difference: int = 0) -> bytes:
        """Return a synthetic INE-shaped 2021 17x17 matrix, not an INE fetch."""
        rows: list[dict[str, Any]] = []
        sorted_codes = sorted(cls.codes)
        first, second = sorted_codes[:2]
        for a in sorted_codes:
            for b in sorted_codes:
                if a == b:
                    n = 0
                else:
                    low, high = sorted((a, b))
                    n = cls.frozen[low, high] + (
                        difference if (low, high) == (first, second) else 0
                    )
                rows.append({
                    "geocod": f"INE-{a}",
                    "dim_3": f"INE-{b}",
                    "valor": n,
                })
        return json.dumps(
            [{"IndicadorCod": INDICATOR_ID, "Dados": {PERIOD: rows}}],
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def write_fixture(
        cls, folder: Path, payload: bytes | None = None
    ) -> tuple[Path, Path, Path, Path]:
        """Write synthetic source, receipt and reviewed fake-code mapping."""
        raw = cls.build_payload() if payload is None else payload
        source = folder / "mock_ine.json"
        receipt = folder / "receipt.json"
        code_map = folder / "code_map.csv"
        report = folder / "audit.json"
        source.write_bytes(raw)
        manifest: dict[str, object] = {
            "indicator_id": INDICATOR_ID,
            "reference_period": PERIOD,
            "geography": SOURCE_GEOGRAPHY,
            "primary_dimension_field": "geocod",
            "secondary_dimension_field": "dim_3",
            "count_field": "valor",
            "source_url": API_URL,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "downloaded_at_utc": datetime(2026, 11, 10).isoformat() + "Z",
        }
        receipt.write_text(
            json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with code_map.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["source_code", "dico"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(
                [{"source_code": raw_code, "dico": dico}
                 for raw_code, dico in sorted(cls.mapping.items())]
            )
        return source, receipt, code_map, report

    def test_frozen_data_are_checksum_pinned_and_have_136_pairs(self) -> None:
        self.assertEqual(len(self.codes), 17)
        self.assertEqual(len(self.frozen), 136)
        self.assertTrue(all(
            isinstance(count, int) and count >= 0
            for count in self.frozen.values()
        ))

    def test_synthetic_exact_match_does_not_authenticate_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            a, b, c, d = self.write_fixture(Path(temp))
            result = run(a, b, c, d)
            self.assertEqual(result["status"], "match")
            self.assertEqual(result["matched_pairs"], 136)
            self.assertEqual(result["mismatched_pairs"], 0)
            self.assertEqual(result["directed_views_validated"], 272)
            self.assertIn("operator-attested only", result["source_authenticity"])
            self.assertEqual(json.loads(d.read_text(encoding="utf-8")), result)

    def test_symmetric_disagreement_reports_one_unique_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source, receipt, code_map, report = self.write_fixture(
                Path(temp), self.build_payload(difference=3)
            )
            result = run(source, receipt, code_map, report)
            self.assertEqual(result["status"], "discrepancies_detected")
            self.assertEqual(result["mismatched_pairs"], 1)
            self.assertEqual(result["matched_pairs"], 135)
            self.assertEqual(result["mismatches"][0]["ine_minus_third_party"], 3)

    def test_wrong_digest_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source, receipt, code_map, report = self.write_fixture(Path(temp))
            receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
            receipt_data["response_sha256"] = "0" * 64
            receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run(source, receipt, code_map, report)
            self.assertFalse(report.exists())

    def test_source_url_is_strict_and_rejects_other_indicators(self) -> None:
        self.assertTrue(_source_url_is_exact_indicator(API_URL))
        for bad in (
            "http://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=0012340&Dim1=S7A2021&lang=PT",
            API_URL.replace("0012340", "0011675"),
            API_URL.replace("S7A2021", "S7A2022"),
            API_URL + "&extra=1",
            API_URL.replace("www.ine.pt", "example.com"),
        ):
            with self.subTest(bad=bad):
                self.assertFalse(_source_url_is_exact_indicator(bad))

    def test_wrong_period_or_indicator_fails_closed(self) -> None:
        source = json.loads(self.build_payload())
        for mutation, phrase in (
            (lambda x: x[0].update(IndicadorCod="1111111"), "indicator code"),
            (lambda x: x[0].update(Dados={"2022": x[0]["Dados"][PERIOD]}),
             "one 2021 data period"),
        ):
            fake = json.loads(self.build_payload())
            mutation(fake)
            with self.subTest(phrase=phrase):
                with self.assertRaisesRegex(ValueError, phrase):
                    parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)

    def test_missing_duplicate_and_asymmetric_pairs_fail(self) -> None:
        fake = json.loads(self.build_payload())
        rows = fake[0]["Dados"][PERIOD]
        rows.remove(next(r for r in rows if r["geocod"] != r["dim_3"]))
        with self.assertRaisesRegex(ValueError, "Incomplete INE AMP"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)
        fake = json.loads(self.build_payload())
        fake[0]["Dados"][PERIOD].append(dict(fake[0]["Dados"][PERIOD][1]))
        with self.assertRaisesRegex(ValueError, "Duplicated INE AMP"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)
        fake = json.loads(self.build_payload())
        first_off_diagonal = next(
            r for r in fake[0]["Dados"][PERIOD]
            if r["geocod"] != r["dim_3"]
        )
        first_off_diagonal["valor"] += 1
        with self.assertRaisesRegex(ValueError, "asymmetric"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)

    def test_missing_or_duplicate_diagonal_fails(self) -> None:
        fake = json.loads(self.build_payload())
        rows = fake[0]["Dados"][PERIOD]
        diagonal = next(r for r in rows if r["geocod"] == r["dim_3"])
        rows.remove(diagonal)
        with self.assertRaisesRegex(ValueError, "missing same-municipality diagonals"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)
        fake = json.loads(self.build_payload())
        rows = fake[0]["Dados"][PERIOD]
        diagonal = next(r for r in rows if r["geocod"] == r["dim_3"])
        rows.append(dict(diagonal))
        with self.assertRaisesRegex(ValueError, "Duplicated INE AMP diagonal"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)

    def test_nonzero_diagonal_and_suppression_fail(self) -> None:
        fake = json.loads(self.build_payload())
        diagonal = next(
            r for r in fake[0]["Dados"][PERIOD]
            if r["geocod"] == r["dim_3"]
        )
        diagonal["valor"] = 1
        with self.assertRaisesRegex(ValueError, "nonzero"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)
        fake = json.loads(self.build_payload())
        fake[0]["Dados"][PERIOD][1]["valor"] = None
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)

    def test_mapping_must_be_complete_and_one_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _, _, file, _ = self.write_fixture(Path(temp))
            self.assertEqual(read_code_map(file, self.codes), self.mapping)
            lines = file.read_text(encoding="utf-8").splitlines()
            file.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly the 17"):
                read_code_map(file, self.codes)
            file.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                read_code_map(file, self.codes)

    def test_unsupported_extra_geography_does_not_fill_missing_amp_pairs(self) -> None:
        fake = json.loads(self.build_payload())
        fake[0]["Dados"][PERIOD][1]["dim_3"] = "OUTSIDE-AMP"
        with self.assertRaisesRegex(ValueError, "Incomplete INE AMP"):
            parse_ine_matrix(json.dumps(fake).encode("utf-8"), self.mapping)

    def test_bad_receipt_metadata_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            a, b, _, _ = self.write_fixture(Path(temp))
            meta = json.loads(b.read_text(encoding="utf-8"))
            meta["geography"] = "Not 2021 geography"
            b.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Wrong INE receipt"):
                read_receipt(b, a.read_bytes())
            meta["geography"] = SOURCE_GEOGRAPHY
            meta["downloaded_at_utc"] = "2026-11-10T17:00:00"
            b.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "specify UTC"):
                read_receipt(b, a.read_bytes())


if __name__ == "__main__":
    unittest.main()
