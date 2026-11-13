"""Offline integrity and numeric tests for one archived real INE Dim2 slice.

Unlike synthetic parser fixtures, the positive control checks the exact bytes
preserved from a real GitHub Actions capture against the pinned workbook
derivative. This establishes only one 16-pair comparison, never migration.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from extract_commuting import municipality_codes_from_panel
from ine_commuting_partition_reconciliation import (
    parse_dim2_slice,
    run,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "ine_0012340_2021_dim2_0104_raw.json"
MANIFEST = ROOT / "data" / "ine_0012340_2021_dim2_first_two_manifest.json"
AUDIT = ROOT / "data" / "ine_0012340_2021_dim2_0104_audit.json"
PINNED_SOURCE_SHA256 = "abbe42d2e98dd1522609da8d058663430f9b42fee99591634371379446c22635"
CODE = "0104"


class PinnedINESliceAuditTests(unittest.TestCase):
    """Pin a real incomplete official capture without extrapolating its scope."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_bytes()
        cls.manifest = MANIFEST.read_bytes()
        cls.codes = municipality_codes_from_panel()

    def payload(self) -> list[dict[str, Any]]:
        """Create a mutable copy of the archived response for negative tests."""
        return json.loads(self.source)

    def encode(self, value: Any) -> bytes:
        """Encode a deliberately modified test-only source, not official data."""
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    def resign_manifest(self, raw: bytes) -> bytes:
        """Simulate a new capture receipt for a structurally changed payload."""
        manifest = json.loads(self.manifest)
        entry = manifest["items"][0]
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
        entry["size_bytes"] = len(raw)
        return self.encode(manifest)

    def test_archived_source_matches_all_sixteen_amplinks_exactly(self) -> None:
        self.assertEqual(hashlib.sha256(self.source).hexdigest(), PINNED_SOURCE_SHA256)
        with tempfile.TemporaryDirectory() as folder:
            report = run(SOURCE, MANIFEST, CODE, Path(folder) / "audit.json")
            self.assertEqual(report["status"], "partial_match")
            self.assertEqual(report["matched_pairs"], 16)
            self.assertEqual(report["mismatched_pairs"], 0)
            self.assertEqual(len(report["comparison_rows"]), 16)
            self.assertTrue(all(row["match"] for row in report["comparison_rows"]))
            self.assertEqual(report["source_records"], 347)
            self.assertEqual(report["amp_geographies_present"], 17)
            self.assertEqual(report["diagonal_count"], 0)
            self.assertFalse(report["full_17_municipality_matrix_reconciled"])
            self.assertIn("not residential relocation", report["data_semantics"])
            self.assertEqual(json.loads(AUDIT.read_text(encoding="utf-8")), report)

    def test_missing_unchanged_source_digest_rejected(self) -> None:
        edited = self.payload()
        edited[0]["Dados"]["2021"][0]["valor"] = "99999"
        raw = self.encode(edited)
        with self.assertRaisesRegex(ValueError, "SHA-256/size"):
            validate_manifest(self.manifest, raw, CODE, self.codes)

    def test_modified_count_with_modified_receipt_reports_difference(self) -> None:
        edited = self.payload()
        row = next(row for row in edited[0]["Dados"]["2021"]
                   if row["dim_3"] == "0107")
        row["valor"] = str(int(row["valor"]) + 3)
        raw = self.encode(edited)
        receipt = self.resign_manifest(raw)
        validate_manifest(receipt, raw, CODE, self.codes)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_file, receipt_file = root / "edited_source.json", root / "edited_receipt.json"
            source_file.write_bytes(raw)
            receipt_file.write_bytes(receipt)
            report = run(source_file, receipt_file, CODE, root / "report.json")
            self.assertEqual(report["status"], "partial_discrepancies_detected")
            self.assertEqual(report["matched_pairs"], 15)
            self.assertEqual(report["mismatched_pairs"], 1)
            changed = [row for row in report["comparison_rows"] if not row["match"]]
            self.assertEqual(changed[0]["counterpart_code"], "0107")
            self.assertEqual(changed[0]["difference"], 3)

    def test_wrong_manifest_fields_fail_closed(self) -> None:
        changes: list[tuple[Callable[[dict[str, Any]], None], str]] = [
            (lambda value: value.update(dimension="Dim3"), "protocol"),
            (lambda value: value.update(source_data_is_residential_migration=True), "scope"),
            (lambda value: value["source_metadata"].update(metadata_unit="Percent"), "metadata"),
            (lambda value: value["items"][0].update(status="failed"), "success count"),
            (lambda value: value["items"][0].update(response_url="https://example.org/data"), "URL"),
            (lambda value: value["items"][0].update(requested_dimension="Dim3"), "record"),
        ]
        for change, label in changes:
            with self.subTest(label=label):
                manifest = json.loads(self.manifest)
                change(manifest)
                with self.assertRaises(ValueError):
                    validate_manifest(self.encode(manifest), self.source, CODE, self.codes)

    def test_changed_or_missing_geocod_fail(self) -> None:
        source = self.payload()
        source[0]["Dados"]["2021"][0]["geocod"] = "0107"
        with self.assertRaisesRegex(ValueError, "Dim2 filter"):
            parse_dim2_slice(self.encode(source), CODE, self.codes)

        source = self.payload()
        source[0]["Dados"]["2021"][0].pop("geocod")
        with self.assertRaisesRegex(ValueError, "lacks the source fields"):
            parse_dim2_slice(self.encode(source), CODE, self.codes)

    def test_missing_and_duplicate_amp_counterpart_fail(self) -> None:
        source = self.payload()
        rows = source[0]["Dados"]["2021"]
        rows.remove(next(row for row in rows if row["dim_3"] == "0107"))
        with self.assertRaisesRegex(ValueError, "Incomplete AMP"):
            parse_dim2_slice(self.encode(source), CODE, self.codes)

        source = self.payload()
        rows = source[0]["Dados"]["2021"]
        rows.append(dict(next(row for row in rows if row["dim_3"] == "0107")))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_dim2_slice(self.encode(source), CODE, self.codes)

    def test_inconsistent_names_and_count_are_rejected(self) -> None:
        for field, value, expected in (
            ("dim_3_t", "Different municipality", "code/name"),
            ("valor", None, "nonnegative integer"),
        ):
            source = self.payload()
            row = next(row for row in source[0]["Dados"]["2021"]
                       if row["dim_3"] == "0107")
            row[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                parse_dim2_slice(self.encode(source), CODE, self.codes)

    def test_nonzero_diagonal_wrong_period_and_indicator_rejected(self) -> None:
        source = self.payload()
        diagonal = next(row for row in source[0]["Dados"]["2021"]
                        if row["dim_3"] == CODE)
        diagonal["valor"] = "1"
        with self.assertRaisesRegex(ValueError, "diagonal"):
            parse_dim2_slice(self.encode(source), CODE, self.codes)

        source = self.payload()
        source[0]["IndicadorCod"] = "not-0012340"
        with self.assertRaisesRegex(ValueError, "indicator"):
            parse_dim2_slice(self.encode(source), CODE, self.codes)

        source = self.payload()
        source[0]["Dados"]["2022"] = source[0]["Dados"].pop("2021")
        with self.assertRaisesRegex(ValueError, "reference period"):
            parse_dim2_slice(self.encode(source), CODE, self.codes)


if __name__ == "__main__":
    unittest.main()
