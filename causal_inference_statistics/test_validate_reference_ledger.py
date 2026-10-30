"""Regression tests for bibliography provenance validation."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_reference_ledger import validate_ledger, valid_evidence_url  # noqa: E402

FIELDS = [
    "reference_id", "reference", "doi", "status", "verified_source", "verification_note"
]


class ReferenceLedgerValidationTests(unittest.TestCase):
    """Exercise valid rows and common provenance failures."""

    def check_rows(self, rows: list[dict[str, str]]) -> list[str]:
        """Write a temporary ledger to test the public validator."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return validate_ledger(path)

    def test_valid_verified_and_pending_rows(self) -> None:
        rows = [
            {"reference_id": "one", "reference": "A (2020)", "doi": "",
             "status": "verified", "verified_source": "https://doi.org/10.1/test",
             "verification_note": "Checked publisher metadata"},
            {"reference_id": "two", "reference": "B (2021)", "doi": "",
             "status": "pending", "verified_source": "", "verification_note": ""},
        ]
        self.assertEqual(self.check_rows(rows), [])

    def test_placeholder_evidence_is_rejected(self) -> None:
        row = {"reference_id": "one", "reference": "A (2020)", "doi": "",
               "status": "verified", "verified_source": "h", "verification_note": "t"}
        self.assertEqual(len(self.check_rows([row])), 2)

    def test_invalid_urls_are_rejected(self) -> None:
        for url in ("", "ftp://example.com/a", "https:///missing-host", "https://site/a b"):
            with self.subTest(url=url):
                self.assertFalse(valid_evidence_url(url))

    def test_pending_evidence_and_duplicate_ids_are_rejected(self) -> None:
        row = {"reference_id": "one", "reference": "A (2020)", "doi": "",
               "status": "pending", "verified_source": "https://example.org",
               "verification_note": "Old verification"}
        problems = self.check_rows([row, row])
        self.assertTrue(any("duplicate" in message for message in problems))
        self.assertTrue(any("pending" in message for message in problems))


if __name__ == "__main__":
    unittest.main()
