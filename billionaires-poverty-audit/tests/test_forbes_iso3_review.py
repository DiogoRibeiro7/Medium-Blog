"""Offline controls for the UN-backed Forbes ISO3 review, not regression data."""

from __future__ import annotations

import csv
from decimal import Decimal
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH: Path = Path(__file__).resolve().parents[1] / "forbes_iso3_review.py"
SPEC = importlib.util.spec_from_file_location("forbes_iso3_review", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ForbesISO3ReviewTests(unittest.TestCase):
    """Verify exact coverage, source authority, and the analysis veto."""

    def test_source_code_review_covers_every_primary_record(self) -> None:
        """An independent geography-code check is not regression approval."""
        rows = review.build_ledger()
        self.assertEqual(len(rows), 80)
        self.assertEqual(sum(row["primary_count"] for row in rows), 3428)
        self.assertEqual(len({row["iso3_candidate"] for row in rows}), 80)
        self.assertTrue(all(row["review_status"] == "iso3_code_crosschecked" for row in rows))
        self.assertTrue(
            all(row["approval_status"] == "not_approved_for_regression" for row in rows)
        )
        self.assertTrue(all(row["evidence_url"].startswith("https://") for row in rows))
        self.assertEqual(sum(row["reference_authority"] == "UNSD_M49" for row in rows), 79)
        self.assertEqual(
            sum(row["reference_authority"] == "UNPD_location_metadata" for row in rows), 1
        )
        self.assertEqual(len(review.PRIMARY_SHA256), 64)
        self.assertEqual(len(review.SECONDARY_SHA256), 64)
        self.assertEqual(len(review.UN_REFERENCE_SHA256), 64)

    def test_aliases_and_areas_have_source_backed_codes(self) -> None:
        """Record the code of each non-identical UN name and named area."""
        rows = {row["primary_label"]: row for row in review.build_ledger()}
        self.assertEqual(rows["Eswatini (Swaziland)"]["iso3_candidate"], "SWZ")
        self.assertEqual(rows["Eswatini (Swaziland)"]["un_geography_name"], "Eswatini")
        self.assertEqual(rows["Afghanistan"]["match_type"], "missing_from_secondary")
        self.assertEqual(rows["St. Kitts and Nevis"]["iso3_candidate"], "KNA")
        self.assertEqual(rows["St. Kitts and Nevis"]["m49_code"], "659")
        self.assertEqual(rows["Taiwan"]["iso3_candidate"], "TWN")
        self.assertEqual(rows["Taiwan"]["m49_code"], "158")
        self.assertEqual(rows["Taiwan"]["reference_authority"], "UNPD_location_metadata")
        self.assertEqual(rows["Hong Kong"]["un_geography_name"],
                         "China, Hong Kong Special Administrative Region")
        for label in review.TERRITORIAL_LABELS:
            self.assertEqual(rows[label]["territory_review"], "required")
        self.assertEqual(rows["Portugal"]["territory_review"], "standard")

    def test_country_count_differences_are_not_hidden_by_aliases(self) -> None:
        """Thirty-five differences and the 69-person gap survive code review."""
        rows = review.build_ledger()
        comparable = [row for row in rows if row["secondary_count"]]
        missing = [row for row in rows if not row["secondary_count"]]
        self.assertEqual(len(comparable), 76)
        self.assertEqual(len(missing), 4)
        self.assertEqual(sum(row["primary_count"] for row in missing), 5)
        self.assertEqual(
            sum(Decimal(row["secondary_count"]) for row in comparable), Decimal(3359)
        )
        self.assertEqual(
            sum(Decimal(row["difference_from_secondary"]) for row in comparable),
            Decimal(64),
        )
        self.assertEqual(
            sum(Decimal(row["difference_from_secondary"]) != 0 for row in comparable),
            35,
        )
        self.assertEqual(sum(row["primary_count"] for row in rows), 3428)

    def test_modified_reference_or_secondary_fails_closed(self) -> None:
        """A SHA-1 Git blob ID can never masquerade as file SHA-256."""
        with tempfile.TemporaryDirectory() as tempdir:
            altered = Path(tempdir) / "reference.csv"
            altered.write_bytes(review.REFERENCE.read_bytes() + b"\n")
            with patch.object(review, "REFERENCE", altered):
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    review.build_ledger()
        with patch.object(review, "SECONDARY_SHA256", "a" * 40):
            with self.assertRaisesRegex(ValueError, "64 hex"):
                review.build_ledger()

    def test_writer_rejects_overwrite(self) -> None:
        """Do not accidentally replace an earlier source-review output."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            previous: list[str] = sys.argv
            try:
                sys.argv = ["forbes_iso3_review.py", "--output", str(path)]
                review.main()
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 80)
                with self.assertRaises(FileExistsError):
                    review.main()
            finally:
                sys.argv = previous


if __name__ == "__main__":
    unittest.main()
