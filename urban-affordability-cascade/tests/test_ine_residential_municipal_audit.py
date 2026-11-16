"""Offline, synthetic tests for the final 17-municipality source audit."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from extract_commuting import municipality_codes_from_panel
from ine_residential_categories_probe import CaptureResponse
from ine_residential_municipal_audit import (
    CATEGORY_FIELDS,
    ROOT,
    audit_archive,
    reconcile_categories,
)
from ine_residential_municipal_capture import capture_amp
from test_ine_residential_municipal_capture import fake_metadata, fake_municipality


class INEResidentialMunicipalAuditTests(unittest.TestCase):
    """Ensure source provenance and official destination totals stay intact."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names = municipality_codes_from_panel()
        cls.counts = {
            "1": 10, "2": 100, "3": 30, "31": 12,
            "32": 8, "33": 6, "34": 4, "T": 140,
        }

    def _reference(self, geography: str, multiplier: int) -> dict[str, str]:
        row = {"geography_code": geography}
        for category, field in CATEGORY_FIELDS.items():
            if category != "3":
                row[field] = str(multiplier * self.counts[category])
        return row

    def test_all_seventeen_synthetic_categories_reconcile_without_od_claim(self) -> None:
        observed = {code: dict(self.counts) for code in self.names}
        reference = {
            "1312": self._reference("1312", 1),
            "11A": self._reference("11A", 17),
        }
        rows = reconcile_categories(observed, self.names, reference)
        self.assertEqual(len(rows), 17)
        self.assertEqual(rows[0]["dico"], "0104")
        self.assertEqual(rows[0]["moved_from_other_municipality"], 6)
        self.assertEqual(rows[0]["eligible_2019_population"], 130)
        self.assertEqual(rows[0]["moved_from_other_municipality_pct_eligible"], "4.615385")

        # A disagreement in one other municipality must change the aggregate
        # margin and fail, even if Porto's individual source remains unchanged.
        changed = {code: dict(self.counts) for code in self.names}
        changed["1317"]["33"] += 1
        changed["1317"]["3"] += 1
        changed["1317"]["T"] += 1
        with self.assertRaisesRegex(ValueError, "INE 11A category"):
            reconcile_categories(changed, self.names, reference)

    def test_archived_source_receipts_are_verified_before_reporting(self) -> None:
        def fake_get(url: str) -> CaptureResponse:
            code = url.split("Dim2=")[1].split("&", 1)[0]
            return CaptureResponse(fake_municipality(code, self.names), url, "application/json")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "archive"
            archive.mkdir()
            metadata = fake_metadata(self.names)
            report = capture_amp(archive, metadata_bytes=metadata, fetcher=fake_get)
            self.assertEqual(report["validated_count"], 17)
            (archive / "ine_0011631_metadata.json.gz").write_bytes(gzip.compress(metadata))

            reference = [self._reference("1312", 1), self._reference("11A", 17)]
            reference_csv = root / "porto_reference.csv"
            with reference_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(reference[0]))
                writer.writeheader()
                writer.writerows(reference)

            rows, provenance = audit_archive(archive, reference_csv=reference_csv)
            self.assertEqual(len(rows), 17)
            self.assertEqual(provenance["porto_previous_other_portuguese_municipality"], 6)
            self.assertEqual(provenance["amp_previous_other_portuguese_municipality"], 102)
            self.assertFalse(provenance["source_data_is_named_origin_destination"])
            self.assertEqual(len(provenance["raw_source_files"]), 17)

            source = archive / "ine_0011631_dim2_1317_2021.json"
            source.write_bytes(source.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "source/manifest SHA"):
                audit_archive(archive, reference_csv=reference_csv)

            # A receipt must also preserve exact official endpoint scoping.
            source.write_bytes(fake_municipality("1317", self.names))
            receipt_path = archive / "municipal_capture_manifest.json"
            tampered_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            tampered_receipt["items"][15]["source_url"] = (
                "https://www.ine.pt/ine/json_indicador/pindica.jsp?"
                "op=2&varcd=0011631&Dim1=S7A2021&Dim2=1312&lang=PT"
            )
            receipt_path.write_text(json.dumps(tampered_receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Archive has wrong municipality identity"):
                audit_archive(archive, reference_csv=reference_csv)

    def test_original_status_and_complete_grid_are_required(self) -> None:
        observed = {code: dict(self.counts) for code in self.names}
        reference = {
            "1312": self._reference("1312", 1),
            "11A": self._reference("11A", 17),
        }
        del observed["1318"]
        with self.assertRaisesRegex(ValueError, "17 AMP"):
            reconcile_categories(observed, self.names, reference)

    def test_frozen_official_sources_reproduce_all_seventeen_margins(self) -> None:
        """Check the committed official source bytes entirely offline."""
        archive = ROOT / "data" / "ine_0011631_amp17_2021_source"
        rows, provenance = audit_archive(archive)
        self.assertEqual(len(rows), 17)
        self.assertEqual(provenance["porto_previous_other_portuguese_municipality"], 15724)
        self.assertEqual(provenance["amp_previous_other_portuguese_municipality"], 124618)
        self.assertFalse(provenance["source_data_is_named_origin_destination"])

        output = ROOT / "data" / "amp_census2021_municipal_residential_categories.csv"
        with output.open(newline="", encoding="utf-8") as stream:
            published = list(csv.DictReader(stream))
        self.assertEqual(len(published), 17)
        for result, row in zip(rows, published):
            self.assertEqual(row, {field: str(value) for field, value in result.items()})
        receipt = json.loads((ROOT / "data" /
                              "amp_census2021_municipal_residential_categories_provenance.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(
            receipt["derived_csv_sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
        )
        self.assertEqual(
            receipt["municipal_capture_manifest_sha256"],
            provenance["municipal_capture_manifest_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
