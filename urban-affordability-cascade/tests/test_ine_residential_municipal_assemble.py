"""Offline contracts for assembling incomplete, independently captured INE runs."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from extract_commuting import municipality_codes_from_panel
from ine_residential_categories_probe import CaptureResponse
from ine_residential_municipal_assemble import ASSEMBLY_PROTOCOL, assemble
from ine_residential_municipal_audit import CATEGORY_FIELDS, audit_archive
from ine_residential_municipal_capture import capture_amp
from test_ine_residential_municipal_capture import fake_metadata, fake_municipality


class INEResidentialMunicipalAssemblyTests(unittest.TestCase):
    """Require all 17 authentic-looking response receipts; fixtures are fake."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names = municipality_codes_from_panel()
        cls.metadata = fake_metadata(cls.names)

    def _attempt(
        self, root: Path, name: str, start: int, count: int,
    ) -> tuple[str, Path, Path]:
        """Create one simulated bounded source capture with exact checksums."""
        candidate = root / f"{name}_candidate"
        municipal = root / f"{name}_municipal"
        candidate.mkdir()
        porto = fake_municipality("1312", self.names)
        (candidate / "ine_0011631_metadata.json").write_bytes(self.metadata)
        (candidate / "ine_0011631_porto_candidate.json").write_bytes(porto)
        probe = {
            "protocol": "ine_0011631_2021_porto_category_source_probe_v1",
            "indicator_id": "0011631",
            "requested_kinds": ["metadata", "porto_dim2_candidate"],
            "source_semantics_verified": False,
            "directed_residential_origin_destination_counts_reconciled": False,
            "items": [
                {
                    "kind": kind,
                    "status": "captured_unverified",
                    "filename": filename,
                    "source_bytes": len(raw),
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                }
                for kind, filename, raw in (
                    ("metadata", "ine_0011631_metadata.json", self.metadata),
                    ("porto_dim2_candidate", "ine_0011631_porto_candidate.json", porto),
                )
            ],
        }
        (candidate / "probe_manifest.json").write_text(
            json.dumps(probe), encoding="utf-8"
        )

        def fake_fetch(url: str) -> CaptureResponse:
            code = url.split("Dim2=")[1].split("&", 1)[0]
            return CaptureResponse(fake_municipality(code, self.names), url, "application/json")

        capture_amp(
            municipal, metadata_bytes=self.metadata, start_index=start,
            count=count, fetcher=fake_fetch,
        )
        return name, candidate, municipal

    def _reference(self, destination: Path) -> Path:
        """Write independent synthetic AMP and Porto control totals."""
        counts = {
            "1": 10, "2": 100, "3": 30, "31": 12,
            "32": 8, "33": 6, "34": 4, "T": 140,
        }
        rows: list[dict[str, str]] = []
        for geography, multiplier in (("1312", 1), ("11A", 17)):
            row = {"geography_code": geography}
            row.update({field: str(counts[category] * multiplier)
                        for category, field in CATEGORY_FIELDS.items() if category != "3"})
            rows.append(row)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return destination

    def test_two_partial_runs_make_one_complete_auditable_archive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self._attempt(root, "run_one", 0, 9)
            second = self._attempt(root, "run_two", 9, 8)
            archive = root / "assembled"
            report = assemble([first, second], archive)
            self.assertEqual(report["archive_assembly_protocol"], ASSEMBLY_PROTOCOL)
            self.assertEqual(report["validated_count"], 17)
            self.assertEqual(len(report["source_attempt_receipts"]), 2)
            rows, provenance = audit_archive(
                archive, reference_csv=self._reference(root / "reference.csv")
            )
            self.assertEqual(len(rows), 17)
            self.assertEqual(provenance["source_attempt_count"], 2)
            self.assertEqual(provenance["amp_previous_other_portuguese_municipality"], 102)

            receipt = archive / "evidence" / "run_two_municipal_capture_manifest.json"
            receipt.write_bytes(receipt.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Changed original source attempt receipt"):
                audit_archive(archive, reference_csv=root / "reference.csv")

    def test_incomplete_source_runs_cannot_generate_archive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self._attempt(root, "run_one", 0, 9)
            destination = root / "assembled"
            with self.assertRaisesRegex(ValueError, "Missing validated INE municipality"):
                assemble([first], destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
