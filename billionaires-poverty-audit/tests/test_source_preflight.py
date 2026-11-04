"""Offline tests of source snapshot checks with explicitly synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from source_pipeline import SOURCE_URLS
from source_preflight import SourceValidationError, summarize_public_sources, verify_manifest


def wdi_document(indicator: str, rows: list[tuple[str, int, float]]) -> str:
    """Create a one-page synthetic WDI API response."""
    return json.dumps([
        {"page": 1, "pages": 1, "per_page": 100, "total": len(rows)},
        [
            {
                "countryiso3code": iso3,
                "date": str(year),
                "value": value,
                "indicator": {"id": indicator},
            }
            for iso3, year, value in rows
        ],
    ])


class SourcePreflightTests(unittest.TestCase):
    """Preflight must fail closed on missing, mismatched, or partial sources."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = {
            "owid_30usd_2021_ppp.csv": (
                "Entity,Code,Year,Poverty $30\n"
                "Example A,AAA,2023,40\n"
                "Example A,AAA,2024,35\n"
                "Example B,BBB,2025,55\n"
                "Example C,CCC,2024,78\n"
            ),
            "wdi_gdp_ppp_constant.json": wdi_document(
                "NY.GDP.PCAP.PP.KD",
                [("AAA", 2024, 40000), ("BBB", 2025, 10000), ("CCC", 2024, 2000)],
            ),
            "wdi_population.json": wdi_document(
                "SP.POP.TOTL",
                [("AAA", 2025, 100000), ("BBB", 2024, 1000000), ("CCC", 2025, 8000000)],
            ),
        }
        records = {}
        for name, content in self.data.items():
            payload = content.encode("utf-8")
            (self.root / name).write_bytes(payload)
            records[name] = {
                "source_url": SOURCE_URLS[name],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        (self.root / "SOURCE_MANIFEST.json").write_text(
            json.dumps({"retrieved_at_utc": "2026-11-04T08:00:00+00:00", "files": records}),
            encoding="utf-8",
        )

    def test_valid_frozen_sources(self) -> None:
        report = summarize_public_sources(self.root, minimum_overlap=3)
        self.assertEqual(report["n_countries_in_all_three"], 3)
        self.assertEqual(report["overlap_iso3"], ["AAA", "BBB", "CCC"])
        self.assertEqual(report["year_spans_of_matched_observations"]["poverty_year"], {"min": 2024, "max": 2025})
        self.assertIn("not necessarily", report["owid_year_semantics"])

    def test_minimum_overlap_detects_incomplete_capture(self) -> None:
        with self.assertRaisesRegex(SourceValidationError, "Only 3"):
            summarize_public_sources(self.root)

    def test_source_corruption_is_rejected(self) -> None:
        with (self.root / "wdi_population.json").open("a", encoding="utf-8") as f:
            f.write("corruption")
        with self.assertRaisesRegex(SourceValidationError, "SHA-256"):
            verify_manifest(self.root)

    def test_missing_manifest_rejected(self) -> None:
        (self.root / "SOURCE_MANIFEST.json").unlink()
        with self.assertRaisesRegex(SourceValidationError, "not found"):
            verify_manifest(self.root)

    def test_missing_source_rejected(self) -> None:
        (self.root / "owid_30usd_2021_ppp.csv").unlink()
        with self.assertRaisesRegex(SourceValidationError, "Missing"):
            verify_manifest(self.root)

    def test_wrong_source_url_rejected(self) -> None:
        path = self.root / "SOURCE_MANIFEST.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["files"]["wdi_population.json"]["source_url"] = "https://example.invalid"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(SourceValidationError, "URL mismatch"):
            verify_manifest(self.root)

    def test_missing_expected_file_in_manifest_rejected(self) -> None:
        path = self.root / "SOURCE_MANIFEST.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        del doc["files"]["wdi_population.json"]
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(SourceValidationError, "exactly"):
            verify_manifest(self.root)

    def test_byte_size_mismatch_rejected(self) -> None:
        path = self.root / "SOURCE_MANIFEST.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["files"]["wdi_population.json"]["bytes"] = 1
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(SourceValidationError, "Byte-size"):
            verify_manifest(self.root)

    def test_minimum_overlap_validation(self) -> None:
        for value in (0, -1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    summarize_public_sources(self.root, minimum_overlap=value)

    def test_wrong_indicator_fails(self) -> None:
        # Update payload and manifest together to test semantic validation after hash validation.
        name = "wdi_gdp_ppp_constant.json"
        payload = wdi_document(
            "SP.POP.TOTL",
            [("AAA", 2024, 40000), ("BBB", 2025, 10000), ("CCC", 2024, 2000)],
        ).encode("utf-8")
        (self.root / name).write_bytes(payload)
        path = self.root / "SOURCE_MANIFEST.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["files"][name]["sha256"] = hashlib.sha256(payload).hexdigest()
        doc["files"][name]["bytes"] = len(payload)
        path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected WDI"):
            summarize_public_sources(self.root, minimum_overlap=3)


if __name__ == "__main__":
    unittest.main()
