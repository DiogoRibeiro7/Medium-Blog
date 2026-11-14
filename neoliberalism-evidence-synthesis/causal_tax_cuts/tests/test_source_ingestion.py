"""Offline tests for WID's documented series and the author CSV probe."""

from __future__ import annotations

import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from causal_tax_cuts.event_source import download_author_file, inspect_csv, probe
from causal_tax_cuts.wid import (
    WID_PERCENTILE,
    WID_VARIABLE,
    download_country,
    parse_wid_csv,
    source_url,
    write_results,
)


def wid_rows() -> list[dict[str, str]]:
    """Two real-format but wholly synthetic World Inequality Database rows."""
    return [
        {
            "country": "US",
            "variable": WID_VARIABLE,
            "percentile": WID_PERCENTILE,
            "year": "1970",
            "value": "0.125",
        },
        {
            "country": "US",
            "variable": WID_VARIABLE,
            "percentile": WID_PERCENTILE,
            "year": "1971",
            "value": "0.14",
        },
    ]


class FakeResponse(io.BytesIO):
    """An in-memory network response with the headers expected by urllib."""

    def __init__(self, data: bytes, *, content_type: str = "text/csv") -> None:
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


class WIDSourceTests(unittest.TestCase):
    """Ensure the intended income concept and exact country mapping."""

    def test_wid_uses_explicit_source_url_and_rejects_non_paper_country(self) -> None:
        self.assertEqual(
            source_url("USA"), "https://wid.world/bulk_download/WID_data_US.csv"
        )
        with self.assertRaisesRegex(ValueError, "Unsupported country"):
            source_url("PRT")

    def test_wid_share_fraction_is_converted_to_percentage(self) -> None:
        observations = parse_wid_csv(
            wid_rows(), country="USA", start_year=1970, end_year=1972
        )
        self.assertEqual(len(observations), 2)
        self.assertAlmostEqual(observations[0].top1_share_pct, 12.5)
        self.assertAlmostEqual(observations[1].top1_share_pct, 14.0)

    def test_wid_does_not_blend_other_income_concepts(self) -> None:
        other = dict(wid_rows()[0], variable="sfiinc992j", value="0.95")
        result = parse_wid_csv(
            [*wid_rows(), other], country="USA", start_year=1970, end_year=1971
        )
        self.assertEqual(len(result), 2)

    def test_wid_rejects_duplicate_and_mislabelled_data(self) -> None:
        rows = wid_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_wid_csv(
                [*rows, rows[0]], country="USA", start_year=1970, end_year=1971
            )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            parse_wid_csv([dict(rows[0], country="GB")], country="USA")

    def test_wid_rejects_invalid_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected a fraction of 1"):
            parse_wid_csv([dict(wid_rows()[0], value="12.5")], country="USA")

    def test_streamed_wid_source_is_hashed_and_has_coverage_audit(self) -> None:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(wid_rows()[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(wid_rows())
        raw = output.getvalue().encode()
        with patch("causal_tax_cuts.wid.urlopen", return_value=FakeResponse(raw)):
            observations, audit = download_country(
                "USA", start_year=1970, end_year=1972
            )
        self.assertEqual(audit.sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(audit.n_observations, 2)
        self.assertEqual(audit.missing_years, (1972,))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            write_results(
                destination, observations, [audit], start_year=1970, end_year=1972
            )
            content = (destination / "wid_top1_pretax.csv").read_text(encoding="utf-8")
            self.assertIn("USA,1970,12.5", content)
            self.assertNotIn("USA,1972", content)
            self.assertTrue((destination / "wid_top1_provenance.json").is_file())


class AuthorEventProbeTests(unittest.TestCase):
    """Distinguish actual CSV payloads from Google sign-in pages."""

    def test_author_csv_detects_actual_columns_without_guessing(self) -> None:
        raw = b"Nation,Date,Shock\nUSA,1982,1\nGBR,1988,1\nFRA,1990,0\n"
        report = inspect_csv(raw)
        self.assertEqual(report["columns"], ["Nation", "Date", "Shock"])
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["sha256"], hashlib.sha256(raw).hexdigest())

    def test_html_sign_in_page_is_never_accepted_as_csv(self) -> None:
        with self.assertRaisesRegex(ValueError, "sign-in"):
            inspect_csv(b"<!DOCTYPE html><html><title>Sign in</title></html>")

    def test_public_download_falls_back_when_first_url_requires_sign_in(self) -> None:
        bad = b"<!DOCTYPE html><html><title>Sign in</title></html>"
        good = b"Nation;Date;Shock\nUSA;1982;1\nGBR;1988;1\nFRA;1990;0\n"
        with patch(
            "causal_tax_cuts.event_source.urlopen",
            side_effect=[
                FakeResponse(bad, content_type="text/html"),
                FakeResponse(good),
            ],
        ):
            content, url, report = download_author_file()
        self.assertEqual(content, good)
        self.assertIn("usercontent.google.com", url)
        self.assertEqual(report["delimiter"], ";")

    def test_probe_leaves_explicit_unavailable_audit_without_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "causal_tax_cuts.event_source.download_author_file",
                side_effect=RuntimeError("sign-in required"),
            ):
                report = probe(Path(directory))
            self.assertEqual(report["status"], "unavailable")
            self.assertTrue((Path(directory) / "author_event_probe.json").is_file())
            self.assertFalse(
                (Path(directory) / "hope_limberg_reforms_11_2020.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
