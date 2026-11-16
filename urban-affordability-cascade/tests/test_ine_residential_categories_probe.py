"""Offline tests for the bounded INE 0011631 residential category probe.

All responses are synthetic protocol fixtures. These tests never contact INE
and do not establish observed origin-destination migration counts.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ine_residential_categories_probe import (
    INDICATOR,
    METADATA_URL,
    PORTO_DATA_CANDIDATE_URL,
    PROTOCOL,
    CaptureResponse,
    capture,
    download_source,
    municipality_candidate_url,
    require_source_url,
)


def fixture_payload(kind: str, *, indicator: str = INDICATOR) -> bytes:
    """Build a minimal fake INE-shaped metadata or period response."""
    if kind == "metadata":
        item = {
            "IndicadorCod": indicator,
            "PrimeiroPeriodo": "2021",
            "UltimoPeriodo": "2021",
            "Dimensoes": {"Descricao_Dim": [{"dim_num": "1"}, {"dim_num": "2"}]},
        }
    elif kind == "porto_dim2_candidate":
        item = {"IndicadorCod": indicator, "Dados": {"2021": [{"valor": "12"}]}}
    else:
        raise ValueError("Unknown synthetic fixture kind")
    return json.dumps([item], ensure_ascii=False).encode("utf-8")


class ResidentialCategorySourceProbeTests(unittest.TestCase):
    """Demand exact source selection and transparent, bounded evidence."""

    def test_exact_official_urls_allow_no_parameter_or_host_substitution(self) -> None:
        self.assertEqual(require_source_url(METADATA_URL), "metadata")
        self.assertEqual(
            require_source_url(PORTO_DATA_CANDIDATE_URL), "porto_dim2_candidate"
        )
        self.assertEqual(require_source_url(municipality_candidate_url("1317")), "municipality_1317")
        rejected = (
            PORTO_DATA_CANDIDATE_URL.replace("www.ine.pt", "evil.example"),
            PORTO_DATA_CANDIDATE_URL.replace("https:", "http:"),
            PORTO_DATA_CANDIDATE_URL.replace("0011631", "0012340"),
            PORTO_DATA_CANDIDATE_URL.replace("1312", "1415"),
            PORTO_DATA_CANDIDATE_URL.replace("S7A2021", "S7A2011"),
            PORTO_DATA_CANDIDATE_URL + "&Dim2=1312",
            METADATA_URL + "#fragment",
        )
        for changed in rejected:
            with self.subTest(url=changed), self.assertRaises(ValueError):
                require_source_url(changed)
        with self.assertRaises(ValueError):
            municipality_candidate_url("1415")

    def test_saves_both_exact_fake_responses_and_unverified_receipts(self) -> None:
        def fake_get(url: str) -> CaptureResponse:
            kind = require_source_url(url)
            return CaptureResponse(fixture_payload(kind), url, "application/json")

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            report = capture(directory, fetcher=fake_get)
            saved = json.loads((directory / "probe_manifest.json").read_text("utf-8"))
            self.assertEqual(report, saved)
            self.assertEqual(report["protocol"], PROTOCOL)
            self.assertEqual(report["completed_json_captures"], 2)
            self.assertEqual(report["requested_kinds"], ["metadata", "porto_dim2_candidate"])
            self.assertFalse(report["source_semantics_verified"])
            self.assertFalse(report["official_previous_municipality_observed"])
            self.assertFalse(report["directed_residential_origin_destination_counts_reconciled"])
            for item in report["items"]:
                raw = (directory / item["filename"]).read_bytes()
                self.assertEqual(raw, fixture_payload(item["kind"]))
                self.assertEqual(item["source_sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(item["source_bytes"], len(raw))
                self.assertTrue(item["indicator_code_matches_request"])
                self.assertEqual(item["status"], "captured_unverified")

    def test_unexpected_indicator_remains_visibly_unverified(self) -> None:
        def fake_get(url: str) -> CaptureResponse:
            kind = require_source_url(url)
            return CaptureResponse(fixture_payload(kind, indicator="unexpected"), url, "")

        with tempfile.TemporaryDirectory() as folder:
            report = capture(Path(folder), fetcher=fake_get)
        self.assertEqual(report["completed_json_captures"], 2)
        self.assertTrue(all(
            item["indicator_code_matches_request"] is False for item in report["items"]
        ))
        self.assertFalse(report["source_semantics_verified"])

    def test_bad_json_and_failed_fetch_leave_no_stale_success_files(self) -> None:
        def fake_get(url: str) -> CaptureResponse:
            kind = require_source_url(url)
            if kind == "metadata":
                return CaptureResponse(b'{"x":1,"x":2}', url, "application/json")
            raise TimeoutError("Synthetic endpoint timeout")

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "ine_0011631_metadata.json").write_bytes(b"old")
            (directory / "ine_0011631_porto_candidate.json").write_bytes(b"old")
            report = capture(directory, fetcher=fake_get)
            self.assertEqual(report["completed_json_captures"], 0)
            self.assertEqual([item["status"] for item in report["items"]], ["failed", "failed"])
            self.assertFalse((directory / "ine_0011631_metadata.json").exists())
            self.assertFalse((directory / "ine_0011631_porto_candidate.json").exists())
            self.assertEqual(report["items"][1]["error_type"], "TimeoutError")

    def test_invalid_time_and_size_fail_before_any_network_request(self) -> None:
        for timeout in (0, -1, 121, float("inf"), float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                download_source(METADATA_URL, timeout_seconds=timeout)
        for max_bytes in (0, -1, True, 12 * 1024 * 1024 + 1):
            with self.subTest(max_bytes=max_bytes), self.assertRaises(ValueError):
                download_source(METADATA_URL, max_bytes=max_bytes)


if __name__ == "__main__":
    unittest.main()
