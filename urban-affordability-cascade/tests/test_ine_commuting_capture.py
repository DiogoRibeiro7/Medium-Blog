"""Offline tests for bounded, schema-agnostic official INE source capture.

All response bodies are synthetic. Tests do not contact the INE website.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ine_commuting_capture import (
    DATA_URL,
    MAX_BYTES,
    META_URL,
    CapturedJSON,
    _endpoint_kind,
    _parse_json,
    capture_sources,
    download_json,
)


class FakeResponse:
    """Small urllib-compatible response mock with a controlled final URL."""

    def __init__(
        self,
        body: bytes,
        *,
        final_url: str = DATA_URL,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.final_url = final_url
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def read(self, amount: int) -> bytes:
        return self.body[:amount]

    def geturl(self) -> str:
        return self.final_url


class FakeOpener:
    """Inject the response into the urllib client without network traffic."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request: Any = None
        self.timeout: float | None = None

    def open(self, request: Any, *, timeout: float) -> FakeResponse:
        self.request = request
        self.timeout = timeout
        return self.response


class SourceCaptureTests(unittest.TestCase):
    """Require truthful provenance and strict URL, JSON and byte boundaries."""

    def test_exact_official_source_allowlist(self) -> None:
        self.assertEqual(_endpoint_kind(META_URL), "metadata")
        self.assertEqual(_endpoint_kind(DATA_URL), "data_2021_candidate")
        self.assertEqual(
            _endpoint_kind(DATA_URL.replace("op=2&varcd=", "varcd=").replace(
                "&Dim1=", "&op=2&Dim1=")),
            "data_2021_candidate",
        )
        invalid = (
            DATA_URL.replace("https", "http"),
            DATA_URL.replace("www.ine.pt", "evil.ine.pt"),
            DATA_URL.replace("0012340", "0012341"),
            DATA_URL.replace("2021", "2022"),
            DATA_URL + "&Dim1=S7A2021",
            DATA_URL + "&other=1",
            DATA_URL + "#fragment",
        )
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                _endpoint_kind(url)

    def test_capture_saves_original_bytes_and_honest_unverified_manifest(self) -> None:
        metadata = b'[{"source": "INE", "dimensions": ["opaque"]}]'
        values = b'[{"IndicadorCod": "0012340", "Dados": {"2021": [1, 2]}}]'

        def fetcher(url: str) -> CapturedJSON:
            return CapturedJSON(
                payload=metadata if url == META_URL else values,
                final_url=url,
                content_type="application/json; charset=utf-8",
            )

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            result = capture_sources(destination, fetcher=fetcher)
            self.assertEqual(result["status"], "captured_unreconciled")
            self.assertFalse(result["semantics_verified"])
            self.assertFalse(result["frozen_pairs_reconciled"])
            self.assertEqual(result["indicator_id"], "0012340")
            self.assertTrue(result["candidate_data_endpoint_not_validated"])
            self.assertEqual(
                (destination / "ine_0012340_metadata.json").read_bytes(), metadata
            )
            self.assertEqual(
                (destination / "ine_0012340_data_2021_candidate.json").read_bytes(), values
            )
            self.assertEqual(
                result["sources"]["data_2021_candidate"]["sha256"],
                hashlib.sha256(values).hexdigest(),
            )
            self.assertEqual(
                result["sources"]["data_2021_candidate"]["schema_outline"]
                ["dados_period_keys"],
                ["2021"],
            )
            on_disk = json.loads(
                (destination / "capture_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(on_disk, result)

    def test_partial_capture_writes_manifest_and_no_fake_ine_data(self) -> None:
        def fetcher(url: str) -> CapturedJSON:
            if url == DATA_URL:
                raise TimeoutError("Synthetic server timeout")
            return CapturedJSON(b'{"name":"metadata"}', url, "application/json")

        with tempfile.TemporaryDirectory() as tmp:
            result = capture_sources(Path(tmp), fetcher=fetcher)
            self.assertEqual(result["status"], "capture_incomplete")
            self.assertEqual(result["sources"]["data_2021_candidate"]["status"],
                             "capture_failed")
            self.assertFalse(Path(tmp, "ine_0012340_data_2021_candidate.json").exists())
            self.assertTrue(Path(tmp, "ine_0012340_metadata.json").exists())
            self.assertTrue(Path(tmp, "capture_manifest.json").exists())

    def test_invalid_json_is_not_saved_as_a_successful_capture(self) -> None:
        def fetcher(url: str) -> CapturedJSON:
            return CapturedJSON(b"<html>forbidden</html>", url, "text/html")

        with tempfile.TemporaryDirectory() as tmp:
            result = capture_sources(Path(tmp), fetcher=fetcher)
            self.assertEqual(result["status"], "capture_incomplete")
            self.assertTrue(all(x["status"] == "capture_failed"
                                for x in result["sources"].values()))
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["capture_manifest.json"])

    def test_json_parser_rejects_duplicates_nonfinite_and_empty(self) -> None:
        for sample in (
            b'{"x": 1,"x": 2}',
            b'{"x": NaN}',
            b'{"x": Infinity}',
            b'{}',
            b'[]',
            b'false',
            b'not json',
            b'\xff',
        ):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                _parse_json(sample)
        self.assertEqual(_parse_json(b'[{"x": 2}]'), [{"x": 2}])

    def test_network_client_is_bounded_and_preserves_source_bytes(self) -> None:
        raw = b'{ "a": [3, 2, 1] }'
        fake = FakeOpener(FakeResponse(raw, headers={
            "Content-Length": str(len(raw)), "Content-Type": "text/plain",
        }))
        with patch("ine_commuting_capture.build_opener", return_value=fake):
            result = download_json(DATA_URL, max_bytes=len(raw), timeout_seconds=2.0)
        self.assertEqual(result.payload, raw)
        self.assertEqual(result.content_type, "text/plain")
        self.assertEqual(fake.timeout, 2.0)
        self.assertEqual(fake.request.get_method(), "GET")
        self.assertEqual(fake.request.full_url, DATA_URL)

    def test_content_length_and_actual_bytes_are_bounded(self) -> None:
        raw = b'{"hello": [1, 2, 3]}'
        for header in ({"Content-Length": str(MAX_BYTES + 1)},
                       {"Content-Length": "invalid"}):
            with self.subTest(header=header):
                fake = FakeOpener(FakeResponse(raw, headers=header))
                with patch("ine_commuting_capture.build_opener", return_value=fake):
                    with self.assertRaisesRegex(ValueError, "Content-Length"):
                        download_json(DATA_URL)
        fake = FakeOpener(FakeResponse(raw))
        with patch("ine_commuting_capture.build_opener", return_value=fake):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                download_json(DATA_URL, max_bytes=5)

    def test_redirect_changed_indicator_is_rejected(self) -> None:
        fake = FakeOpener(FakeResponse(b'[1]', final_url=META_URL))
        with patch("ine_commuting_capture.build_opener", return_value=fake):
            with self.assertRaisesRegex(ValueError, "Redirected URL"):
                download_json(DATA_URL)

    def test_invalid_limit_and_timeout_fail_without_http(self) -> None:
        for timeout in (float("nan"), 0.0, -2.0, 999.0, True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                download_json(DATA_URL, timeout_seconds=timeout)
        for limit in (0, -1, MAX_BYTES + 1, True, 2.2):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                download_json(DATA_URL, max_bytes=limit)

    def test_failure_does_not_preserve_stale_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "ine_0012340_data_2021_candidate.json")
            target.write_bytes(b"old data from another run")

            def always_fail(url: str) -> CapturedJSON:
                raise TimeoutError("No response")

            manifest = capture_sources(Path(tmp), fetcher=always_fail)
            self.assertEqual(manifest["status"], "capture_incomplete")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()