"""Synthetic capture-integrity tests; never contact live INE endpoints.

An exact source-byte match authenticates the manifest-to-artifact relation,
not the source institution or the meaning of Census dimensions.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ine_commuting_capture import (
    DATA_URL,
    META_URL,
    CapturedJSON,
    capture_sources,
)
from ine_commuting_capture_review import review_capture, run


METADATA_JSON = b'{"indicator":"0012340","dimensions":["synthetic"]}'
DATA_JSON = b'[{"IndicadorCod":"0012340","Dados":{"2021":[{"dummy":"synthetic"}]}}]'


def fake_fetcher(url: str) -> CapturedJSON:
    """Return a synthetic response for each exact candidate request."""
    return CapturedJSON(
        payload=METADATA_JSON if url == META_URL else DATA_JSON,
        final_url=url,
        content_type="application/json",
    )


def partial_fetcher(url: str) -> CapturedJSON:
    """Simulate a real capture with unavailable candidate data response."""
    if url == DATA_URL:
        raise TimeoutError("synthetic timeout")
    return fake_fetcher(url)


class INECaptureReviewTests(unittest.TestCase):
    """Exercise integrity boundaries and explicit non-verification semantics."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.directory = Path(self._temp.name)

    def _capture(self, *, partial: bool = False) -> dict[str, object]:
        return capture_sources(
            self.directory,
            fetcher=partial_fetcher if partial else fake_fetcher,
        )

    def _alter_manifest(self, mutation: object) -> None:
        """Apply a typed test mutation to the captured JSON manifest."""
        manifest_path = self.directory / "capture_manifest.json"
        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not callable(mutation):
            raise TypeError("Mutation must be callable")
        mutation(obj)
        manifest_path.write_text(
            json.dumps(obj, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_complete_capture_is_integrity_checked_but_not_verified(self) -> None:
        self._capture()
        result = review_capture(self.directory)
        self.assertEqual(result["status"], "integrity_checked_unreconciled")
        self.assertEqual(result["saved_source_files"], 2)
        self.assertEqual(result["expected_source_files"], 2)
        self.assertIs(result["source_semantics_verified"], False)
        self.assertIs(result["frozen_pairs_reconciled"], False)
        self.assertIs(result["INE_publisher_authenticated_independently"], False)
        self.assertEqual(
            result["sources"]["metadata"]["sha256"],
            hashlib.sha256(METADATA_JSON).hexdigest(),
        )
        self.assertIn(
            "2021",
            result["sources"]["data_2021_candidate"]["schema_outline"]["dados_period_keys"],
        )

    def test_partial_capture_is_not_falsely_marked_complete(self) -> None:
        self._capture(partial=True)
        result = review_capture(self.directory)
        self.assertEqual(result["status"], "partial_capture_integrity_checked")
        self.assertEqual(result["saved_source_files"], 1)
        self.assertEqual(result["sources"]["data_2021_candidate"]["status"], "unavailable")
        self.assertFalse(result["source_semantics_verified"])

    def test_checksum_mismatch_is_rejected(self) -> None:
        self._capture()
        source = self.directory / "ine_0012340_metadata.json"
        source.write_bytes(b'{"indicator":"0012341"}')
        with self.assertRaisesRegex(ValueError, "byte count|digest mismatch"):
            review_capture(self.directory)

    def test_modified_json_with_same_length_is_rejected_by_hash(self) -> None:
        self._capture()
        source = self.directory / "ine_0012340_metadata.json"
        old = source.read_bytes()
        edited = old.replace(b"synthetic", b"fictional")
        self.assertEqual(len(edited), len(old))
        source.write_bytes(edited)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            review_capture(self.directory)

    def test_tampered_manifest_hash_is_rejected(self) -> None:
        self._capture()
        self._alter_manifest(
            lambda obj: obj["sources"]["metadata"].update(sha256="0" * 64)
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            review_capture(self.directory)

    def test_missing_source_with_success_status_fails(self) -> None:
        self._capture()
        (self.directory / "ine_0012340_metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "captured JSON is missing"):
            review_capture(self.directory)

    def test_failed_capture_with_stale_source_is_rejected(self) -> None:
        self._capture(partial=True)
        (self.directory / "ine_0012340_data_2021_candidate.json").write_bytes(DATA_JSON)
        with self.assertRaisesRegex(ValueError, "failed capture must not include"):
            review_capture(self.directory)

    def test_wrong_indicator_and_false_verification_flags_fail(self) -> None:
        for update in (
            lambda obj: obj.update(indicator_id="0012341"),
            lambda obj: obj.update(semantics_verified=True),
            lambda obj: obj.update(frozen_pairs_reconciled=True),
            lambda obj: obj.update(candidate_data_endpoint_not_validated=False),
        ):
            self._capture()
            self._alter_manifest(update)
            with self.assertRaises(ValueError):
                review_capture(self.directory)

    def test_wrong_request_url_and_response_url_fail(self) -> None:
        self._capture()
        self._alter_manifest(
            lambda obj: obj["sources"]["metadata"].update(requested_url=DATA_URL)
        )
        with self.assertRaisesRegex(ValueError, "requested official endpoint"):
            review_capture(self.directory)
        self._capture()
        self._alter_manifest(
            lambda obj: obj["sources"]["metadata"].update(response_url=DATA_URL)
        )
        with self.assertRaisesRegex(ValueError, "response URL"):
            review_capture(self.directory)

    def test_missing_and_duplicate_source_entries_fail(self) -> None:
        self._capture()
        self._alter_manifest(lambda obj: obj["sources"].pop("metadata"))
        with self.assertRaisesRegex(ValueError, "exactly the two"):
            review_capture(self.directory)
        self._capture()
        self._alter_manifest(
            lambda obj: obj["sources"].update({"unexpected": {}})
        )
        with self.assertRaisesRegex(ValueError, "exactly the two"):
            review_capture(self.directory)

    def test_inconsistent_overall_status_is_rejected(self) -> None:
        self._capture(partial=True)
        self._alter_manifest(
            lambda obj: obj.update(status="captured_unreconciled")
        )
        with self.assertRaisesRegex(ValueError, "Overall capture status"):
            review_capture(self.directory)

    def test_timestamp_is_required_to_be_utc(self) -> None:
        self._capture()
        self._alter_manifest(
            lambda obj: obj.update(captured_at_utc="2026-11-11T10:00:00+01:00")
        )
        with self.assertRaisesRegex(ValueError, "UTC"):
            review_capture(self.directory)

    def test_structural_outline_must_agree_with_raw_json(self) -> None:
        self._capture()
        self._alter_manifest(
            lambda obj: obj["sources"]["metadata"].update(
                schema_outline={"root_type": "object"}
            )
        )
        with self.assertRaisesRegex(ValueError, "structural fingerprint"):
            review_capture(self.directory)

    def test_source_filename_cannot_escape_capture_directory(self) -> None:
        self._capture()
        self._alter_manifest(
            lambda obj: obj["sources"]["metadata"].update(
                filename="../../unrelated.json"
            )
        )
        with self.assertRaisesRegex(ValueError, "expected fixed basename"):
            review_capture(self.directory)

    def test_output_report_is_deterministic_and_no_live_requests(self) -> None:
        self._capture()
        output = self.directory / "report" / "review.json"
        first = run(self.directory, output)
        saved = output.read_bytes()
        second = run(self.directory, output)
        self.assertEqual(first, second)
        self.assertEqual(saved, output.read_bytes())
        self.assertEqual(json.loads(saved), first)
        self.assertEqual(first["status"], "integrity_checked_unreconciled")

    def test_reject_missing_manifest_and_invalid_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "Capture manifest"):
            review_capture(self.directory)
        with self.assertRaisesRegex(ValueError, "Capture directory"):
            review_capture(self.directory / "missing")


if __name__ == "__main__":
    unittest.main()
