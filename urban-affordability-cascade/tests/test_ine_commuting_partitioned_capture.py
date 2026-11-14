"""Offline tests of INE 2021 candidate municipality-partition capture.

All downloaded responses in these tests are synthetic. A successful capture
does not establish the effectiveness of the Dim2/Dim3 filtering parameter.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from extract_commuting import municipality_codes_from_panel
from ine_commuting_capture import CapturedJSON
from ine_commuting_partitioned_capture import (
    candidate_slice_url,
    capture_partition,
    main,
    require_slice_url,
    validate_metadata,
)


def mock_metadata() -> bytes:
    """Create a 17-municipality metadata skeleton matching the official schema."""
    codes = municipality_codes_from_panel()
    categories = {}
    for dimension in (2, 3):
        for code, name in codes.items():
            categories[f"Dim_Num{dimension}_{code}"] = [{
                "dim_num": str(dimension), "cat_id": code, "categ_cod": code,
                "categ_dsg": name, "categ_nivel": "5",
            }]
    return json.dumps([{
        "IndicadorCod": "0012340",
        "PrimeiroPeriodo": "2021",
        "UltimoPeriodo": "2021",
        "UnidadeMedida": "Número (N.º)",
        "Dimensoes": {
            "Descricao_Dim": [{"dim_num": str(i)} for i in (1, 2, 3)],
            "Categoria_Dim": [categories],
        },
    }], ensure_ascii=False).encode("utf-8")


def fake_response(url: str) -> CapturedJSON:
    """Return synthetic JSON, never an official observation."""
    value = b'[{"IndicadorCod":"0012340","Dados":{"2021":[{"value":"synthetic"}]}}]'
    return CapturedJSON(value, url, "application/json")


class PartitionedINECaptureTests(unittest.TestCase):
    """Test metadata gates, bounded geography and error evidence."""

    def setUp(self) -> None:
        self.source = mock_metadata()
        self.codes = municipality_codes_from_panel()
        self.digest = hashlib.sha256(self.source).hexdigest()

    def test_metadata_has_all_seventeen_dico_codes_on_two_dimensions(self) -> None:
        report = validate_metadata(self.source, self.codes)
        self.assertEqual(report["amp_municipalities_validated_in_both_dimensions"], 17)
        self.assertEqual(report["geographic_dimensions"], [2, 3])

    def test_metadata_capture_timestamp_does_not_change_definitions(self) -> None:
        document = json.loads(self.source)
        document[0]["DataExtracao"] = "2026-11-12T15:00:00Z"
        changed = json.dumps(document, ensure_ascii=False).encode("utf-8")
        self.assertNotEqual(hashlib.sha256(changed).hexdigest(), self.digest)
        report = validate_metadata(changed, self.codes)
        self.assertEqual(report["amp_municipalities_validated_in_both_dimensions"], 17)

    def test_metadata_changed_municipality_name_rejected(self) -> None:
        document = json.loads(self.source)
        document[0]["Dimensoes"]["Categoria_Dim"][0]["Dim_Num2_1312"][0][
            "categ_dsg"
        ] = "Wrong name"
        source = json.dumps(document, ensure_ascii=False).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "mismatched AMP"):
            validate_metadata(source, self.codes)

    def test_unknown_dico_and_changed_query_rejected(self) -> None:
        for invalid in ("AAA1", "131", "13122", "PT"):
            with self.assertRaises(ValueError):
                candidate_slice_url(invalid)
        url = candidate_slice_url("1312")
        require_slice_url(url, dico="1312", dimension="Dim2")
        require_slice_url(candidate_slice_url("1312", dimension="Dim3"),
                          dico="1312", dimension="Dim3")
        for invalid in (
            url.replace("1312", "1316"),
            url.replace("www.ine.pt", "example.org"),
            url.replace("S7A2021", "S7A2022"),
            url + "&Dim2=1312",
        ):
            with self.assertRaises(ValueError):
                require_slice_url(invalid, dico="1312", dimension="Dim2")

    def test_bounded_batch_writes_raw_responses_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = capture_partition(Path(temp), metadata_bytes=self.source,
                                       start_index=0, count=2,
                                       fetcher=fake_response)
            self.assertEqual(result["capture_count"], 2)
            self.assertEqual(result["completed_json_captures"], 2)
            self.assertFalse(result["candidate_slice_semantics_verified"])
            self.assertFalse(result["all_136_pairs_reconciled"])
            for row in result["items"]:
                self.assertEqual(row["status"], "captured_unverified")
                self.assertTrue((Path(temp) / row["filename"]).is_file())
                self.assertEqual(
                    hashlib.sha256((Path(temp) / row["filename"]).read_bytes()).hexdigest(),
                    row["sha256"],
                )
                self.assertFalse(row["filter_effect_verified"])

    def test_third_amp_index_targets_only_santa_maria_da_feira(self) -> None:
        """Pin the next bounded source-probe selection without network access."""
        with tempfile.TemporaryDirectory() as temp:
            result = capture_partition(
                Path(temp), metadata_bytes=self.source,
                start_index=2, count=1, dimension="Dim2",
                fetcher=fake_response,
            )
            self.assertEqual(result["requested_codes"], ["0109"])
            self.assertEqual(result["completed_json_captures"], 1)
            self.assertEqual(result["items"][0]["municipality"], "Santa Maria da Feira")
            self.assertEqual(
                result["items"][0]["candidate_url"],
                candidate_slice_url("0109", dimension="Dim2"),
            )

    def test_fourth_amp_index_targets_only_oliveira_de_azemeis(self) -> None:
        """Keep the fourth bounded source request fixed to DICO 0113."""
        with tempfile.TemporaryDirectory() as temp:
            report = capture_partition(
                Path(temp), metadata_bytes=self.source,
                start_index=3, count=1, dimension="Dim2",
                fetcher=fake_response,
            )
            self.assertEqual(report["requested_codes"], ["0113"])
            self.assertEqual(report["completed_json_captures"], 1)
            self.assertEqual(report["items"][0]["municipality"], "Oliveira de Azeméis")
            self.assertEqual(
                report["items"][0]["candidate_url"],
                candidate_slice_url("0113", dimension="Dim2"),
            )

    def test_fifth_amp_index_targets_only_sao_joao_da_madeira(self) -> None:
        """Keep the fifth bounded source request fixed to DICO 0116."""
        with tempfile.TemporaryDirectory() as temp:
            report = capture_partition(
                Path(temp), metadata_bytes=self.source,
                start_index=4, count=1, dimension="Dim2",
                fetcher=fake_response,
            )
            self.assertEqual(report["requested_codes"], ["0116"])
            self.assertEqual(report["completed_json_captures"], 1)
            self.assertEqual(report["items"][0]["municipality"], "São João da Madeira")
            self.assertEqual(
                report["items"][0]["candidate_url"],
                candidate_slice_url("0116", dimension="Dim2"),
            )

    def test_sixth_amp_index_targets_only_vale_de_cambra(self) -> None:
        """Keep the sixth bounded source request fixed to DICO 0119."""
        with tempfile.TemporaryDirectory() as temp:
            report = capture_partition(
                Path(temp), metadata_bytes=self.source,
                start_index=5, count=1, dimension="Dim2",
                fetcher=fake_response,
            )
            self.assertEqual(report["requested_codes"], ["0119"])
            self.assertEqual(report["completed_json_captures"], 1)
            self.assertEqual(report["items"][0]["municipality"], "Vale de Cambra")
            self.assertEqual(
                report["items"][0]["candidate_url"],
                candidate_slice_url("0119", dimension="Dim2"),
            )

    def test_failed_partition_remains_in_manifest_without_stale_success(self) -> None:
        def partial_fetch(url: str) -> CapturedJSON:
            if "Dim2=0107" in url:
                raise TimeoutError("synthetic timeout")
            return fake_response(url)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "ine_0012340_dim2_0107_candidate.json").write_text("stale")
            report = capture_partition(path, metadata_bytes=self.source,
                                       count=2, fetcher=partial_fetch)
            self.assertEqual(report["completed_json_captures"], 1)
            self.assertEqual(report["items"][1]["status"], "failed")
            self.assertFalse((path / "ine_0012340_dim2_0107_candidate.json").exists())
            self.assertTrue((path / "partition_manifest.json").exists())

    def test_reject_unbounded_and_unknown_batch_ranges(self) -> None:
        for args in (
            {"count": 0}, {"count": 4}, {"start_index": -1},
            {"start_index": 16, "count": 2},
            {"dimension": "Dim4"}, {"start_index": True},
        ):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    capture_partition(Path(temp), metadata_bytes=self.source,
                                      fetcher=fake_response, **args)

    def test_cli_applies_a_longer_bounded_slice_timeout(self) -> None:
        """Pass 75 seconds to the live downloader, using no real network calls."""
        observed: list[tuple[str, float]] = []

        def fake_download(url: str, *, timeout_seconds: float) -> CapturedJSON:
            observed.append((url, timeout_seconds))
            return fake_response(url)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / "metadata.json"
            metadata.write_bytes(self.source)
            with patch(
                "ine_commuting_partitioned_capture.download_slice",
                side_effect=fake_download,
            ), redirect_stdout(io.StringIO()):
                main([
                    "--metadata-json", str(metadata),
                    "--output", str(root / "capture"),
                    "--start-index", "4", "--count", "1",
                    "--dimension", "Dim2", "--timeout-seconds", "75",
                ])
            self.assertEqual(observed, [(candidate_slice_url("0116"), 75.0)])
            report = json.loads((root / "capture" / "partition_manifest.json").read_text())
            self.assertEqual(report["requested_codes"], ["0116"])
            self.assertEqual(report["completed_json_captures"], 1)

    def test_cli_rejects_nonfinite_or_unbounded_timeouts(self) -> None:
        """Reject invalid timeouts before metadata or network access."""
        for timeout in ("nan", "inf", "0", "-1", "121"):
            with self.subTest(timeout=timeout), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as captured:
                    main(["--output", "/tmp/unused-ine-timeout-test", "--timeout-seconds", timeout])
                self.assertEqual(captured.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
