"""Synthetic *test-only* tables for the future Census residential OD intake."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from residential_od_intake import (  # noqa: E402
    CONTRACT_VERSION,
    CSV_FIELDS,
    OUTSIDE_AMP,
    REFERENCE_END,
    REFERENCE_START,
    amp_codes,
    run,
)


class SyntheticODIntakeTests(unittest.TestCase):
    """Strict accounting; no example or synthetic flow is committed as data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.codes = amp_codes()
        cls.other_code = next(code for code in cls.codes if code != "1312")
        cls.second_other = next(code for code in cls.codes if code not in ("1312", cls.other_code))

    def rows(self) -> list[dict[str, str]]:
        """Construct a 306-cell synthetic cross-tab with correct destination sums."""
        counts = {
            (self.other_code, "1312"): 5_000,
            (OUTSIDE_AMP, "1312"): 10_724,
            ("1312", self.other_code): 25_000,
            (OUTSIDE_AMP, self.other_code): 83_894,
        }
        output: list[dict[str, str]] = []
        for origin in (*self.codes, OUTSIDE_AMP):
            for dest in self.codes:
                output.append({
                    "origin_dico": origin,
                    "destination_dico": dest,
                    "count": str(counts.get((origin, dest), 0)),
                    "disclosure_status": "structural_zero" if origin == dest else "observed",
                })
        return output

    def write_fixture(
        self, directory: Path, *, rows: list[dict[str, str]] | None = None,
        metadata_changes: dict[str, Any] | None = None,
    ) -> tuple[Path, Path, Path]:
        csv_path = directory / "synthetic_od_fixture.csv"
        meta_path = directory / "synthetic_metadata.json"
        output_path = directory / "derived_audit.json"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows() if rows is None else rows)
        metadata: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "publisher": "INE",
            "reference_start": REFERENCE_START,
            "census_reference": REFERENCE_END,
            "population_universe": "2021 usual residents with previous residence in another Portuguese municipality",
            "origin_scope": "AMP17_PLUS_REST_PT",
            "destination_scope": "AMP17",
            "unit": "persons",
            "domestic_previous_residence_only": True,
            "source_title": "SYNTHETIC TEST FIXTURE – NOT INE DATA",
            "source_citation": "Synthetic test fixture, generated only during unit tests",
            "license": "Synthetic test data",
            "disclosure_policy": "Empty count with status suppressed; no zero imputation",
            "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }
        metadata.update(metadata_changes or {})
        meta_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        return csv_path, meta_path, output_path

    def test_complete_17_destination_and_18_origin_grid(self) -> None:
        data = self.rows()
        self.assertEqual(len(self.codes), 17)
        self.assertEqual(len(data), 18 * 17)
        self.assertEqual(sum(row["disclosure_status"] == "structural_zero" for row in data), 17)

    def test_exact_synthetic_directional_exchange_is_identified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path, meta, output = self.write_fixture(Path(directory))
            result = run(csv_path, meta, output)
            self.assertEqual(result["suppressed_cells"], 0)
            self.assertEqual(result["official_destination_margins"]["porto_other_municipality"], 15_724)
            self.assertEqual(result["official_destination_margins"]["rest_amp_other_municipality"], 108_894)
            self.assertEqual(result["porto_to_other_amp_bounds"], {"lower": 25_000, "upper": 25_000})
            self.assertEqual(result["other_amp_to_porto_bounds"], {"lower": 5_000, "upper": 5_000})
            self.assertEqual(result["net_toward_other_amp_bounds"], {"lower": 20_000, "upper": 20_000})
            self.assertTrue(result["direction_point_identified"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            self.assertEqual(result["source"]["title"], "SYNTHETIC TEST FIXTURE – NOT INE DATA")

    def test_suppression_forced_by_remaining_mass_is_still_identified(self) -> None:
        data = self.rows()
        for row in data:
            if (row["origin_dico"], row["destination_dico"]) == ("1312", self.other_code):
                row.update(count="", disclosure_status="suppressed")
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            result = run(source, meta, out)
            self.assertEqual(result["suppressed_cells"], 1)
            self.assertEqual(result["porto_to_other_amp_bounds"], {"lower": 25_000, "upper": 25_000})

    def test_two_suppressions_allow_both_net_exchange_signs(self) -> None:
        data = self.rows()
        unknown = {("1312", self.other_code), (OUTSIDE_AMP, self.second_other)}
        for row in data:
            if (row["origin_dico"], row["destination_dico"]) in unknown:
                row.update(count="", disclosure_status="suppressed")
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            result = run(source, meta, out)
            self.assertEqual(result["porto_to_other_amp_bounds"], {"lower": 0, "upper": 25_000})
            self.assertEqual(result["other_amp_to_porto_bounds"], {"lower": 5_000, "upper": 5_000})
            self.assertEqual(result["net_toward_other_amp_bounds"], {"lower": -5_000, "upper": 20_000})
            self.assertFalse(result["direction_point_identified"])

    def test_suppression_in_porto_destination_expands_reverse_flow_bound(self) -> None:
        data = self.rows()
        unknown = {(self.other_code, "1312"), (OUTSIDE_AMP, "1312")}
        for row in data:
            if (row["origin_dico"], row["destination_dico"]) in unknown:
                row.update(count="", disclosure_status="suppressed")
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            result = run(source, meta, out)
            self.assertEqual(result["other_amp_to_porto_bounds"], {"lower": 0, "upper": 15_724})
            self.assertEqual(result["net_toward_other_amp_bounds"], {"lower": 9_276, "upper": 25_000})

    def test_missing_or_duplicate_pair_is_rejected(self) -> None:
        data = self.rows()
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete 18x17"):
                run(source, meta, out)
            source, meta, out = self.write_fixture(Path(directory), rows=data + [data[-1]])
            with self.assertRaisesRegex(ValueError, "Duplicate residential OD"):
                run(source, meta, out)

    def test_unknown_geography_or_renumbered_code_is_rejected(self) -> None:
        data = self.rows()
        data[0]["origin_dico"] = "9999"
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            with self.assertRaisesRegex(ValueError, "Unknown origin/destination"):
                run(source, meta, out)

    def test_wrong_diagonal_or_structural_zero_is_rejected(self) -> None:
        data = self.rows()
        diagonal = next(row for row in data if row["origin_dico"] == row["destination_dico"])
        diagonal["count"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            with self.assertRaisesRegex(ValueError, "diagonal must be structural zero"):
                run(source, meta, out)

    def test_noncanonical_number_and_suppressed_zero_are_rejected(self) -> None:
        data = self.rows()
        row = next(r for r in data if r["origin_dico"] == OUTSIDE_AMP and r["destination_dico"] == "1312")
        with tempfile.TemporaryDirectory() as directory:
            row["count"] = "010724"
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            with self.assertRaisesRegex(ValueError, "Invalid published nonnegative integer"):
                run(source, meta, out)
            row.update(count="0", disclosure_status="suppressed")
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            with self.assertRaisesRegex(ValueError, "Suppressed cell must have an empty count"):
                run(source, meta, out)

    def test_out_of_margin_counts_fail_even_if_every_cell_is_integer(self) -> None:
        data = self.rows()
        row = next(r for r in data if r["origin_dico"] == OUTSIDE_AMP and r["destination_dico"] == "1312")
        row["count"] = "10725"
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            with self.assertRaisesRegex(ValueError, "exceed an official destination margin"):
                run(source, meta, out)

    def test_under_margin_complete_table_cannot_pass(self) -> None:
        data = self.rows()
        row = next(r for r in data if r["origin_dico"] == "1312" and r["destination_dico"] == self.other_code)
        row["count"] = "24999"
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), rows=data)
            with self.assertRaisesRegex(ValueError, "does not reconcile with the Census margin"):
                run(source, meta, out)

    def test_wrong_census_window_or_scope_metadata_fails(self) -> None:
        for changes in (
            {"reference_start": "2019-01-01"},
            {"origin_scope": "PORTO_ONLY"},
            {"unit": "households"},
            {"domestic_previous_residence_only": False},
        ):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                source, meta, out = self.write_fixture(Path(directory), metadata_changes=changes)
                with self.assertRaisesRegex(ValueError, "Incorrect source metadata field"):
                    run(source, meta, out)

    def test_source_digest_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory), metadata_changes={"source_csv_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run(source, meta, out)

    def test_no_person_level_columns_in_aggregate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, meta, out = self.write_fixture(Path(directory))
            first = source.read_text(encoding="utf-8").splitlines()
            first[0] += ",individual_id"
            source.write_text("\n".join(first) + "\n", encoding="utf-8")
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            metadata["source_csv_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            meta.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema; personal fields forbidden"):
                run(source, meta, out)


if __name__ == "__main__":
    unittest.main()
