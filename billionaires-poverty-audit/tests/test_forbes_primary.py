"""Offline synthetic-only checks of the Forbes 2026 primary-list auditor."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from forbes_primary import (
    aggregate_country_fields,
    compare_secondary,
    parse_annual_list,
    run_primary_audit,
)


def synthetic_json(n: int = 6, *, api_country: bool = True) -> bytes:
    """Create example records that are never treated as empirical people."""
    people = [
        {
            "uri": f"synthetic-{i}",
            "countryOfCitizenship": "Country A" if i < 3 else "Country B",
            **({"country": "Country B" if i % 2 else "Country A"}
               if api_country else {}),
        }
        for i in range(n)
    ]
    return json.dumps({"personList": {"persons": people}}).encode("utf-8")


class ForbesPrimaryTests(unittest.TestCase):
    """Fail closed when a person list is partial, altered or ambiguous."""

    def test_complete_synthetic_list(self) -> None:
        persons, info = parse_annual_list(synthetic_json(), expected_count=6)
        self.assertEqual(len(persons), 6)
        self.assertEqual(info["json_array_path"], "personList.persons")

    def test_plural_primary_list_still_checks_every_record(self) -> None:
        """Accept a known Forbes array spelling without changing audit gates."""
        document = json.loads(synthetic_json())
        section = document["personList"]
        section["personsLists"] = section.pop("persons")
        payload = json.dumps(document).encode("utf-8")
        persons, info = parse_annual_list(payload, expected_count=6)
        self.assertEqual(info["json_array_path"], "personList.personsLists")
        self.assertEqual(info["n_uris"], 6)
        counts, coverage = aggregate_country_fields(persons)
        self.assertEqual(counts["citizenship"]["Country A"], 3)
        self.assertEqual(coverage["citizenship"]["assigned"], 6)
        with self.assertRaisesRegex(ValueError, "Expected 3428"):
            parse_annual_list(payload)

    def test_plural_primary_output_remains_aggregated(self) -> None:
        """Record only country totals and an explicit source array path."""
        document = json.loads(synthetic_json())
        section = document["personList"]
        section["personsLists"] = section.pop("persons")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit"
            report = run_primary_audit(
                output, payload=json.dumps(document).encode("utf-8"),
                expected_count=6,
            )
            self.assertEqual(report["schema"]["json_array_path"], "personList.personsLists")
            self.assertEqual(
                sorted(file.name for file in output.iterdir()),
                ["audit.json", "counts_api_country.csv", "counts_citizenship.csv"],
            )

    def test_refuse_partial_annual_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 3428"):
            parse_annual_list(synthetic_json())

    def test_reject_html(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_annual_list(b"<html>blocked</html>")

    def test_duplicate_identifier(self) -> None:
        doc = json.loads(synthetic_json())
        doc["personList"]["persons"][0]["uri"] = "synthetic-1"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_annual_list(json.dumps(doc).encode(), expected_count=6)

    def test_distinct_geography_fields(self) -> None:
        persons, _ = parse_annual_list(synthetic_json(), expected_count=6)
        counts, coverage = aggregate_country_fields(persons)
        self.assertEqual(counts["citizenship"]["Country A"], 3)
        self.assertEqual(counts["api_country"]["Country B"], 3)
        self.assertEqual(coverage["citizenship"]["missing"], 0)

    def test_missing_country_never_inferred_as_residence(self) -> None:
        persons, _ = parse_annual_list(
            synthetic_json(api_country=False), expected_count=6
        )
        counts, coverage = aggregate_country_fields(persons)
        self.assertFalse(counts["api_country"])
        self.assertEqual(coverage["api_country"]["missing"], 6)

    def test_citizenship_coverage_gate(self) -> None:
        persons, _ = parse_annual_list(synthetic_json(), expected_count=6)
        del persons[0]["countryOfCitizenship"]
        with self.assertRaisesRegex(ValueError, "Citizenship field"):
            aggregate_country_fields(persons)

    def test_exact_label_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "secondary.csv"
            pd.DataFrame([
                {"country": "Country A", "iso3": "AAA", "billionaires": 2,
                 "billionaire_method": "citizenship"},
                {"country": "Country C", "iso3": "CCC", "billionaires": 9,
                 "billionaire_method": "citizenship"},
            ]).to_csv(csv, index=False)
            result = compare_secondary(
                Counter({"Country A": 3, "Country B": 3}), csv
            )
            by_name = {entry["country_label"]: entry for entry in result}
            self.assertEqual(by_name["Country A"]["difference"], 1)
            self.assertIsNone(by_name["Country B"]["secondary_count"])
            self.assertIsNone(by_name["Country C"]["forbes_count"])

    def test_aggregate_outputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outputs"
            audit = run_primary_audit(
                path, payload=synthetic_json(), expected_count=6
            )
            self.assertEqual(audit["schema"]["n_persons"], 6)
            self.assertEqual(len(audit["source_sha256"]), 64)
            self.assertEqual(
                sorted(f.name for f in path.iterdir()),
                ["audit.json", "counts_api_country.csv", "counts_citizenship.csv"],
            )
            with self.assertRaises(FileExistsError):
                run_primary_audit(path, payload=synthetic_json(), expected_count=6)

    def test_schema_change_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema changed"):
            parse_annual_list(json.dumps({"items": []}).encode(), expected_count=1)


if __name__ == "__main__":
    unittest.main()
