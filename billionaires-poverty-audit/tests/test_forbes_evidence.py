"""Offline invariants for the frozen, aggregate-only Forbes 2026 API capture.

The live source is not queried by these tests. A later source retrieval must
create a *new* dated evidence directory instead of replacing this snapshot.
"""

from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import unittest
from typing import Any, cast


ROOT: Path = (
    Path(__file__).resolve().parents[1]
    / "evidence"
    / "forbes-2026-run-42099896845"
)
SECONDARY: Path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "forbes_2026_citizenship_secondary.csv"
)
EXPECTED_RAW_SHA256: str = (
    "42fd8cec21e84360adb812fb206afdcb7814483ffd718b3087e8f73502c1d17c"
)
AGGREGATES: tuple[str, ...] = (
    "counts_api_country.csv",
    "counts_citizenship.csv",
    "secondary_discrepancies.csv",
)


def load_json(name: str) -> dict[str, Any]:
    """Read a single archived audit object with basic schema checking."""
    document: Any = json.loads((ROOT / name).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def load_csv(name: str) -> list[dict[str, str]]:
    """Read one immutable UTF-8 CSV, rejecting malformed/null cells."""
    with (ROOT / name).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(
        key is None or value is None
        for record in rows
        for key, value in record.items()
    ):
        raise ValueError(f"{name} has missing, malformed or empty rows")
    # DictReader permits None in its annotations; the guard excludes it.
    return cast(list[dict[str, str]], rows)


def counted_rows(name: str, *, attribution: str, source_field: str) -> dict[str, int]:
    """Validate that these are positive aggregates of the specified field."""
    counts: dict[str, int] = {}
    for row in load_csv(name):
        label: str = row["country_label"]
        count: int = int(row["billionaires"])
        if (
            not label.strip()
            or label in counts
            or count <= 0
            or row["attribution"] != attribution
            or row["source_field"] != source_field
        ):
            raise ValueError(f"Invalid {name} country aggregate")
        counts[label] = count
    return counts


class ForbesFrozenEvidenceTests(unittest.TestCase):
    """Pin the artifact bytes and independently check count reconciliation."""

    def test_all_original_member_hashes_are_unchanged(self) -> None:
        """Refuse edits to a snapshot instead of silently rewriting history."""
        manifest = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        expected: dict[str, str] = {}
        for line in manifest:
            digest, separator, filename = line.partition("  ")
            self.assertEqual(separator, "  ")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertNotIn(filename, expected)
            self.assertNotIn("..", Path(filename).parts)
            expected[filename] = digest
            observed = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            self.assertEqual(observed, digest, filename)
        self.assertEqual(len(expected), 8)
        self.assertTrue(set(AGGREGATES).issubset(expected))
        self.assertIn("audit.json", expected)
        self.assertIn("reconciliation/summary.json", expected)

    def test_source_provenance_and_aggregate_coverage(self) -> None:
        """The valid 3,428-person response is distinct from API country."""
        audit: dict[str, Any] = load_json("audit.json")
        self.assertEqual(audit["source_sha256"], EXPECTED_RAW_SHA256)
        self.assertEqual(audit["source_bytes"], 6_502_586)
        self.assertEqual(audit["schema"]["n_persons"], 3_428)
        self.assertEqual(audit["schema"]["json_array_path"], "personList.personsLists")
        self.assertEqual(audit["source_retrieval"]["mode"], "single_response")
        for filename, digest in audit["aggregated_output_sha256"].items():
            self.assertIn(filename, AGGREGATES)
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), digest)

        citizenship = counted_rows(
            "counts_citizenship.csv",
            attribution="citizenship",
            source_field="countryOfCitizenship",
        )
        api_country = counted_rows(
            "counts_api_country.csv",
            attribution="api_country",
            source_field="country",
        )
        self.assertEqual(len(citizenship), 80)
        self.assertEqual(sum(citizenship.values()), 3_428)
        self.assertEqual(len(api_country), 79)
        self.assertEqual(sum(api_country.values()), 3_407)
        self.assertEqual(audit["coverage"]["citizenship"]["missing"], 0)
        self.assertEqual(audit["coverage"]["api_country"]["missing"], 21)
        self.assertIn("MUST NOT be labelled residence", audit["geography_warning"])

    def test_reconciliation_is_explicitly_unapproved(self) -> None:
        """Unmatched labels are unresolved, not verified zero counts."""
        summary: dict[str, Any] = load_json("reconciliation/summary.json")
        self.assertFalse(summary["approved_for_publication_or_regression"])
        self.assertFalse(summary["historical_2026_vintage_verified"])
        self.assertFalse(summary["iso3_mappings_independently_verified"])
        self.assertFalse(summary["residence_attribution_verified"])
        self.assertEqual(summary["n_exact_label_candidates"], 75)
        self.assertEqual(summary["n_unmapped_primary_labels"], 5)
        self.assertEqual(summary["primary_people_with_unresolved_label"], 6)
        self.assertEqual(summary["primary_people_with_exact_label_candidate"], 3_422)
        self.assertEqual(summary["source_sha256"], EXPECTED_RAW_SHA256)
        self.assertEqual(
            summary["secondary_transcription_sha256"],
            hashlib.sha256(SECONDARY.read_bytes()).hexdigest(),
        )
        exact = load_csv("reconciliation/exact_label_candidates.csv")
        missing = load_csv("reconciliation/unmapped_primary_labels.csv")
        secondary_missing = load_csv(
            "reconciliation/secondary_without_exact_primary_label.csv"
        )
        self.assertEqual(len(exact), 75)
        self.assertEqual(len(missing), 5)
        self.assertEqual(len(secondary_missing), 1)
        self.assertEqual(secondary_missing[0]["secondary_label"], "Eswatini")
        self.assertEqual(sum(int(row["primary_count"]) for row in missing), 6)
        self.assertTrue(
            all(row["review_status"] == "unverified_secondary_candidate" for row in exact)
        )

    def test_secondary_differences_are_real_and_not_silently_zeroed(self) -> None:
        """35 shared labels differ; five primary labels lack secondary rows."""
        comparisons = load_csv("secondary_discrepancies.csv")
        matched = [row for row in comparisons if row["exact_name_match"] == "True"]
        changed = [row for row in matched if Decimal(row["difference"]) != 0]
        self.assertEqual(len(matched), 75)
        self.assertEqual(len(changed), 35)
        self.assertEqual(
            sum(Decimal(row["difference"]) for row in matched), Decimal(64)
        )
        self.assertEqual(
            sum(
                Decimal(row["secondary_count"])
                for row in comparisons
                if row["secondary_count"]
            ),
            Decimal(3_359),
        )
        self.assertEqual(
            sum(
                Decimal(row["forbes_count"])
                for row in comparisons
                if row["forbes_count"]
            ),
            Decimal(3_428),
        )
        self.assertEqual(
            {row["country_label"] for row in comparisons if not row["secondary_count"]},
            {
                "Afghanistan", "Eswatini (Swaziland)", "Pakistan", "Peru",
                "St. Kitts and Nevis",
            },
        )
        self.assertEqual(
            {row["country_label"] for row in comparisons if not row["forbes_count"]},
            {"Eswatini"},
        )


if __name__ == "__main__":
    unittest.main()
