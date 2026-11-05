"""Synthetic-only checks for conservative Forbes country-label reconciliation."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from forbes_primary import run_primary_audit
from forbes_reconciliation import reconcile


def fake_people() -> bytes:
    """Construct six fictional records, never empirical Forbes evidence."""
    persons = [
        {
            "uri": f"fictional-{index}",
            "countryOfCitizenship": "Country A" if index < 3 else "Country B",
        }
        for index in range(6)
    ]
    return json.dumps({"personList": {"persons": persons}}).encode("utf-8")


class ForbesReconciliationTests(unittest.TestCase):
    """Check that source integrity and unresolved geography remain visible."""

    def setUp(self) -> None:
        """Use a fresh fictional primary audit and tiny secondary table."""
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.primary = self.root / "primary"
        self.secondary = self.root / "secondary.csv"
        self.output = self.root / "review"
        run_primary_audit(self.primary, payload=fake_people(), expected_count=6)
        # Only the TEST fixture simulates the metadata of a live response.
        # The production CLI rejects real `provided_test_fixture` ledgers.
        audit_path = self.primary / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["source_retrieval"]["mode"] = "single_response"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        self.secondary.write_text(
            "iso3,country,billionaires,billionaire_method,forbes_year\n"
            "AAA,Country A,2,citizenship,2026\n"
            "CCC,Country C,4,citizenship,2026\n",
            encoding="utf-8",
        )

    def test_exact_spelling_is_only_a_candidate(self) -> None:
        """Never treat a matching secondary name as verified primary ISO3."""
        summary = reconcile(self.primary, self.secondary, self.output, expected_people=6)
        self.assertEqual(summary["n_exact_label_candidates"], 1)
        self.assertEqual(summary["n_unmapped_primary_labels"], 1)
        self.assertEqual(summary["n_secondary_labels_absent_from_primary"], 1)
        self.assertEqual(summary["primary_people_with_unresolved_label"], 3)
        self.assertFalse(summary["approved_for_publication_or_regression"])
        self.assertFalse(summary["iso3_mappings_independently_verified"])
        with (self.output / "exact_label_candidates.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["iso3_candidate"], "AAA")
        self.assertEqual(rows[0]["difference"], "1")
        self.assertEqual(rows[0]["review_status"], "unverified_secondary_candidate")
        with (self.output / "secondary_without_exact_primary_label.csv").open(newline="") as stream:
            absent = list(csv.DictReader(stream))
        self.assertIn("not_a_verified_zero", absent[0]["reason"])

    def test_fixture_provenance_is_not_accepted(self) -> None:
        """Production reconciliation requires a real retrieval-mode ledger."""
        path = self.primary / "audit.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["source_retrieval"]["mode"] = "provided_test_fixture"
        path.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fixture"):
            reconcile(self.primary, self.secondary, self.output, expected_people=6)

    def test_altered_aggregate_is_rejected_by_digest(self) -> None:
        """Aggregate CSV must still be the one written by the source audit."""
        path = self.primary / "counts_citizenship.csv"
        path.write_text(path.read_text() + "Country C,1,citizenship,countryOfCitizenship\n")
        with self.assertRaisesRegex(ValueError, "aggregate changed"):
            reconcile(self.primary, self.secondary, self.output, expected_people=6)

    def test_api_country_cannot_be_silently_joined(self) -> None:
        """Even a correctly rehashed CSV with ambiguous geography must fail."""
        path = self.primary / "counts_citizenship.csv"
        path.write_text(path.read_text().replace("countryOfCitizenship", "country"))
        audit_path = self.primary / "audit.json"
        audit = json.loads(audit_path.read_text())
        audit["aggregated_output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "countryOfCitizenship"):
            reconcile(self.primary, self.secondary, self.output, expected_people=6)

    def test_assigned_count_must_match_primary_ledger(self) -> None:
        """No missing people can be hidden by the country mapping process."""
        audit_path = self.primary / "audit.json"
        audit = json.loads(audit_path.read_text())
        audit["coverage"]["citizenship"]["assigned"] = 5
        audit["coverage"]["citizenship"]["missing"] = 1
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "disagree"):
            reconcile(self.primary, self.secondary, self.output, expected_people=6)

    def test_duplicate_iso3_is_not_accepted(self) -> None:
        """A secondary transcription cannot silently merge two territories."""
        self.secondary.write_text(
            self.secondary.read_text().replace("CCC,Country C", "AAA,Country C")
        )
        with self.assertRaisesRegex(ValueError, "Duplicate secondary"):
            reconcile(self.primary, self.secondary, self.output, expected_people=6)

    def test_refuse_replacing_an_existing_ledger(self) -> None:
        """Preserve original evidence and require a new destination per run."""
        reconcile(self.primary, self.secondary, self.output, expected_people=6)
        with self.assertRaises(FileExistsError):
            reconcile(self.primary, self.secondary, self.output, expected_people=6)


if __name__ == "__main__":
    unittest.main()
