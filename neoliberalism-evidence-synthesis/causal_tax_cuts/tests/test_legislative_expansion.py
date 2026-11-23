"""Synthetic tests for statutory context and unchanged tax-cut event risk sets."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.legislative_expansion import (
    ATTRIBUTION,
    audit_legislative_coverage,
    read_evidence,
    run,
)

FIELDS = [
    "country",
    "author_year",
    "statute_title",
    "enacted_on",
    "provision_identifier",
    "first_applicable_period_start",
    "last_applicable_period_day",
    "taxable_period_kind",
    "official_source_url",
    "period_rule_source_url",
    "attribution_status",
]


def write_csv(path: Path, fields: list[str], rows: list[tuple[str, ...]]) -> None:
    """Generate an explicitly synthetic CSV, without imputation."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def case(
    country: str,
    year: int,
    *,
    enacted: str,
    period_start: str,
    host: str = "www.govinfo.gov",
    status: str = ATTRIBUTION,
) -> tuple[str, ...]:
    """Produce a fictional statute-provision context for a known flag-year."""
    url = f"https://{host}/example/{country}/{year}"
    return (
        country,
        str(year),
        "Synthetic Tax Act",
        enacted,
        "Section 1 (fictional rate rule)",
        period_start,
        "",
        "Calendar-year taxpayers only, not a composite effective day",
        url,
        url,
        status,
    )


def fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Write the 4-flag, 3-onset, 2-donor-eligible reference rectangle."""
    flags, onsets = root / "flags.csv", root / "onsets.csv"
    old, new, audit = root / "old.csv", root / "new.csv", root / "audit.json"
    write_csv(
        flags,
        ["country", "author_year", "run_start"],
        [
            ("AAA", "2000", "2000"),
            ("AAA", "2001", "2000"),
            ("BBB", "2002", "2002"),
            ("CCC", "2003", "2003"),
        ],
    )
    write_csv(
        onsets,
        ["country", "first_flag_year", "baseline_donor_eligible"],
        [
            ("AAA", "2000", "True"),
            ("BBB", "2002", "True"),
            ("CCC", "2003", "False"),
        ],
    )
    write_csv(old, FIELDS, [case("AAA", 2000, enacted="2000-07-01", period_start="2000-01-01")])
    write_csv(new, FIELDS, [case("BBB", 2002, enacted="2001-12-01", period_start="2002-01-01")])
    audit.write_text(
        json.dumps(
            {
                "status": "author_flag_onsets_and_provision_periods_only",
                "n_author_flag_years": 4,
                "n_author_flag_runs": 3,
                "n_baseline_donor_eligible_onsets": 2,
                "n_provision_context_rows": 1,
                "author_source_sha256": "a" * 64,
                "joined_panel_sha256": "b" * 64,
                "source_sha256": {
                    "author_flag_years": hashlib.sha256(flags.read_bytes()).hexdigest(),
                    "tax_provision_ledger": hashlib.sha256(old.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return flags, onsets, old, new, audit


class LegislativeExpansionTests(unittest.TestCase):
    """Enforce provenance, missingness and noncausal legislation semantics."""

    def test_complete_flag_register_and_unchanged_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = fixture(root)
            result = run(*files, root / "output")
            self.assertEqual(result["n_author_flag_years"], 4)
            self.assertEqual(result["n_original_onsets"], 3)
            self.assertEqual(result["n_baseline_donor_eligible_onsets"], 2)
            self.assertEqual(result["n_total_legal_context_flag_years"], 2)
            self.assertEqual(result["n_reviewed_eligible_onsets"], 2)
            self.assertEqual(result["n_reviewed_period_starts_before_enactment"], 1)
            self.assertEqual(result["n_eligible_onsets_causal_identified"], 0)
            with (root / "output/legislative_review_all_flag_years.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["causal_estimand_identified"] == "False" for row in rows))
            self.assertEqual(
                next(row for row in rows if row["author_year"] == "2001")[
                    "legal_context_status"
                ],
                "unreviewed",
            )

    def test_results_identical_given_identical_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = fixture(root)
            output = root / "output"
            run(*files, output)
            before = (output / "legislative_coverage_audit.json").read_bytes()
            run(*files, output)
            self.assertEqual(before, (output / "legislative_coverage_audit.json").read_bytes())
            self.assertNotIn(b"measured_at_utc", before)

    def test_duplicate_evidence_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, old, new, _ = fixture(root)
            write_csv(new, FIELDS, [case("AAA", 2000, enacted="2000-07-01", period_start="2000-01-01")])
            with self.assertRaisesRegex(ValueError, "Duplicate legal evidence"):
                read_evidence(old, new)

    def test_statute_for_unknown_author_flag_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flags, onsets, old, new, previous = fixture(root)
            write_csv(new, FIELDS, [case("ZZZ", 2002, enacted="2001-12-01", period_start="2002-01-01")])
            evidence, old_count, new_count = read_evidence(old, new)
            with flags.open(newline="", encoding="utf-8") as handle:
                flag_rows = list(csv.DictReader(handle))
            with onsets.open(newline="", encoding="utf-8") as handle:
                onset_rows = list(csv.DictReader(handle))
            with self.assertRaisesRegex(ValueError, "nonexistent 2-SD"):
                audit_legislative_coverage(
                    flag_rows,
                    onset_rows,
                    evidence,
                    json.loads(previous.read_text(encoding="utf-8")),
                    legacy_count=old_count,
                    expansion_count=new_count,
                )

    def test_unofficial_domain_or_verified_causal_mapping_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, old, new, _ = fixture(root)
            for invalid in (
                case("BBB", 2002, enacted="2001-12-01", period_start="2002-01-01", host="evil.example"),
                case("BBB", 2002, enacted="2001-12-01", period_start="2002-01-01", status="verified_causal_mapping"),
            ):
                with self.subTest(invalid=invalid[-1]):
                    write_csv(new, FIELDS, [invalid])
                    with self.assertRaises(ValueError):
                        read_evidence(old, new)

    def test_future_law_and_shifted_period_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, old, new, _ = fixture(root)
            for invalid in (
                case("BBB", 2002, enacted="2003-01-01", period_start="2002-01-01"),
                case("BBB", 2002, enacted="2001-12-01", period_start="2003-01-01"),
            ):
                with self.subTest(invalid=invalid[3]):
                    write_csv(new, FIELDS, [invalid])
                    with self.assertRaises(ValueError):
                        read_evidence(old, new)

    def test_previous_ledger_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flags, onsets, old, new, previous = fixture(root)
            old.write_text(old.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer match"):
                run(flags, onsets, old, new, previous, root / "out")

    def test_missing_original_onset_does_not_change_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            flags, onsets, old, new, previous = fixture(root)
            write_csv(
                onsets,
                ["country", "first_flag_year", "baseline_donor_eligible"],
                [("AAA", "2000", "True"), ("BBB", "2002", "True")],
            )
            with self.assertRaisesRegex(ValueError, "would change"):
                run(flags, onsets, old, new, previous, root / "out")


if __name__ == "__main__":
    unittest.main()
