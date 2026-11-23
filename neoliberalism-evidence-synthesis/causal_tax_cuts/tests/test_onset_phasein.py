"""Synthetic risk-set and statutory-period tests for tax-cut onset audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.onset_phasein import (
    ATTRIBUTION,
    audit_onsets,
    load_flags,
    load_provisions,
    load_support,
    save_audit,
)


def _csv(path: Path, fields: list[str], rows: list[tuple[str, ...]]) -> None:
    """Build minimal synthetic input with declared source headings."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    """Four flagged years in three runs, one of which lacks WID."""
    flags = root / "flags.csv"
    support = root / "support.csv"
    provisions = root / "provisions.csv"
    _csv(
        flags,
        [
            "country",
            "author_year",
            "run_start",
            "run_end",
            "run_year_count",
            "year_position_in_run",
            "wid_outcome_observed",
        ],
        [
            ("AAA", "2000", "2000", "2001", "2", "1", "True"),
            ("AAA", "2001", "2000", "2001", "2", "2", "True"),
            ("BBB", "2003", "2003", "2003", "1", "1", "True"),
            ("CCC", "2004", "2004", "2004", "1", "1", "False"),
        ],
    )
    _csv(
        support,
        ["country", "reform_year", "status", "n_horizon_five_donors"],
        [
            ("AAA", "2000", "baseline_donors_available", "2"),
            ("AAA", "2001", "invalid_treatment_history", "0"),
            ("BBB", "2003", "baseline_donors_available", "1"),
        ],
    )
    _csv(
        provisions,
        [
            "country",
            "author_year",
            "statute_title",
            "enacted_on",
            "provision_identifier",
            "first_applicable_period_start",
            "last_applicable_period_day",
            "taxable_period_kind",
            "official_source_url",
            "attribution_status",
        ],
        [
            (
                "AAA",
                "2000",
                "Synthetic law",
                "2000-07-01",
                "Section 1",
                "2000-04-06",
                "2001-04-05",
                "Fiscal year, synthetic",
                "https://www.legislation.gov.uk/ukpga/1988/39/section/24/enacted",
                ATTRIBUTION,
            )
        ],
    )
    return flags, support, provisions


class OnsetAuditTests(unittest.TestCase):
    """Check that onset counts and timing do not imply changed treatment."""

    def test_reform_onset_counts_and_future_run_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = _fixture(Path(tmp))
            flags = load_flags(a)
            support = load_support(b)
            periods = load_provisions(
                c, {(row.country, row.year) for row in flags}
            )
            onsets, report = audit_onsets(flags, support, periods)
            self.assertEqual(report["n_author_flag_years"], 4)
            self.assertEqual(report["n_author_flag_runs"], 3)
            self.assertEqual(report["n_continuation_flags"], 1)
            self.assertEqual(report["n_first_year_flags_with_wid"], 2)
            self.assertEqual(report["n_baseline_donor_eligible_onsets"], 2)
            self.assertEqual(report["n_eligible_onsets_with_repeat_by_five_years"], 1)
            self.assertEqual(report["n_repeat_next_year_same_run"], 1)
            self.assertEqual(report["n_repeat_new_run_within_five_years"], 0)
            self.assertEqual(report["n_provision_periods_starting_before_enactment"], 1)
            self.assertEqual(report["n_author_flags_with_identified_composite_effective_day"], 0)
            aa = next(row for row in onsets if row.country == "AAA")
            self.assertEqual(aa.next_flag_within_five_years, 2001)
            self.assertEqual(aa.next_flag_relation, "same_contiguous_flag_run")
            cc = next(row for row in onsets if row.country == "CCC")
            self.assertEqual(cc.previously_reported_support_status, "missing_wid_outcome")

    def test_reject_duplicate_and_gap_inside_contiguous_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, _, _ = _fixture(Path(tmp))
            raw = a.read_text()
            with a.open("a", encoding="utf-8") as handle:
                handle.write("AAA,2000,2000,2001,2,1,True\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_flags(a)
            a.write_text(raw.replace("AAA,2001,2000,2001,2,2,True\n", ""))
            with self.assertRaisesRegex(ValueError, "Incomplete flagged-year"):
                load_flags(a)

    def test_reject_unobserved_or_duplicate_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = _fixture(Path(tmp))
            flags = load_flags(a)
            status = load_support(b)
            del status["AAA", 2001]
            periods = load_provisions(
                c, {(row.country, row.year) for row in flags}
            )
            with self.assertRaisesRegex(ValueError, "does not cover each"):
                audit_onsets(flags, status, periods)
            with b.open("a", encoding="utf-8") as handle:
                handle.write("AAA,2000,baseline_donors_available,2\n")
            with self.assertRaisesRegex(ValueError, "Duplicate eligibility"):
                load_support(b)

    def test_reject_false_flag_specific_legal_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, _, c = _fixture(Path(tmp))
            flags = load_flags(a)
            contents = c.read_text(encoding="utf-8")
            c.write_text(contents.replace(ATTRIBUTION, "proves_exact_author_flag"))
            with self.assertRaisesRegex(ValueError, "cannot certify"):
                load_provisions(c, {(row.country, row.year) for row in flags})
            c.write_text(contents.replace("www.legislation.gov.uk", "example.org"))
            with self.assertRaisesRegex(ValueError, "official source"):
                load_provisions(c, {(row.country, row.year) for row in flags})

    def test_reject_future_unknown_legal_period(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, _, c = _fixture(Path(tmp))
            flags = load_flags(a)
            contents = c.read_text(encoding="utf-8")
            c.write_text(contents.replace("2001-04-05", "1999-04-05"))
            with self.assertRaisesRegex(ValueError, "ends before"):
                load_provisions(c, {(row.country, row.year) for row in flags})

    def test_reconcile_prior_archived_source_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = _fixture(root)
            flags = load_flags(a)
            support = load_support(b)
            periods = load_provisions(c, {(row.country, row.year) for row in flags})
            onsets, report = audit_onsets(flags, support, periods)
            upstream = root / "timing.json"
            donor = root / "donor.json"
            upstream.write_text(json.dumps({
                "author_source_sha256": "a" * 64,
                "joined_panel_sha256": "b" * 64,
                "n_author_flag_years": 4,
                "n_descriptive_consecutive_runs": 3,
                "n_flag_years_with_joined_wid_outcome": 3,
            }))
            donor.write_text(json.dumps({
                "joined_panel_sha256": "b" * 64,
                "n_eligible_reforms_before_future_censoring": 2,
            }))
            destination = root / "result"
            save_audit(
                destination, onsets, periods, report,
                flags_path=a, support_path=b, provision_path=c,
                timing_audit_path=upstream, donor_audit_path=donor,
            )
            audit = json.loads((destination / "tax_cut_onset_audit.json").read_text())
            self.assertEqual(audit["n_baseline_donor_eligible_onsets"], 2)
            with (destination / "tax_cut_onset_eligibility.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            donor.write_text(json.dumps({
                "joined_panel_sha256": "c" * 64,
                "n_eligible_reforms_before_future_censoring": 2,
            }))
            with self.assertRaisesRegex(ValueError, "different source panels"):
                save_audit(
                    destination, onsets, periods, report,
                    flags_path=a, support_path=b, provision_path=c,
                    timing_audit_path=upstream, donor_audit_path=donor,
                )


if __name__ == "__main__":
    unittest.main()
