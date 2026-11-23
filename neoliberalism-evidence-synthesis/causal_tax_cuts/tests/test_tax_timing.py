"""Synthetic, source-contract tests for annual tax-cut flag timing."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from causal_tax_cuts.tax_timing import (
    CONTEXT_ONLY,
    audit_annual_timing,
    load_author_flags,
    load_joined_outcomes,
    load_statute_contexts,
    save_timing_audit,
)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write an explicit synthetic CSV without missing header values."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root: Path) -> tuple[Path, Path, Path]:
    """Two countries, three flagged years, two contiguous sequences."""
    events = root / "events.csv"
    data = [
        {
            "iso3": country,
            "year": str(year),
            "taxcut_2SD": str(
                int((country, year) in {("AAA", 2001), ("AAA", 2002), ("BBB", 2004)})
            ),
        }
        for country in ("AAA", "BBB")
        for year in range(2000, 2005)
    ]
    write_csv(events, data + [{"iso3": "KOR", "year": "2001", "taxcut_2SD": "1"}])
    joined = root / "joined.csv"
    write_csv(
        joined,
        [
            {
                "country": row["iso3"],
                "year": row["year"],
                "treated": row["taxcut_2SD"],
            }
            for row in data
            if (row["iso3"], row["year"]) != ("AAA", "2002")
        ],
    )
    ledger = root / "ledger.csv"
    write_csv(
        ledger,
        [
            {
                "country": "AAA",
                "author_year": "2001",
                "title": "Illustrative statute",
                "enacted_on": "2000-11-11",
                "primary_source_url": (
                    "https://www.govinfo.gov/content/pkg/SYNTHETIC/example"
                ),
                "tax_period_text": "",
                "evidence_scope": CONTEXT_ONLY,
            }
        ],
    )
    return events, joined, ledger


def load_fixture(
    root: Path,
) -> tuple[
    dict[tuple[str, int], int],
    set[tuple[str, int]],
    dict[tuple[str, int], object],
    tuple[Path, Path, Path],
]:
    """Construct full data in the test's synthetic rectangle."""
    files = fixture(root)
    author = load_author_flags(
        files[0],
        countries=frozenset({"AAA", "BBB"}),
        start_year=2000,
        end_year=2004,
        expected_flags=3,
    )
    observed = load_joined_outcomes(files[1], author)
    context = load_statute_contexts(files[2], author)
    return author, observed, context, files


class TaxTimingTests(unittest.TestCase):
    """Guard against interpreting annual flags as legal action dates."""

    def test_source_import_discards_kor_only_and_never_adds_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            author, _, _, _ = load_fixture(Path(directory))
            self.assertEqual(len(author), 10)
            self.assertEqual(sum(author.values()), 3)
            self.assertNotIn(("KOR", 2001), author)

    def test_consecutive_flag_run_is_not_separate_law_or_moved_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            author, observed, context, _ = load_fixture(Path(directory))
            years, runs, report = audit_annual_timing(author, observed, context)
            self.assertEqual(len(years), 3)
            self.assertEqual(len(runs), 2)
            self.assertEqual(report["n_multiyear_runs"], 1)
            self.assertEqual(report["n_flag_years_inside_multiyear_runs"], 2)
            self.assertEqual(report["n_flag_years_after_previous_flag_same_country"], 1)
            self.assertEqual(report["missing_flag_year_ids"], ["AAA-2002"])
            self.assertEqual(report["n_flag_years_with_joined_wid_outcome"], 2)
            self.assertEqual(report["n_flags_with_verified_implementation_date"], 0)
            aaa = sorted(
                (item for item in years if item.country == "AAA"),
                key=lambda item: item.author_year,
            )
            self.assertEqual([item.author_year for item in aaa], [2001, 2002])
            self.assertEqual([item.run_start for item in aaa], [2001, 2001])
            self.assertEqual([item.year_position_in_run for item in aaa], [1, 2])
            self.assertEqual(aaa[0].statute_enacted_on, "2000-11-11")
            self.assertEqual(aaa[1].statute_context_scope, "no_statute_checked")

    def test_joined_panel_cannot_recode_original_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            author, _, _, files = load_fixture(Path(directory))
            source = files[1]
            contents = source.read_text(encoding="utf-8")
            source.write_text(contents.replace("AAA,2001,1", "AAA,2001,0"))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_joined_outcomes(source, author)

    def test_reject_missing_author_zero_instead_of_imputing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            author, _, _, files = load_fixture(Path(directory))
            source = files[0]
            records = list(csv.DictReader(source.open(encoding="utf-8")))
            write_csv(
                source,
                [
                    r
                    for r in records
                    if not (r["iso3"] == "AAA" and r["year"] == "2000")
                ],
            )
            with self.assertRaisesRegex(ValueError, "Unobserved author policy"):
                load_author_flags(
                    source,
                    countries=frozenset({"AAA", "BBB"}),
                    start_year=2000,
                    end_year=2004,
                    expected_flags=3,
                )
            self.assertEqual(author["AAA", 2000], 0)

    def test_reject_duplicate_and_ambiguous_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, files = load_fixture(Path(directory))
            with files[0].open("a", encoding="utf-8") as handle:
                handle.write("AAA,2001,1\n")
            with self.assertRaisesRegex(ValueError, "Duplicate author"):
                load_author_flags(
                    files[0],
                    countries=frozenset({"AAA", "BBB"}),
                    start_year=2000,
                    end_year=2004,
                    expected_flags=3,
                )

    def test_legislative_context_must_be_primary_and_noncausal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            author, _, _, files = load_fixture(Path(directory))
            source = files[2]
            content = source.read_text(encoding="utf-8")
            source.write_text(content.replace(CONTEXT_ONLY, "verified_reform_date"))
            with self.assertRaisesRegex(ValueError, "cannot certify"):
                load_statute_contexts(source, author)
            source.write_text(content.replace("www.govinfo.gov", "example.com"))
            with self.assertRaisesRegex(ValueError, "official statute"):
                load_statute_contexts(source, author)
            source.write_text(content.replace("2000-11-11", "2002-01-01"))
            with self.assertRaisesRegex(ValueError, "Future statute"):
                load_statute_contexts(source, author)

    def test_statute_without_matching_coded_flag_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            author, _, _, files = load_fixture(Path(directory))
            source = files[2]
            content = source.read_text(encoding="utf-8")
            source.write_text(content.replace("AAA,2001", "AAA,2000"))
            with self.assertRaisesRegex(ValueError, "not an observed author flag"):
                load_statute_contexts(source, author)

    def test_stale_original_reference_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            author, observed, context, files = load_fixture(root)
            flags, runs, report = audit_annual_timing(author, observed, context)
            previous = root / "prior.json"
            previous.write_text(
                json.dumps(
                    {
                        "joined_panel_sha256": hashlib.sha256(
                            files[1].read_bytes()
                        ).hexdigest(),
                        "n_original_observed_reform_events": 2,
                    }
                )
            )
            save_timing_audit(
                root / "out",
                flags,
                runs,
                report,
                author_path=files[0],
                joined_path=files[1],
                ledger_path=files[2],
                prior_reference=previous,
            )
            result = json.loads((root / "out/author_timing_audit.json").read_text())
            self.assertEqual(result["frozen_reference_status"], "same_source_validated")
            self.assertEqual(result["n_descriptive_consecutive_runs"], 2)
            previous.write_text(
                json.dumps(
                    {
                        "joined_panel_sha256": result["joined_panel_sha256"],
                        "n_original_observed_reform_events": 3,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Same-vintage reform count"):
                save_timing_audit(
                    root / "out",
                    flags,
                    runs,
                    report,
                    author_path=files[0],
                    joined_path=files[1],
                    ledger_path=files[2],
                    prior_reference=previous,
                )


if __name__ == "__main__":
    unittest.main()
