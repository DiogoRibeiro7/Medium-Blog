"""Offline provenance and publication-veto tests for Forbes March 2026."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


# The standalone study runs without needing an installable Python package.
STUDY: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
import forbes_march_vintage as vintage  # noqa: E402


class ForbesMarchVintageTests(unittest.TestCase):
    """Check the historical-source comparison, not a poverty regression."""

    def test_frozen_official_top_ten_provenance(self) -> None:
        """The March 11 Forbes counts are separately cited and SHA-locked."""
        entries = vintage.march_top_ten()
        self.assertEqual(len(entries), 10)
        self.assertEqual(sum(item.count for item in entries.values()), 2494)
        self.assertEqual([item.rank for item in entries.values()], list(range(1, 11)))
        expected: dict[str, int] = {
            "United States": 989, "China": 539, "India": 229,
            "Germany": 212, "Russia": 147, "Italy": 89,
            "Canada": 82, "Hong Kong": 71, "Brazil": 70, "Taiwan": 66,
        }
        self.assertEqual({key: item.count for key, item in entries.items()}, expected)
        self.assertEqual(len(vintage.MARCH_TOP10_SHA256), 64)

    def test_november_is_not_exact_march_vintage(self) -> None:
        """Matching global totals must not erase three observed mismatches."""
        summary, rows = vintage.audit_march_vintage()
        self.assertEqual(summary["global_people_march_published"], 3428)
        self.assertEqual(summary["primary_people_november_api"], 3428)
        self.assertEqual(summary["primary_distinct_labels"], 80)
        self.assertEqual(summary["published_march_top10_labels"], 10)
        self.assertEqual(summary["march_top10_primary_matches"], 7)
        self.assertEqual(summary["march_top10_secondary_matches"], 10)
        self.assertEqual(
            {row["country"]: (row["march_published"], row["november_api"])
             for row in summary["march_top10_primary_conflicts"]},
            {"United States": (989, 991), "Russia": (147, 149), "Italy": (89, 90)},
        )
        self.assertFalse(summary["november_api_exact_march_distribution"])
        self.assertFalse(summary["historic_full_80_country_vintage_verified"])
        self.assertFalse(summary["approved_for_publication_or_regression"])
        self.assertEqual(len(rows), 80)
        self.assertTrue(all(row["approved_for_regression"] == "false" for row in rows))

    def test_secondary_differences_and_unknown_march_counts_remain_visible(self) -> None:
        """An unavailable March country count is unknown, never zero."""
        summary, rows = vintage.audit_march_vintage()
        self.assertEqual(summary["secondary_people_total"], 3359)
        self.assertEqual(summary["primary_minus_secondary_total"], 69)
        self.assertEqual(summary["comparable_secondary_labels"], 76)
        self.assertEqual(summary["comparable_secondary_disagreements"], 35)
        self.assertEqual(summary["comparable_primary_minus_secondary"], 64)
        self.assertEqual(summary["people_absent_from_secondary"], 5)
        self.assertEqual(
            set(summary["primary_labels_absent_from_secondary"]),
            {"Afghanistan", "Pakistan", "Peru", "St. Kitts and Nevis"},
        )
        unknown = [row for row in rows if not row["march_published_count"]]
        self.assertEqual(len(unknown), 70)
        self.assertTrue(all(row["primary_minus_march"] == "" for row in unknown))
        self.assertTrue(all(row["secondary_minus_march"] == "" for row in unknown))
        self.assertTrue(all(
            row["march_comparison_status"] == "not_independently_checked_against_march"
            for row in unknown
        ))
        afghanistan = next(row for row in rows if row["primary_label"] == "Afghanistan")
        self.assertEqual(afghanistan["secondary_citizenship_count"], "")
        self.assertEqual(afghanistan["primary_citizenship_count"], 1)

    def test_tampered_march_transcription_fails_closed(self) -> None:
        """The reference cannot silently drift without a reviewed new hash."""
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "altered_march.csv"
            altered.write_bytes(vintage.MARCH_TOP10.read_bytes() + b"\n")
            with patch.object(vintage, "MARCH_TOP10", altered):
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    vintage.march_top_ten()

    def test_writer_is_reproducible_and_rejects_overwrite(self) -> None:
        """Persist only generated aggregate comparisons and a source ledger."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "march_audit"
            expected = vintage.write_report(output)
            with (output / "summary.json").open(encoding="utf-8") as stream:
                actual = json.load(stream)
            self.assertEqual(expected, actual)
            with (output / "country_comparison.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 80)
            self.assertEqual(len([row for row in rows if row["march_published_count"]]), 10)
            with self.assertRaises(FileExistsError):
                vintage.write_report(output)


if __name__ == "__main__":
    unittest.main()
