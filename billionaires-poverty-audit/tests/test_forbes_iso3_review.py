"""Offline contract tests for the provisional Forbes ISO3 review ledger."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "forbes_iso3_review.py"
SPEC = importlib.util.spec_from_file_location("forbes_iso3_review", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ForbesISO3ReviewTests(unittest.TestCase):
    """Enforce full coverage and a conservative, non-publication default."""

    def test_all_frozen_labels_are_provisional(self) -> None:
        rows = review.build_ledger()
        self.assertEqual(len(rows), 80)
        self.assertEqual(sum(row["primary_count"] for row in rows), 3428)
        self.assertEqual(len({row["iso3_candidate"] for row in rows}), 80)
        self.assertTrue(all(row["review_status"] == "unverified_candidate" for row in rows))
        self.assertTrue(all(not row["evidence_url"] for row in rows))

    def test_aliases_and_territories_are_explicit(self) -> None:
        rows = {row["primary_label"]: row for row in review.build_ledger()}
        self.assertEqual(rows["Eswatini (Swaziland)"]["iso3_candidate"], "SWZ")
        self.assertEqual(rows["Afghanistan"]["match_type"], "missing_from_secondary")
        self.assertEqual(rows["St. Kitts and Nevis"]["iso3_candidate"], "KNA")
        for label in review.TERRITORIAL_LABELS:
            self.assertEqual(rows[label]["territory_review"], "required")

    def test_writer_refuses_overwrite(self) -> None:
        # Exercise the deterministic output contract without modifying frozen inputs.
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            previous = sys.argv
            try:
                sys.argv = ["forbes_iso3_review.py", "--output", str(path)]
                review.main()
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 80)
                with self.assertRaises(FileExistsError):
                    review.main()
            finally:
                sys.argv = previous


if __name__ == "__main__":
    unittest.main()
