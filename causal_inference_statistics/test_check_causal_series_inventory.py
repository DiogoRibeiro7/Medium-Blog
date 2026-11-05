"""Unit tests for the causal-series supplemental-archive checker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_causal_series_inventory import PARTS, SUPPLEMENTAL_TABLES, audit_inventory, check_table


class InventoryTests(unittest.TestCase):
    """Check expected counts, missing artifacts and CSV integrity failures."""

    def test_manifest_has_unique_paths(self) -> None:
        self.assertEqual(len(SUPPLEMENTAL_TABLES), 12)
        self.assertEqual(len(set(SUPPLEMENTAL_TABLES)), 12)
        self.assertEqual(len(PARTS), 20)

    def test_valid_table(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "results.csv"
            target.write_text("method,value\na,1.0\n", encoding="utf-8")
            self.assertEqual(check_table(target), [])

    def test_duplicate_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "results.csv"
            target.write_text("method,method\na,1.0\n", encoding="utf-8")
            self.assertTrue(check_table(target))

    def test_extra_or_missing_data_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "results.csv"
            target.write_text("method,value\na,1,3\n", encoding="utf-8")
            self.assertTrue(check_table(target))
            target.write_text("method,value\na\n", encoding="utf-8")
            self.assertTrue(check_table(target))

    def test_inventory_detects_missing_parts_and_archives(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            count, errors = audit_inventory(Path(name))
            self.assertEqual(count, 32)
            self.assertTrue(any("missing part 1" in err for err in errors))
            self.assertTrue(any("missing table" in err for err in errors))

    def test_unregistered_supplemental_output_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name) / "part1"
            folder.mkdir()
            (folder / "other_results.csv").write_text("a\n1\n", encoding="utf-8")
            _, errors = audit_inventory(Path(name))
            self.assertTrue(any("untracked supplemental table" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
