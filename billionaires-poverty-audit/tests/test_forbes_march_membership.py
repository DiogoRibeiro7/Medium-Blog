"""Tests for independent March-versus-November Forbes membership evidence."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "forbes_march_membership.py"
SPEC = importlib.util.spec_from_file_location("forbes_march_membership", PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class MarchMembershipTests(unittest.TestCase):
    """Separate verified membership from unverified numerical count vintage."""

    def test_all_eighty_published_labels_match(self) -> None:
        result = audit.audit_membership()
        self.assertEqual(result["march_labels"], 80)
        self.assertEqual(result["november_labels"], 80)
        self.assertTrue(result["membership_matches"])
        self.assertEqual(result["march_only"], [])
        self.assertEqual(result["november_only"], [])
        self.assertFalse(result["counts_verified"])
        self.assertFalse(result["residence_verified"])

    def test_detects_missing_country(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            march = Path(temp) / "march.csv"
            november = Path(temp) / "november.csv"
            march.write_text("country_label\nPortugal\nSpain\n", encoding="utf-8")
            november.write_text("country_label\nPortugal\nItaly\n", encoding="utf-8")
            result = audit.audit_membership(march, november)
            self.assertFalse(result["membership_matches"])
            self.assertEqual(result["march_only"], ["Spain"])
            self.assertEqual(result["november_only"], ["Italy"])

    def test_rejects_duplicate_source_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            march = Path(temp) / "march.csv"
            november = Path(temp) / "november.csv"
            march.write_text("country_label\nPortugal\nPortugal\n", encoding="utf-8")
            november.write_text("country_label\nPortugal\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.audit_membership(march, november)


if __name__ == "__main__":
    unittest.main()
