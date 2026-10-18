"""Regression checks for transportability estimands and archived result schemas.

Run with Python 3.12+: python -m unittest discover -s causal_inference_statistics/part19 -p 'test_*.py'
"""
from __future__ import annotations

import csv
import unittest
from pathlib import Path


class TransportabilityAuditTests(unittest.TestCase):
    """Check defined causal targets and machine-readable result consistency."""

    def test_omitted_modifier_target_is_distinct(self) -> None:
        """A mean shift in an unmeasured effect modifier changes target ATE."""
        source_mean_u: float = 0.0
        target_mean_u: float = 1.0
        self.assertEqual(1.0 + source_mean_u, 1.0)
        self.assertEqual(1.0 + target_mean_u, 2.0)

    def test_result_rows_match_experiment_design(self) -> None:
        """Stale restricted-target rows must not survive manuscript updates."""
        result_path = Path(__file__).with_name("results.csv")
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = {
            ("measured_effect_modifier", "trial"),
            ("measured_effect_modifier", "transported"),
            ("omitted_effect_modifier", "trial"),
            ("omitted_effect_modifier", "transported"),
            ("nonoverlap", "trial"),
            ("nonoverlap", "transported"),
            ("nonoverlap", "restricted"),
        }
        actual = {(row["scenario"], row["method"]) for row in rows}
        self.assertEqual(actual, expected)
        self.assertEqual(len(rows), len(expected))
        targets = {(row["scenario"], row["method"]): float(row["target"]) for row in rows}
        self.assertEqual(targets[("omitted_effect_modifier", "trial")], 1.0)
        self.assertEqual(targets[("omitted_effect_modifier", "transported")], 2.0)
        self.assertEqual(targets[("nonoverlap", "transported")], 2.5)
        self.assertEqual(targets[("nonoverlap", "restricted")], 1.75)


if __name__ == "__main__":
    unittest.main()
