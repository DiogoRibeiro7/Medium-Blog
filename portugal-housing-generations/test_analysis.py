"""Offline regression checks for the Eurostat parser and descriptive statistics."""

import json
import unittest
from pathlib import Path

from analysis import parse_hpi, summarize


class AnalysisTests(unittest.TestCase):
    """Validate observable source values and indexing invariants."""

    def setUp(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_prc_hpi_a_PT_2026-10-29.json"
        self.payload = json.loads(path.read_text(encoding="utf-8"))

    def test_expected_values(self) -> None:
        series = parse_hpi(self.payload)
        self.assertEqual(series[2015], 100.0)
        self.assertEqual(series[2025], 263.85)
        self.assertEqual(summarize(series)["owner_gap_percentage_points"], -19)

    def test_rejects_other_geography(self) -> None:
        self.payload["dimension"]["geo"]["category"]["index"] = {"ES": 0}
        with self.assertRaises(ValueError):
            parse_hpi(self.payload)


if __name__ == "__main__":
    unittest.main()
