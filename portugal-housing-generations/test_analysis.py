"""Offline regression checks for the Eurostat parser and descriptive statistics."""

import json
import unittest
from pathlib import Path

from analysis import parse_hpi, parse_income, relative_affordability, summarize


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

    def test_price_income_index(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_2026-10-29.json"
        income = parse_income(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(income[2015], 8435.0)
        self.assertEqual(income[2025], 14465.0)
        index = relative_affordability(parse_hpi(self.payload), income)
        self.assertEqual(index[2015], 100.0)
        self.assertGreater(index[2025], 150)

    def test_income_validation(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_2026-10-29.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["dimension"]["statinfo"]["category"]["index"] = {"MEAN_EI": 0}
        with self.assertRaises(ValueError):
            parse_income(raw)


if __name__ == "__main__":
    unittest.main()
