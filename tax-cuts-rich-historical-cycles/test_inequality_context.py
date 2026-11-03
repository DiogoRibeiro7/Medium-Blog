"""Structural and interpretation tests for Census income inequality evidence."""

import csv
import tempfile
import unittest
from pathlib import Path

from inequality_context import GiniPoint, load_gini, render_gini


class InequalityContextTests(unittest.TestCase):
    """Verify complete years, key points, and explicit methodological break."""

    def test_landmarks_and_range(self) -> None:
        rows = load_gini()
        self.assertEqual(len(rows), 52)
        observed = {row.year: row.household_money_income_gini for row in rows}
        for year, value in ((1967, 0.397), (1980, 0.403), (1988, 0.426), (1992, 0.433), (1993, 0.454), (2000, 0.462), (2003, 0.464), (2017, 0.489), (2018, 0.486)):
            with self.subTest(year=year):
                self.assertAlmostEqual(observed[year], value)

    def test_invalid_inputs(self) -> None:
        for year, value in ((1964, 0.4), (1980, -1.), (1980, float("nan")), (1980.0, 0.4)):
            with self.subTest(year=year, value=value), self.assertRaises((TypeError, ValueError)):
                GiniPoint(year, value)

    def test_gap_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("year", "household_money_income_gini"))
                writer.writerows((row.year, row.household_money_income_gini) for row in load_gini() if row.year != 1993)
            with self.assertRaisesRegex(ValueError, "52 consecutive"):
                load_gini(path)

    def test_1993_break_and_disclaimer(self) -> None:
        svg = render_gini(load_gini())
        self.assertEqual(svg, render_gini(load_gini()))
        self.assertEqual(svg.count("<polyline "), 2)
        self.assertIn("1993 survey redesign", svg)
        self.assertIn("not after-tax resources or wealth", svg)


if __name__ == "__main__":
    unittest.main()
