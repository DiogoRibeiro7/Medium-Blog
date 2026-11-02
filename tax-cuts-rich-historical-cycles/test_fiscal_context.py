"""Structural, landmark, and artifact tests for the federal-receipts series."""

import csv
import tempfile
import unittest
from pathlib import Path

from fiscal_context import FiscalPoint, load_receipts, render_comparison
from historical_rates import load_rates


class FiscalContextTests(unittest.TestCase):
    """Validate source transcription and prohibit malformed chronologies."""

    def test_full_range_and_landmarks(self) -> None:
        points = load_receipts()
        self.assertEqual(len(points), 55)
        values = {point.year: point.receipts_pct_gdp for point in points}
        for year, expected in {1964: 16.45282, 1981: 18.68613, 1983: 16.52602, 1988: 17.36367, 2000: 19.75613, 2003: 15.55730, 2009: 14.53916, 2017: 16.90229, 2018: 16.11475}.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(values[year], expected, places=5)

    def test_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "broken.csv"
            rows = [(point.year, point.receipts_pct_gdp) for point in load_receipts() if point.year != 1983]
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("year", "receipts_pct_gdp"))
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Missing, reordered, or duplicate"):
                load_receipts(target)

    def test_invalid_observation(self) -> None:
        for year, rate in ((1960, 15.0), (1985, -1.0), (1985, float("nan")), (1985.0, 16.0)):
            with self.subTest(year=year, rate=rate), self.assertRaises((ValueError, TypeError)):
                FiscalPoint(year, rate)

    def test_deterministic_caveated_chart(self) -> None:
        rates, receipts = load_rates(), load_receipts()
        image = render_comparison(rates, receipts)
        self.assertEqual(image, render_comparison(rates, receipts))
        self.assertIn("fiscal-year numerator", image)
        self.assertIn("does not estimate tax-cut effects", image)


if __name__ == "__main__":
    unittest.main()
