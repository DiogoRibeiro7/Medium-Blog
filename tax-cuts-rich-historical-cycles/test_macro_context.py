"""Unit tests and original-source landmarks for BEA macro evidence."""

import csv
import tempfile
import unittest
from pathlib import Path

from macro_context import MacroPoint, load_macro, render_macro


class MacroContextTests(unittest.TestCase):
    """Ensure numeric evidence is complete and figure output reproducible."""

    def test_landmarks(self) -> None:
        rows = load_macro()
        self.assertEqual(len(rows), 55)
        values = {row.year: row for row in rows}
        for year, growth, share in (
            (1964, 5.8, 16.4),
            (1982, -1.8, 17.4),
            (1984, 7.2, 20.3),
            (1988, 4.2, 17.9),
            (2000, 4.1, 19.9),
            (2003, 2.8, 17.7),
            (2009, -2.6, 13.3),
            (2017, 2.5, 17.7),
            (2018, 3.0, 18.0),
        ):
            with self.subTest(year=year):
                self.assertAlmostEqual(values[year].real_gdp_growth_pct, growth)
                self.assertAlmostEqual(values[year].private_investment_share_pct, share)

    def test_invalid_numeric(self) -> None:
        for args in ((1960, 3., 16.), (1980, float("nan"), 16.), (1980, 3., -1.), (1980.0, 3., 16.)):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                MacroPoint(*args)

    def test_missing_year_rejected(self) -> None:
        original = load_macro()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "gap.csv"
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("year", "real_gdp_growth_pct", "private_investment_share_pct"))
                writer.writerows((r.year, r.real_gdp_growth_pct, r.private_investment_share_pct) for r in original if r.year != 2001)
            with self.assertRaisesRegex(ValueError, "consecutive"):
                load_macro(target)

    def test_chart_is_deterministic_and_caveated(self) -> None:
        rows = load_macro()
        svg = render_macro(rows)
        self.assertEqual(svg, render_macro(rows))
        self.assertIn("Historical co-movement is not an estimated effect", svg)
        self.assertIn("Investment includes inventories", svg)


if __name__ == "__main__":
    unittest.main()
