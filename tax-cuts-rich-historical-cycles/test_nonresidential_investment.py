"""Tests for frozen BEA nonresidential investment evidence."""

import tempfile
import unittest
from pathlib import Path

from nonresidential_investment import InvestmentPoint, load_investment, render_investment


class NonresidentialInvestmentTests(unittest.TestCase):
    def test_55_year_anchors(self) -> None:
        rows = load_investment()
        self.assertEqual(len(rows), 55)
        self.assertEqual((rows[0].year, rows[0].nonresidential_fixed_pct), (1964, 10.6))
        self.assertEqual((rows[-1].year, rows[-1].nonresidential_fixed_pct), (2018, 13.8))

    def test_expected_categories(self) -> None:
        for row in load_investment():
            self.assertLess(row.nonresidential_fixed_pct, row.gross_private_pct)

    def test_invalid_values(self) -> None:
        with self.assertRaises(TypeError):
            InvestmentPoint(1964, True, 12.)
        with self.assertRaises(ValueError):
            InvestmentPoint(1964, -1., 12.)

    def test_reject_incomplete_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "short.csv"
            p.write_text("year,nonresidential_fixed_investment_share_pct\n1964,10.6\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_investment(nonres_path=p)

    def test_svg_determinism(self) -> None:
        rows = load_investment()
        self.assertEqual(render_investment(rows), render_investment(rows))
        self.assertIn("No tax-reform effect", render_investment(rows))


if __name__ == "__main__":
    unittest.main()
