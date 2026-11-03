"""Validation checks for OMB/FRED fiscal-balance and debt data."""

import tempfile
import unittest
from pathlib import Path

from fiscal_balance_debt import FiscalObservation, load_fiscal, render_fiscal


class FiscalBalanceDebtTests(unittest.TestCase):
    def test_full_window_and_landmarks(self) -> None:
        rows = load_fiscal()
        self.assertEqual(len(rows), 55)
        self.assertAlmostEqual(rows[0].balance_pct, -0.86418)
        self.assertAlmostEqual(rows[-1].debt_pct, 76.24519)

    def test_reagan_and_recession_landmarks(self) -> None:
        by_year = {r.year: r for r in load_fiscal()}
        self.assertLess(by_year[1983].balance_pct, -5)
        self.assertLess(by_year[2009].balance_pct, -9)
        self.assertGreater(by_year[2018].debt_pct, by_year[2017].debt_pct)

    def test_invalid_types_and_bounds(self) -> None:
        with self.assertRaises(TypeError):
            FiscalObservation(1964, True, 30.)
        with self.assertRaises(ValueError):
            FiscalObservation(1964, -2., -1.)

    def test_missing_year_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "incomplete.csv"
            p.write_text("year,federal_surplus_deficit_gdp_pct,debt_held_public_gdp_pct\n1964,-1,40\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fiscal(p)

    def test_svg_is_deterministic_and_cautious(self) -> None:
        rows = load_fiscal()
        self.assertEqual(render_fiscal(rows), render_fiscal(rows))
        self.assertIn("no tax-cut effect", render_fiscal(rows))


if __name__ == "__main__":
    unittest.main()
