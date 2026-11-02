"""Accounting identity and rendering tests for OMB federal series."""

import tempfile
import unittest
from pathlib import Path

from revenue_outlays import FiscalComponents, load_components, render_components


class RevenueOutlaysTests(unittest.TestCase):
    def test_all_55_years_reconcile(self) -> None:
        self.assertEqual(len(load_components()), 55)
        self.assertTrue(all(abs(r.receipts-r.outlays-r.balance) < 0.00015 for r in load_components()))

    def test_landmarks(self) -> None:
        rows = {r.year: r for r in load_components()}
        self.assertGreater(rows[2000].receipts, rows[2000].outlays)
        self.assertGreater(rows[2009].outlays, rows[2009].receipts)

    def test_failure_when_identity_breaks(self) -> None:
        with self.assertRaises(ValueError):
            FiscalComponents(2018, 17., 20., -1.)

    def test_reject_truncated_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "short.csv"
            p.write_text("year,federal_net_outlays_gdp_pct\n1964,17\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_components(outlays_path=p)

    def test_deterministic_chart(self) -> None:
        rows = load_components()
        self.assertEqual(render_components(rows), render_components(rows))
        self.assertIn("not the causal effect", render_components(rows))


if __name__ == "__main__":
    unittest.main()
