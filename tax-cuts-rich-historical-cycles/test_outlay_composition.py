"""Source landmarks, schema, accounting and rendering tests for OMB outlays."""

import tempfile
import unittest
from pathlib import Path

from outlay_composition import OutlayMix, load_outlays, render_outlays


class OutlayCompositionTests(unittest.TestCase):
    def test_complete_series_and_anchors(self) -> None:
        rows = load_outlays()
        self.assertEqual(len(rows), 55)
        self.assertEqual((rows[0].year, rows[0].defense), (1964, 8.3))
        self.assertEqual((rows[-1].year, rows[-1].human), (2018, 14.2))

    def test_source_rounding_and_offsets(self) -> None:
        for row in load_outlays():
            self.assertLessEqual(abs(row.defense+row.human+row.physical+row.interest+row.other+row.offsets-row.total), 0.301)
            self.assertLessEqual(row.offsets, 0)

    def test_invalid_observations(self) -> None:
        with self.assertRaises(TypeError):
            OutlayMix(1964, True, 3., 2., 1., 1., -1., 7.)
        with self.assertRaises(ValueError):
            OutlayMix(1964, 3., 2., 1., 1., 1., -1., 20.)

    def test_missing_year_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as location:
            p = Path(location) / "short.csv"
            p.write_text("year,national_defense_pct_gdp,human_resources_pct_gdp,physical_resources_pct_gdp,net_interest_pct_gdp,other_functions_pct_gdp,undistributed_offsets_pct_gdp,total_outlays_pct_gdp\n1964,8.3,5.3,1.4,1.2,2.5,-0.9,17.9\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_outlays(p)

    def test_deterministic_svg(self) -> None:
        rows = load_outlays()
        self.assertEqual(render_outlays(rows), render_outlays(rows))
        self.assertIn("do not identify tax-law effects", render_outlays(rows))


if __name__ == "__main__":
    unittest.main()
