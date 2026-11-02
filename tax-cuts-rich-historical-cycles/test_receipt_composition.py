"""Tests for OMB receipt mix, rounding, and deterministic visualization."""

import tempfile
import unittest
from xml.etree import ElementTree
from pathlib import Path

from receipt_composition import ReceiptMix, load_mix, render_mix


class ReceiptCompositionTests(unittest.TestCase):
    def test_55_years_and_source_anchors(self) -> None:
        rows = load_mix()
        self.assertEqual(len(rows), 55)
        self.assertEqual((rows[0].year, rows[0].corporate), (1964, 3.6))
        self.assertEqual((rows[-1].year, rows[-1].corporate), (2018, 1.0))

    def test_rounding_reconciles(self) -> None:
        self.assertTrue(all(abs(r.individual+r.corporate+r.social_insurance+r.excise+r.other-r.total) <= 0.251 for r in load_mix()))

    def test_rejects_bad_observation(self) -> None:
        with self.assertRaises(ValueError):
            ReceiptMix(1964, 1, 2, 3, 4, 5, 100)
        with self.assertRaises(TypeError):
            ReceiptMix(1964, True, 2, 3, 4, 5, 14)

    def test_rejects_missing_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "short.csv"
            p.write_text("year,individual_income_pct_gdp,corporate_income_pct_gdp,social_insurance_pct_gdp,excise_pct_gdp,other_pct_gdp,total_receipts_pct_gdp\n1964,7.4,3.6,3.3,2.1,0.7,17.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mix(p)

    def test_svg_determinism(self) -> None:
        rows = load_mix()
        self.assertEqual(render_mix(rows), render_mix(rows))
        self.assertIn("do not isolate effects", render_mix(rows))
        root = ElementTree.fromstring(render_mix(rows))
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("Social insurance &amp; retirement", render_mix(rows))


if __name__ == "__main__":
    unittest.main()