"""Regression checks for public equation assets after editorial migration."""

from __future__ import annotations

import unittest
from pathlib import Path

from prepare_medium_publication import EQUATION_CAPTIONS, EQUATIONS


class EquationAssetTests(unittest.TestCase):
    """Keep scientific equation PNGs available without private manuscripts."""

    def test_equation_artworks_are_present_and_valid_png(self) -> None:
        self.assertEqual(len(EQUATION_CAPTIONS), 15)
        for number in range(1, len(EQUATION_CAPTIONS) + 1):
            with self.subTest(equation=number):
                asset: Path = EQUATIONS / f"equation_{number:02d}.png"
                self.assertTrue(asset.is_file())
                self.assertGreater(asset.stat().st_size, 500)
                with asset.open("rb") as file:
                    self.assertEqual(file.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
