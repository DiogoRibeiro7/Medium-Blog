"""Structural and reproducibility tests for the CBO 2018 evidence subset."""

import tempfile
import unittest
from pathlib import Path

from cbo_distribution import HouseholdIncome, load_income, render_income


class CboDistributionTests(unittest.TestCase):
    def test_selected_cbo_anchors(self) -> None:
        rows = load_income()
        self.assertEqual([(r.before, r.after) for r in rows], [(22500, 37700), (77500, 71100), (321700, 243900)])

    def test_distributional_directions(self) -> None:
        low, mid, high = load_income()
        self.assertGreater(low.after, low.before)
        self.assertLess(mid.after, mid.before)
        self.assertLess(high.after, high.before)

    def test_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("group,before,after\nLowest quintile,1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_income(path)

    def test_rejects_wrong_types_and_groups(self) -> None:
        with self.assertRaises(TypeError):
            HouseholdIncome("Lowest quintile", True, 1)
        with self.assertRaises(ValueError):
            HouseholdIncome("Unknown", 1, 2)
        with self.assertRaises(ValueError):
            render_income(list(reversed(load_income())))

    def test_svg_is_reproducible_and_cautious(self) -> None:
        first = render_income(load_income())
        self.assertEqual(first, render_income(load_income()))
        self.assertIn("NOT an estimate", first)
        self.assertIn("$321,700", first)


if __name__ == "__main__":
    unittest.main()
