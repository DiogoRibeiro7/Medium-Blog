"""Deterministic contract tests for the illustrative political bargaining model."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from analysis import (
    FIGURE_NAMES,
    bargaining_solution,
    discounted_delay_loss,
    effective_outside_option,
    generate,
)


class BargainingTests(unittest.TestCase):
    """Tests of mathematics and explicit domain constraints."""

    def test_equal_outside_options_and_weights(self) -> None:
        self.assertEqual(bargaining_solution(0.2, 0.2, 0.5), (0.5, 0.5))

    def test_asymmetric_options_affect_a_formally_equal_bargain(self) -> None:
        r, i = bargaining_solution(0.05, 0.55, 0.5)
        self.assertAlmostEqual(r, 0.25)
        self.assertAlmostEqual(i, 0.75)

    def test_endpoint_weights(self) -> None:
        self.assertEqual(bargaining_solution(0.1, 0.2, 0), (0.1, 0.9))
        self.assertAlmostEqual(bargaining_solution(0.1, 0.2, 1)[0], 0.8)

    def test_zero_residual(self) -> None:
        result = bargaining_solution(0.4, 0.6, 0.5)
        self.assertAlmostEqual(result[0], 0.4)
        self.assertAlmostEqual(result[1], 0.6)

    def test_solution_respects_both_outside_options(self) -> None:
        for a in (0.0, 0.2, 0.5):
            for b in (0.0, 0.1, 0.3):
                for w in (0.0, 0.25, 0.5, 0.8, 1.0):
                    r, i = bargaining_solution(a, b, w)
                    self.assertGreaterEqual(r + 1e-12, a)
                    self.assertGreaterEqual(i + 1e-12, b)
                    self.assertAlmostEqual(r + i, 1.0)

    def test_invalid_inputs(self) -> None:
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    bargaining_solution(invalid, 0.2)
        with self.assertRaises(ValueError):
            bargaining_solution(0.7, 0.7)
        with self.assertRaises(TypeError):
            bargaining_solution(True, 0.3)

    def test_weight_monotonicity(self) -> None:
        shares = [bargaining_solution(0.1, 0.4, float(w))[0]
                  for w in (0.1, 0.2, 0.5, 0.9)]
        self.assertEqual(shares, sorted(shares))

    def test_restraint_cost_is_explicitly_assumed(self) -> None:
        self.assertAlmostEqual(effective_outside_option(0.35, 0.5, 0.3), 0.2)
        self.assertEqual(effective_outside_option(0.2, 1.0, 0.3), 0.0)
        with self.assertRaises(ValueError):
            effective_outside_option(0.2, -0.1)

    def test_delay_formula_and_limit(self) -> None:
        self.assertAlmostEqual(discounted_delay_loss(10, 2, 0), 20)
        self.assertAlmostEqual(discounted_delay_loss(10, 2, 0.03),
                               2 * (-math.expm1(-0.3)) / 0.03)
        self.assertEqual(discounted_delay_loss(0, 2, 0.03), 0.0)
        with self.assertRaises(ValueError):
            discounted_delay_loss(1, 1, -0.03)

    def test_delay_harm_increases_with_waiting_time(self) -> None:
        values = [discounted_delay_loss(year, 1, 0.03) for year in (0, 1, 5, 10)]
        self.assertEqual(values, sorted(values))


class OutputTests(unittest.TestCase):
    """Check that the public results match the declared reproducibility contract."""

    def test_all_files_are_generated_and_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figures, tables = generate(Path(directory))
            self.assertEqual(len(figures), 8)
            self.assertEqual(len(tables), 4)
            self.assertEqual({p.name for p in figures}, set(FIGURE_NAMES))
            self.assertTrue(all(path.stat().st_size > 10_000 for path in figures))
            for path in tables:
                with path.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertGreater(len(rows), 0)
            with (Path(directory) / "results/bargaining_scenarios.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                cases = list(csv.DictReader(stream))
            self.assertEqual(len(cases), 3)
            self.assertAlmostEqual(float(cases[0]["reformer_share"]), 0.25)


if __name__ == "__main__":
    unittest.main()
