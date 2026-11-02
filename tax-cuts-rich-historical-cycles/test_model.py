"""Unit and integration checks for the illustrative sensitivity model."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from model import Scenario, elasticity_grid, evaluate, write_outputs


class ScenarioTests(unittest.TestCase):
    """Test invariants, comparative statics, and deterministic artifacts."""

    def test_default_revenue_loss(self) -> None:
        result = evaluate(Scenario())
        self.assertGreater(result["actual_loss"], 0)
        self.assertLess(result["actual_loss"], result["static_loss"])

    def test_zero_behavior_yields_static_loss(self) -> None:
        result = evaluate(Scenario(base_elasticity=0, capital_elasticity=0))
        self.assertAlmostEqual(result["actual_loss"], result["static_loss"])
        self.assertAlmostEqual(result["feedback_fraction"], 0)

    def test_elasticity_increases_feedback(self) -> None:
        low = evaluate(Scenario(base_elasticity=0.1))
        high = evaluate(Scenario(base_elasticity=0.9))
        self.assertGreater(high["feedback_fraction"], low["feedback_fraction"])

    def test_public_complement_reduces_output(self) -> None:
        weak = evaluate(Scenario(public_productivity=0.0))
        strong = evaluate(Scenario(public_productivity=0.6))
        self.assertLess(strong["output_ratio"], weak["output_ratio"])

    def test_invalid_inputs(self) -> None:
        for values in ({"rate_after": 0.5}, {"taxable_base": 0}, {"base_elasticity": -1}, {"rate_before": float("nan")}, {"public_spending_share": 1.5}):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                Scenario(**values)

    def test_grid_and_artifacts(self) -> None:
        self.assertEqual(len(elasticity_grid()), 9)
        with tempfile.TemporaryDirectory() as location:
            output = Path(location)
            write_outputs(output)
            first_csv = (output / "sensitivity.csv").read_bytes()
            first_svg = (output / "sensitivity.svg").read_bytes()
            write_outputs(output)
            self.assertEqual(first_csv, (output / "sensitivity.csv").read_bytes())
            self.assertEqual(first_svg, (output / "sensitivity.svg").read_bytes())
            self.assertNotIn(b"\r", first_csv, "CSV output must use LF on every OS")
            with (output / "sensitivity.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(math.isfinite(float(row["output_change_pct"])) for row in rows))
            self.assertIn(b"<svg", first_svg)


if __name__ == "__main__":
    unittest.main()
