"""Regression tests for the original causal-identification article figures."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_figures as figures  # noqa: E402


class IdentificationIllustrationTests(unittest.TestCase):
    """Verify that the constructed examples communicate the stated quantities."""

    def test_selection_can_generate_thirty_percent_gap(self) -> None:
        """Compare users with different composition but zero treatment effect."""
        high_retention = 0.70
        casual_retention = 0.45
        adopter_high_share = 0.80
        other_high_share = 0.20

        adopter_retention = (
            adopter_high_share * high_retention
            + (1.0 - adopter_high_share) * casual_retention
        )
        other_retention = (
            other_high_share * high_retention
            + (1.0 - other_high_share) * casual_retention
        )

        self.assertAlmostEqual(adopter_retention, 0.65)
        self.assertAlmostEqual(other_retention, 0.50)
        self.assertAlmostEqual(
            (adopter_retention - other_retention) / other_retention,
            0.30,
        )

    def test_did_counterfactual_arithmetic(self) -> None:
        """The hypothetical continuation differs from naive DiD."""
        naive_did = (65.0 - 50.0) - (45.0 - 40.0)
        effect_under_continued_trend = 65.0 - 60.0
        self.assertEqual(naive_did, 10.0)
        self.assertEqual(effect_under_continued_trend, 5.0)

    def test_all_figures_render(self) -> None:
        """Generate four figure pairs without requiring data or network access."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(figures, "OUTPUT_DIRECTORY", output):
                figures.main()

            images = sorted(output.glob("*.png"))
            vectors = sorted(output.glob("*.svg"))
            self.assertEqual(len(images), 4)
            self.assertEqual(len(vectors), 4)
            self.assertTrue(all(path.stat().st_size > 1_000 for path in images))


if __name__ == "__main__":
    unittest.main()
