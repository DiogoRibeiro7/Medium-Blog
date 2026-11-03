"""Tests for the Part 9 synthetic-control pre-period holdout check."""

from __future__ import annotations

import unittest

import numpy as np

from sc_preperiod_holdout import generate_panel, holdout_gap, run


class SyntheticControlHoldoutTests(unittest.TestCase):
    """Verify that anticipation is detected without false claims about shocks."""

    def test_panel_shapes_and_finite_diagnostics(self) -> None:
        rng = np.random.default_rng(71)
        donors, observed = generate_panel(rng, "stable")
        self.assertEqual(donors.shape, (36, 8))
        self.assertEqual(observed.shape, (36,))
        self.assertTrue(np.isfinite(holdout_gap(donors, observed)).all())

    def test_anticipation_vs_unobservable_post_shock(self) -> None:
        rows = {str(item["scenario"]): item for item in run()}
        self.assertLess(float(rows["anticipation"]["mean_holdout_gap"]), -0.7)
        self.assertGreater(
            float(rows["anticipation"]["fraction_holdout_gap_below_minus_0_4"]), 0.95
        )
        for name in ("stable", "latent_post_shift"):
            self.assertLess(abs(float(rows[name]["mean_holdout_gap"])), 0.05)
            self.assertLess(float(rows[name]["fraction_holdout_gap_below_minus_0_4"]), 0.05)


if __name__ == "__main__":
    unittest.main()
