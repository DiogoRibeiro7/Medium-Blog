"""Tests for the independent cutoff falsification demonstration."""

from __future__ import annotations

import unittest

import numpy as np

from rd_falsification import density_screen, run, sample_design


class FalsificationTests(unittest.TestCase):
    """Verify both diagnostic screens are sensitive to distinct mechanisms."""

    def test_known_density_asymmetry(self) -> None:
        rng = np.random.default_rng(100)
        x, baseline = sample_design(rng, "density_sorting")
        self.assertGreater(density_screen(x, 0.2), 2.0)
        self.assertEqual(x.shape, baseline.shape)

    def test_independent_simulation_diagnostics(self) -> None:
        results = {str(row["scenario"]): row for row in run()}
        for screen in ("density_screen_rejection_fraction", "covariate_screen_rejection_fraction"):
            self.assertLess(float(results["smooth_valid"][screen]), 0.12)
        self.assertGreater(float(results["density_sorting"]["density_screen_rejection_fraction"]), 0.9)
        self.assertGreater(float(results["latent_prognosis_jump"]["covariate_screen_rejection_fraction"]), 0.8)
        self.assertLess(float(results["latent_prognosis_jump"]["density_screen_rejection_fraction"]), 0.12)
        self.assertLess(float(results["density_sorting"]["covariate_screen_rejection_fraction"]), 0.12)


if __name__ == "__main__":
    unittest.main()
