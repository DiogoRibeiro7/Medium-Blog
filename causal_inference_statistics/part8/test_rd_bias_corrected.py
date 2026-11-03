"""Tests for the fixed-bandwidth pilot-quadratic RD bias correction."""

from __future__ import annotations

import unittest

import numpy as np

from rd_bias_corrected import corrected_rd, run


class RDBiasCorrectionTests(unittest.TestCase):
    """Check exact quadratic recovery and inferential behavior."""

    def test_exact_quadratic_boundary_effect(self) -> None:
        """A quadratic trend has no residual approximation bias after correction."""
        x = np.linspace(-1, 1, 2001)
        y = 1.0 + 0.6 * x + 2 * x**2 * (x >= 0) - (x >= 0)
        for h in (0.15, 0.3, 0.6, 1.0):
            estimate, se = corrected_rd(x, y, h)
            self.assertAlmostEqual(estimate, -1, places=9)
            self.assertGreaterEqual(se, 0)

    def test_curvature_coverage_and_invalid_continuity(self) -> None:
        """Correct smoothing bias, while keeping identification failure visible."""
        output = {(str(r["scenario"]), float(r["bandwidth"])): r for r in run()}
        curved = output[("curvature", 1.0)]
        invalid = output[("sorting", 1.0)]
        self.assertGreater(float(curved["corrected_coverage"]), 0.90)
        self.assertLess(float(curved["ordinary_coverage"]), 0.60)
        self.assertLess(float(invalid["corrected_coverage"]), 0.05)
        self.assertAlmostEqual(float(invalid["mean_corrected"]), -0.3, delta=0.04)


if __name__ == "__main__":
    unittest.main()
