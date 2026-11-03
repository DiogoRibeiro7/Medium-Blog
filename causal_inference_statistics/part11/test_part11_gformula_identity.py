"""Finite-sample algebraic checks for Part 11's linear g-formula.

Run from this directory with ``python -m unittest -v test_part11_gformula_identity``.
"""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from experiment import g_formula, simple_adjustment, simulate


class GFormulaIdentityTests(unittest.TestCase):
    """Separate the direct treatment coefficients from the mediated path."""

    def test_correct_regime_contrast_equals_direct_plus_indirect(self) -> None:
        """Linear plug-in standardization has an exact sample identity."""
        rng = np.random.default_rng(20261103)
        for sample_size in (100, 500, 2000):
            with self.subTest(sample_size=sample_size):
                data = simulate(rng, sample_size)
                l0, a0, l1, a1, y = data
                transition = LinearRegression().fit(np.column_stack((l0, a0)), l1)
                outcome = LinearRegression().fit(np.column_stack((l0, a0, l1, a1)), y)
                direct: float = float(outcome.coef_[1] + outcome.coef_[3])
                indirect: float = float(transition.coef_[1] * outcome.coef_[2])
                contrast: float = g_formula(data, 1.0, 1.0) - g_formula(data, 0.0, 0.0)
                self.assertAlmostEqual(contrast, direct + indirect, places=11)
                self.assertAlmostEqual(simple_adjustment(data), direct, places=11)

    def test_omitted_intermediate_regime_contrast_is_reduced_coefficient_sum(self) -> None:
        """The misspecified comparison does not use its transition model."""
        rng = np.random.default_rng(20261104)
        data = simulate(rng, 700)
        l0, a0, _l1, a1, y = data
        reduced = LinearRegression().fit(np.column_stack((l0, a0, a1)), y)
        expected: float = float(reduced.coef_[1] + reduced.coef_[2])
        observed: float = g_formula(data, 1.0, 1.0, include_l1=False) - g_formula(
            data, 0.0, 0.0, include_l1=False
        )
        self.assertAlmostEqual(observed, expected, places=11)


if __name__ == "__main__":
    unittest.main()
