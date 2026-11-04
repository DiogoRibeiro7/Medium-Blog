"""Verify oracle sequential IPW reconstructs interventional moments."""

from __future__ import annotations

import unittest

from part15_oracle_ipw_moments import oracle_regime_moments


class OracleIPWMomentTests(unittest.TestCase):
    """Check moment recovery for both sustained treatment strategies."""

    def test_weighted_moments_recover_known_regimes(self) -> None:
        """Reweighting restores E[L0]=0, E[L1(a)]=1.2a and E[Y(a,a)]=1+3.36a."""
        observed = oracle_regime_moments()
        for regime in (0, 1):
            with self.subTest(regime=regime):
                moments = observed[regime]
                self.assertGreater(moments.observed_count, 10000)
                self.assertAlmostEqual(moments.weighted_baseline, 0.0, delta=0.06)
                self.assertAlmostEqual(moments.weighted_intermediate, 1.2 * regime, delta=0.06)
                self.assertAlmostEqual(moments.weighted_outcome, 1.0 + 3.36 * regime, delta=0.09)
        contrast = observed[1].weighted_outcome - observed[0].weighted_outcome
        self.assertAlmostEqual(contrast, 3.36, delta=0.12)

    def test_invalid_sample_size_is_rejected(self) -> None:
        """The diagnostic requires a usable simulated population."""
        for invalid in (0, 99, True):
            with self.subTest(n=invalid):
                with self.assertRaises(ValueError):
                    oracle_regime_moments(n=invalid)


if __name__ == "__main__":
    unittest.main()
