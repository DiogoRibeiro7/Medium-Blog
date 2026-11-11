"""Tests of the shared public-goods tools."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publicgoods import (  # noqa: E402
    Behaviour,
    Sanctions,
    equivalent_share,
    g_star,
    make_draws,
    s_star,
    simulate,
    welfare_compulsory,
    welfare_voluntary,
)


class TestWelfareAlgebra(unittest.TestCase):
    def test_break_even_share_equalises_welfare(self):
        for G in (1.2, 1.6, 3.0, 10.0):
            for lam in (1.0, 1.09, 1.33, 1.56):
                for w in (0.0, 0.1, 0.3):
                    s = s_star(G, lam, w)
                    self.assertAlmostEqual(float(welfare_voluntary(s, G)), float(welfare_compulsory(G, lam, w)))

    def test_g_star_inverts_s_star(self):
        for s in (0.1, 0.3, 0.6):
            for lam in (1.09, 1.33):
                for w in (0.0, 0.1):
                    G = float(g_star(s, lam, w))
                    self.assertAlmostEqual(float(s_star(G, lam, w)), s)

    def test_g_star_infinite_when_waste_exceeds_gap(self):
        self.assertTrue(np.isinf(g_star(0.8, 1.2, 0.3)))

    def test_costless_tax_beats_any_partial_provision(self):
        # With lam = 1 and no waste the tax provides everything at no extra cost: s* = 1.
        self.assertAlmostEqual(float(s_star(2.5, 1.0, 0.0)), 1.0)

    def test_equivalent_share_of_voluntary_welfare(self):
        self.assertAlmostEqual(float(equivalent_share(welfare_voluntary(0.37, 1.6), 1.6)), 0.37)


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.draws = make_draws(7, 2000, 4, 20)

    def test_welfare_matches_contributions_without_sanctions(self):
        p = simulate(Behaviour(), self.draws, 0.4, 20)
        np.testing.assert_allclose(p.welfare, 1.0 + 0.6 * p.contribution, atol=1e-12)
        self.assertTrue(np.all(p.sanction_cost == 0))

    def test_perfect_matching_without_noise_keeps_contributions(self):
        b = Behaviour(beta_mean=1 - 1e-4, beta_conc=1e6, belief0=0.5, belief0_sd=0.0, noise_sd=0.0)
        p = simulate(b, self.draws, 0.4, 20)
        np.testing.assert_allclose(p.contribution, 0.5, atol=1e-3)

    def test_imperfect_matching_declines(self):
        b = Behaviour(beta_mean=0.8, belief0=0.6, noise_sd=0.0)
        p = simulate(b, self.draws, 0.4, 20)
        self.assertTrue(np.all(np.diff(p.contribution) < 0))

    def test_sanctions_cost_accounting(self):
        s = Sanctions(gamma=5.0, eta=0.1, floor_decay=1.0)
        p = simulate(Behaviour(), self.draws, 0.4, 5, s)
        # Welfare is the public-goods payoff less the sanction cost.
        no_cost = 1.0 + 0.6 * p.contribution
        np.testing.assert_allclose(p.welfare, no_cost - p.sanction_cost, atol=1e-12)

    def test_sanctions_raise_contributions(self):
        b = Behaviour()
        v = simulate(b, self.draws, 0.4, 10)
        p = simulate(b, self.draws, 0.4, 10, Sanctions(gamma=10.0, eta=0.1, floor_decay=1.0))
        self.assertGreater(p.contribution[-1], v.contribution[-1])

    def test_antisocial_punishment_lowers_welfare(self):
        b = Behaviour()
        s = Sanctions(gamma=10.0, eta=0.1, floor_decay=1.0)
        p0 = simulate(b, self.draws, 0.4, 10, s)
        p3 = simulate(b, self.draws, 0.4, 10, Sanctions(gamma=10.0, eta=0.1, floor_decay=1.0, antisocial=0.3))
        self.assertLess(p3.welfare.sum(), p0.welfare.sum())


if __name__ == "__main__":
    unittest.main()
