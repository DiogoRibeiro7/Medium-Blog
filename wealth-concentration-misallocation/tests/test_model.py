import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import (Economy, Params, allocate, coupled_draws, lognormal_sigma_for_top_share,  # noqa: E402
                   spearman, top_share)


def test_top_share_matches_lognormal_formula():
    rng = np.random.default_rng(0)
    for target in (0.10, 0.25, 0.40):
        a = np.exp(lognormal_sigma_for_top_share(target) * rng.standard_normal(400_000))
        assert top_share(a) == pytest.approx(target, abs=0.015)


def test_copula_delivers_spearman():
    for rho in (0.0, 0.5, 0.9):
        a, z = coupled_draws(100_000, 0.25, 1.0, rho, np.random.default_rng(1))
        assert spearman(a, z) == pytest.approx(rho, abs=0.01)
        assert a.mean() == pytest.approx(1.0)


def test_unconstrained_allocation_is_first_best():
    a, z = coupled_draws(5_000, 0.30, 1.0, 0.0, np.random.default_rng(2))
    assert allocate(a, z, np.inf, 0.5).efficiency == pytest.approx(1.0, rel=1e-12)


def test_wealth_proportional_to_first_best_capital_is_efficient():
    rng = np.random.default_rng(3)
    z = np.exp(rng.standard_normal(5_000))
    a = z ** 2.0  # gamma = 0.5 -> first-best k proportional to z^(1/(1-gamma))
    a /= a.mean()
    assert allocate(a, z, 1.01, 0.5).efficiency == pytest.approx(1.0, abs=1e-9)


def test_capital_market_clears_and_respects_collateral():
    a, z = coupled_draws(5_000, 0.30, 1.0, 0.3, np.random.default_rng(4))
    al = allocate(a, z, 1.25, 0.5)
    assert al.k.sum() == pytest.approx(a.sum(), rel=1e-9)
    assert np.all(al.k <= 1.25 * a * (1 + 1e-6))
    assert 0 < al.efficiency < 1


def test_decoupled_concentration_lowers_efficiency():
    eff = [allocate(*coupled_draws(20_000, t, 1.0, 0.0, np.random.default_rng(5)), 1.25, 0.5).efficiency
           for t in (0.05, 0.20, 0.40)]
    assert eff[0] > eff[1] > eff[2]


def test_dynamic_markets_clear_and_taxes_collect():
    p = Params(n=2_000, tau_a=0.02)
    eco = Economy(p, np.random.default_rng(6))
    st = eco.initial_state()
    for _ in range(20):
        eco.step(st)
    a0 = st.a.copy()
    k, n, w, R = eco.clear(st)
    assert k.sum() == pytest.approx(a0.sum(), rel=1e-9)
    assert n.sum() == pytest.approx(p.n, rel=1e-9)
    assert np.all(k <= p.lam * a0 * (1 + 1e-6))
    rec = eco.step(st)
    assert rec.revenue == pytest.approx(0.02 * a0.sum(), rel=1e-9)
    assert 0 < rec.efficiency <= 1
