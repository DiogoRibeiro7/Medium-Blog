import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model as m  # noqa: E402
import scenarios as s  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return s.food_world()


def test_gini_round_trip():
    hh = m.Households.from_gini(0.309, 0.4)
    # Gini of a lognormal: 2 Phi(sigma / sqrt 2) - 1
    from scipy.special import ndtr

    assert 2 * ndtr(hh.sigma / np.sqrt(2)) - 1 == pytest.approx(0.309, abs=1e-12)


def test_reference_price_hits_baseline_share():
    hh = m.Households.from_gini(0.309, 0.4)
    p0 = m.reference_price(hh, 0.05)
    assert float(hh.priced_out(p0)) == pytest.approx(0.05, abs=1e-12)


def test_need_demand_is_monotone_and_bounded():
    hh = m.Households.from_gini(0.309, 0.4)
    p = np.exp(np.linspace(-5, 3, 400))
    d = hh.demand(p)
    assert np.all(np.diff(d) < 0)
    assert d.max() <= 1.0 and d.min() > 0.0


def test_inverse_demand(world):
    for q in (0.6, 0.9, 1.0, 1.1):
        p = float(world.dem.inverse(q))
        assert float(world.dem.total(p)) == pytest.approx(q, rel=1e-4)


def test_food_elasticity_in_range(world):
    assert -0.3 < world.dem.elasticity() < -0.1


def test_price_is_world_price_while_borrowing_fills_the_gap(world):
    for K in (0.0, 0.3, 0.7, 0.95):
        p, b = world.price(K, 1.0)
        assert p == 1.0
        assert b == pytest.approx(world._q_w - K)


def test_cap_raises_price_when_binding(world):
    p, b = world.price(0.2, 0.3)
    assert p > 1.0 and b == 0.3


def test_fold_matches_equilibrium_scan(world):
    phi_c, _ = s.fold_open(world.cap, 1.0, world._q_w, s.DELTA0)
    below = [k for k, st in world.equilibria(phi_c - 0.005, s.DELTA0, 1.0) if st and k > 0]
    above = [k for k, st in world.equilibria(phi_c + 0.005, s.DELTA0, 1.0) if st and k > 0]
    assert below and not above


def test_collapsed_state_cannot_restart_at_world_price(world):
    # unit cost of a tiny sector exceeds the world price even with no effort premium
    assert world.cap.cost(0.0, 0.0) > 1.0


def test_capped_borrowing_keeps_a_sector_beyond_the_fold(world):
    phi_c, _ = s.fold_open(world.cap, 1.0, world._q_w, s.DELTA0)
    st = [k for k, ok in world.equilibria(phi_c + 0.05, s.DELTA0, 0.3) if ok and k > 0.3]
    assert st


def test_growth_asymmetry():
    cap = m.Capacity()
    # strongly positive vs strongly negative margin at full scale
    up = cap.growth(1.0, 10.0, 0.025) / 1.0
    down = cap.growth(1.0, -10.0, 0.025) / 1.0
    assert down < -0.11 and 0.03 < up < 0.05


def test_batch_simulator_matches_scalar(world):
    phi = 0.25
    K0 = max(k for k, st in world.equilibria(phi, s.DELTA0, 1.0) if st)
    rate = s.wave_rate(0.519, 5.0)
    w = m.FoodModel(hh=world.hh, cap=world.cap, p0=world.p0, phi=m.const(phi), delta=m.step(s.DELTA0, rate, 5.0, 10.0), M=m.const(1.0))
    scalar = w.simulate(K0, 0.0, 40.0, 0.1)["K"][-1]
    caps = {k: np.array([getattr(world.cap, k)]) for k in ("c_inf", "dc", "K_h", "n", "beta", "g_max", "l_max", "pi_s", "K_g", "eps")}
    batch = s.simulate_food_batch(
        world.hh, world.p0, caps, np.array([phi]), np.array([1.0]),
        lambda t: np.where((t >= 5.0) & (t < 10.0), rate, s.DELTA0), np.array([K0]), 40.0, 0.1,
    )[0]
    assert batch == pytest.approx(scalar, rel=1e-6, abs=1e-9)


def test_housing_steady_state_is_stationary():
    params = {"kappa": 0.1, "x_bust": 0.9, "x_boom": 1.3, "household_growth": (0.0073, 0.0138)}
    w = s.housing_world(params, cost_shock=m.const(0.0), nonprimary=m.const(0.44), growth_need=m.const(0.0073))
    H, K = w.steady_state(0.0)
    d = w.rhs(0.0, np.array([H, K]))
    assert abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9
