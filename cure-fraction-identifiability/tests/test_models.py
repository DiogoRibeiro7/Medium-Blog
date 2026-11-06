import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa: E402
from models import fit, km_at, km_plateau, maller_zhou_alpha  # noqa: E402
from simulate import Q95, draw_latency, latency_tail, simulate_dataset  # noqa: E402


def test_cure_model_recovers_known_fraction():
    models._check()


def test_weibull_cure_recovers_known_fraction():
    rng = np.random.default_rng(3)
    n = 5000
    cured = rng.random(n) < 0.4
    T = 4.0 * rng.weibull(1.5, n)
    C = rng.uniform(0, 40, n)
    t = np.where(cured, C, np.minimum(T, C))
    e = (~cured & (T <= C)).astype(int)
    f = fit(t, e, "weibull", True)
    assert abs(f.pi - 0.4) < 0.03
    assert abs(np.exp(f.params[2]) - 1.5) < 0.1


def test_no_cure_fit_has_zero_fraction_and_proper_survival():
    rng = np.random.default_rng(4)
    t = np.exp(1 + 0.5 * rng.standard_normal(500))
    e = np.ones(500, int)
    f = fit(t, e, "lognormal", False)
    assert f.pi == 0.0
    assert f.survival(1e6)[0] < 1e-6


def test_kaplan_meier_simple():
    t = np.array([1, 2, 3, 4])
    e = np.array([1, 1, 0, 1])
    assert np.allclose(km_at(t, e, [0.5, 1, 2, 3, 4]), [1.0, 0.75, 0.5, 0.5, 0.0])
    assert km_plateau(t, e) == 0.0
    assert np.isclose(km_plateau(np.array([1, 2, 3, 9]), np.array([1, 0, 1, 0])), 0.75 * 0.5)


def test_maller_zhou_alpha():
    # Y*_(n) = 2, Y_(n) = 10: the window (2*2 - 10, 2] = (-6, 2] holds both events, N_n = 2
    t = np.array([1.0, 2.0, 10.0])
    e = np.array([1, 1, 0])
    assert np.isclose(maller_zhou_alpha(t, e), (1 - 2 / 3) ** 3)
    # last observation is an event: the interval is empty, alpha = 1
    assert maller_zhou_alpha(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 1])) == 1.0


def test_latency_scaled_to_q95():
    rng = np.random.default_rng(5)
    for lat in ("lognormal-0.5", "lognormal-1.0", "weibull-1.5"):
        x = draw_latency(lat, 200_000, rng)
        assert abs(np.quantile(x, 0.95) - Q95) < 0.2
        assert abs(latency_tail(lat, Q95) - 0.05) < 1e-6


def test_simulated_follow_up_is_bounded_by_tau():
    rng = np.random.default_rng(6)
    t, e = simulate_dataset(0.3, "lognormal-1.0", 2000, 0.5, rng)
    assert t.max() <= 0.5 * Q95 + 1e-9
    assert 0 < e.mean() < 0.7
