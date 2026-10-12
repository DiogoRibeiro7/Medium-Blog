import numpy as np
import pytest

from eos import toy


@pytest.mark.parametrize("u", [0.3, 1.0, 2.5])
def test_hessian_at_minimum_has_eigenvalues_zero_and_u2_plus_v2(u):
    v = 1.0 / u
    eig = np.linalg.eigvalsh(toy.hessian(u, v))
    assert eig[0] == pytest.approx(0.0, abs=1e-12)
    assert eig[1] == pytest.approx(u * u + v * v, rel=1e-12)
    assert eig[1] == pytest.approx(toy.minimum_sharpness(toy.imbalance(u, v)), rel=1e-12)


def test_gradient_flow_conserves_imbalance_and_lands_on_predicted_point():
    u0, v0 = 2.2, 0.15
    _, u, v = toy.gradient_flow(u0, v0, t_max=200.0)
    d = toy.imbalance(u, v)
    assert np.max(np.abs(d - d[0])) < 1e-8
    ue, ve = toy.flow_endpoint(u0, v0)
    assert u[-1] == pytest.approx(ue, rel=1e-7)
    assert v[-1] == pytest.approx(ve, rel=1e-7)


@pytest.mark.parametrize("eta", [0.1, 0.45, 0.8])
def test_gd_imbalance_identity_is_exact(eta):
    u, v = toy.gd_trajectory(1.7, 0.4, eta, 300)
    ok = np.isfinite(u)
    u, v = u[ok], v[ok]
    d = toy.imbalance(u, v)
    r = toy.residual(u, v)
    predicted = d[:-1] * (1.0 - eta**2 * r[:-1] ** 2)
    np.testing.assert_allclose(d[1:], predicted, rtol=1e-10, atol=1e-12)


def test_stability_bound_matches_linearised_map():
    eta = 0.6
    dmax = toy.stable_imbalance_bound(eta)
    for frac, stable in [(0.95, True), (1.05, False)]:
        d = frac * dmax
        u = np.sqrt((d + np.sqrt(d * d + 4)) / 2)
        v = 1 / u
        jac = np.eye(2) - eta * toy.hessian(u, v)
        mult = np.linalg.eigvals(jac)
        # one multiplier is exactly 1: the direction along the curve of minimisers
        assert np.min(np.abs(mult - 1.0)) < 1e-10
        transverse = mult[np.argmax(np.abs(mult - 1.0))]
        assert (abs(transverse) < 1) == stable


def test_balanced_map_is_gd_on_diagonal():
    for s in [0.4, 1.1, -0.7]:
        u, v = toy.gd_step(s, s, 0.9)
        assert u == pytest.approx(v)
        assert u == pytest.approx(toy.balanced_map(s, 0.9))


def test_period_two_just_beyond_eta_one():
    pts = toy.orbit(1.05, 1.01, burn=20000, keep=256)
    assert toy.period_of(pts) == 2
    pts = toy.orbit(0.95, 1.2, burn=20000, keep=256)
    assert toy.period_of(pts) == 1
