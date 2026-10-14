import numpy as np
import pytest

from pinncross.problems import Problem


@pytest.mark.parametrize("name,d", [("oscillator", 1), ("oscillator", 4),
                                    ("coupled", 2), ("coupled", 3), ("coupled", 5)])
def test_psi0_is_an_eigenfunction(name, d):
    """H psi0 = E0 psi0, checked with a fourth-order finite-difference Laplacian."""
    P = Problem(name, d)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.7, size=(20, d))
    h = 1e-3

    def psi(y):
        return np.exp(-P.phi(y))

    lap = np.zeros(x.shape[0])
    for k in range(d):
        e = np.zeros(d)
        e[k] = h
        lap += (-psi(x + 2 * e) + 16 * psi(x + e) - 30 * psi(x)
                + 16 * psi(x - e) - psi(x - 2 * e)) / (12 * h * h)
    residual = -0.5 * lap + P.potential(x) * psi(x) - P.E0 * psi(x)
    assert np.max(np.abs(residual / psi(x))) < 1e-6


def test_coupled_reduces_to_oscillator_without_coupling():
    P = Problem("coupled", 3, g=0.0)
    x = np.random.default_rng(1).normal(size=(10, 3))
    np.testing.assert_allclose(P.potential(x), 0.5 * (x**2).sum(1), atol=1e-12)


def test_grid_potential_matches_pointwise_formula():
    P = Problem("coupled", 3)
    nodes = np.linspace(-2.0, 2.0, 5)
    Vg = P.potential_on_grid(nodes)
    lg = P.log_psi0_on_grid(nodes)
    for idx in [(0, 1, 2), (4, 4, 0), (2, 3, 1)]:
        pt = nodes[list(idx)]
        assert Vg[idx] == pytest.approx(P.potential(pt), rel=1e-12)
        assert lg[idx] == pytest.approx(-P.phi(pt), rel=1e-12)


def test_rejection_sampler_matches_quadrature_in_2d():
    """E[x1^2] under |psi0|^2 from 400k exact draws vs 2D quadrature."""
    P = Problem("coupled", 2)
    rng = np.random.default_rng(2)
    X = P.sample_psi0_squared(400_000, rng)
    t = np.linspace(-5, 5, 801)
    xx, yy = np.meshgrid(t, t, indexing="ij")
    pts = np.stack([xx, yy], -1)
    dens = np.exp(-2.0 * P.phi(pts))
    exact = float((dens * xx**2).sum() / dens.sum())
    se = X[:, 0].__pow__(2).std() / np.sqrt(X.shape[0])
    assert abs(np.mean(X[:, 0] ** 2) - exact) < 5 * se
