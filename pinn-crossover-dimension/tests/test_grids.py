import numpy as np
import pytest

from pinncross.grids import (GridHamiltonian, kinetic_1d, separable_oscillator,
                             solve_ground_state)
from pinncross.problems import Problem


@pytest.mark.parametrize("method", ["fd2", "fd4", "sinc"])
def test_matrix_free_operator_equals_dense_kronecker_sum(method):
    P = Problem("coupled", 2)
    n, L = 9, 4.0
    H = GridHamiltonian(P, method, n, L)
    T = kinetic_1d(method, n, L)
    eye = np.eye(n)
    dense = np.kron(T, eye) + np.kron(eye, T) + np.diag(H.V.ravel())
    v = np.random.default_rng(0).normal(size=(n, n))
    np.testing.assert_allclose(H.apply(v).ravel(), dense @ v.ravel(), atol=1e-9)


@pytest.mark.parametrize("method", ["fd2", "fd4", "sinc"])
def test_lobpcg_reproduces_the_separable_answer(method):
    P = Problem("oscillator", 3)
    r = solve_ground_state(P, method, 18, 5.0, track_memory=False)
    s = separable_oscillator(method, 18, 5.0, 3)
    assert r["E"] == pytest.approx(s["E"], rel=1e-10)


def test_coupled_2d_eigenvalue_matches_dense_solver():
    P = Problem("coupled", 2)
    n, L = 24, 5.0
    H = GridHamiltonian(P, "sinc", n, L)
    T = kinetic_1d("sinc", n, L)
    eye = np.eye(n)
    dense = np.kron(T, eye) + np.kron(eye, T) + np.diag(H.V.ravel())
    lam = np.linalg.eigvalsh(dense)[0]
    r = solve_ground_state(P, "sinc", n, L, track_memory=False)
    assert r["E"] == pytest.approx(lam, rel=1e-10)


@pytest.mark.parametrize("method,order", [("fd2", 2.0), ("fd4", 4.0)])
def test_finite_difference_convergence_order(method, order):
    errs = [separable_oscillator(method, n, 6.0, 1)["rel_energy_error"] for n in (40, 80, 160)]
    slopes = np.diff(np.log(errs)) / np.diff(np.log([41.0, 81.0, 161.0]))
    assert np.all(np.abs(-slopes - order) < 0.15)


def test_sinc_converges_spectrally():
    assert separable_oscillator("sinc", 40, 6.0, 1)["rel_energy_error"] < 1e-12
