"""Tensor-product grid solvers for the ground state, matrix-free, in any dimension.

Three discretisations of the kinetic operator -1/2 d^2/dx^2 on N interior nodes of
[-L, L] (spacing h = 2L / (N + 1)):

* ``fd2``  - the 3-point second-order stencil,
* ``fd4``  - the 5-point fourth-order stencil (nodes outside the box are taken as 0),
* ``sinc`` - the sinc discrete variable representation of Colbert and Miller (1992),
  a spectral method whose error falls exponentially with N for smooth, decaying psi.

In d dimensions H = sum_k T_k + diag(V), applied without ever forming a matrix. The
lowest eigenpair comes from LOBPCG with a kinetic-energy preconditioner: the inverse
of (symbol of T, summed over axes) + c, applied with a type-I discrete sine transform.
The start vector is the constant vector, so nothing about the answer is used.
"""

from __future__ import annotations

import time
import tracemalloc
from functools import lru_cache

import numpy as np
from scipy import fft as sfft
from scipy import sparse
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse.linalg import eigsh

from .problems import Problem

METHODS = ("fd2", "fd4", "sinc")
FD_STENCILS = {
    # coefficients of d^2/dx^2 * h^2 at offsets 0, 1, 2, ...
    "fd2": (-2.0, 1.0),
    "fd4": (-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0),
}


def nodes(n: int, L: float) -> np.ndarray:
    return np.linspace(-L, L, n + 2)[1:-1]


def kinetic_1d(method: str, n: int, L: float) -> np.ndarray:
    """Dense N x N kinetic matrix T (only used in 1D, for tests and the separable case)."""
    h = 2.0 * L / (n + 1)
    if method in FD_STENCILS:
        c = FD_STENCILS[method]
        T = np.zeros((n, n))
        for off, coef in enumerate(c):
            T += np.diag(np.full(n - off, coef), off)
            if off:
                T += np.diag(np.full(n - off, coef), -off)
        return -0.5 * T / h**2
    if method == "sinc":
        i = np.arange(n)
        k = i[:, None] - i[None, :]
        off = 2.0 * (-1.0) ** k / np.where(k == 0, 1, k) ** 2
        return np.where(k == 0, np.pi**2 / 3.0, off) / (2.0 * h**2)
    raise ValueError(method)


def kinetic_symbol(method: str, n: int, L: float) -> np.ndarray:
    """Eigenvalue-like symbol of T at the DST-I frequencies theta_j = j pi / (N + 1)."""
    h = 2.0 * L / (n + 1)
    th = np.arange(1, n + 1) * np.pi / (n + 1)
    if method == "fd2":
        return (1.0 - np.cos(th)) / h**2
    if method == "fd4":
        return (2.5 - 8.0 * np.cos(th) / 3.0 + np.cos(2.0 * th) / 6.0) / (2.0 * h**2)
    if method == "sinc":
        return th**2 / (2.0 * h**2)
    raise ValueError(method)


class GridHamiltonian:
    """H = sum_k T_k + V on the grid nodes^d, as functions on arrays of shape (N,)*d."""

    def __init__(self, problem: Problem, method: str, n: int, L: float):
        if method not in METHODS:
            raise ValueError(method)
        self.problem, self.method, self.n, self.L = problem, method, n, L
        self.d = problem.d
        self.h = 2.0 * L / (n + 1)
        self.x = nodes(n, L)
        self.shape = (n,) * self.d
        self.size = n**self.d
        self.V = problem.potential_on_grid(self.x)
        if method == "sinc":
            self.T1 = kinetic_1d("sinc", n, L)
        sym = kinetic_symbol(method, n, L)
        shift = 1.0  # generic shift; any positive constant keeps the preconditioner SPD
        D = np.full(self.shape, shift)
        for k in range(self.d):
            s = [1] * self.d
            s[k] = n
            D = D + sym.reshape(s)
        self.precond_inv = 1.0 / D
        del D
        # potential-aware scaling: S = sqrt((t + c) / (t + V+ + c)) with t the kinetic
        # diagonal; S = 1 where V is small, and S ~ 1/sqrt(V) in the box corners where
        # the potential reaches 10^4-10^5, so M = S (T + c)^-1 S also damps those modes.
        if method == "sinc":
            tdiag = self.d * np.pi**2 / (6.0 * self.h**2)
        else:
            tdiag = self.d * (-0.5 * FD_STENCILS[method][0] / self.h**2)
        self.scale = np.sqrt((tdiag + shift) / (tdiag + np.maximum(self.V, 0.0) + shift))

    # kinetic term -------------------------------------------------------------------

    def _kinetic_fd(self, v: np.ndarray) -> np.ndarray:
        coefs = FD_STENCILS[self.method]
        scale = -0.5 / self.h**2
        out = (coefs[0] * self.d * scale) * v
        for k in range(self.d):
            for off, c in enumerate(coefs[1:], start=1):
                lo = [slice(None)] * self.d
                hi = [slice(None)] * self.d
                lo[k] = slice(None, -off)
                hi[k] = slice(off, None)
                lo, hi = tuple(lo), tuple(hi)
                out[hi] += (c * scale) * v[lo]
                out[lo] += (c * scale) * v[hi]
        return out

    def _kinetic_sinc(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v)
        for k in range(self.d):
            out += np.moveaxis(np.tensordot(self.T1, v, axes=([1], [k])), 0, k)
        return out

    def apply(self, v: np.ndarray) -> np.ndarray:
        if self.method == "sinc":
            out = self._kinetic_sinc(v)
        else:
            out = self._kinetic_fd(v)
        out += self.V * v
        return out

    def precondition(self, r: np.ndarray) -> np.ndarray:
        t = sfft.dstn(r * self.scale, type=1, norm="ortho", workers=-1)
        t *= self.precond_inv
        t = sfft.idstn(t, type=1, norm="ortho", workers=-1)
        t *= self.scale
        return t


def lobpcg_lowest(apply, precondition, x0: np.ndarray, rtol_energy: float = 1e-13,
                  rtol_residual: float = 1e-10, maxiter: int = 500):
    """Lowest eigenpair of a symmetric operator by single-vector LOBPCG (Knyazev 2001).

    Rayleigh-Ritz on span{x, M r, p} at every step. Stops when the eigenvalue has
    changed by less than rtol_energy (relative) in three consecutive iterations, or the
    residual norm falls below rtol_residual * |lambda|. Energy-based stopping matters
    here: on the coupled problem the potential reaches 10^4-10^5 in the box corners,
    where the eigenvector is ~0 and residual components converge slowly without
    affecting the eigenvalue. Holds about seven vectors of the grid's size.
    """
    x = x0 / np.linalg.norm(x0)
    Ax = apply(x)
    lam = float(np.vdot(x, Ax))
    p = Ap = None
    quiet = 0
    history = [lam]
    for it in range(1, maxiter + 1):
        r = Ax - lam * x
        rn = float(np.linalg.norm(r))
        if rn < rtol_residual * abs(lam):
            break
        w = precondition(r)
        del r
        w -= np.vdot(x, w) * x
        w /= np.linalg.norm(w)
        Aw = apply(w)
        basis, abasis = [x, w], [Ax, Aw]
        if p is not None:
            basis.append(p)
            abasis.append(Ap)
        k = len(basis)
        G = np.array([[np.vdot(basis[i], basis[j]) for j in range(k)] for i in range(k)])
        K = np.array([[np.vdot(basis[i], abasis[j]) for j in range(k)] for i in range(k)])
        K = 0.5 * (K + K.T)
        try:
            mu, C = eigh(K, G)
        except np.linalg.LinAlgError:          # p nearly dependent: drop it
            basis, abasis = basis[:2], abasis[:2]
            mu, C = eigh(K[:2, :2], G[:2, :2])
        c = C[:, 0]
        new_p = c[1] * w + (c[2] * p if len(basis) == 3 else 0.0)
        new_Ap = c[1] * Aw + (c[2] * Ap if len(basis) == 3 else 0.0)
        x = c[0] * x + new_p
        Ax = c[0] * Ax + new_Ap
        del w, Aw
        p, Ap = new_p, new_Ap
        nx = np.linalg.norm(x)
        x /= nx
        Ax /= nx
        pn = np.linalg.norm(p)
        p /= pn
        Ap /= pn
        new_lam = float(np.vdot(x, Ax))
        quiet = quiet + 1 if abs(new_lam - lam) < rtol_energy * abs(new_lam) else 0
        lam = new_lam
        history.append(lam)
        if quiet >= 3:
            break
    rn = float(np.linalg.norm(Ax - lam * x))
    return lam, x, it, rn, history


def solve_ground_state(problem: Problem, method: str, n: int, L: float,
                       maxiter: int = 500, track_memory: bool = True) -> dict:
    """Lowest eigenpair of the d-dimensional grid Hamiltonian, with cost measurements."""
    if track_memory:
        tracemalloc.start()
    t_cpu, t_wall = time.process_time(), time.perf_counter()
    H = GridHamiltonian(problem, method, n, L)
    x0 = np.ones(H.shape)
    E, v, iters, res, _ = lobpcg_lowest(H.apply, H.precondition, x0, maxiter=maxiter)
    cpu = time.process_time() - t_cpu
    wall = time.perf_counter() - t_wall
    v = v.ravel()
    logp = problem.log_psi0_on_grid(H.x).ravel()
    p = np.exp(logp - logp.max())
    cosang = abs(float(v @ p)) / (np.linalg.norm(v) * np.linalg.norm(p))
    l2 = float(np.sqrt(max(0.0, 2.0 - 2.0 * cosang)))
    peak = None
    if track_memory:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return {
        "problem": problem.name, "d": problem.d, "g": problem.g, "method": method,
        "n": n, "L": L, "unknowns": H.size,
        "E": E, "E0": problem.E0, "rel_energy_error": abs(E - problem.E0) / problem.E0,
        "l2_error_nodal": l2, "iterations": iters, "final_residual": res,
        "cpu_s": cpu, "wall_s": wall, "peak_bytes": peak,
    }


@lru_cache(maxsize=None)
def _oscillator_1d(method: str, n: int, L: float) -> tuple[float, float]:
    """Lowest eigenvalue of the 1D grid oscillator and the cosine between its
    eigenvector and the exact ground state at the nodes."""
    x = nodes(n, L)
    if method in FD_STENCILS:
        coefs = FD_STENCILS[method]
        h = 2.0 * L / (n + 1)
        if len(coefs) == 2:      # tridiagonal: O(N) memory even for very large N
            diag = -0.5 * coefs[0] / h**2 + 0.5 * x**2
            off = np.full(n - 1, -0.5 * coefs[1] / h**2)
            w, V = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
        else:                    # pentadiagonal: sparse shift-invert near the bottom
            offs = list(range(-(len(coefs) - 1), len(coefs)))
            bands = [np.full(n - abs(k), -0.5 * coefs[abs(k)] / h**2) for k in offs]
            H = sparse.diags(bands, offs, format="csc") + sparse.diags(0.5 * x**2)
            w, V = eigsh(H, k=1, sigma=0.0, which="LM")
    else:
        H1 = kinetic_1d(method, n, L) + np.diag(0.5 * x**2)
        w, V = eigh(H1, subset_by_index=[0, 0])
    v = V[:, 0]
    p = np.exp(-0.5 * x**2)
    c = abs(float(v @ p)) / (np.linalg.norm(v) * np.linalg.norm(p))
    return float(w[0]), c


def separable_oscillator(method: str, n: int, L: float, d: int) -> dict:
    """Exact grid answer for the d-dimensional oscillator from one 1D eigenproblem.

    On a tensor-product grid the oscillator Hamiltonian is a Kronecker sum of identical
    1D operators, so its lowest eigenvalue is d times the 1D one and its eigenvector is
    the d-fold tensor power of the 1D eigenvector. Nothing is extrapolated.
    """
    e1, c = _oscillator_1d(method, int(n), float(L))
    E = d * e1
    return {
        "problem": "oscillator", "d": d, "method": method, "n": n, "L": L,
        "unknowns": n**d, "E": E, "E0": d / 2.0,
        "rel_energy_error": abs(E - d / 2.0) / (d / 2.0),
        "l2_error_nodal": float(np.sqrt(max(0.0, 2.0 - 2.0 * c**d))),
    }


def oscillator_n_needed(method: str, L: float, target: float, nmax: int = 20000,
                        metric: str = "rel_energy_error", d: int = 1,
                        nmin: int = 2) -> int | None:
    """Smallest N per axis from which the oscillator's exact grid error stays below target.

    Bisection, then a check at 1.25, 1.5, 2, 3 and 4 times the answer: at coarse N the
    discretisation and box-truncation errors can cancel by accident, and an N that only
    reaches the target by such a cancellation is rejected (the search restarts above the
    first larger N that fails). Returns None if even nmax does not reach the target.
    """
    def err(n):
        return separable_oscillator(method, n, L, d)[metric]

    lo, hi = nmin, nmax
    if err(hi) >= target:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err(mid) < target:
            hi = mid
        else:
            lo = mid
    for k in (1.25, 1.5, 2.0, 3.0, 4.0):
        m = min(int(np.ceil(hi * k)), nmax)
        if err(m) >= target:
            return oscillator_n_needed(method, L, target, nmax, metric, d, nmin=m)
    return hi
