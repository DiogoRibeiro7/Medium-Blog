"""Local experimental design for the one-mode inverse-diffusion problem.

The forward map is u(x, t) = A sin(pi x) exp(-kappa pi^2 t).  In the log
parameters (alpha, beta) = (log A, log kappa) the sensitivities are
du/dalpha = u and du/dbeta = -kappa pi^2 t u, so the Fisher information of
independent Gaussian observations depends on the design only through the
weighted moments S_r = sum_i u_i^2 t_i^r.  Everything here is local: it is
evaluated at an assumed (A, kappa), which is the classical difficulty of
non-linear design (Box and Lucas, 1959; Chaloner and Verdinelli, 1995).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]


def fisher_information(
    x: FloatArray, t: FloatArray, amplitude: float, diffusivity: float, sigma: float
) -> FloatArray:
    """2x2 Fisher information in (log A, log kappa) for Gaussian noise of known sigma."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x.shape != t.shape or sigma <= 0:
        raise ValueError("x and t must share a shape and sigma must be positive")
    u = amplitude * np.sin(np.pi * x) * np.exp(-diffusivity * np.pi**2 * t)
    c = diffusivity * np.pi**2
    w = u**2
    s0, s1, s2 = np.sum(w), np.sum(w * t), np.sum(w * t**2)
    return np.array([[s0, -c * s1], [-c * s1, c**2 * s2]]) / sigma**2


def laplace_sd_log_kappa(information: FloatArray, prior_sd: FloatArray) -> float:
    """Posterior SD of log kappa under the local Gaussian (Laplace) approximation."""
    precision = information + np.diag(1.0 / np.asarray(prior_sd, dtype=np.float64) ** 2)
    return float(np.sqrt(np.linalg.inv(precision)[1, 1]))


def box_lucas_second_time(first_time: float, diffusivity: float) -> float:
    """D-optimal second time for a two-time design of A exp(-c t): t1 + 1/c (Box and Lucas, 1959)."""
    if diffusivity <= 0:
        raise ValueError("diffusivity must be positive")
    return float(first_time + 1.0 / (diffusivity * np.pi**2))


@dataclass(frozen=True)
class TwoTimeDesign:
    """n observations split between an early time and a later time at the same x locations."""

    early: float
    late: float
    n_early: int
    n_total: int

    def times(self) -> FloatArray:
        return np.array([self.early] * self.n_early + [self.late] * (self.n_total - self.n_early))


def early_mask(x: FloatArray, n_early: int) -> NDArray[np.bool_]:
    """Spread the early-time observations evenly across the sorted x locations.

    np.round(np.linspace(0, n - 1, k)) gives k distinct indices whenever k <= n,
    so both times see the same range of sin(pi x).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if not 1 <= n_early < n:
        raise ValueError("n_early must lie in [1, n - 1]")
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(x)[np.round(np.linspace(0, n - 1, n_early)).astype(int)]] = True
    return mask


def assign_times(x: FloatArray, design: TwoTimeDesign) -> FloatArray:
    """Times aligned to x for a two-time design."""
    return np.where(early_mask(x, design.n_early), design.early, design.late)


def two_time_design(
    x: FloatArray,
    first_time: float,
    amplitude: float,
    diffusivity: float,
    sigma: float,
    prior_sd: FloatArray,
    *,
    latest_time: float = 1.5,
) -> TwoTimeDesign:
    """Best integer two-time design for the Laplace SD of log kappa, at an assumed (A, kappa).

    For each split of the n observations between the two times, the late time is
    optimised on (first_time, latest_time]; the best split is returned.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    best: tuple[float, TwoTimeDesign] | None = None
    for n_early in range(1, n):
        mask = early_mask(x, n_early)

        def criterion(late: float, mask: NDArray[np.bool_] = mask) -> float:
            t = np.where(mask, first_time, late)
            return laplace_sd_log_kappa(fisher_information(x, t, amplitude, diffusivity, sigma), prior_sd)

        opt = minimize_scalar(criterion, bounds=(first_time + 1e-3, latest_time), method="bounded",
                              options={"xatol": 1e-6})
        if best is None or opt.fun < best[0]:
            best = (float(opt.fun), TwoTimeDesign(first_time, float(opt.x), n_early, n))
    assert best is not None
    return best[1]


def window_map(
    x: FloatArray,
    starts: FloatArray,
    widths: FloatArray,
    amplitude: float,
    diffusivity: float,
    sigma: float,
    prior_sd: FloatArray,
) -> FloatArray:
    """Laplace SD of log kappa for evenly spaced times on [start, start + width]."""
    out = np.empty((len(widths), len(starts)))
    n = len(x)
    for i, width in enumerate(widths):
        for j, start in enumerate(starts):
            t = np.linspace(start, start + width, n)
            out[i, j] = laplace_sd_log_kappa(fisher_information(x, t, amplitude, diffusivity, sigma), prior_sd)
    return out
