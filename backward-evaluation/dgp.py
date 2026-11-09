"""Gaussian AR(1) processes with parameters that change over time.

    y_t = mu_t + phi_t (y_{t-1} - mu_{t-1}) + sigma_t e_t,   e_t ~ N(0, 1)

Each scenario is a function of the row count that returns the three parameter paths.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Paths = tuple[np.ndarray, np.ndarray, np.ndarray]


def simulate(mu: np.ndarray, phi: np.ndarray, sigma: np.ndarray, rng: np.random.Generator, burn: int = 200) -> np.ndarray:
    """Simulate with a burn-in drawn from the first regime."""
    n = len(mu)
    y = np.empty(n + burn)
    m0, p0, s0 = mu[0], phi[0], sigma[0]
    prev = m0 + rng.normal() * s0 / np.sqrt(1 - p0**2)
    for t in range(burn):
        prev = m0 + p0 * (prev - m0) + s0 * rng.normal()
        y[t] = prev
    e = rng.normal(size=n)
    prev_mu = m0
    for t in range(n):
        prev = mu[t] + phi[t] * (prev - prev_mu) + sigma[t] * e[t]
        prev_mu = mu[t]
        y[burn + t] = prev
    return y[burn:]


def step(n: int, at: int, before: float, after: float) -> np.ndarray:
    v = np.full(n, before, dtype=float)
    v[at:] = after
    return v


def scenarios(n: int, at: int) -> dict[str, Callable[[], Paths]]:
    """Parameter paths of the scenarios used in the article (one break at ``at``)."""
    one = np.ones(n)
    return {
        "stationary": lambda: (0 * one, 0.5 * one, one),
        "mean_shift": lambda: (step(n, at, 0.0, 1.5), 0.5 * one, one),
        "variance_down": lambda: (0 * one, 0.5 * one, step(n, at, 2.0, 1.0)),
        "variance_up": lambda: (0 * one, 0.5 * one, step(n, at, 1.0, 2.0)),
        "persistence_up": lambda: (0 * one, step(n, at, 0.2, 0.8), one),
        "drift": lambda: (0 * one, np.linspace(0.1, 0.85, n), one),
    }


LABELS = {
    "stationary": "Stationary",
    "mean_shift": "Mean shift",
    "variance_down": "Volatility falls",
    "variance_up": "Volatility rises",
    "persistence_up": "Persistence rises",
    "drift": "Gradual drift in persistence",
}
