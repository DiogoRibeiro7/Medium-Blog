"""A known data-generating mechanism and a correctly specified generator for it.

The mechanism (n records):

    x3 ~ Bernoulli(0.4)
    (x1, x2) | x3 ~ Normal((0.5 x3, 0), [[1, 0.5], [0.5, 1]])
    y = 1 + 0.5 x1 - 0.3 x2 + 0.8 x3 + 0.4 x1 x3 + e,   e ~ Normal(0, 1)

The analyst regresses y on x1, x2, x3 and x1:x3 by least squares, so the true
coefficients are known exactly. `Parametric` synthesises from the same family
the data came from: it is the generator for which the combining rules were
derived, and the yardstick for the generic generators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generators import Encoder
from inference import ols

TRUE_BETA = np.array([1.0, 0.5, -0.3, 0.8, 0.4])
TERMS = ["intercept", "x1", "x2", "x3", "x1:x3"]
COV = np.array([[1.0, 0.5], [0.5, 1.0]])


def draw(n: int, rng: np.random.Generator) -> pd.DataFrame:
    x3 = (rng.random(n) < 0.4).astype(float)
    z = rng.multivariate_normal([0.0, 0.0], COV, size=n)
    x1 = z[:, 0] + 0.5 * x3
    x2 = z[:, 1]
    y = TRUE_BETA @ np.vstack([np.ones(n), x1, x2, x3, x1 * x3]) + rng.standard_normal(n)
    return pd.DataFrame(
        {"x3": np.where(x3 == 1, "yes", "no"), "x1": x1, "x2": x2, "y": y}
    )


def design(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns of M in order x3 (code 0 = no, 1 = yes), x1, x2, y."""
    x3, x1, x2, y = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
    X = np.column_stack([np.ones(len(M)), x1, x2, x3, x1 * x3])
    return X, y


def fit_analysis(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return ols(*design(M))


def encoder() -> Encoder:
    return Encoder.from_frame(draw(50, np.random.default_rng(0)))


class Parametric:
    """Maximum-likelihood fit of the true family, then sampling from it."""

    name = "parametric"

    def __init__(self, enc: Encoder | None = None):
        self.enc = enc

    def fit(self, M: np.ndarray, rng: np.random.Generator) -> "Parametric":
        x3 = M[:, 0]
        self.p = x3.mean()
        Z = M[:, 1:3]
        self.mu = [Z[x3 == g].mean(axis=0) if np.any(x3 == g) else Z.mean(axis=0) for g in (0, 1)]
        resid = Z - np.where(x3[:, None] == 1, self.mu[1], self.mu[0])
        self.cov = resid.T @ resid / len(M)
        X, y = design(M)
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.sigma = np.sqrt(np.mean((y - X @ self.beta) ** 2))
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x3 = (rng.random(n) < self.p).astype(float)
        Z = rng.multivariate_normal([0.0, 0.0], self.cov, size=n)
        Z = Z + np.where(x3[:, None] == 1, self.mu[1], self.mu[0])
        M = np.column_stack([x3, Z, np.zeros(n)])
        X, _ = design(M)
        M[:, 3] = X @ self.beta + self.sigma * rng.standard_normal(n)
        return M
