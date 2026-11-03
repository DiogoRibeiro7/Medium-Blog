"""Exact-integration audit of the Part 10 naive treatment contrast.

For X~N(0,1), treatment assignment expit(-.3+.8X-.45X**2),
and untreated prognosis 1+.7X+.8X**2, the naive group difference has
two opposing confounding components. Gauss-Hermite quadrature isolates them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit


@dataclass(frozen=True)
class ConfoundingDecomposition:
    """Population selection contributions to the unadjusted contrast."""

    treatment_probability: float
    linear_component: float
    quadratic_component: float
    total_bias: float
    naive_contrast: float


def population_contrast(nodes: int = 100) -> ConfoundingDecomposition:
    """Integrate the conditional X moments given treatment status.

    Hermite nodes integrate exp(-x**2) weighted functions. The sqrt(2)
    transformation and division by sqrt(pi) recover N(0,1) expectations.
    """
    if not isinstance(nodes, int) or not 20 <= nodes <= 180:
        raise ValueError("Nodes must be an integer between 20 and 180")
    roots, weights = hermgauss(nodes)
    x: NDArray[np.float64] = np.sqrt(2.0) * roots
    normalized: NDArray[np.float64] = weights / np.sqrt(np.pi)
    propensity: NDArray[np.float64] = expit(-0.3 + 0.8 * x - 0.45 * x**2)
    treated_share: float = float(np.dot(normalized, propensity))
    control_share: float = 1.0 - treated_share

    def conditional_difference(function: NDArray[np.float64]) -> float:
        treated = np.dot(normalized, propensity * function) / treated_share
        control = np.dot(normalized, (1.0 - propensity) * function) / control_share
        return float(treated - control)

    linear: float = 0.7 * conditional_difference(x)
    quadratic: float = 0.8 * conditional_difference(x**2)
    total: float = linear + quadratic
    return ConfoundingDecomposition(
        treatment_probability=treated_share,
        linear_component=linear,
        quadratic_component=quadratic,
        total_bias=total,
        naive_contrast=1.0 + total,
    )


def main() -> None:
    """Print the reproducible theoretical decomposition."""
    result = population_contrast()
    for field, value in vars(result).items():
        print(f"{field}: {value:.9f}")


if __name__ == "__main__":
    main()
