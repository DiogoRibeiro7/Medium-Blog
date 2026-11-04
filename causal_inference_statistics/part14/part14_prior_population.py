"""Expected-design ridge benchmarks for Part 14's Gaussian causal model.

These coefficients replace empirical cross-products by their population
expectations at a specified sample size. They are *not* exact finite-sample
expectations of random posterior ratios.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedDesignCoefficients:
    """Treatment slopes from the two population-regularized normal equations."""

    unadjusted: float
    adjusted: float


def population_ridge(n: int, precision: float) -> ExpectedDesignCoefficients:
    """Solve the expected design normal equations for both likelihoods.

    For X=U+eX and Y=X+2U+eY, the unadjusted moments are
    E[X²]=2, E[XY]=4. In the adjusted model, the [X,U] covariance
    matrix is [[2,1],[1,1]] and E[[X,U]Y]=[4,3].
    """
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not 0.0 < precision < float("inf"):
        raise ValueError("precision must be finite and positive")
    size: float = float(n)
    penalty: float = float(precision)
    unadjusted: float = 4.0 * size / (2.0 * size + penalty)
    adjusted: float = (size * size + 4.0 * size * penalty) / (
        size * size + 3.0 * size * penalty + penalty * penalty
    )
    return ExpectedDesignCoefficients(unadjusted, adjusted)


if __name__ == "__main__":
    for sample_size in (100, 1000, 5000):
        for prior_precision in (0.1, 10.0, 100.0):
            slopes = population_ridge(sample_size, prior_precision)
            print(f"n={sample_size} lambda={prior_precision:g} "
                  f"unadjusted={slopes.unadjusted:.6f} adjusted={slopes.adjusted:.6f}")
