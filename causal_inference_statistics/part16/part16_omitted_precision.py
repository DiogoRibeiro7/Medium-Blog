"""Population precision benchmark for Part 16's omitted-confounder regression.

Derive the residual variance and leading OLS standard error from the known
Gaussian DGP. These are large-sample approximations, not finite-sample exact
expectations of estimated standard errors or confidence-interval coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist


STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class ObservedPrecision:
    """Asymptotic coefficient, residual variance, SE, and causal coverage."""

    coefficient: float
    residual_variance: float
    standard_error: float
    approximate_causal_coverage: float


def observed_only_precision(strength: float, n: int = 2000) -> ObservedPrecision:
    """Evaluate the misspecified regression Y on intercept, X and W.

    Residual X is λU+eX; conditioning a Gaussian U on this residual leaves
    Var(U|X,W)=1/(1+λ²). The conditional regression error is
    1.5(U-E[U|X,W])+eY, and its variance is 1+2.25/(1+λ²).
    """
    if not isfinite(strength):
        raise ValueError("strength must be finite")
    if not isinstance(n, int) or isinstance(n, bool) or n < 10:
        raise ValueError("n must be an integer >= 10")
    residual_treatment_variance: float = 1.0 + strength * strength
    coefficient: float = 1.0 + 1.5 * strength / residual_treatment_variance
    residual_variance: float = 1.0 + 2.25 / residual_treatment_variance
    standard_error: float = sqrt(residual_variance / (n * residual_treatment_variance))
    offset: float = (coefficient - 1.0) / standard_error
    coverage: float = STANDARD_NORMAL.cdf(1.96 - offset) - STANDARD_NORMAL.cdf(-1.96 - offset)
    return ObservedPrecision(coefficient, residual_variance, standard_error, coverage)


if __name__ == "__main__":
    for strength in (0.0, 0.25, 0.5, 1.0):
        result = observed_only_precision(strength)
        print(f"lambda={strength:g} coefficient={result.coefficient:.6f} "
              f"SE={result.standard_error:.6f} coverage={result.approximate_causal_coverage:.6f}")
