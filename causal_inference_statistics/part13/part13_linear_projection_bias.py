"""Exact population bias from linearly residualizing nonlinear nuisances.

The Part 13 nonlinear DGP has independent standard-normal X coordinates. Its
linear projections leave mutually orthogonal nonlinear residual components.
No external numerical dependencies are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class ProjectionBias:
    """Population components of the misspecified partialling-out coefficient."""

    sine_residual_variance: float
    omitted_treatment_variance: float
    omitted_treatment_outcome_covariance: float
    coefficient: float


def nonlinear_linear_projection_limit(treatment_noise_variance: float = 1.0) -> ProjectionBias:
    """Compute the limiting DML coefficient when only linear X is partialled out.

    Projection of sin(X0) on X0 is exp(-1/2) X0. The centered quadratic
    X1**2-1 and interaction X0*X2 are orthogonal to the linear basis and to
    the sine residual. The treatment residual includes independent noise V.
    """
    if treatment_noise_variance <= 0.0:
        raise ValueError("treatment_noise_variance must be strictly positive")
    sine_variance: float = (1.0 - exp(-2.0)) / 2.0 - exp(-1.0)
    m_variance: float = 1.4**2 * sine_variance + 0.9**2 * 2.0 + 0.5**2
    mg_covariance: float = 1.4 * 1.2 * sine_variance + 0.9 * 1.3 * 2.0 + 0.5 * 0.8
    coefficient: float = 1.0 + mg_covariance / (treatment_noise_variance + m_variance)
    return ProjectionBias(sine_variance, m_variance, mg_covariance, coefficient)


if __name__ == "__main__":
    result = nonlinear_linear_projection_limit()
    print(f"Misspecified linear nuisance population limit: {result.coefficient:.9f}")
    print(f"Exact asymptotic bias relative to tau=1: {result.coefficient - 1.0:.9f}")
