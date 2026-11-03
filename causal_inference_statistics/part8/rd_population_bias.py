"""Population target of the Part 8 triangular-kernel local linear RD fit.

This module uses exact polynomial moments rather than a Monte Carlo average.
The running variable is uniform on [-1, 1], and the cutoff is zero.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationRD:
    """The local-linear population discontinuity and its causal discrepancy."""

    scenario: str
    bandwidth: float
    causal_effect: float
    fitted_jump: float
    smoothing_bias: float


def triangular_moment(order: int) -> float:
    """Return integral_0^1 t**order (1-t) dt for nonnegative order."""
    if not isinstance(order, int) or order < 0:
        raise ValueError("Moment order must be a nonnegative integer")
    return 1.0 / ((order + 1) * (order + 2))


def quadratic_boundary_intercept(coefficient: float, bandwidth: float) -> float:
    """Intercept fitted to `coefficient*x**2` for 0 <= x <= h.

    The weighted population regression includes an intercept and slope,
    using triangular weights (1-x/h). A scale change x=h*t yields exact
    normal equations with moments of t**k (1-t).
    """
    if not 0 < bandwidth <= 1:
        raise ValueError("Bandwidth must lie in (0, 1]")
    moments: tuple[float, ...] = tuple(triangular_moment(k) for k in range(4))
    determinant: float = moments[0] * moments[2] - moments[1] ** 2
    intercept_factor: float = (
        moments[2] ** 2 - moments[1] * moments[3]
    ) / determinant
    return coefficient * bandwidth**2 * intercept_factor


def population_target(scenario: str, bandwidth: float) -> PopulationRD:
    """Compute the population local-linear target for each Part 8 design."""
    if scenario not in {"smooth", "curvature", "sorting"}:
        raise ValueError("Unknown Part 8 scenario")
    effect: float = -1.0
    # 'sorting' adds a +0.7 untreated-outcome jump but no density manipulation.
    fitted: float = effect + (0.7 if scenario == "sorting" else 0.0)
    if scenario == "curvature":
        fitted += quadratic_boundary_intercept(2.0, bandwidth)
    return PopulationRD(scenario, bandwidth, effect, fitted, fitted - effect)
