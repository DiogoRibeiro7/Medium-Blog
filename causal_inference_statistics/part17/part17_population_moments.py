"""Population OLS benchmarks for Part 17's continuous-treatment experiment.

The standard errors are large-sample approximations for samples of size ``n``.
They are not exact finite-sample expectations of the fitted OLS standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class RegressionMoment:
    """Population slope and leading-order standard error for one regression."""

    slope: float
    asymptotic_se: float


def population_moments(scenario: str, n: int = 1200) -> dict[str, RegressionMoment]:
    """Derive target, negative-control, and placebo OLS benchmarks.

    All independent Gaussian inputs have unit variance. The target slope is
    observational and need not equal the structural unit-shift effect of one.

    Args:
        scenario: Assignment mechanism used by ``experiment.py``.
        n: Positive sample size used for the asymptotic SE approximation.

    Raises:
        ValueError: If ``scenario`` is unknown or ``n`` is too small.
        TypeError: If ``n`` is not a plain integer.
    """
    if type(n) is not int:
        raise TypeError("n must be an integer")
    if n < 3:
        raise ValueError("n must be at least 3")

    # Tuples contain (Var(A), slope(Y~A), Var(Y|A), slope(N~A),
    # Var(N|A), Var(Y)); Var(P)=1 and Cov(P,Y)=0 throughout.
    parameters: dict[str, tuple[float, float, float, float, float, float]] = {
        "no_confounding": (1.0, 1.0, 3.0, 0.0, 2.0, 4.0),
        "shared_confounder": (2.0, 1.75, 3.125, 0.5, 1.5, 9.25),
        "outcome_specific_confounder": (2.0, 1.75, 3.125, 0.0, 2.0, 9.25),
    }
    if scenario not in parameters:
        raise ValueError(f"unknown scenario: {scenario}")

    var_a, target_slope, target_residual, nc_slope, nc_residual, var_y = parameters[scenario]
    return {
        "target": RegressionMoment(target_slope, sqrt(target_residual / (n * var_a))),
        "negative_control_outcome": RegressionMoment(nc_slope, sqrt(nc_residual / (n * var_a))),
        "placebo_exposure": RegressionMoment(0.0, sqrt(var_y / n)),
    }
