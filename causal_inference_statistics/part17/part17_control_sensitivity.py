"""Analytical sensitivity of a negative-control outcome to confounder overlap.

This is a population and large-sample calculation, not a finite-sample Monte
Carlo result. It extends the binary Part 17 scenarios without changing the
original experiment or its archived results.

The Gaussian data-generating mechanism is::

    A = C + e_A
    Y = A + 1.5 C + V + e_Y
    N = rho C + sqrt(1 - rho**2) V + e_N

All five source variables are mutually independent standard normals. The
negative-control outcome N has no structural dependence on A. The causal
effect of A on Y is 1, while the observational slope is always 1.75.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist


STANDARD_NORMAL: NormalDist = NormalDist()
TRUE_EFFECT: float = 1.0
OBSERVATIONAL_SLOPE: float = 1.75


@dataclass(frozen=True)
class ControlSensitivity:
    """Large-sample diagnostic at a given confounder-control alignment."""

    alignment: float
    sample_size: int
    significance: float
    target_causal_effect: float
    target_observational_slope: float
    target_bias: float
    negative_control_slope: float
    negative_control_residual_variance: float
    approximate_standard_error: float
    approximate_rejection_probability: float


def control_sensitivity(
    alignment: float,
    sample_size: int = 1200,
    significance: float = 0.05,
) -> ControlSensitivity:
    """Compute the power of a two-sided *unadjusted* negative-control test.

    ``alignment`` is the coefficient rho in N's confounder component, not an
    observed correlation or a fitted sensitivity parameter. By construction
    Var(N)=2, Var(A)=2, and Cov(N,A)=rho. Under Gaussian conditioning,
    Var(N|A)=2-rho**2/2 and the leading slope SE is
    sqrt((2-rho**2/2)/(2*n)). The two-sided normal test uses a known nominal
    level and a Gaussian approximation to the sampling distribution.

    The calculation does not cover model-selection, measurement error,
    uncertain negative-control validity, or multiple-testing adjustments.
    """
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, (int, float))
        or not isfinite(alignment)
        or abs(alignment) > 1.0
    ):
        raise ValueError("alignment must be a finite real number in [-1, 1]")
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 30
    ):
        raise ValueError("sample_size must be an integer >= 30")
    if (
        isinstance(significance, bool)
        or not isinstance(significance, (int, float))
        or not isfinite(significance)
        or not 0.0 < significance < 1.0
    ):
        raise ValueError("significance must be a finite number in (0, 1)")

    rho: float = float(alignment)
    alpha: float = float(significance)
    negative_control_slope: float = rho / 2.0
    # Conditional Gaussian variance; no heteroskedasticity is introduced.
    residual_variance: float = 2.0 - rho * rho / 2.0
    approximate_se: float = sqrt(residual_variance / (2.0 * sample_size))
    critical_value: float = STANDARD_NORMAL.inv_cdf(1.0 - alpha / 2.0)
    noncentrality: float = abs(negative_control_slope) / approximate_se

    # P(|Z + delta| > z), where Z is standard normal and delta >= 0.
    power: float = STANDARD_NORMAL.cdf(-critical_value - noncentrality)
    power += STANDARD_NORMAL.cdf(noncentrality - critical_value)

    return ControlSensitivity(
        alignment=rho,
        sample_size=sample_size,
        significance=alpha,
        target_causal_effect=TRUE_EFFECT,
        target_observational_slope=OBSERVATIONAL_SLOPE,
        target_bias=OBSERVATIONAL_SLOPE - TRUE_EFFECT,
        negative_control_slope=negative_control_slope,
        negative_control_residual_variance=residual_variance,
        approximate_standard_error=approximate_se,
        approximate_rejection_probability=power,
    )


def main() -> None:
    """Print a deterministic power grid; the original CSV is untouched."""
    print("alignment,target_bias,control_slope,approx_se,approx_power")
    for alignment in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        result: ControlSensitivity = control_sensitivity(alignment)
        print(
            f"{result.alignment:.2f},{result.target_bias:.4f},"
            f"{result.negative_control_slope:.4f},"
            f"{result.approximate_standard_error:.6f},"
            f"{result.approximate_rejection_probability:.6f}"
        )


if __name__ == "__main__":
    main()
