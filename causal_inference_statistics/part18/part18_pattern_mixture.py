"""Exact pattern-mixture sensitivity for randomized trials with missing outcomes.

The calculation reconstructs a *realized full-data arm-mean contrast* under
assumed shifts between unobserved and observed outcomes in each treatment arm.
It does not identify the missing-outcome means from observed data, calculate
sampling uncertainty, or make an unadjusted contrast equal to a finite-sample
causal effect. Randomization supports its interpretation as an ATE estimator.

Part 18's original experiment and archived CSV are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _finite_real(value: float, name: str) -> float:
    """Validate a scalar before using it in a sensitivity calculation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class ObservedArm:
    """Respondent mean and response fraction for one randomized arm.

    The respondent mean exists only if at least one outcome was observed;
    hence the response fraction must be strictly positive and no more than 1.
    """

    respondent_mean: float
    response_fraction: float

    def __post_init__(self) -> None:
        """Reject invalid summaries before carrying them into an analysis."""
        _finite_real(self.respondent_mean, "respondent_mean")
        fraction: float = _finite_real(self.response_fraction, "response_fraction")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("response_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ContrastBounds:
    """Sharp arithmetic bounds conditional on prespecified shift intervals."""

    respondent_contrast: float
    lower: float
    upper: float


def _shift_interval(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    """Validate an inclusive interval of missing-minus-observed shifts."""
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError(f"{name} must be a pair of finite bounds")
    lower: float = _finite_real(bounds[0], f"{name} lower")
    upper: float = _finite_real(bounds[1], f"{name} upper")
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    return lower, upper


def completed_arm_mean(arm: ObservedArm, missing_shift: float) -> float:
    """Reconstruct the full mean under E[Y|R=0]-E[Y|R=1]=shift.

    The equality is a mixture identity for each arm, not a missing-at-random
    claim. It is exact for realized arm means when the shift equals the actual
    nonrespondent-minus-respondent mean, and hypothetical otherwise.
    """
    delta: float = _finite_real(missing_shift, "missing_shift")
    return arm.respondent_mean + (1.0 - arm.response_fraction) * delta


def completed_contrast(
    treated: ObservedArm,
    control: ObservedArm,
    treated_missing_shift: float = 0.0,
    control_missing_shift: float = 0.0,
) -> float:
    """Return the reconstructed treated-minus-control full-data contrast."""
    return completed_arm_mean(treated, treated_missing_shift) - completed_arm_mean(
        control, control_missing_shift
    )


def contrast_bounds(
    treated: ObservedArm,
    control: ObservedArm,
    treated_shifts: tuple[float, float],
    control_shifts: tuple[float, float],
) -> ContrastBounds:
    """Bound the reconstructed contrast over rectangular shift assumptions.

    The contrast increases with the treated shift and decreases with the
    control shift. No stochastic independence of the two shifts is assumed;
    the interval records the extrema over the specified rectangular region.
    """
    treated_min, treated_max = _shift_interval(treated_shifts, "treated_shifts")
    control_min, control_max = _shift_interval(control_shifts, "control_shifts")

    return ContrastBounds(
        respondent_contrast=treated.respondent_mean - control.respondent_mean,
        lower=completed_contrast(treated, control, treated_min, control_max),
        upper=completed_contrast(treated, control, treated_max, control_min),
    )


def treated_shift_for_zero_contrast(
    treated: ObservedArm,
    control: ObservedArm,
    control_missing_shift: float = 0.0,
) -> float:
    """Solve exactly for a treated-arm shift giving zero full-data contrast.

    A solution in the treated shift is unique only if some treated outcomes
    are missing. The specified control shift remains an assumption.
    """
    control_delta: float = _finite_real(control_missing_shift, "control_missing_shift")
    if treated.response_fraction == 1.0:
        raise ValueError("treated arm must have missing outcomes for a unique shift")
    observed_difference: float = treated.respondent_mean - control.respondent_mean
    return (
        -observed_difference + (1.0 - control.response_fraction) * control_delta
    ) / (1.0 - treated.response_fraction)


def main() -> None:
    """Print an illustrative tipping-point calculation, not study results."""
    treated = ObservedArm(respondent_mean=2.0, response_fraction=0.55)
    control = ObservedArm(respondent_mean=1.2, response_fraction=0.70)
    assumed_bounds: ContrastBounds = contrast_bounds(
        treated, control, treated_shifts=(-2.0, 0.0), control_shifts=(-2.0, 0.0)
    )
    print("Illustrative values (not generated by Part 18's experiment)")
    print(f"Respondent mean contrast: {assumed_bounds.respondent_contrast:.4f}")
    print(f"Contrast sensitivity interval: [{assumed_bounds.lower:.4f}, "
          f"{assumed_bounds.upper:.4f}]")
    print("Treated missing-minus-observed shift for zero contrast "
          f"(control shift 0): {treated_shift_for_zero_contrast(treated, control):.4f}")


if __name__ == "__main__":
    main()
