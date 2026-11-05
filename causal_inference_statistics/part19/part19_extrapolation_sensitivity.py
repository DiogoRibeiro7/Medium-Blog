"""Sharp transportability counterexample for Part 19's nonoverlap scenario.

Trial X is uniform on [-1, 1], while target X is uniform on [0.5, 2.5].
The alternative treatment-effect curve equals the original 1 + X within
trial support, but can depart quadratically for target-only values X > 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TransportSensitivity:
    """Population effects under a trial-indistinguishable alternative curve."""

    curvature: float
    trial_ate: float
    supported_target_ate: float
    full_target_ate: float
    linear_extrapolation_bias: float


def target_effect_with_curvature(curvature: float) -> TransportSensitivity:
    """Return target ATE if tau(x)=1+x+curvature*max(x-1,0)**2.

    The trial data distribution, including all treatment/outcome observations,
    is unchanged for every curvature because the added term vanishes on the
    entire trial covariate support [-1, 1]. The target-only interval (1,2.5]
    contributes E[max(X-1,0)^2] = 9/16 under Uniform[0.5,2.5].

    Args:
        curvature: Finite coefficient of the unsupported quadratic effect.

    Raises:
        ValueError: If curvature is not finite.
    """
    if not isfinite(curvature):
        raise ValueError("curvature must be finite")

    extrapolated = 2.5  # E_target[1 + X], under the fitted linear effect.
    actual = extrapolated + (9.0 / 16.0) * curvature
    return TransportSensitivity(
        curvature=curvature,
        trial_ate=1.0,
        supported_target_ate=1.75,
        full_target_ate=actual,
        linear_extrapolation_bias=extrapolated - actual,
    )


if __name__ == "__main__":
    for coefficient in (0.0, 0.5, 1.0, 2.0):
        print(target_effect_with_curvature(coefficient))
