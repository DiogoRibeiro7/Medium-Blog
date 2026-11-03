"""Population policy benchmarks for Part 12's Gaussian effect model.

Only standard-library mathematics is required. This complements the finite-test-set
Monte Carlo simulation without modifying its training or allocation procedures.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt
from statistics import NormalDist


STANDARD_NORMAL = NormalDist()
CAPACITY: float = 0.20


@dataclass(frozen=True)
class PopulationPolicy:
    """Population selection probability and conditional mean treatment effect."""

    cutoff: float
    left_tail_probability: float
    mean_selected_effect: float


def density(value: float) -> float:
    """Evaluate the standard-normal density."""
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def linear_policy(capacity: float = CAPACITY) -> PopulationPolicy:
    """Select the largest X observations and integrate their true effects."""
    if not 0.0 < capacity < 1.0:
        raise ValueError("capacity must lie strictly between zero and one")
    cutoff: float = STANDARD_NORMAL.inv_cdf(1.0 - capacity)
    ordinate: float = density(cutoff)
    benefit: float = 0.5 + (1.2 + 0.4 * cutoff) * ordinate / capacity
    return PopulationPolicy(cutoff, 0.0, benefit)


def oracle_policy(capacity: float = CAPACITY) -> PopulationPolicy:
    """Integrate the two tails maximizing the convex quadratic true effect.

    tau(x) = 0.4(x + 1.5)^2 - 0.8. An equal-effect cutoff at x=r
    has its second root at x=-3-r. The oracle selects both outer tails.
    """
    if not 0.0 < capacity < 1.0:
        raise ValueError("capacity must lie strictly between zero and one")
    low: float = STANDARD_NORMAL.inv_cdf(1.0 - capacity)
    high: float = low + 1.0
    while STANDARD_NORMAL.cdf(-3.0 - high) + 1.0 - STANDARD_NORMAL.cdf(high) > capacity:
        high += 1.0
    for _ in range(80):
        midpoint: float = (low + high) / 2.0
        selected: float = STANDARD_NORMAL.cdf(-3.0 - midpoint) + 1.0 - STANDARD_NORMAL.cdf(midpoint)
        if selected > capacity:
            low = midpoint
        else:
            high = midpoint
    right: float = (low + high) / 2.0
    left: float = -3.0 - right
    lower_probability: float = STANDARD_NORMAL.cdf(left)
    # Integrals: E[X 1_(X>r)] = phi(r); E[(X²-1) 1_(X>r)] = r phi(r).
    # Corresponding left-tail integrals are -phi(l) and -l phi(l).
    first_moment: float = density(right) - density(left)
    centered_second_moment: float = right * density(right) - left * density(left)
    benefit: float = 0.5 + (1.2 * first_moment + 0.4 * centered_second_moment) / capacity
    return PopulationPolicy(right, lower_probability, benefit)


if __name__ == "__main__":
    linear = linear_policy()
    oracle = oracle_policy()
    print(f"Linear selection cutoff: {linear.cutoff:.9f}")
    print(f"Population linear selected mean: {linear.mean_selected_effect:.9f}")
    print(f"Population oracle selected mean: {oracle.mean_selected_effect:.9f}")
    print(f"Oracle extra left-tail fraction: {oracle.left_tail_probability:.9f}")
    print(f"Population oracle gain over linear: {oracle.mean_selected_effect-linear.mean_selected_effect:.9f}")
