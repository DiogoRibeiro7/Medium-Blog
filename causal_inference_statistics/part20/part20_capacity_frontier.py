"""Population capacity frontier for the Part 20 linear treatment-benefit DGP.

The original simulation assigns a benefit B(X) = 0.5 + 1.5 X with X ~ N(0,1)
and treats exactly 20% of the population. This analytical companion compares
an *exactly q* allocation with an *at most q* allocation that may leave spare
capacity rather than treat people whose benefit is negative.

These are population expectations under the known structural DGP, not
estimates, finite-sample uncertainty intervals, or learned-policy guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import NormalDist


NORMAL: NormalDist = NormalDist()
INTERCEPT: float = 0.5
SLOPE: float = 1.5
BENEFIT_ZERO_X: float = -INTERCEPT / SLOPE


@dataclass(frozen=True)
class CapacityFrontier:
    """Expected population gain and allocation shares for one capacity level."""

    capacity: float
    positive_benefit_share: float
    exact_treated_share: float
    optional_treated_share: float
    exact_population_gain: float
    optional_population_gain: float
    forced_negative_benefit_share: float
    avoidable_population_loss: float


def _top_share_population_gain(treated_share: float) -> float:
    """Compute E[(0.5+1.5X)1{X exceeds its upper-share cutoff}].

    For 0 < t < 1 and z=Phi^{-1}(1-t), E[X 1{X>z}]=phi(z).
    At the endpoints, the gain is 0 for nobody and E[B]=0.5 for all.
    """
    if treated_share == 0.0:
        return 0.0
    if treated_share == 1.0:
        return INTERCEPT
    cutoff: float = NORMAL.inv_cdf(1.0 - treated_share)
    return INTERCEPT * treated_share + SLOPE * NORMAL.pdf(cutoff)


def population_capacity_frontier(capacity: float) -> CapacityFrontier:
    """Compare top-CATE policies with and without a mandate to fill capacity.

    The ranking is oracle-correct because the structural treatment benefit is
    strictly increasing in X. At-most-capacity treatment stops at B(X)>0.
    Ties and allocation costs are absent from this continuous Gaussian DGP.
    """
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, (float, int))
        or not isfinite(capacity)
        or not 0.0 <= capacity <= 1.0
    ):
        raise ValueError("capacity must be a finite real number in [0, 1]")

    q: float = float(capacity)
    # If treating is optional, nobody with B(X)<0 should be assigned treatment.
    positive_share: float = 1.0 - NORMAL.cdf(BENEFIT_ZERO_X)
    optional_share: float = min(q, positive_share)
    exact_gain: float = _top_share_population_gain(q)
    optional_gain: float = _top_share_population_gain(optional_share)
    harmful_share: float = max(q - positive_share, 0.0)
    return CapacityFrontier(
        capacity=q,
        positive_benefit_share=positive_share,
        exact_treated_share=q,
        optional_treated_share=optional_share,
        exact_population_gain=exact_gain,
        optional_population_gain=optional_gain,
        forced_negative_benefit_share=harmful_share,
        avoidable_population_loss=max(optional_gain - exact_gain, 0.0),
    )


def main() -> None:
    """Print deterministic, analytically derived capacity scenarios."""
    print("capacity,exact_gain,at_most_gain,at_most_share,avoidable_loss")
    for capacity in (0.0, 0.2, 0.5, 0.65, 0.8, 1.0):
        result: CapacityFrontier = population_capacity_frontier(capacity)
        print(
            f"{result.capacity:.2f},{result.exact_population_gain:.6f},"
            f"{result.optional_population_gain:.6f},"
            f"{result.optional_treated_share:.6f},"
            f"{result.avoidable_population_loss:.6f}"
        )


if __name__ == "__main__":
    main()
