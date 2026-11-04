"""Oracle-propensity checks for Part 15's longitudinal IPW identification.

This independent simulation validates the weighted intermediate and outcome
means under sustained treatment histories. It does not replace the original
estimated-propensity Monte Carlo experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RegimeMoments:
    """Weighted estimates of moments under a sustained static regime."""

    observed_count: int
    weighted_baseline: float
    weighted_intermediate: float
    weighted_outcome: float


def expit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute logistic probabilities for the moderate simulated logits."""
    return 1.0 / (1.0 + np.exp(-values))


def oracle_regime_moments(*, n: int = 300_000, seed: int = 20261104) -> dict[int, RegimeMoments]:
    """Estimate E[L0(a)], E[L1(a)] and E[Y(a,a)] using true IP weights.

    Independent Bernoulli assignments use the known DGP probabilities. For
    each observed regime, normalized inverse weights reconstruct the baseline
    population and its intervention-induced intermediate distribution.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 100:
        raise ValueError("n must be an integer >= 100")
    rng: np.random.Generator = np.random.default_rng(seed)
    l0: NDArray[np.float64] = rng.standard_normal(n)
    p0: NDArray[np.float64] = expit(0.8 * l0)
    a0: NDArray[np.int64] = rng.binomial(1, p0)
    l1: NDArray[np.float64] = 0.8 * l0 + 1.2 * a0 + rng.standard_normal(n)
    p1: NDArray[np.float64] = expit(0.8 * l1)
    a1: NDArray[np.int64] = rng.binomial(1, p1)
    y: NDArray[np.float64] = 1.0 + 0.5 * l0 + 1.3 * l1 + 0.7 * a0 + 1.1 * a1 + rng.standard_normal(n)
    results: dict[int, RegimeMoments] = {}
    for regime in (0, 1):
        selected: NDArray[np.bool_] = (a0 == regime) & (a1 == regime)
        probability: NDArray[np.float64] = (
            (p0 if regime else 1.0 - p0)[selected]
            * (p1 if regime else 1.0 - p1)[selected]
        )
        weights: NDArray[np.float64] = 1.0 / probability
        results[regime] = RegimeMoments(
            observed_count=int(selected.sum()),
            weighted_baseline=float(np.average(l0[selected], weights=weights)),
            weighted_intermediate=float(np.average(l1[selected], weights=weights)),
            weighted_outcome=float(np.average(y[selected], weights=weights)),
        )
    return results


if __name__ == "__main__":
    for regime, moments in oracle_regime_moments().items():
        print(f"regime=({regime},{regime}) {moments}")
