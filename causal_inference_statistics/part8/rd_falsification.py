"""Independent Part 8 falsification diagnostics under three known mechanisms.

The count-based density screen is deliberately *not* a McCrary/Cattaneo
density estimator. It is appropriate here because each DGP has a locally
flat running-variable density on both sides of zero. It is not a generic
density-discontinuity test for arbitrary real-world RD applications.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from experiment import local_linear

SEED: int = 20261103
REPLICATIONS: int = 400
SAMPLE_SIZE: int = 2000
BANDWIDTH: float = 0.2
CRITICAL_VALUE: float = 1.959963984540054
SCENARIOS: tuple[str, ...] = ("smooth_valid", "latent_prognosis_jump", "density_sorting")


def sample_design(
    rng: np.random.Generator, scenario: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate running scores and a pre-treatment diagnostic covariate.

    In density_sorting, rejection sampling gives double the score density
    just above the threshold, without modifying the covariate distribution.
    In latent_prognosis_jump, the covariate itself shifts at the threshold,
    while score density remains uniform. This is a *measured proxy* for an
    underlying selection discontinuity, not a randomized treatment effect.
    """
    if scenario not in SCENARIOS:
        raise ValueError("Unknown falsification scenario")
    accepted: list[NDArray[np.float64]] = []
    count: int = 0
    while count < SAMPLE_SIZE:
        candidates = rng.uniform(-1.0, 1.0, SAMPLE_SIZE)
        if scenario == "density_sorting":
            # Selection probability is 1 to the right and 0.5 to the left.
            keep = (candidates >= 0) | (rng.uniform(size=SAMPLE_SIZE) < 0.5)
            candidates = candidates[keep]
        accepted.append(candidates)
        count += candidates.size
    x: NDArray[np.float64] = np.concatenate(accepted)[:SAMPLE_SIZE]
    baseline: NDArray[np.float64] = rng.normal(size=SAMPLE_SIZE)
    if scenario == "latent_prognosis_jump":
        baseline = baseline + 0.7 * (x >= 0)
    return x, baseline


def density_screen(x: NDArray[np.float64], bandwidth: float) -> float:
    """Compute the equal-bin count z-statistic at the cutoff.

    Conditional on the number of observations in symmetric bins, left
    counts are Binomial(total, 1/2) under locally flat equal densities.
    """
    left: int = int(np.count_nonzero((-bandwidth < x) & (x < 0)))
    right: int = int(np.count_nonzero((0 <= x) & (x < bandwidth)))
    if left + right == 0:
        raise ValueError("No observations in the symmetric cutoff bins")
    return (right - left) / np.sqrt(left + right)


def run() -> list[dict[str, str | int | float]]:
    """Estimate false-positive and detection frequencies across mechanisms."""
    rng: np.random.Generator = np.random.default_rng(SEED)
    rows: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        density_rejections: list[bool] = []
        covariate_rejections: list[bool] = []
        covariate_jumps: list[float] = []
        for _ in range(REPLICATIONS):
            x, baseline = sample_design(rng, scenario)
            density_rejections.append(abs(density_screen(x, BANDWIDTH)) > CRITICAL_VALUE)
            jump, se, _ = local_linear(x, baseline, BANDWIDTH)
            covariate_jumps.append(jump)
            covariate_rejections.append(abs(jump / se) > CRITICAL_VALUE)
        rows.append(
            {
                "scenario": scenario,
                "n": SAMPLE_SIZE,
                "replications": REPLICATIONS,
                "bandwidth": BANDWIDTH,
                "density_screen_rejection_fraction": float(np.mean(density_rejections)),
                "covariate_screen_rejection_fraction": float(np.mean(covariate_rejections)),
                "mean_predetermined_covariate_jump": float(np.mean(covariate_jumps)),
            }
        )
    return rows


def main() -> None:
    """Write deterministic diagnostic results alongside the experiment."""
    records = run()
    output = Path(__file__).with_name("rd_falsification_results.csv")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
