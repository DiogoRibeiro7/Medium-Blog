"""Pre-treatment holdout diagnostics for Part 9 synthetic controls.

Python >=3.12; NumPy and SciPy required. Uses an independent deterministic
seed and the same structural design as experiment.py. The final four
pre-treatment periods are *not* used to fit donor weights.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from experiment import EFFECT, N_DONORS, POST, PRE, SCENARIOS, synthetic_weights

SEED: int = 20261103
REPLICATIONS: int = 300
HOLDOUT: int = 4


def generate_panel(
    rng: np.random.Generator, scenario: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Regenerate the published factor process with a separate random seed."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown synthetic-control scenario")
    times: NDArray[np.int64] = np.arange(PRE + POST)
    common: NDArray[np.float64] = 0.04 * times + 0.4 * np.sin(times / 4)
    factor: NDArray[np.float64] = np.cumsum(rng.normal(scale=0.20, size=times.size))
    loadings: NDArray[np.float64] = rng.uniform(-1.5, 1.5, size=N_DONORS)
    intercepts: NDArray[np.float64] = rng.normal(scale=1, size=N_DONORS)
    donors: NDArray[np.float64] = (
        common[:, None] + intercepts[None, :] + factor[:, None] * loadings[None, :]
        + rng.normal(scale=0.28, size=(times.size, N_DONORS))
    )
    true_weights: NDArray[np.float64] = np.zeros(N_DONORS)
    selected: NDArray[np.int64] = rng.choice(N_DONORS, size=3, replace=False)
    true_weights[selected] = rng.dirichlet(np.ones(3))
    observed: NDArray[np.float64] = (
        donors @ true_weights + rng.normal(scale=0.20, size=times.size)
    )
    observed[PRE:] += EFFECT
    if scenario == "latent_post_shift":
        observed[PRE:] += 1.5
    elif scenario == "anticipation":
        observed[PRE - HOLDOUT:PRE] -= 0.8
    return donors, observed


def holdout_gap(
    donors: NDArray[np.float64], treated: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Fit on early pre-periods; validate on the last four pre-periods."""
    if donors.shape != (PRE + POST, N_DONORS) or treated.shape != (PRE + POST,):
        raise ValueError("Unexpected panel shape")
    train_end: int = PRE - HOLDOUT
    weights: NDArray[np.float64] = synthetic_weights(
        donors[:train_end], treated[:train_end]
    )
    training_residual: NDArray[np.float64] = (
        treated[:train_end] - donors[:train_end] @ weights
    )
    held_residual: NDArray[np.float64] = (
        treated[train_end:PRE] - donors[train_end:PRE] @ weights
    )
    return (
        float(np.mean(held_residual)),
        float(np.sqrt(np.mean(held_residual**2))),
        float(np.sqrt(np.mean(training_residual**2))),
    )


def run() -> list[dict[str, str | int | float]]:
    """Summarize holdout residuals across each structural scenario."""
    rng: np.random.Generator = np.random.default_rng(SEED)
    results: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        replicates: NDArray[np.float64] = np.asarray([
            holdout_gap(*generate_panel(rng, scenario))
            for _ in range(REPLICATIONS)
        ])
        results.append({
            "scenario": scenario,
            "replications": REPLICATIONS,
            "training_periods": PRE - HOLDOUT,
            "holdout_periods": HOLDOUT,
            "mean_holdout_gap": float(np.mean(replicates[:, 0])),
            "mean_holdout_rmspe": float(np.mean(replicates[:, 1])),
            "mean_training_rmspe": float(np.mean(replicates[:, 2])),
            "fraction_holdout_gap_below_minus_0_4": float(np.mean(replicates[:, 0] < -0.4)),
        })
    return results


def main() -> None:
    """Write an independently reproducible CSV snapshot."""
    records = run()
    path = Path(__file__).with_name("sc_preperiod_holdout_results.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
