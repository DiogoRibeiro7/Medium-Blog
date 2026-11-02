"""Distinguish associational and causal confidence-interval coverage in Part 1.

Run ``python coverage_targets.py`` from this directory to generate a separate
audit CSV. No original experiment outputs are modified.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment import REPETITIONS, SAMPLE_SIZES, SEED, TRUE_EFFECT, estimate

METHODS: tuple[str, ...] = ("naive", "adjusted", "randomized")
ASSOCIATIONAL_TARGETS: dict[str, float] = {
    "naive": 2.0,
    "adjusted": TRUE_EFFECT,
    "randomized": TRUE_EFFECT,
}


def main() -> None:
    """Record interval coverage relative to two conceptually distinct targets."""
    rng = np.random.default_rng(SEED + 10_001)
    rows: list[dict[str, str | int | float]] = []

    for n in SAMPLE_SIZES:
        observations: dict[str, list[tuple[float, float]]] = {name: [] for name in METHODS}
        for _ in range(REPETITIONS):
            u, ex, ey = rng.normal(size=(3, n))
            x = u + ex
            y = TRUE_EFFECT * x + 2.0 * u + ey
            randomized_x = rng.normal(size=n)
            randomized_y = TRUE_EFFECT * randomized_x + 2.0 * u + ey
            observations["naive"].append(estimate(y, x))
            observations["adjusted"].append(estimate(y, x, u))
            observations["randomized"].append(estimate(randomized_y, randomized_x))

        for method in METHODS:
            data = np.asarray(observations[method], dtype=float)
            coefficients, standard_errors = data.T
            if not np.isfinite(data).all() or np.any(standard_errors <= 0):
                raise ValueError(f"Invalid estimates for {method}, n={n}")
            target = ASSOCIATIONAL_TARGETS[method]
            causal_coverage = np.mean(np.abs(coefficients - TRUE_EFFECT) <= 1.96 * standard_errors)
            associational_coverage = np.mean(np.abs(coefficients - target) <= 1.96 * standard_errors)
            rows.append(
                {
                    "n": n,
                    "replications": REPETITIONS,
                    "method": method,
                    "causal_target": TRUE_EFFECT,
                    "population_regression_target": target,
                    "mean_estimate": round(float(np.mean(coefficients)), 5),
                    "mean_standard_error": round(float(np.mean(standard_errors)), 5),
                    "coverage_causal": round(float(causal_coverage), 4),
                    "coverage_regression": round(float(associational_coverage), 4),
                }
            )

    destination = Path(__file__).with_name("coverage_targets.csv")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} method/sample-size summaries to {destination}")


if __name__ == "__main__":
    main()
