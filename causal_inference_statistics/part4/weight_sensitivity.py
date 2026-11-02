"""Supplementary propensity estimation and clipping sensitivity experiment.

Run ``python weight_sensitivity.py`` in this directory to regenerate the
standalone output without altering Part 4's original results.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from experiment import N, REPETITIONS, SEED, STRENGTHS, TRUE_ATE, Result, simulate

PROPENSITY_SOURCES: tuple[Literal["fitted", "oracle"], ...] = ("fitted", "oracle")
CLIPPING_THRESHOLDS: tuple[float, ...] = (0.01, 0.05, 0.10)


def summarize(results: list[Result]) -> dict[str, float]:
    """Describe estimation error and practical overlap across replications."""
    if not results:
        raise ValueError("At least one replicate is required")
    raw = np.asarray([r.hajek for r in results], dtype=float)
    clipped = np.asarray([r.clipped for r in results], dtype=float)
    fractions = np.asarray([r.clipped_fraction for r in results], dtype=float)
    ess_treated = np.asarray([r.ess_treated for r in results], dtype=float)
    ess_control = np.asarray([r.ess_control for r in results], dtype=float)
    if any(not np.isfinite(v).all() for v in (raw, clipped, fractions, ess_treated, ess_control)):
        raise ValueError("Non-finite simulation results")
    return {
        "hajek_bias": round(float(np.mean(raw - TRUE_ATE)), 4),
        "hajek_rmse": round(float(np.sqrt(np.mean((raw - TRUE_ATE) ** 2))), 4),
        "clipped_bias": round(float(np.mean(clipped - TRUE_ATE)), 4),
        "clipped_rmse": round(float(np.sqrt(np.mean((clipped - TRUE_ATE) ** 2))), 4),
        "mean_clipped_fraction": round(float(np.mean(fractions)), 4),
        "mean_ess_treated_unclipped": round(float(np.mean(ess_treated)), 2),
        "mean_ess_control_unclipped": round(float(np.mean(ess_control)), 2),
    }


def main() -> None:
    """Run paired-seed scenarios and record reproducible sensitivity summaries."""
    rows: list[dict[str, str | int | float]] = []
    for strength in STRENGTHS:
        for source in PROPENSITY_SOURCES:
            for threshold in CLIPPING_THRESHOLDS:
                # Reset per scenario: comparisons use identical generated data.
                rng = np.random.default_rng(SEED + int(strength * 100))
                results = [
                    simulate(
                        rng,
                        strength,
                        propensity_source=source,
                        clipping_threshold=threshold,
                    )
                    for _ in range(REPETITIONS)
                ]
                rows.append(
                    {
                        "strength": strength,
                        "propensity_source": source,
                        "clipping_threshold": threshold,
                        "n": N,
                        "replications": REPETITIONS,
                        **summarize(results),
                    }
                )
    output = Path(__file__).with_name("weight_sensitivity.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sensitivity scenarios to {output}")


if __name__ == "__main__":
    main()
