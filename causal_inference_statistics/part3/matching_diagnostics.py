"""Supplementary Monte Carlo balance and reuse diagnostics for Part 3.

Run from this directory: ``python matching_diagnostics.py``. The original
``results.csv`` and its main estimands remain unchanged.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment import N, REPETITIONS, SEED, Estimate, simulate

METRICS: tuple[str, ...] = (
    "x_pre_match_smd",
    "x_absolute_smd",
    "matched_fraction",
    "distinct_control_fraction",
    "maximum_control_reuse",
)


def summarize(replicates: list[Estimate]) -> list[dict[str, str | float]]:
    """Return interpretable distribution summaries for matching diagnostics."""
    if not replicates:
        raise ValueError("At least one simulation replicate is required")

    rows: list[dict[str, str | float]] = []
    for metric in METRICS:
        values = np.asarray([getattr(item, metric) for item in replicates], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values for {metric}")
        rows.append(
            {
                "metric": metric,
                "mean": round(float(np.mean(values)), 4),
                "median": round(float(np.median(values)), 4),
                "p95": round(float(np.quantile(values, 0.95)), 4),
            }
        )
    return rows


def main() -> None:
    """Reproduce each scenario and write a standalone diagnostic CSV."""
    rows: list[dict[str, str | float | bool | int]] = []
    for hidden in (False, True):
        # Use the same independent per-scenario seed as the main experiment.
        rng = np.random.default_rng(SEED)
        replicates = [simulate(rng, hidden=hidden, n=N) for _ in range(REPETITIONS)]
        for summary in summarize(replicates):
            rows.append(
                {
                    "hidden_confounding": hidden,
                    "n": N,
                    "replications": REPETITIONS,
                    **summary,
                }
            )

    destination = Path(__file__).with_name("matching_diagnostics.csv")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} diagnostic summaries to {destination}")


if __name__ == "__main__":
    main()
