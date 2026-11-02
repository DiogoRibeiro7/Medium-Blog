"""Check HC2 against Neyman's difference-in-means variance under randomization.

This supplementary experiment preserves Part 2's original results.csv.
For an intercept and binary treatment, unadjusted HC2 and the usual Neyman
variance estimate are algebraically identical: s1²/n1 + s0²/n0.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from experiment import REPETITIONS, SAMPLE_SIZES, SEED, TAU, fit


def neyman_standard_error(y: NDArray[np.float64], a: NDArray[np.float64]) -> float:
    """Estimate standard error of the randomized difference in means.

    Uses unbiased within-arm sample variances. With constant individual
    treatment effects, the otherwise unobservable treatment-effect variance
    term in the finite-population formula is exactly zero.
    """
    if y.ndim != 1 or y.shape != a.shape or not np.isfinite(y).all():
        raise ValueError("Expected equal-length finite one-dimensional arrays")
    if not np.isin(a, [0.0, 1.0]).all():
        raise ValueError("Treatment must be binary")
    treated = y[a == 1.0]
    control = y[a == 0.0]
    if min(treated.size, control.size) < 2:
        raise ValueError("At least two observations per treatment arm required")
    variance = np.var(treated, ddof=1) / treated.size + np.var(control, ddof=1) / control.size
    return float(np.sqrt(variance))


def simulate(n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """One complete-randomization replicate using the Part 2 structural DGP."""
    if not isinstance(n, int) or n < 4 or n % 2:
        raise ValueError("n must be an even integer >= 4")
    u = rng.normal(size=n)
    error = rng.normal(size=n)
    a = np.zeros(n, dtype=float)
    a[rng.permutation(n)[: n // 2]] = 1.0
    y = TAU * a + 2.0 * u + error
    coefficient, hc2_se = fit(y, a)
    neyman_se = neyman_standard_error(y, a)
    return coefficient, hc2_se, neyman_se


def main() -> None:
    """Summarize Monte Carlo agreement, nominal coverage, and uncertainty."""
    rng = np.random.default_rng(SEED + 20_002)
    rows: list[dict[str, float | int]] = []
    for n in SAMPLE_SIZES:
        data = np.asarray([simulate(n, rng) for _ in range(REPETITIONS)], dtype=float)
        estimate, hc2, neyman = data.T
        absolute_error = np.abs(hc2 - neyman)
        rows.append(
            {
                "n": n,
                "replications": REPETITIONS,
                "mean_estimate": round(float(np.mean(estimate)), 5),
                "mean_hc2_se": round(float(np.mean(hc2)), 5),
                "mean_neyman_se": round(float(np.mean(neyman)), 5),
                "maximum_se_difference": float(np.max(absolute_error)),
                "hc2_coverage": round(float(np.mean(np.abs(estimate - TAU) <= 1.96 * hc2)), 4),
                "neyman_coverage": round(float(np.mean(np.abs(estimate - TAU) <= 1.96 * neyman)), 4),
                "monte_carlo_sd": round(float(np.std(estimate, ddof=1)), 5),
            }
        )
    destination = Path(__file__).with_name("randomization_uncertainty.csv")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["maximum_se_difference"] > 1e-10:
            raise AssertionError("HC2 and Neyman standard errors disagree")
    print(f"Wrote {len(rows)} sample-size summaries to {destination}")


if __name__ == "__main__":
    main()
