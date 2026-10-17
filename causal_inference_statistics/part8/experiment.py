"""Regression discontinuity Monte Carlo: curvature, bandwidth, and sorting.

Python >=3.12. Requires NumPy only. Outcomes and all reported results are synthetic.
Run: python experiment.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SEED: int = 20261017
REPLICATIONS: int = 400
SAMPLE_SIZE: int = 2000
TRUE_EFFECT: float = -1.0
BANDWIDTHS: tuple[float, ...] = (0.15, 0.30, 0.60, 1.00)
SCENARIOS: tuple[str, ...] = ("smooth", "curvature", "sorting")


def local_linear(x: NDArray[np.float64], y: NDArray[np.float64], bandwidth: float) -> tuple[float, float, int]:
    """Estimate sharp RD using separate triangular-weighted local linear fits.

    Returns the intercept discontinuity, a conditional HC1 sandwich standard
    error (bandwidth treated as fixed), and the number inside the window.
    """
    if x.ndim != 1 or y.shape != x.shape or not (0 < bandwidth <= 1):
        raise ValueError("x and y must be equal-length vectors and bandwidth in (0,1]")
    indices = (np.abs(x) < bandwidth)
    a = x[indices]
    b = y[indices]
    treated = (a >= 0).astype(float)
    if min(np.sum(treated), np.sum(1 - treated)) < 12:
        raise ValueError("At least 12 units on each side are required")
    design = np.column_stack((np.ones(a.size), a, treated, treated * a))
    weights = 1.0 - np.abs(a) / bandwidth
    root_w = np.sqrt(weights)
    beta = np.linalg.lstsq(design * root_w[:, None], b * root_w, rcond=None)[0]
    residuals = b - design @ beta
    bread = np.linalg.inv(design.T @ (weights[:, None] * design))
    # HC1 is a conventional fixed-bandwidth approximation, not RD robust-bias correction.
    meat = design.T @ (((weights * residuals) ** 2)[:, None] * design)
    covariance = (a.size / (a.size - design.shape[1])) * bread @ meat @ bread
    return float(beta[2]), float(np.sqrt(covariance[2, 2])), int(a.size)


def simulate(rng: np.random.Generator, scenario: str) -> dict[float, tuple[float, float, int]]:
    """Generate one independent study under a known cutoff treatment effect."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    x = rng.uniform(-1.0, 1.0, SAMPLE_SIZE)
    u = rng.normal(size=SAMPLE_SIZE)
    # Outcome is continuous at the cutoff absent intervention in valid scenarios.
    control = 1.0 + 0.6 * x + u
    if scenario == "curvature":
        control += 2.0 * x**2 * (x >= 0)  # smooth value, unequal curvature across sides
    if scenario == "sorting":
        # This is a local discontinuity in unobserved prognosis, NOT a causal effect.
        # Sorting is encoded as treatment-side selection in latent prognosis.
        control += 0.7 * (x >= 0)
    y = control + TRUE_EFFECT * (x >= 0)
    return {h: local_linear(x, y, h) for h in BANDWIDTHS}


def main() -> None:
    """Produce empirical bias, RMSE, coverage and sample-size summaries."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        estimates: dict[float, list[tuple[float, float, int]]] = {h: [] for h in BANDWIDTHS}
        for _ in range(REPLICATIONS):
            observations = simulate(rng, scenario)
            for h, result in observations.items():
                estimates[h].append(result)
        for h, data in estimates.items():
            a = np.asarray(data)
            coefficient, se, count = a.T
            rows.append({
                "scenario": scenario,
                "bandwidth": h,
                "replications": REPLICATIONS,
                "sample_size": SAMPLE_SIZE,
                "true_effect": TRUE_EFFECT,
                "mean_estimate": float(coefficient.mean()),
                "bias": float(coefficient.mean() - TRUE_EFFECT),
                "rmse": float(np.sqrt(np.mean((coefficient - TRUE_EFFECT) ** 2))),
                "mean_se": float(se.mean()),
                "coverage": float(np.mean(np.abs(coefficient - TRUE_EFFECT) <= 1.96 * se)),
                "mean_window_n": float(count.mean()),
            })
    output = Path(__file__).with_name("results.csv")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
