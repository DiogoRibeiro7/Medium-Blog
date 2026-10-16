"""Regression discontinuity Monte Carlo, Python >=3.12, NumPy only."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEED: int = 20261016
REPS: int = 400
N: int = 2000
TAU: float = 1.0
BANDWIDTHS: tuple[float, ...] = (0.25, 0.5, 1.0)
SCENARIOS: tuple[str, ...] = ("valid", "curved", "sorting")


def local_linear(r: np.ndarray, y: np.ndarray, h: float) -> tuple[float, int]:
    """Estimate the jump at zero using side-specific triangular-kernel local lines."""
    if h <= 0 or r.shape != y.shape or r.ndim != 1:
        raise ValueError("Invalid inputs or bandwidth")
    intercepts: list[float] = []
    retained = 0
    for right in (False, True):
        keep = ((r >= 0) if right else (r < 0)) & (np.abs(r) < h)
        x, values = r[keep], y[keep]
        if x.size < 5:
            raise ValueError("Too few observations in a cutoff neighbourhood")
        weights = 1 - np.abs(x) / h
        design = np.column_stack((np.ones(x.size), x))
        coefficients = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None
        )[0]
        intercepts.append(float(coefficients[0]))
        retained += x.size
    return intercepts[1] - intercepts[0], retained


def draw(rng: np.random.Generator, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    """Construct running variable and outcome for one causal assignment mechanism."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    latent = rng.uniform(-1, 1, N)
    running = latent.copy()
    if scenario == "sorting":
        preference = rng.normal(size=N)
        movers = (latent < 0) & (latent > -0.25) & (preference > 0)
        running[movers] = -latent[movers]
        outcome = TAU * (running >= 0) + 0.6 * running + preference + rng.normal(size=N)
    else:
        outcome = TAU * (running >= 0) + 0.6 * running + rng.normal(size=N)
        if scenario == "curved":
            outcome += 2.0 * (running >= 0) * running**2
            outcome -= 2.0 * (running < 0) * running**2
    return running, outcome


def run() -> list[dict[str, str | int | float]]:
    """Summarize estimate means, biases, dispersion, RMSE and retained observations."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        replications: dict[float, list[tuple[float, int]]] = {h: [] for h in BANDWIDTHS}
        for _ in range(REPS):
            running, outcome = draw(rng, scenario)
            for bandwidth in BANDWIDTHS:
                replications[bandwidth].append(local_linear(running, outcome, bandwidth))
        for bandwidth, result in replications.items():
            array = np.asarray(result)
            effects = array[:, 0]
            rows.append({
                "scenario": scenario, "bandwidth": bandwidth, "n": N, "reps": REPS,
                "mean": float(effects.mean()),
                "bias": float(effects.mean() - TAU),
                "sd": float(effects.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean((effects - TAU) ** 2))),
                "mean_retained": float(array[:, 1].mean()),
            })
    return rows


if __name__ == "__main__":
    records = run()
    with Path(__file__).with_name("results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)
