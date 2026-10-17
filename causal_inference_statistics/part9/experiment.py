"""Monte Carlo study of synthetic controls with stable, shifted and anticipatory outcomes.

Python >=3.12; requires numpy and scipy. Run this file to write results.csv.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

SEED: int = 20261017
REPS: int = 300
N_DONORS: int = 8
PRE: int = 24
POST: int = 12
EFFECT: float = -2.0
SCENARIOS: tuple[str, ...] = ("stable", "latent_post_shift", "anticipation")


def synthetic_weights(donors: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """Minimize pre-period mean squared error with simplex-constrained weights."""
    if donors.ndim != 2 or treated.ndim != 1 or donors.shape[0] != treated.size:
        raise ValueError("Invalid donor or treated-unit input shape")
    n = donors.shape[1]
    result = minimize(
        lambda w: np.mean((treated - donors @ w) ** 2),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-11, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"Synthetic control optimizer failed: {result.message}")
    return np.asarray(result.x)


def simulate(rng: np.random.Generator, scenario: str) -> tuple[float, float, float, float]:
    """Generate one factor-model panel and estimate a known treatment effect."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    times = np.arange(PRE + POST)
    common = 0.04 * times + 0.4 * np.sin(times / 4)
    factor = np.cumsum(rng.normal(scale=0.20, size=times.size))
    loadings = rng.uniform(-1.5, 1.5, size=N_DONORS)
    intercepts = rng.normal(scale=1, size=N_DONORS)
    donors = (
        common[:, None] + intercepts[None, :] + factor[:, None] * loadings[None, :]
        + rng.normal(scale=0.28, size=(times.size, N_DONORS))
    )
    true_weights = np.zeros(N_DONORS)
    picked = rng.choice(N_DONORS, size=3, replace=False)
    true_weights[picked] = rng.dirichlet(np.ones(3))
    potential = donors @ true_weights + rng.normal(scale=0.20, size=times.size)
    observed = potential.copy()
    observed[PRE:] += EFFECT
    if scenario == "latent_post_shift":
        observed[PRE:] += 1.5  # Treated-specific untreated post-policy shock
    elif scenario == "anticipation":
        observed[PRE-4:PRE] -= 0.8  # Training period no longer untreated
    weights = synthetic_weights(donors[:PRE], observed[:PRE])
    fitted = donors @ weights
    effect_sc = float(np.mean(observed[PRE:] - fitted[PRE:]))
    pre_rmspe = float(np.sqrt(np.mean((observed[:PRE] - fitted[:PRE]) ** 2)))
    before_after = float(np.mean(observed[PRE:]) - np.mean(observed[PRE-8:PRE]))
    donor_average = donors.mean(axis=1)
    did = float(
        before_after - (np.mean(donor_average[PRE:]) - np.mean(donor_average[PRE-8:PRE]))
    )
    return effect_sc, pre_rmspe, before_after, did


def main() -> None:
    """Aggregate 300 independent Monte Carlo panels for each scenario."""
    rng = np.random.default_rng(SEED)
    records: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        samples = np.asarray([simulate(rng, scenario) for _ in range(REPS)])
        sc, pre_rmspe, naive, did = samples.T
        for method, estimates in (
            ("synthetic_control", sc), ("before_after", naive), ("equal_donor_did", did)
        ):
            records.append({
                "scenario": scenario, "method": method, "replications": REPS,
                "true_effect": EFFECT, "mean": float(estimates.mean()),
                "median": float(np.median(estimates)),
                "bias": float(estimates.mean() - EFFECT),
                "sd": float(estimates.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean((estimates - EFFECT) ** 2))),
                "mean_pre_rmspe": float(pre_rmspe.mean()),
            })
    output = Path(__file__).with_name("results.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
