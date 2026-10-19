"""Robustness of treatment targeting under nonlinear CATE and deployment shift.

Python >=3.12, NumPy. Run: python validate_policy.py
Produces validation_results.csv. All policy values are evaluated using held-out
structural treatment benefits; no outcome labels from the test set train models.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

SEED: int = 20261019
REPS: int = 500
N_TRAIN: int = 1000
N_TEST: int = 2000
CAPACITY: float = 0.2
SCENARIOS: tuple[str, ...] = ("nonlinear", "shifted", "small_noisy_trial")

def simulate(rng: np.random.Generator, scenario: str) -> dict[str, float]:
    """Compare linear and nonlinear CATE-based allocation on held-out people."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    n = 300 if scenario == "small_noisy_trial" else N_TRAIN
    x = rng.normal(size=n)
    a = rng.binomial(1, .5, size=n).astype(float)
    benefit = .5 + 1.5*x - .8*(x*x-1)
    noise_sd = 2.0 if scenario == "small_noisy_trial" else 1.0
    y = 3.0 - 1.2*x - a*benefit + rng.normal(scale=noise_sd, size=n)
    xt = rng.normal(loc=1.2 if scenario == "shifted" else 0., size=N_TEST)
    true_benefit = .5 + 1.5*xt - .8*(xt*xt-1)
    risk = 3.0 - 1.2*xt
    z = np.column_stack((np.ones(n), x, x*x-1))
    zt = np.column_stack((np.ones(N_TEST), xt, xt*xt-1))
    ranking: dict[str, np.ndarray] = {
        "random": rng.random(N_TEST),
        "baseline_risk": risk,
        "oracle": true_benefit,
    }
    for name, columns in (("linear_cate", 2), ("quadratic_cate", 3)):
        it = z[:, :columns]
        coefficients = np.linalg.lstsq(np.column_stack((z, a[:, None]*it)), y, rcond=None)[0]
        ranking[name] = -(zt[:, :columns] @ coefficients[z.shape[1]:])
    k = int(round(CAPACITY*N_TEST))
    return {
        name: float(true_benefit[np.argsort(score, kind="stable")[-k:]].mean())
        for name, score in ranking.items()
    }

def main() -> None:
    """Report policy value and regret in three predeclared stress-test regimes."""
    rng = np.random.default_rng(SEED)
    results: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        observations = [simulate(rng, scenario) for _ in range(REPS)]
        oracle = np.array([obs["oracle"] for obs in observations])
        for policy in observations[0]:
            values = np.array([obs[policy] for obs in observations])
            results.append({
                "scenario": scenario, "policy": policy,
                "replications": REPS, "mean_benefit": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "mean_regret": float(np.mean(oracle-values)),
                "mean_population_gain": float(CAPACITY*values.mean()),
            })
    path = Path(__file__).with_name("validation_results.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for item in results:
        print(item)

if __name__ == "__main__":
    main()
