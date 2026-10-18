"""Monte Carlo sequential g-computation in a two-decision causal system.

Requires Python >=3.12, numpy and scikit-learn.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

SEED: int = 20261017
REPS: int = 300
N: int = 2000

def simulate(rng: np.random.Generator, n: int) -> tuple[NDArray[np.float64], ...]:
    """Generate observational data with treatment-confounder feedback."""
    if not isinstance(n, int) or n < 20:
        raise ValueError("n must be an integer >= 20")
    l0 = rng.normal(size=n)
    a0 = rng.binomial(1, 1 / (1 + np.exp(-0.8 * l0))).astype(float)
    l1 = 0.8 * l0 + 1.2 * a0 + rng.normal(size=n)
    a1 = rng.binomial(1, 1 / (1 + np.exp(-0.8 * l1))).astype(float)
    y = 1 + 0.5 * l0 + 1.3 * l1 + 0.7 * a0 + 1.1 * a1 + rng.normal(size=n)
    return l0, a0, l1, a1, y

def g_formula(data: tuple[NDArray[np.float64], ...], a0: float, a1: float, *, include_l1: bool = True) -> float:
    """Predict the intervention mean, integrating linear intermediate evolution."""
    l0, observed_a0, l1, observed_a1, y = data
    l_model = LinearRegression().fit(np.column_stack([l0, observed_a0]), l1)
    x = np.column_stack([l0, observed_a0, l1, observed_a1]) if include_l1 else np.column_stack([l0, observed_a0, observed_a1])
    y_model = LinearRegression().fit(x, y)
    new_l1 = l_model.predict(np.column_stack([l0, np.full(l0.size, a0)]))
    new_x = np.column_stack([l0, np.full(l0.size, a0), new_l1, np.full(l0.size, a1)]) if include_l1 else np.column_stack([l0, np.full(l0.size, a0), np.full(l0.size, a1)])
    return float(np.mean(y_model.predict(new_x)))

def simple_adjustment(data: tuple[NDArray[np.float64], ...]) -> float:
    """Return direct conditional coefficient sum, not the total regime effect."""
    l0, a0, l1, a1, y = data
    model = LinearRegression().fit(np.column_stack([l0, a0, l1, a1]), y)
    return float(model.coef_[1] + model.coef_[3])

def main() -> None:
    """Compute Monte Carlo estimate, bias, dispersion and RMSE."""
    rng = np.random.default_rng(SEED)
    records: dict[str, list[float]] = {key: [] for key in ("g_correct", "g_omit_l1", "naive_adjustment")}
    for _ in range(REPS):
        data = simulate(rng, N)
        for key, include in (("g_correct", True), ("g_omit_l1", False)):
            records[key].append(g_formula(data, 1., 1., include_l1=include) - g_formula(data, 0., 0., include_l1=include))
        records["naive_adjustment"].append(simple_adjustment(data))
    truth = 0.7 + 1.1 + 1.3 * 1.2
    rows: list[dict[str, str | int | float]] = []
    for name, estimates in records.items():
        v = np.asarray(estimates)
        rows.append({"method": name, "n": N, "replications": REPS, "true_effect": truth,
                     "mean": float(v.mean()), "bias": float(v.mean() - truth),
                     "sd": float(v.std(ddof=1)), "rmse": float(np.sqrt(np.mean((v - truth) ** 2))),
                     "q05": float(np.quantile(v, .05)), "q95": float(np.quantile(v, .95))})
    with Path(__file__).with_name("results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)

if __name__ == "__main__":
    main()
