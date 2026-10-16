"""Monte Carlo experiment: confounders, colliders, mediators and irrelevant controls.

Requires Python 3.12+ and NumPy. Writes results.csv alongside this file.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

SEED: int = 20261016
REPETITIONS: int = 1000
SAMPLE_SIZES: tuple[int, ...] = (200, 1000, 5000)
TRUE_EFFECT: float = 1.0

def slope(y: NDArray[np.float64], a: NDArray[np.float64], z: NDArray[np.float64] | None = None) -> float:
    """Fit OLS with intercept; return the treatment coefficient."""
    if y.ndim != 1 or a.shape != y.shape or (z is not None and z.shape != y.shape):
        raise ValueError("Inputs must be equally sized vectors")
    if not np.isfinite(y).all() or not np.isfinite(a).all() or (z is not None and not np.isfinite(z).all()):
        raise ValueError("Inputs must be finite")
    design = np.column_stack((np.ones(y.size), a)) if z is None else np.column_stack((np.ones(y.size), a, z))
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])

def run() -> list[dict[str, str | int | float]]:
    """Evaluate four causal structures at three sample sizes."""
    rng = np.random.default_rng(SEED)
    output: list[dict[str, str | int | float]] = []
    for n in SAMPLE_SIZES:
        estimates: dict[str, list[float]] = {
            f"{scenario}:{adjustment}": []
            for scenario in ("collider", "confounder", "mediator", "irrelevant")
            for adjustment in ("unadjusted", "adjusted")
        }
        for _ in range(REPETITIONS):
            a, u, e_y, e_z = rng.normal(size=(4, n))
            # Collider: A -> C <- U -> Y, alongside A -> Y.
            y_c = TRUE_EFFECT * a + u + e_y
            c = a + u + e_z
            # Confounder: U -> T and U -> Y.
            t = u + a
            y_f = TRUE_EFFECT * t + u + e_y
            # Mediator: A -> M -> Y, with an additional A -> Y path.
            m = a + e_z
            y_m = a + m + e_y
            # Irrelevant control: Z independent of treatment and outcome.
            v = e_z
            y_i = TRUE_EFFECT * a + e_y
            cases = {
                "collider": (a, y_c, c),
                "confounder": (t, y_f, u),
                "mediator": (a, y_m, m),
                "irrelevant": (a, y_i, v),
            }
            for name, (treatment, outcome, control) in cases.items():
                estimates[f"{name}:unadjusted"].append(slope(outcome, treatment))
                estimates[f"{name}:adjusted"].append(slope(outcome, treatment, control))
        for name, values in estimates.items():
            scenario, adjustment = name.split(":")
            target = 2.0 if scenario == "mediator" else TRUE_EFFECT
            arr = np.asarray(values)
            output.append({
                "n": n, "scenario": scenario, "model": adjustment,
                "target_total_effect": target, "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)),
                "bias": float(arr.mean() - target),
                "rmse": float(np.sqrt(np.mean((arr - target) ** 2))),
            })
    return output

if __name__ == "__main__":
    rows = run()
    output = Path(__file__).with_name("results.csv")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['n']:>5} {row['scenario']:<11} {row['model']:<10} "
              f"target={row['target_total_effect']:.1f} mean={row['mean']:.4f} RMSE={row['rmse']:.4f}")
