"""Monte Carlo study of IV relevance, exclusion and OLS endogeneity.

Python >=3.12; NumPy required. Runs 500 independent replicates per design.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

SEED: int = 20261016
REPLICATIONS: int = 500
SAMPLE_SIZE: int = 1000
TRUE_EFFECT: float = 1.0
SCENARIOS: dict[str, tuple[float, float]] = {
    "valid_strong": (1.0, 0.0),
    "valid_moderate": (0.3, 0.0),
    "valid_weak": (0.1, 0.0),
    "invalid_strong": (1.0, 0.3),
}


def run() -> list[dict[str, float | str | int]]:
    """Compare OLS and just-identified IV under four structural models."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str | int]] = []
    for name, (first_stage, direct_effect) in SCENARIOS.items():
        estimates: list[tuple[float, float, float]] = []
        for _ in range(REPLICATIONS):
            z, u, v, error = rng.standard_normal((4, SAMPLE_SIZE))
            treatment = first_stage * z + u + v
            outcome = TRUE_EFFECT * treatment + u + error + direct_effect * z
            cov_zy = np.cov(z, outcome, bias=True)[0, 1]
            cov_zx = np.cov(z, treatment, bias=True)[0, 1]
            ols = np.cov(treatment, outcome, bias=True)[0, 1] / np.var(treatment)
            iv = cov_zy / cov_zx
            r_squared = float(np.corrcoef(z, treatment)[0, 1] ** 2)
            f_stat = (SAMPLE_SIZE - 2) * r_squared / (1 - r_squared)
            estimates.append((float(ols), float(iv), float(f_stat)))
        array = np.asarray(estimates)
        for column, label in enumerate(("ols", "iv")):
            values = array[:, column]
            rows.append({
                "scenario": name,
                "method": label,
                "n": SAMPLE_SIZE,
                "replications": REPLICATIONS,
                "true_effect": TRUE_EFFECT,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "bias": float(values.mean() - TRUE_EFFECT),
                "rmse": float(np.sqrt(np.mean((values - TRUE_EFFECT) ** 2))),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
                "mean_first_stage_F": float(array[:, 2].mean()),
                "median_first_stage_F": float(np.median(array[:, 2])),
            })
    return rows


def main() -> None:
    """Write a stable CSV snapshot of the generated Monte Carlo metrics."""
    records = run()
    path = Path(__file__).with_name("results.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
