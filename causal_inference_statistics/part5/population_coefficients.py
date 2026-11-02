"""Independently verify Part 5 population OLS targets against Monte Carlo output.

Run from the Part 5 directory. The four independent Gaussian shocks have
unit variance. Population covariances are therefore dot products of structural
loading vectors; no simulated regression or fitted model is required.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeAlias

Loading: TypeAlias = tuple[float, float, float, float]

# Columns correspond to independent structural disturbances (a, u, ey, ez).
# For each scenario: (treatment, outcome, adjustment variable, total effect).
SCENARIOS: dict[str, tuple[Loading, Loading, Loading, float]] = {
    "collider": ((1, 0, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1), 1.0),
    "confounder": ((1, 1, 0, 0), (1, 2, 1, 0), (0, 1, 0, 0), 1.0),
    "mediator": ((1, 0, 0, 0), (2, 0, 1, 1), (1, 0, 0, 1), 2.0),
    "irrelevant": ((1, 0, 0, 0), (1, 0, 1, 0), (0, 0, 0, 1), 1.0),
}


def covariance(left: Loading, right: Loading) -> float:
    """Covariance of linear forms of independent unit-variance shocks."""
    if len(left) != 4 or len(right) != 4:
        raise ValueError("Expected four independent-shock loadings")
    return float(sum(x * y for x, y in zip(left, right, strict=True)))


def population_slope(treatment: Loading, outcome: Loading, adjustment: Loading | None) -> float:
    """Return unadjusted or partial-regression population treatment slope."""
    var_t = covariance(treatment, treatment)
    cov_ty = covariance(treatment, outcome)
    if adjustment is None:
        if var_t <= 0:
            raise ValueError("Treatment variance must be positive")
        return cov_ty / var_t
    var_z = covariance(adjustment, adjustment)
    cov_tz = covariance(treatment, adjustment)
    determinant = var_t * var_z - cov_tz**2
    if determinant <= 0:
        raise ValueError("Treatment and adjustment variable are collinear")
    return (cov_ty * var_z - covariance(adjustment, outcome) * cov_tz) / determinant


def main() -> None:
    """Check each structural coefficient against the n=5000 Monte Carlo means."""
    root = Path(__file__).resolve().parent
    with (root / "results.csv").open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))
    rows: list[dict[str, str | float | int]] = []
    for scenario, (t, y, z, total_effect) in SCENARIOS.items():
        for model in ("unadjusted", "adjusted"):
            target = population_slope(t, y, None if model == "unadjusted" else z)
            matches = [
                record for record in observations
                if record["scenario"] == scenario and record["model"] == model and record["n"] == "5000"
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one n=5000 result for {scenario}/{model}")
            reported_total = float(matches[0]["target_total_effect"])
            if abs(reported_total - total_effect) > 1e-12:
                raise AssertionError("Manuscript total-effect target disagrees with SEM")
            empirical = float(matches[0]["mean"])
            discrepancy = abs(empirical - target)
            if discrepancy > 0.02:
                raise AssertionError(f"Population discrepancy for {scenario}/{model}: {discrepancy}")
            rows.append({
                "scenario": scenario,
                "model": model,
                "n": 5000,
                "causal_total_effect": total_effect,
                "population_ols_coefficient": target,
                "monte_carlo_mean": round(empirical, 6),
                "absolute_discrepancy": round(discrepancy, 6),
            })
    destination = root / "population_coefficients.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} population regression coefficients")


if __name__ == "__main__":
    main()
