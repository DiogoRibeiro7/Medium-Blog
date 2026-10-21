"""Simulation of matching, overlap, and hidden confounding.

Requires Python >=3.12, numpy, scipy, scikit-learn.
Run: python experiment.py
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

SEED: int = 20261016
N: int = 1500
REPETITIONS: int = 300
TRUE_ATT: float = 1.0
CALIPER: float = 0.2


@dataclass(frozen=True)
class Estimate:
    """One replication: outcome contrasts and matched-sample diagnostics."""
    naive: float
    matched: float
    matched_fraction: float
    x_difference: float
    u_difference: float


def simulate(rng: np.random.Generator, *, hidden: bool, n: int = N) -> Estimate:
    """Match treated subjects to nearest propensity-score controls with replacement.

    The matching distance is the fitted propensity-score logit. Reuse of control
    observations means naive independent-pair standard errors are inappropriate.
    """
    if n < 20 or not isinstance(hidden, bool):
        raise ValueError("n must be >= 20 and hidden must be boolean")
    x: NDArray[np.float64] = rng.standard_normal(n)
    u: NDArray[np.float64] = rng.standard_normal(n)
    treatment_prob = expit(0.9 * x + (1.2 * u if hidden else 0.0))
    treatment: NDArray[np.int_] = (rng.random(n) < treatment_prob).astype(int)
    outcome = TRUE_ATT * treatment + 1.5 * x + 1.5 * u + rng.standard_normal(n)
    if treatment.min() == treatment.max():
        raise ValueError("Both treatment groups must be present")

    propensity = LogisticRegression(C=1e6, max_iter=300)
    propensity.fit(x.reshape(-1, 1), treatment)
    logit = propensity.decision_function(x.reshape(-1, 1))
    treated = np.flatnonzero(treatment)
    controls = np.flatnonzero(1 - treatment)
    sorted_controls = controls[np.argsort(logit[controls])]
    sorted_scores = logit[sorted_controls]

    insertion = np.searchsorted(sorted_scores, logit[treated])
    lower = np.clip(insertion - 1, 0, len(sorted_controls) - 1)
    upper = np.clip(insertion, 0, len(sorted_controls) - 1)
    neighbor = np.where(
        np.abs(logit[treated] - sorted_scores[lower])
        <= np.abs(logit[treated] - sorted_scores[upper]),
        lower,
        upper,
    )
    distance = np.abs(logit[treated] - sorted_scores[neighbor])
    keep = distance <= CALIPER * np.std(logit)
    matched_treated = treated[keep]
    matched_control = sorted_controls[neighbor[keep]]
    if matched_treated.size == 0:
        raise ValueError("Caliper excluded every treated subject")

    return Estimate(
        naive=float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean()),
        matched=float((outcome[matched_treated] - outcome[matched_control]).mean()),
        matched_fraction=float(matched_treated.size / treated.size),
        x_difference=float(x[matched_treated].mean() - x[matched_control].mean()),
        u_difference=float(u[matched_treated].mean() - u[matched_control].mean()),
    )


def main() -> None:
    """Print Monte Carlo means and between-replicate standard deviations."""
    for hidden in (False, True):
        rng = np.random.default_rng(SEED)
        results = [simulate(rng, hidden=hidden) for _ in range(REPETITIONS)]
        array = np.array([
            [r.naive, r.matched, r.matched_fraction, r.x_difference, r.u_difference]
            for r in results
        ])
        print(f"hidden_confounding={hidden}, n={N}, replicates={REPETITIONS}")
        print("metric,mean,sd")
        for name, values in zip(
            ("naive", "matched", "fraction", "x_difference", "u_difference"),
            array.T,
            strict=True,
        ):
            print(f"{name},{values.mean():.4f},{values.std():.4f}")


if __name__ == "__main__":
    # Preserve the original grouped console report and also archive a tidy CSV.
    import csv
    import io
    from contextlib import redirect_stdout
    from pathlib import Path

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main()
    original = buffer.getvalue()
    rows: list[dict[str, str | int | float | bool]] = []
    scenario: bool | None = None
    for raw in original.splitlines():
        if raw.startswith("hidden_confounding="):
            scenario = raw.split(",")[0].split("=")[1] == "True"
        elif raw and raw != "metric,mean,sd":
            if scenario is None:
                raise ValueError("Result metric found before scenario header")
            name, mean, sd = raw.split(",")
            rows.append({
                "hidden_confounding": scenario,
                "n": N,
                "replications": REPETITIONS,
                "metric": name,
                "mean": float(mean),
                "sd": float(sd),
            })
    destination = Path(__file__).with_name("results.csv")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(original, end="")
