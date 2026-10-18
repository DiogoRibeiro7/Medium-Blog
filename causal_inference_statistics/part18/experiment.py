"""Monte Carlo analysis of complete-case selection and outcome-missingness corrections."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

SEED: int = 20261018
N: int = 2000
REPS: int = 300
TRUE_ATE: float = 1.0
SCENARIOS: tuple[str, ...] = ("mcar", "mar", "mnar")


def fit(y: np.ndarray, a: np.ndarray, x: np.ndarray) -> float:
    """Return treatment coefficient from linear regression adjusted for baseline x."""
    design = np.column_stack((np.ones(len(y)), a, x))
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def simulate(rng: np.random.Generator, scenario: str) -> tuple[float, float, float, float, float]:
    """Generate one randomized trial with full, conditional, or outcome-driven observation."""
    x = rng.normal(size=N)
    a = rng.binomial(1, .5, size=N)
    y = TRUE_ATE * a + 1.5 * x + rng.normal(size=N)
    if scenario == "mcar":
        p = np.full(N, .65)
    elif scenario == "mar":
        p = expit(.8 - 1.0 * x + .6 * a)
    elif scenario == "mnar":
        p = expit(.8 - 1.0 * x + .6 * a - .9 * y)
    else:
        raise ValueError("unknown scenario")
    observed = rng.random(N) < p
    full = fit(y, a, x)
    cc = fit(y[observed], a[observed], x[observed])
    # Observation model includes recorded A,X, but cannot include unobserved Y.
    model = LogisticRegression(C=1e6, max_iter=500).fit(np.column_stack((a, x)), observed)
    pred = np.clip(model.predict_proba(np.column_stack((a, x)))[:, 1], .01, .99)
    weights = 1 / pred[observed]
    design = np.column_stack((np.ones(observed.sum()), a[observed], x[observed]))
    root = np.sqrt(weights)
    ipw = float(np.linalg.lstsq(design * root[:, None], y[observed] * root, rcond=None)[0][1])
    # A correctly specified constant conditional effect makes OR equal to CC here.
    outcome_model = np.linalg.lstsq(design, y[observed], rcond=None)[0]
    outcome = float(outcome_model[1])
    return full, cc, ipw, outcome, float(observed.mean())


def main() -> None:
    """Persist Monte Carlo mean, bias, RMSE and observed fraction."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        arr = np.array([simulate(rng, scenario) for _ in range(REPS)])
        for j, name in enumerate(("full_data", "complete_case", "response_ipw", "outcome_regression")):
            values = arr[:, j]
            rows.append(dict(
                scenario=scenario, method=name, n=N, replications=REPS, true_effect=TRUE_ATE,
                mean=float(values.mean()), bias=float(values.mean() - TRUE_ATE),
                sd=float(values.std(ddof=1)),
                rmse=float(np.sqrt(np.mean((values - TRUE_ATE) ** 2))),
                mean_observed_fraction=float(arr[:, 4].mean()),
            ))
    with Path(__file__).with_name("results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
