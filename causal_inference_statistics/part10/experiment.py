"""Double-robust AIPW under correct and misspecified nuisance models.

Requires Python >=3.12, NumPy, SciPy, and scikit-learn.
Run this script to write results.csv.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold

SEED: int = 20261017
REPS: int = 200
N: int = 1200
ATE: float = 1.0


def evaluate(seed: int, p_correct: bool, m_correct: bool) -> tuple[float, float, float, float]:
    """Return naive, outcome regression, normalized IPW and cross-fitted AIPW."""
    if not isinstance(seed, int) or not isinstance(p_correct, bool) or not isinstance(m_correct, bool):
        raise TypeError("Expected integer seed and boolean specification flags")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    # Positive assignment probabilities, but treatment and prognosis share X.
    propensity = expit(-0.3 + 0.8 * x - 0.45 * x * x)
    a = rng.binomial(1, propensity)
    baseline = 1 + 0.7 * x + 0.8 * x * x
    y = baseline + ATE * a + rng.normal(scale=1, size=N)
    naive = y[a == 1].mean() - y[a == 0].mean()
    folds = KFold(n_splits=2, shuffle=True, random_state=seed)
    phat = np.zeros(N)
    mu0 = np.zeros(N)
    mu1 = np.zeros(N)
    for tr, te in folds.split(x):
        pt = np.column_stack([x[tr], x[tr] ** 2]) if p_correct else x[tr, None]
        pe = np.column_stack([x[te], x[te] ** 2]) if p_correct else x[te, None]
        pmodel = LogisticRegression(C=1e6, max_iter=500).fit(pt, a[tr])
        phat[te] = pmodel.predict_proba(pe)[:, 1]
        mt = np.column_stack([x[tr], x[tr] ** 2]) if m_correct else x[tr, None]
        me = np.column_stack([x[te], x[te] ** 2]) if m_correct else x[te, None]
        for treatment, predictions in ((0, mu0), (1, mu1)):
            model = LinearRegression().fit(mt[a[tr] == treatment], y[tr][a[tr] == treatment])
            predictions[te] = model.predict(me)
    phat = np.clip(phat, 0.01, 0.99)
    or_est = float(np.mean(mu1 - mu0))
    ipw = float(np.sum(a * y / phat) / np.sum(a / phat)
                - np.sum((1 - a) * y / (1 - phat)) / np.sum((1 - a) / (1 - phat)))
    aipw = float(np.mean(mu1 - mu0 + a * (y - mu1) / phat
                         - (1 - a) * (y - mu0) / (1 - phat)))
    return float(naive), or_est, ipw, aipw


def main() -> None:
    """Write reproducible Monte Carlo point-estimator summaries."""
    rows: list[dict[str, bool | str | int | float]] = []
    for j, (pc, mc) in enumerate(((True, True), (True, False),
                                   (False, True), (False, False))):
        results = np.array([evaluate(SEED + 1000 * j + k, pc, mc)
                            for k in range(REPS)])
        for col, name in enumerate(("naive", "outcome_regression", "normalized_ipw", "aipw")):
            vals = results[:, col]
            rows.append({
                "propensity_correct": pc,
                "outcome_correct": mc,
                "method": name,
                "replicates": REPS,
                "n": N,
                "true_effect": ATE,
                "mean": float(vals.mean()),
                "bias": float(vals.mean() - ATE),
                "sd": float(vals.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean((vals - ATE) ** 2))),
                "q05": float(np.quantile(vals, .05)),
                "q95": float(np.quantile(vals, .95)),
            })
    output = Path(__file__).with_name("results.csv")
    with output.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["propensity_correct"], row["outcome_correct"],
              row["method"], f"mean={row['mean']:.4f} RMSE={row['rmse']:.4f}")


if __name__ == "__main__":
    main()
