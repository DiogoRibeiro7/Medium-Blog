"""Bayesian causal identification and regularization experiment.

Python 3.12+; NumPy only. Known Gaussian observation variances isolate
posterior concentration and confounding from variance estimation.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

SEED: int = 20261018
REPLICATIONS: int = 300
SAMPLE_SIZES: tuple[int, ...] = (100, 1000, 5000)
PRIOR_PRECISIONS: tuple[float, ...] = (0.1, 10.0, 100.0)
TRUE_EFFECT: float = 1.0


def posterior(y: NDArray[np.float64], design: NDArray[np.float64], *, variance: float, precision: float) -> tuple[float, float]:
    """Return treatment posterior mean and SD under conjugate Gaussian prior.

    The intercept is flat; remaining coefficients have independent
    N(0, variance/precision) priors. Residual variance is known.
    """
    if y.ndim != 1 or design.ndim != 2 or design.shape[0] != y.size or design.shape[1] < 2:
        raise ValueError("Outcome and design dimensions do not match")
    if not np.isfinite(y).all() or not np.isfinite(design).all() or variance <= 0 or precision <= 0:
        raise ValueError("Inputs must be finite and variance/precision positive")
    penalty = np.diag([0.0] + [precision] * (design.shape[1] - 1))
    gram = design.T @ design + penalty
    coef = np.linalg.solve(gram, design.T @ y)
    var = variance * np.linalg.inv(gram)[1, 1]
    return float(coef[1]), float(np.sqrt(var))


def main() -> None:
    """Run factorial Monte Carlo benchmark and persist the complete summaries."""
    rng = np.random.default_rng(SEED)
    records: dict[tuple[int, str, float], list[tuple[float, float]]] = {}
    for n in SAMPLE_SIZES:
        for _ in range(REPLICATIONS):
            u, ex, ey = rng.normal(size=(3, n))
            x = u + ex
            y = TRUE_EFFECT * x + 2.0 * u + ey
            designs = {
                "observational_unadjusted": (np.column_stack((np.ones(n), x)), 3.0),
                "observational_adjusted": (np.column_stack((np.ones(n), x, u)), 1.0),
            }
            for label, (design, variance) in designs.items():
                for p in PRIOR_PRECISIONS:
                    records.setdefault((n, label, p), []).append(posterior(y, design, variance=variance, precision=p))
    rows: list[dict[str, int | float | str]] = []
    for (n, design, p), values in records.items():
        a = np.asarray(values)
        mu, sd = a[:, 0], a[:, 1]
        rows.append(dict(n=n, design=design, prior_precision=p, replications=REPLICATIONS,
                         true_effect=TRUE_EFFECT, mean=float(mu.mean()), bias=float(mu.mean()-TRUE_EFFECT),
                         rmse=float(np.sqrt(np.mean((mu-TRUE_EFFECT)**2))),
                         posterior_sd=float(sd.mean()),
                         causal_coverage=float(np.mean(np.abs(mu-TRUE_EFFECT)<=1.96*sd)),
                         positive_posterior_fraction=float(np.mean(mu/sd>1.645))))
    path = Path(__file__).with_name("results.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
