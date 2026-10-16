"""Monte Carlo study of inverse probability weighting and overlap.

Requirements: Python >=3.12, numpy, scipy, scikit-learn.
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
TRUE_ATE: float = 1.0
STRENGTHS: tuple[float, ...] = (0.7, 1.5, 3.0)
TRIM: float = 0.05


@dataclass(frozen=True)
class Result:
    """Metrics for one simulated dataset."""
    naive: float
    ipw: float
    hajek: float
    clipped: float
    ess_treated: float
    ess_control: float
    max_weight: float
    clipped_fraction: float
    min_propensity: float
    max_propensity: float


def effective_size(weights: NDArray[np.float64]) -> float:
    """Return Kish's effective sample size for positive finite weights."""
    if weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be a nonempty, nonnegative finite vector")
    return float(weights.sum() ** 2 / np.square(weights).sum())


def simulate(rng: np.random.Generator, strength: float, n: int = N) -> Result:
    """Generate observational treatment data and evaluate four estimators."""
    if n < 20 or not np.isfinite(strength) or strength < 0:
        raise ValueError("n >= 20 and finite nonnegative strength required")
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    true_p = expit(strength * x)
    a = rng.binomial(1, true_p).astype(float)
    y = TRUE_ATE * a + 2.0 * x + noise
    if a.min() == a.max():
        raise ValueError("both groups required")
    model = LogisticRegression(C=1e6, max_iter=200)
    model.fit(x[:, None], a.astype(int))
    p: NDArray[np.float64] = model.predict_proba(x[:, None])[:, 1]
    treated = a == 1
    naive = float(y[treated].mean() - y[~treated].mean())
    wt = a / p
    wc = (1 - a) / (1 - p)
    ipw = float(np.mean(wt * y - wc * y))
    hajek = float(np.sum(wt * y) / wt.sum() - np.sum(wc * y) / wc.sum())
    pc = np.clip(p, TRIM, 1 - TRIM)
    wtc = a / pc
    wcc = (1 - a) / (1 - pc)
    clipped = float(np.sum(wtc * y) / wtc.sum() - np.sum(wcc * y) / wcc.sum())
    w_actual = np.where(treated, 1 / p, 1 / (1 - p))
    return Result(
        naive, ipw, hajek, clipped,
        effective_size(wt[treated]), effective_size(wc[~treated]),
        float(w_actual.max()),
        float(np.mean((p < TRIM) | (p > 1 - TRIM))),
        float(p.min()), float(p.max()),
    )


def main() -> None:
    """Print Monte Carlo means, deviations, and RMSE where applicable."""
    rng = np.random.default_rng(SEED)
    fields = list(Result.__dataclass_fields__)
    print("strength,metric,mean,sd,rmse_if_effect")
    for strength in STRENGTHS:
        results = [simulate(rng, strength) for _ in range(REPETITIONS)]
        arr = np.array([[getattr(r, field) for field in fields] for r in results])
        for j, field in enumerate(fields):
            values = arr[:, j]
            rmse = (
                np.sqrt(np.mean((values - TRUE_ATE) ** 2))
                if field in {"naive", "ipw", "hajek", "clipped"} else float("nan")
            )
            print(f"{strength:.1f},{field},{values.mean():.4f},{values.std():.4f},{rmse:.4f}")


if __name__ == "__main__":
    main()
