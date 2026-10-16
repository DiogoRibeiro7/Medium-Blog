"""Monte Carlo experiment on observational selection and randomization.

Python 3.12+, NumPy only. All analyses use HC2 robust standard errors.
"""
from __future__ import annotations

import numpy as np

SEED: int = 20261016
REPETITIONS: int = 1000
SAMPLE_SIZES: tuple[int, ...] = (200, 1000, 5000)
TAU: float = -1.0


def fit(y: np.ndarray, treatment: np.ndarray, covariate: np.ndarray | None = None) -> tuple[float, float]:
    """Estimate a treatment coefficient and HC2 standard error."""
    if y.ndim != 1 or treatment.shape != y.shape or (covariate is not None and covariate.shape != y.shape):
        raise ValueError("Inputs must have the same one-dimensional shape")
    if y.size < 5:
        raise ValueError("At least five observations are required")
    columns = [np.ones_like(y), treatment]
    if covariate is not None:
        columns.append(covariate)
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    inverse = np.linalg.inv(design.T @ design)
    leverage = np.sum((design @ inverse) * design, axis=1)
    omega = residual**2 / (1.0 - leverage)
    covariance = inverse @ (design.T @ (omega[:, None] * design)) @ inverse
    return float(coefficients[1]), float(np.sqrt(covariance[1, 1]))


def main() -> None:
    """Report Monte Carlo estimates, coverage and covariate imbalance."""
    rng = np.random.default_rng(SEED)
    print("n,design,mean_effect,bias,rmse,mean_se,coverage,mean_abs_imbalance")
    for n in SAMPLE_SIZES:
        records: dict[str, list[tuple[float, float, float]]] = {
            label: [] for label in ("selected", "randomized", "randomized_adjusted")
        }
        for _ in range(REPETITIONS):
            u = rng.normal(size=n)
            error = rng.normal(size=n)
            selected_probability = 1.0 / (1.0 + np.exp(-1.5 * u))
            selected = (rng.random(n) < selected_probability).astype(float)
            randomized = np.zeros(n)
            randomized[rng.permutation(n)[: n // 2]] = 1.0
            outcome_selected = TAU * selected + 2.0 * u + error
            outcome_random = TAU * randomized + 2.0 * u + error
            for label, treatment, outcome, adjustment in (
                ("selected", selected, outcome_selected, None),
                ("randomized", randomized, outcome_random, None),
                ("randomized_adjusted", randomized, outcome_random, u),
            ):
                estimate, se = fit(outcome, treatment, adjustment)
                imbalance = float(u[treatment == 1].mean() - u[treatment == 0].mean())
                records[label].append((estimate, se, abs(imbalance)))
        for label, observations in records.items():
            result = np.asarray(observations)
            coefficients, errors, imbalance = result[:, 0], result[:, 1], result[:, 2]
            bias = coefficients.mean() - TAU
            rmse = np.sqrt(np.mean((coefficients - TAU) ** 2))
            coverage = np.mean(np.abs(coefficients - TAU) <= 1.96 * errors)
            print(f"{n},{label},{coefficients.mean():.4f},{bias:.4f},{rmse:.4f},"
                  f"{errors.mean():.4f},{coverage:.3f},{imbalance.mean():.4f}")


if __name__ == "__main__":
    main()
