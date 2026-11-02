"""Compare conventional IV intervals with Anderson--Rubin tests.

Run with Python >=3.12, NumPy and SciPy installed. All stochastic inputs
come from a dedicated seed; the original Part 7 experiment is unchanged.

For a candidate coefficient b, the Anderson--Rubin test regresses Y-bX
on an intercept and Z. Under a valid Gaussian just-identified instrument,
its t test has exact finite-sample size at the true structural coefficient.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

SEED: int = 20261102
REPLICATIONS: int = 500
SAMPLE_SIZE: int = 1000
TRUE_EFFECT: float = 1.0
ALPHA: float = 0.05
SCENARIOS: dict[str, tuple[float, float]] = {
    "valid_strong": (1.0, 0.0),
    "valid_moderate": (0.3, 0.0),
    "valid_weak": (0.1, 0.0),
    "invalid_strong": (1.0, 0.3),
}


def test_statistics(
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return IV estimate, conventional Wald SE and AR t at the true effect."""
    n: int = len(z)
    assert n >= 3 and len(x) == n and len(y) == n
    zc: NDArray[np.float64] = z - z.mean()
    xc: NDArray[np.float64] = x - x.mean()
    yc: NDArray[np.float64] = y - y.mean()
    zz: float = float(zc @ zc)
    zx: float = float(zc @ xc)
    zy: float = float(zc @ yc)
    estimate: float = zy / zx

    # Homoskedastic just-identified IV sandwich, with regression degrees of freedom.
    iv_residual: NDArray[np.float64] = yc - estimate * xc
    iv_variance: float = float(iv_residual @ iv_residual) / (n - 2)
    wald_se: float = np.sqrt(iv_variance * zz) / abs(zx)

    # Test H0: beta=TRUE_EFFECT without dividing by the estimated first stage.
    null_residual: NDArray[np.float64] = yc - TRUE_EFFECT * xc
    zw: float = float(zc @ null_residual)
    reduced_residual: NDArray[np.float64] = null_residual - (zw / zz) * zc
    ar_variance: float = float(reduced_residual @ reduced_residual) / (n - 2)
    ar_t: float = zw / np.sqrt(ar_variance * zz)
    return estimate, wald_se, ar_t


def run() -> list[dict[str, str | float | int]]:
    """Report rejection rates and interval coverage for each design."""
    rng: np.random.Generator = np.random.default_rng(SEED)
    cutoff: float = float(student_t.ppf(1 - ALPHA / 2, SAMPLE_SIZE - 2))
    rows: list[dict[str, str | float | int]] = []
    for scenario, (first_stage, direct_effect) in SCENARIOS.items():
        wald_covers: list[bool] = []
        ar_accepts: list[bool] = []
        first_stage_f: list[float] = []
        for _ in range(REPLICATIONS):
            z, u, v, noise = rng.standard_normal((4, SAMPLE_SIZE))
            x: NDArray[np.float64] = first_stage * z + u + v
            y: NDArray[np.float64] = TRUE_EFFECT * x + u + noise + direct_effect * z
            estimate, se, ar_t = test_statistics(z, x, y)
            wald_covers.append(abs(estimate - TRUE_EFFECT) <= cutoff * se)
            ar_accepts.append(abs(ar_t) <= cutoff)
            r: float = float(np.corrcoef(z, x)[0, 1])
            first_stage_f.append((SAMPLE_SIZE - 2) * r * r / (1 - r * r))
        rows.append(
            {
                "scenario": scenario,
                "n": SAMPLE_SIZE,
                "replications": REPLICATIONS,
                "nominal_coverage": 1 - ALPHA,
                "wald_coverage_of_true_effect": float(np.mean(wald_covers)),
                "ar_nonrejection_of_true_effect": float(np.mean(ar_accepts)),
                "fraction_first_stage_f_below_10": float(np.mean(np.array(first_stage_f) < 10)),
                "median_first_stage_f": float(np.median(first_stage_f)),
            }
        )
    return rows


def main() -> None:
    """Persist the deterministic inference comparison as a CSV snapshot."""
    output: Path = Path(__file__).with_name("weak_iv_inference_results.csv")
    records = run()
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        print(row)


if __name__ == "__main__":
    main()
