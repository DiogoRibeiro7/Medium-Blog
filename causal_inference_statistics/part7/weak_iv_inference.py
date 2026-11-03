"""Compare conventional IV intervals with Anderson--Rubin tests.

Run with Python >=3.12, NumPy and SciPy installed. All stochastic inputs
come from a dedicated seed; the original Part 7 experiment is unchanged.

For a candidate coefficient b, the Anderson--Rubin test regresses Y-bX
on an intercept and Z. Under a valid Gaussian just-identified instrument,
its t test has exact finite-sample size at the true structural coefficient.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ARConfidenceSet:
    """A union of closed intervals; infinite endpoints indicate unboundedness."""

    intervals: tuple[tuple[float, float], ...]

    @property
    def shape(self) -> str:
        """Classify the geometry of the inverted confidence set."""
        if len(self.intervals) == 2:
            return "disconnected"
        lower, upper = self.intervals[0]
        if np.isneginf(lower) and np.isposinf(upper):
            return "all_real"
        if np.isinf(lower) or np.isinf(upper):
            return "half_line"
        return "bounded"

    def contains(self, value: float) -> bool:
        """Check membership in any component, including infinite intervals."""
        return any(low <= value <= high for low, high in self.intervals)


def invert_anderson_rubin(
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    cutoff: float,
) -> ARConfidenceSet:
    """Analytically invert the two-sided homoskedastic AR t test.

    The accepted coefficients satisfy a quadratic inequality derived from
    the regression of Y-beta*X on an intercept and Z. This avoids a finite
    candidate grid, which would incorrectly truncate unbounded sets.
    """
    n: int = len(z)
    if n < 3 or len(x) != n or len(y) != n or cutoff <= 0:
        raise ValueError("Expected matched arrays of length >=3 and positive cutoff")
    zc = z - z.mean()
    xc = x - x.mean()
    yc = y - y.mean()
    zz: float = float(zc @ zc)
    if zz <= 0:
        raise ValueError("The instrument must have nonzero sample variance")
    zy: float = float(zc @ yc)
    zx: float = float(zc @ xc)
    yp = yc - (zy / zz) * zc
    xp = xc - (zx / zz) * zc
    factor: float = cutoff * cutoff * zz
    a: float = (n - 2) * zx * zx - factor * float(xp @ xp)
    b: float = -2 * (n - 2) * zx * zy + 2 * factor * float(xp @ yp)
    c: float = (n - 2) * zy * zy - factor * float(yp @ yp)
    scale: float = max(abs(a), abs(b), abs(c), 1.0)
    tolerance: float = 1e-12 * scale
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            if c <= tolerance:
                return ARConfidenceSet(((-np.inf, np.inf),))
            raise ArithmeticError("Empty AR set: inspect numerical conditioning")
        root = -c / b
        return ARConfidenceSet(((-np.inf, root),) if b > 0 else ((root, np.inf),))
    discriminant: float = b * b - 4 * a * c
    if discriminant < -1e-12 * max(b * b, abs(4 * a * c), 1.0):
        if a < 0:
            return ARConfidenceSet(((-np.inf, np.inf),))
        raise ArithmeticError("Empty AR set: inspect numerical conditioning")
    root_delta: float = np.sqrt(max(discriminant, 0.0))
    low: float = min((-b - root_delta) / (2 * a), (-b + root_delta) / (2 * a))
    high: float = max((-b - root_delta) / (2 * a), (-b + root_delta) / (2 * a))
    if a > 0:
        return ARConfidenceSet(((low, high),))
    return ARConfidenceSet(((-np.inf, low), (high, np.inf)))


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
        ar_sets: list[ARConfidenceSet] = []
        first_stage_f: list[float] = []
        for _ in range(REPLICATIONS):
            z, u, v, noise = rng.standard_normal((4, SAMPLE_SIZE))
            x: NDArray[np.float64] = first_stage * z + u + v
            y: NDArray[np.float64] = TRUE_EFFECT * x + u + noise + direct_effect * z
            estimate, se, ar_t = test_statistics(z, x, y)
            wald_covers.append(abs(estimate - TRUE_EFFECT) <= cutoff * se)
            ar_accepts.append(abs(ar_t) <= cutoff)
            ar_sets.append(invert_anderson_rubin(z, x, y, cutoff))
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
                "ar_set_coverage_of_true_effect": float(np.mean([item.contains(TRUE_EFFECT) for item in ar_sets])),
                "ar_bounded_fraction": float(np.mean([item.shape == "bounded" for item in ar_sets])),
                "ar_disconnected_fraction": float(np.mean([item.shape == "disconnected" for item in ar_sets])),
                "ar_half_line_fraction": float(np.mean([item.shape == "half_line" for item in ar_sets])),
                "ar_all_real_fraction": float(np.mean([item.shape == "all_real" for item in ar_sets])),
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
