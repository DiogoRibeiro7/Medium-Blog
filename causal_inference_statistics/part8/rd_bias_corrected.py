"""Pilot-quadratic bias correction for Part 8's local-linear RD estimator.

This is a transparent, fixed-bandwidth benchmark inspired by robust bias
correction; it is *not* a package implementation of CCT's optimized
bandwidth selection or full coverage-error-optimal procedure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from experiment import BANDWIDTHS, REPLICATIONS, SAMPLE_SIZE, SCENARIOS, SEED, TRUE_EFFECT, local_linear


def _side_weights(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bandwidth: float,
    pilot_bandwidth: float,
) -> tuple[float, NDArray[np.float64]]:
    """Get one-side bias-corrected intercept and its full influence weights.

    The local-linear intercept is corrected using the quadratic coefficient
    estimated on a potentially wider pilot bandwidth. Both linear weights
    are combined before variance estimation, retaining the covariance
    between the first fit and the pilot fit.
    """
    n: int = len(x)
    active = x < bandwidth
    pilot = x < pilot_bandwidth
    if np.count_nonzero(active) < 12 or np.count_nonzero(pilot) < 12:
        raise ValueError("Insufficient observations on this side")
    xa: NDArray[np.float64] = x[active]
    xp: NDArray[np.float64] = x[pilot]
    a: NDArray[np.float64] = np.column_stack((np.ones(xa.size), xa))
    p: NDArray[np.float64] = np.column_stack((np.ones(xp.size), xp, xp**2))
    wa: NDArray[np.float64] = 1 - xa / bandwidth
    wp: NDArray[np.float64] = 1 - xp / pilot_bandwidth

    normal_a = a.T @ (wa[:, None] * a)
    normal_p = p.T @ (wp[:, None] * p)
    # The intercept of x**2 projected onto a local line is its leading bias.
    curvature_projection: float = float(
        np.linalg.solve(normal_a, a.T @ (wa * xa**2))[0]
    )
    intercept_weights = (np.linalg.solve(normal_a, np.array([1.0, 0.0])) @ a.T) * wa
    quadratic_weights = (np.linalg.solve(normal_p, np.array([0.0, 0.0, 1.0])) @ p.T) * wp
    influence: NDArray[np.float64] = np.zeros(n, dtype=float)
    influence[active] += intercept_weights
    influence[pilot] -= curvature_projection * quadratic_weights
    intercept: float = float(influence @ y)
    return intercept, influence


def corrected_rd(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bandwidth: float,
) -> tuple[float, float]:
    """Return corrected jump and an HC0 variance including pilot uncertainty."""
    if x.ndim != 1 or x.shape != y.shape or not 0 < bandwidth <= 1:
        raise ValueError("Matched one-dimensional inputs and h in (0,1] required")
    pilot: float = min(1.0, 2 * bandwidth)
    positive: NDArray[np.bool_] = x >= 0
    negative: NDArray[np.bool_] = ~positive
    estimate: float = 0.0
    influence: NDArray[np.float64] = np.zeros(x.size)
    residual: NDArray[np.float64] = np.zeros(x.size)
    for side, sign in ((negative, -1.0), (positive, 1.0)):
        xs: NDArray[np.float64] = np.abs(x[side])
        ys: NDArray[np.float64] = y[side]
        intercept, side_weights = _side_weights(xs, ys, bandwidth, pilot)
        estimate += sign * intercept
        influence[side] = sign * side_weights
        selected = xs < pilot
        design: NDArray[np.float64] = np.column_stack(
            (np.ones(np.count_nonzero(selected)), xs[selected], xs[selected] ** 2)
        )
        weights = 1 - xs[selected] / pilot
        fitted = np.linalg.lstsq(design * np.sqrt(weights)[:, None], ys[selected] * np.sqrt(weights), rcond=None)[0]
        local_residual = np.zeros(xs.size)
        local_residual[selected] = ys[selected] - design @ fitted
        residual[side] = local_residual
    # A nonnegative heteroskedasticity-consistent quadratic form. The
    # combined influence vector accounts for pilot/main overlap exactly.
    se: float = float(np.sqrt(np.sum((influence * residual) ** 2)))
    return estimate, se


def run() -> list[dict[str, str | int | float]]:
    """Compare the original intervals with pilot-quadratic correction."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        values: dict[float, list[tuple[float, float, float, float]]] = {h: [] for h in BANDWIDTHS}
        for _ in range(REPLICATIONS):
            x = rng.uniform(-1.0, 1.0, SAMPLE_SIZE)
            noise = rng.normal(size=SAMPLE_SIZE)
            control = 1.0 + 0.6 * x + noise
            if scenario == "curvature":
                control += 2.0 * x**2 * (x >= 0)
            if scenario == "sorting":
                control += 0.7 * (x >= 0)
            y = control + TRUE_EFFECT * (x >= 0)
            for h in BANDWIDTHS:
                ordinary, ordinary_se, _ = local_linear(x, y, h)
                corrected, corrected_se = corrected_rd(x, y, h)
                values[h].append((ordinary, ordinary_se, corrected, corrected_se))
        for h, sequence in values.items():
            array = np.asarray(sequence)
            ordinary, ordinary_se, corrected, corrected_se = array.T
            rows.append(
                {
                    "scenario": scenario,
                    "bandwidth": h,
                    "pilot_bandwidth": min(1.0, 2 * h),
                    "replications": REPLICATIONS,
                    "mean_corrected": float(np.mean(corrected)),
                    "corrected_rmse": float(np.sqrt(np.mean((corrected - TRUE_EFFECT) ** 2))),
                    "ordinary_coverage": float(np.mean(np.abs(ordinary - TRUE_EFFECT) <= 1.96 * ordinary_se)),
                    "corrected_coverage": float(np.mean(np.abs(corrected - TRUE_EFFECT) <= 1.96 * corrected_se)),
                    "mean_corrected_se": float(np.mean(corrected_se)),
                }
            )
    return rows


def main() -> None:
    """Export reproducible comparison results to a separate CSV."""
    rows = run()
    output = Path(__file__).with_name("rd_bias_corrected_results.csv")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
