"""Monte Carlo study of confounding bias (Python 3.12+)."""
from __future__ import annotations

import numpy as np

SEED: int = 20261016
REPETITIONS: int = 1000
SAMPLE_SIZES: tuple[int, ...] = (100, 1000, 10000)
TRUE_EFFECT: float = 1.0


def estimate(y: np.ndarray, x: np.ndarray, confounder: np.ndarray | None = None) -> tuple[float, float]:
    """Return OLS treatment coefficient and its homoskedastic standard error."""
    if y.ndim != 1 or x.shape != y.shape or (confounder is not None and confounder.shape != y.shape):
        raise ValueError("Expected equally sized one-dimensional arrays.")
    design = np.column_stack((np.ones(y.size), x)) if confounder is None else np.column_stack((np.ones(y.size), x, confounder))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coefficients
    variance = float(residuals @ residuals) / (y.size - design.shape[1])
    standard_error = np.sqrt(variance * np.linalg.inv(design.T @ design)[1, 1])
    return float(coefficients[1]), float(standard_error)


def main() -> None:
    """Compare observational and randomized analyses over independent samples."""
    rng = np.random.default_rng(SEED)
    print("design,n,mean_est,bias,rmse,mean_se,coverage")
    for n in SAMPLE_SIZES:
        results: dict[str, list[tuple[float, float]]] = {name: [] for name in ("naive", "adjusted", "randomized")}
        for _ in range(REPETITIONS):
            u, ex, ey = rng.normal(size=(3, n))
            x = u + ex
            y = TRUE_EFFECT * x + 2.0 * u + ey
            xr = rng.normal(size=n)
            yr = TRUE_EFFECT * xr + 2.0 * u + ey
            results["naive"].append(estimate(y, x))
            results["adjusted"].append(estimate(y, x, u))
            results["randomized"].append(estimate(yr, xr))
        for method, data in results.items():
            a = np.asarray(data)
            estimates, se = a[:, 0], a[:, 1]
            bias = np.mean(estimates - TRUE_EFFECT)
            rmse = np.sqrt(np.mean((estimates - TRUE_EFFECT) ** 2))
            coverage = np.mean(np.abs(estimates - TRUE_EFFECT) <= 1.96 * se)
            print(f"{method},{n},{np.mean(estimates):.4f},{bias:.4f},{rmse:.4f},{np.mean(se):.4f},{coverage:.3f}")


if __name__ == "__main__":
    # Capture the already machine-readable Monte Carlo table unchanged.
    import io
    from contextlib import redirect_stdout
    from pathlib import Path

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main()
    output = buffer.getvalue()
    Path(__file__).with_name("results.csv").write_text(output, encoding="utf-8")
    print(output, end="")
