"""Reproducible experiments on dependent observations and statistical uncertainty.

Only seeded, synthetic stationary AR(1) and explicitly nonstationary regime
series are used. Publication mode writes eight figures and ten CSV tables;
--quick creates suffixed, smaller CI artifacts without overwriting them.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter
from scipy.stats import norm

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BLUE, ORANGE, TEAL, RED, GRAY = "#2463a0", "#d07832", "#25877a", "#b64e50", "#636c76"
Z975 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class Scenario:
    """Explicit sampling and simulation parameters for all experiments."""

    seed: int = 20261030
    n: int = 240
    nominal_coverage: float = 0.95
    main_phi: float = 0.8
    forecast_phi: float = 0.55
    future_mean: float = 2.5

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 20:
            raise ValueError("n must be an integer >=20")
        for name in ("main_phi", "forecast_phi"):
            validate_phi(getattr(self, name))
        if not 0 < self.nominal_coverage < 1:
            raise ValueError("nominal coverage must lie between zero and one")
        if not np.isfinite(self.future_mean):
            raise ValueError("future mean must be finite")


def validate_phi(phi: float) -> None:
    """Reject nonstationary coefficients for the stationary AR(1) experiment."""

    if not np.isscalar(phi) or not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("Stationary AR(1) requires -1 < phi < 1")


def stationary_ar1(
    rng: np.random.Generator, n: int, phi: float, replicates: int = 1
) -> FloatArray:
    """Simulate independent stationary Gaussian AR(1) paths with marginal variance 1.

    Each path starts at its stationary marginal distribution, rather than at
    a nonstationary deterministic zero, and innovations have SD sqrt(1-phi²).
    Output shape is (replicates, n), even when replicates=1.
    """

    validate_phi(phi)
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >=2")
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    innovations = rng.standard_normal((replicates, n))
    innovations[:, 1:] *= math.sqrt(1 - phi * phi)
    # A stationary draw at t=0 seeds the recursive filter.
    return np.asarray(lfilter([1.0], [1.0, -phi], innovations, axis=1), dtype=float)


def exact_mean_variance(phi: float, n: int) -> float:
    """Exact Var(sample mean) for the stationary unit-variance Gaussian AR(1)."""

    validate_phi(phi)
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be >=2")
    lag = np.arange(1, n, dtype=float)
    inflation = 1.0 + 2.0 * float(np.sum((1.0 - lag / n) * phi**lag))
    return float(inflation / n)


def asymptotic_long_run_variance(phi: float) -> float:
    """Sum of all stationary autocovariances for unit marginal variance."""

    validate_phi(phi)
    return float((1 + phi) / (1 - phi))


def effective_sample_size(phi: float, n: int, *, finite: bool = True) -> float:
    """Precision-equivalent iid sample count, not a literal count of new units."""

    if finite:
        return float(1.0 / exact_mean_variance(phi, n))
    return float(n / asymptotic_long_run_variance(phi))


def sample_acf(x: FloatArray, max_lag: int) -> FloatArray:
    """Biased, mean-centered sample autocorrelation with lag zero equal to one."""

    xx = np.asarray(x, dtype=float)
    if xx.ndim != 1 or xx.size < 2 or np.any(~np.isfinite(xx)):
        raise ValueError("Expected finite one-dimensional series")
    if not isinstance(max_lag, int) or not 0 <= max_lag < len(xx):
        raise ValueError("max_lag must lie in 0..n-1")
    residuals = xx - xx.mean()
    gamma_zero = float(np.dot(residuals, residuals))
    if gamma_zero <= 0:
        raise ValueError("Constant series has undefined autocorrelation")
    out = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        out[lag] = np.dot(residuals[: len(xx) - lag], residuals[lag:]) / gamma_zero
    return out


def iid_mean_se(x: FloatArray) -> float:
    """Incorrect for autocorrelated means unless independence actually holds."""

    xx = np.asarray(x, dtype=float)
    if xx.ndim != 1 or xx.size < 2 or np.any(~np.isfinite(xx)):
        raise ValueError("Expected finite one-dimensional series")
    return float(np.std(xx, ddof=1) / np.sqrt(len(xx)))


def bartlett_hac_long_run_variance(x: FloatArray, bandwidth: int) -> float:
    """Bartlett/Newey–West lag-window estimate of the long-run variance."""

    xx = np.asarray(x, dtype=float)
    if xx.ndim != 1 or xx.size < 2 or np.any(~np.isfinite(xx)):
        raise ValueError("Expected finite one-dimensional series")
    if not isinstance(bandwidth, int) or not 0 <= bandwidth < len(xx):
        raise ValueError("Bandwidth must lie in 0..n-1")
    z = xx - xx.mean()
    n = len(xx)
    value = float(np.dot(z, z) / n)
    for lag in range(1, bandwidth + 1):
        gamma = float(np.dot(z[:-lag], z[lag:]) / n)
        value += 2 * (1 - lag / (bandwidth + 1)) * gamma
    # Bartlett lag-window covariance is nonnegative mathematically; protect
    # floating-point rounding without masking an invalid negative estimate.
    if value < -1e-10:
        raise ArithmeticError("Unexpected negative Bartlett long-run variance")
    return float(max(value, 0.0))


def hac_mean_se(x: FloatArray, bandwidth: int) -> float:
    """HAC standard error of the mean; normal approximation is asymptotic."""

    return float(np.sqrt(bartlett_hac_long_run_variance(x, bandwidth) / len(x)))


def default_bandwidth(n: int) -> int:
    """Simple declared tuning rule: ceil(2 n^(1/3)); not universally optimal."""

    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be >=2")
    return min(n - 1, max(1, int(math.ceil(2 * n ** (1 / 3)))))


def normal_mean_interval(x: FloatArray, se: float) -> tuple[float, float]:
    """Symmetric nominal 95% normal interval using a supplied standard error."""

    xx = np.asarray(x, dtype=float)
    if xx.ndim != 1 or xx.size == 0 or not np.isfinite(se) or se < 0:
        raise ValueError("Expected finite nonnegative SE and one-dimensional series")
    middle = float(np.mean(xx))
    return middle - Z975 * se, middle + Z975 * se


def iid_bootstrap_means(rng: np.random.Generator, x: FloatArray, draws: int) -> FloatArray:
    """IID nonparametric bootstrap means; intentionally breaks serial dependence."""

    xx = np.asarray(x, dtype=float)
    if xx.ndim != 1 or xx.size < 2 or not np.all(np.isfinite(xx)):
        raise ValueError("Expected finite one-dimensional series")
    if not isinstance(draws, int) or draws < 2:
        raise ValueError("draws must be >=2")
    indices = rng.integers(0, len(xx), size=(draws, len(xx)))
    return np.asarray(xx[indices].mean(axis=1), dtype=float)


def moving_block_means(
    rng: np.random.Generator, x: FloatArray, block_length: int, draws: int
) -> FloatArray:
    """Overlapping moving-block bootstrap means, concatenated to length n.

    This is a non-circular block bootstrap and is designed for approximately
    stationary series. It is not generally valid across structural breaks.
    """

    xx = np.asarray(x, dtype=float)
    if xx.ndim != 1 or xx.size < 2 or not np.all(np.isfinite(xx)):
        raise ValueError("Expected finite one-dimensional series")
    n = xx.size
    if not isinstance(block_length, int) or not 1 <= block_length <= n:
        raise ValueError("block_length must be between 1 and n")
    if not isinstance(draws, int) or draws < 2:
        raise ValueError("draws must be >=2")
    nblocks = math.ceil(n / block_length)
    starts = rng.integers(0, n - block_length + 1, size=(draws, nblocks))
    offsets = np.arange(block_length, dtype=int)
    indices = (starts[:, :, None] + offsets).reshape(draws, -1)[:, :n]
    return np.asarray(xx[indices].mean(axis=1), dtype=float)


def basic_bootstrap_interval(x: FloatArray, means: FloatArray) -> tuple[float, float]:
    """Basic two-sided percentile-of-centered bootstrap-mean interval."""

    xx, values = np.asarray(x, dtype=float), np.asarray(means, dtype=float)
    if xx.ndim != 1 or values.ndim != 1 or xx.size < 2 or values.size < 2:
        raise ValueError("Expected nonempty one-dimensional arrays")
    if np.any(~np.isfinite(xx)) or np.any(~np.isfinite(values)):
        raise ValueError("Expected finite samples and means")
    center = float(xx.mean())
    differences = values - center
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return center - float(upper), center - float(lower)


def contains(interval: tuple[float, float], value: float = 0.0) -> bool:
    """Whether the declared scalar parameter belongs to the closed interval."""

    low, high = interval
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError("Invalid confidence interval")
    return bool(low <= value <= high)


def coverage_and_width(rows: list[dict[str, Any]], name: str) -> tuple[float, float]:
    """Aggregate nominal-interval coverage and mean width over replicates."""

    if not rows:
        raise ValueError("Rows must be nonempty")
    return (
        float(np.mean([row[f"{name}_covered"] for row in rows])),
        float(np.mean([row[f"{name}_width"] for row in rows])),
    )


def fit_ar_predictor(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Fit one-lag autoregression with intercept by least squares."""

    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xx.ndim != 1 or yy.ndim != 1 or xx.shape != yy.shape or len(xx) < 3:
        raise ValueError("Aligned one-dimensional x/y with >=3 pairs required")
    if np.any(~np.isfinite(xx)) or np.any(~np.isfinite(yy)):
        raise ValueError("Data must be finite")
    design = np.column_stack((np.ones_like(xx), xx))
    coefficients = np.linalg.lstsq(design, yy, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def forecast_mse(pair_x: FloatArray, pair_y: FloatArray, intercept: float, slope: float) -> float:
    """Mean squared error on a separately specified evaluation population."""

    xx, yy = np.asarray(pair_x, dtype=float), np.asarray(pair_y, dtype=float)
    if xx.shape != yy.shape or xx.size == 0:
        raise ValueError("Aligned nonempty evaluation pairs required")
    return float(np.mean((yy - intercept - slope * xx) ** 2))


def changed_mean_ar1(
    rng: np.random.Generator, n: int, phi: float, change_at: int, new_mean: float
) -> FloatArray:
    """Generate a known structural break: a new conditional mean after change_at."""

    validate_phi(phi)
    if not isinstance(n, int) or n < 40 or not isinstance(change_at, int) or not 20 <= change_at < n - 10:
        raise ValueError("Invalid time series or change-point length")
    if not np.isfinite(new_mean):
        raise ValueError("new_mean must be finite")
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    sigma = math.sqrt(1 - phi * phi)
    for t in range(1, n):
        mu = new_mean if t >= change_at else 0.0
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * e[t]
    return np.asarray(x, dtype=float)


def one_forecast_validation(
    rng: np.random.Generator, *, n: int = 1200, change_at: int = 800,
    phi: float = 0.55, new_mean: float = 2.5
) -> tuple[dict[str, float], FloatArray]:
    """Contrast random mixed-regime holdout with genuine future holdout.

    The random split trains on observations from both sides of the future
    structural break. Its test estimand differs from forward deployment.
    """

    series = changed_mean_ar1(rng, n, phi, change_at, new_mean)
    inputs, targets = series[:-1], series[1:]
    time_index = np.arange(1, n)
    indices = rng.permutation(len(inputs))
    split = int(0.8 * len(inputs))
    random_train, random_eval = indices[:split], indices[split:]
    a_rand, b_rand = fit_ar_predictor(inputs[random_train], targets[random_train])
    random_rmse = math.sqrt(forecast_mse(inputs[random_eval], targets[random_eval], a_rand, b_rand))

    # A pre-shift holdout estimates pre-shift performance with no future data.
    pre_train = time_index < int(0.70 * change_at)
    pre_eval = (time_index >= int(0.70 * change_at)) & (time_index < change_at)
    a_pre, b_pre = fit_ar_predictor(inputs[pre_train], targets[pre_train])
    pre_rmse = math.sqrt(forecast_mse(inputs[pre_eval], targets[pre_eval], a_pre, b_pre))

    forward_train = time_index < change_at
    future = time_index >= change_at
    a_future, b_future = fit_ar_predictor(inputs[forward_train], targets[forward_train])
    future_rmse = math.sqrt(forecast_mse(inputs[future], targets[future], a_future, b_future))
    return {
        "random_mixed_holdout_rmse": random_rmse,
        "chronological_prechange_holdout_rmse": pre_rmse,
        "genuine_future_after_shift_rmse": future_rmse,
        "forward_train_slope": b_future,
        "random_mixed_train_slope": b_rand,
    }, series


def _write_rows(root: Path, name: str, rows: list[dict[str, Any]], suffix: str) -> None:
    if not rows:
        raise ValueError("No rows to write")
    location = root / "results" / f"{name}{suffix}.csv"
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, root: Path, filename: str, suffix: str) -> None:
    fig.tight_layout(pad=1.5)
    fig.savefig(root / "figures" / f"{filename}{suffix}.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.grid": True,
        "grid.alpha": 0.2, "axes.axisbelow": True,
    })


def run_study(
    config: Scenario | None = None, *, quick: bool = False, output_dir: Path | None = None
) -> dict[str, float]:
    """Regenerate eight PNG figures and ten deterministic CSV evidence tables."""

    scenario = config or Scenario()
    root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    _style()

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        _write_rows(root, name, rows, suffix)

    n = scenario.n
    phis = (-0.5, 0.0, 0.4, 0.8, 0.9)
    replications = 48 if quick else 400
    boot_draws = 80 if quick else 190
    block_length = default_bandwidth(n)
    rng = np.random.default_rng(scenario.seed)

    # Figure 1: sample path geometry, with all series on the same scale.
    trace_phis = (-0.5, 0.0, 0.5, 0.85)
    trace_rows: list[dict[str, Any]] = []
    traces: dict[float, FloatArray] = {}
    for j, phi in enumerate(trace_phis):
        trace = stationary_ar1(np.random.default_rng(scenario.seed + 10 + j), 360, phi)[0]
        traces[phi] = trace
        trace_rows.extend({
            "phi": phi, "time_index": t, "value": float(value)
        } for t, value in enumerate(trace))
    write("series_realizations", trace_rows)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.0), sharex=True, sharey=True)
    for ax, phi in zip(axes, trace_phis):
        ax.plot(traces[phi], color=BLUE if phi >= 0 else ORANGE, lw=1.1)
        ax.axhline(0, color=GRAY, lw=0.8, ls="--")
        ax.set_ylabel(f"φ={phi:g}")
    axes[-1].set_xlabel("Observation index")
    axes[0].set_title("Figure 1 · Same marginal variance, different temporal structures")
    _save(fig, root, "fig1_dependent_series", suffix)

    # Figure 2: theoretical and empirical ACF, with iid reference lines.
    acf_rows: list[dict[str, float]] = []
    max_lag = 35
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.2), sharey=True)
    for ax, phi in zip(axes, (-0.5, 0.85)):
        empirical = sample_acf(traces[phi], max_lag)
        lags = np.arange(max_lag + 1)
        theoretical = phi ** lags
        acf_rows.extend({
            "phi": phi, "lag": int(k), "theoretical_acf": float(theoretical[k]),
            "sample_acf": float(empirical[k]),
            "iid_white_noise_reference": float(1.96 / np.sqrt(len(traces[phi]))),
        } for k in lags)
        ax.plot(lags, theoretical, color=BLUE, lw=2.2, label="Population ACF")
        ax.plot(lags, empirical, "o", color=ORANGE, markersize=3.2, alpha=0.8, label="Sample ACF")
        ax.axhline(1.96 / np.sqrt(len(traces[phi])), color=GRAY, ls="--", lw=0.9)
        ax.axhline(-1.96 / np.sqrt(len(traces[phi])), color=GRAY, ls="--", lw=0.9)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set(xlabel="Lag", title=f"φ={phi:g}; dashed: iid reference only")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig2_autocorrelation", suffix)
    write("acf_diagnostics", acf_rows)

    # Figure 3: exact finite-sample variance inflation and precision-equivalent n.
    phi_grid = np.linspace(-0.8, 0.95, 180)
    ess_rows = [{
        "phi": float(phi),
        "n": n,
        "exact_variance_of_mean": exact_mean_variance(float(phi), n),
        "variance_inflation_relative_to_iid": n * exact_mean_variance(float(phi), n),
        "effective_n_finite": effective_sample_size(float(phi), n),
        "effective_n_asymptotic": effective_sample_size(float(phi), n, finite=False),
        "asymptotic_long_run_variance": asymptotic_long_run_variance(float(phi)),
    } for phi in phi_grid]
    write("effective_sample_size", ess_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(phi_grid, [row["variance_inflation_relative_to_iid"] for row in ess_rows], color=BLUE, lw=2)
    axes[0].axhline(1, color=GRAY, ls="--")
    axes[0].set(xlabel="AR(1) coefficient φ", ylabel="Variance inflation", title="Exact finite-n mean variance versus iid")
    axes[1].plot(phi_grid, [row["effective_n_finite"] for row in ess_rows], color=ORANGE, lw=2.2, label="Exact finite n")
    axes[1].plot(phi_grid, [row["effective_n_asymptotic"] for row in ess_rows], color=TEAL, ls="--", lw=1.4, label="Asymptotic")
    axes[1].axhline(n, color=GRAY, ls=":", label=f"Nominal n={n}")
    axes[1].set(xlabel="AR(1) coefficient φ", ylabel="Precision-equivalent iid n", title="Effective sample size is not the row count")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig3_effective_sample_size", suffix)

    # Figure 4: finite Monte Carlo sampling distributions versus exact normals.
    sampling_rows: list[dict[str, Any]] = []
    sampling_n = 400 if quick else 2100
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x_density = np.linspace(-0.65, 0.65, 400)
    for j, phi in enumerate((0.0, scenario.main_phi)):
        samples = stationary_ar1(
            np.random.default_rng(scenario.seed + 100 + j), n, phi, sampling_n
        ).mean(axis=1)
        sampling_rows.extend({
            "phi": phi, "replicate": int(i), "sample_mean": float(value)
        } for i, value in enumerate(samples))
        color = BLUE if phi == 0 else ORANGE
        ax.hist(samples, bins=45, density=True, color=color, alpha=0.22, label=f"Simulated means, φ={phi:g}")
        ax.plot(x_density, norm.pdf(x_density, scale=math.sqrt(exact_mean_variance(phi, n))), color=color, lw=2.2, label=f"Exact Normal SD, φ={phi:g}")
    ax.set(xlabel="Sample mean of n observations", ylabel="Sampling density", title="Figure 4 · More rows do not imply independent information")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, root, "fig4_sampling_distribution", suffix)
    write("mean_sampling", sampling_rows)

    # Figure 5: repeated 95% CI coverage, including an unattainable oracle.
    coverage_rows: list[dict[str, Any]] = []
    coverage_summary: list[dict[str, Any]] = []
    for j, phi in enumerate(phis):
        paths = stationary_ar1(
            np.random.default_rng(scenario.seed + 1000 + j), n, phi, replications
        )
        local_rows: list[dict[str, Any]] = []
        se_oracle = math.sqrt(exact_mean_variance(phi, n))
        for replicate, x in enumerate(paths):
            brng = np.random.default_rng(scenario.seed + 300000 + j * 2000 + replicate)
            intervals = {
                "iid_normal": normal_mean_interval(x, iid_mean_se(x)),
                "iid_bootstrap": basic_bootstrap_interval(x, iid_bootstrap_means(brng, x, boot_draws)),
                "bartlett_hac": normal_mean_interval(x, hac_mean_se(x, block_length)),
                "moving_block": basic_bootstrap_interval(
                    x, moving_block_means(brng, x, block_length, boot_draws)
                ),
                "oracle_normal": normal_mean_interval(x, se_oracle),
            }
            row: dict[str, Any] = {
                "phi": phi, "replicate": replicate,
                "sample_mean": float(np.mean(x)), "n": n,
                "hac_bandwidth": block_length, "block_length": block_length,
            }
            for label, interval in intervals.items():
                row[f"{label}_covered"] = int(contains(interval))
                row[f"{label}_width"] = float(interval[1] - interval[0])
            local_rows.append(row)
            coverage_rows.append(row)
        for method in (
            "iid_normal", "iid_bootstrap", "bartlett_hac", "moving_block", "oracle_normal"
        ):
            coverage, width = coverage_and_width(local_rows, method)
            coverage_summary.append({
                "phi": phi, "method": method, "nominal_coverage": 0.95,
                "empirical_coverage": coverage, "mean_interval_width": width,
                "replications": replications,
                "monte_carlo_standard_error": math.sqrt(coverage * (1 - coverage) / replications),
            })
    write("coverage_replicates", coverage_rows)
    write("coverage_summary", coverage_summary)
    fig, ax = plt.subplots(figsize=(10.1, 4.8))
    colors = {
        "iid_normal": ORANGE, "iid_bootstrap": RED,
        "bartlett_hac": TEAL, "moving_block": BLUE, "oracle_normal": GRAY,
    }
    labels = {
        "iid_normal": "IID Normal", "iid_bootstrap": "IID bootstrap",
        "bartlett_hac": "Bartlett HAC", "moving_block": "Moving blocks",
        "oracle_normal": "Known-covariance oracle",
    }
    for method in colors:
        rr = [row for row in coverage_summary if row["method"] == method]
        ax.errorbar(
            [row["phi"] for row in rr],
            [row["empirical_coverage"] for row in rr],
            yerr=[1.96 * row["monte_carlo_standard_error"] for row in rr],
            marker="o", color=colors[method], lw=1.7, ms=4,
            capsize=2, label=labels[method],
        )
    ax.axhline(0.95, color=GRAY, ls="--", lw=1.3)
    ax.set(xlabel="Serial correlation φ", ylabel="Empirical interval coverage", ylim=(0, 1.05), title="Figure 5 · Nominal 95% can be far from actual coverage")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, root, "fig5_interval_coverage", suffix)

    # Figure 6: lag-window sensitivity at positive dependence.
    bandwidths = (0, 3, 6, 10, 14, 20, 30, 45)
    diagnostic_paths = stationary_ar1(
        np.random.default_rng(scenario.seed + 60000),
        n, scenario.main_phi, 65 if quick else 350,
    )
    lag_rows: list[dict[str, float]] = []
    for lag in bandwidths:
        intervals = [normal_mean_interval(x, hac_mean_se(x, lag)) for x in diagnostic_paths]
        coverage = float(np.mean([contains(interval) for interval in intervals]))
        lag_rows.append({
            "phi": scenario.main_phi, "n": n, "bandwidth": lag,
            "replications": len(diagnostic_paths),
            "empirical_coverage": coverage,
            "mean_estimated_long_run_variance": float(np.mean([
                bartlett_hac_long_run_variance(x, lag) for x in diagnostic_paths
            ])),
            "known_long_run_variance": asymptotic_long_run_variance(scenario.main_phi),
        })
    write("hac_bandwidth", lag_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.3))
    axes[0].plot(bandwidths, [row["empirical_coverage"] for row in lag_rows], "o-", color=TEAL, lw=2)
    axes[0].axhline(0.95, color=GRAY, ls="--", label="Nominal 95%")
    axes[0].set(xlabel="Bartlett HAC maximum lag", ylabel="Empirical coverage", ylim=(0, 1.05), title="Coverage depends on bandwidth")
    axes[0].legend(frameon=False)
    axes[1].plot(bandwidths, [row["mean_estimated_long_run_variance"] for row in lag_rows], "o-", color=ORANGE, lw=2)
    axes[1].axhline(asymptotic_long_run_variance(scenario.main_phi), color=GRAY, ls="--", label="Population long-run variance")
    axes[1].set(xlabel="Bartlett HAC maximum lag", ylabel="Mean estimated long-run variance", title="Bias–variance tradeoff is visible")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig6_hac_bandwidth", suffix)

    # Figure 7: non-circular moving-block bootstrap sensitivity.
    lengths = (1, 4, 8, 13, 20, 30, 45)
    block_paths = stationary_ar1(
        np.random.default_rng(scenario.seed + 70000),
        n, scenario.main_phi, 48 if quick else 280,
    )
    block_rows: list[dict[str, float]] = []
    for block_size in lengths:
        cover: list[bool] = []
        widths: list[float] = []
        for rep, x in enumerate(block_paths):
            brng = np.random.default_rng(scenario.seed + 80000 + block_size * 1000 + rep)
            bounds = basic_bootstrap_interval(
                x, moving_block_means(brng, x, block_size, 80 if quick else 190)
            )
            cover.append(contains(bounds))
            widths.append(bounds[1] - bounds[0])
        block_rows.append({
            "phi": scenario.main_phi, "n": n, "block_length": block_size,
            "replications": len(block_paths),
            "empirical_coverage": float(np.mean(cover)),
            "mean_interval_width": float(np.mean(widths)),
            "population_oracle_95_width": 2 * Z975 * math.sqrt(exact_mean_variance(scenario.main_phi, n)),
        })
    write("block_length", block_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(lengths, [row["empirical_coverage"] for row in block_rows], "o-", color=BLUE, lw=2)
    axes[0].axhline(0.95, color=GRAY, ls="--", label="Nominal")
    axes[0].set(xlabel="Moving-block length", ylabel="Empirical coverage", ylim=(0, 1.05), title="One block length cannot suit every series")
    axes[0].legend(frameon=False)
    axes[1].plot(lengths, [row["mean_interval_width"] for row in block_rows], "o-", color=ORANGE, lw=2)
    axes[1].axhline(block_rows[0]["population_oracle_95_width"], color=GRAY, ls="--", label="Known Gaussian width")
    axes[1].set(xlabel="Moving-block length", ylabel="Mean interval width", title="Longer blocks preserve more dependence")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig7_block_length", suffix)

    # Figure 8: random holdout sees future regime, unlike genuine deployment.
    forecast_rows: list[dict[str, float]] = []
    example_path: FloatArray | None = None
    for rep in range(12 if quick else 90):
        frng = np.random.default_rng(scenario.seed + 90000 + rep)
        performance, observed_series = one_forecast_validation(
            frng, phi=scenario.forecast_phi, new_mean=scenario.future_mean
        )
        forecast_rows.append({"replicate": rep, **performance})
        if rep == 0:
            example_path = observed_series
    write("forecast_validation", forecast_rows)
    assert example_path is not None
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.5))
    axes[0].plot(example_path, color=BLUE, lw=1)
    axes[0].axvline(800, color=RED, ls="--", label="Structural break")
    axes[0].set(xlabel="Time index", ylabel="Observed value", title="One realization: future has a new regime")
    axes[0].legend(frameon=False)
    fields = (
        ("random_mixed_holdout_rmse", "Shuffled\nmixed", ORANGE),
        ("chronological_prechange_holdout_rmse", "Past-only\nvalidation", TEAL),
        ("genuine_future_after_shift_rmse", "Genuine\nfuture", RED),
    )
    boxes = axes[1].boxplot(
        [[row[field] for row in forecast_rows] for field, _, _ in fields],
        tick_labels=[label for _, label, _ in fields], patch_artist=True,
    )
    for patch, (_, _, color) in zip(boxes["boxes"], fields):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
    axes[1].set(ylabel="One-step prediction RMSE", title="Different holdouts answer different questions")
    _save(fig, root, "fig8_forecast_validation", suffix)

    def coverage_at(phi: float, method: str) -> float:
        return next(float(r["empirical_coverage"]) for r in coverage_summary if r["phi"] == phi and r["method"] == method)

    summary = {
        "n": float(n), "main_phi": scenario.main_phi,
        "variance_inflation_phi_0_8": n * exact_mean_variance(0.8, n),
        "effective_n_phi_0_8": effective_sample_size(0.8, n),
        "long_run_variance_phi_0_8": asymptotic_long_run_variance(0.8),
        "naive_coverage_phi_0_8": coverage_at(0.8, "iid_normal"),
        "iid_bootstrap_coverage_phi_0_8": coverage_at(0.8, "iid_bootstrap"),
        "hac_coverage_phi_0_8": coverage_at(0.8, "bartlett_hac"),
        "block_coverage_phi_0_8": coverage_at(0.8, "moving_block"),
        "oracle_coverage_phi_0_8": coverage_at(0.8, "oracle_normal"),
        "hac_bandwidth": float(block_length), "bootstrap_draws": float(boot_draws),
        "coverage_replications_per_phi": float(replications),
        "forecast_random_mixed_rmse_mean": float(np.mean([r["random_mixed_holdout_rmse"] for r in forecast_rows])),
        "forecast_past_only_rmse_mean": float(np.mean([r["chronological_prechange_holdout_rmse"] for r in forecast_rows])),
        "forecast_genuine_future_rmse_mean": float(np.mean([r["genuine_future_after_shift_rmse"] for r in forecast_rows])),
    }
    write("summary", [{"measure": key, "value": value} for key, value in summary.items()])
    return summary


def main() -> None:
    """Command-line entrypoint for the reproducible synthetic study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="CI-scale suffixed artifacts")
    parser.add_argument("--output", type=Path, default=None, help="Root for figures/results")
    args = parser.parse_args()
    metrics = run_study(quick=args.quick, output_dir=args.output)
    for label, value in metrics.items():
        print(f"{label}: {value:.6f}")
    print("Wrote 8 figures and 10 CSV tables")


if __name__ == "__main__":
    main()
