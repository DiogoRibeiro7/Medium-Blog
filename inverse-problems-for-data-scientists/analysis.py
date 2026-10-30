"""Reproducible synthetic inverse-problem and sensor-design experiments.

The public study deliberately separates structural non-identifiability from
noise-amplified practical ill-conditioning. Nothing represents field data.

Examples
--------
    python analysis.py
    python analysis.py --quick
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
SEED: int = 20261030
N: int = 64
NOISE_SD: float = 0.012
DECAY: float = 13.0
SENSOR_BUDGET: int = 20
TRUNCATION_RANK: int = 24
REFERENCE_ALPHA: FloatArray = np.logspace(-10, -0.7, 112)
FIGURE_NAMES: tuple[str, ...] = (
    "fig1_exact_nullspace.png",
    "fig2_singular_spectrum.png",
    "fig3_reconstruction_vs_fit.png",
    "fig4_picard_and_filters.png",
    "fig5_parameter_selection.png",
    "fig6_bayesian_uncertainty.png",
    "fig7_sensor_design.png",
    "fig8_design_risk_distribution.png",
)
RESULT_NAMES: tuple[str, ...] = (
    "summary.csv",
    "singular_modes.csv",
    "regularization_path.csv",
    "method_comparison.csv",
    "bayesian_intervals.csv",
    "sensor_designs.csv",
    "design_budget.csv",
    "monte_carlo_methods.csv",
    "sensor_trials.csv",
    "observed_signal.csv",
)


@dataclass(frozen=True)
class Study:
    """Computed evidence for one seeded, synthetic inverse-problem study."""

    grid: FloatArray
    truth: FloatArray
    operator: FloatArray
    basis: FloatArray
    singular_values: FloatArray
    observed: FloatArray
    noiseless_observed: FloatArray
    exact_null_vector: FloatArray
    alternate_truth: FloatArray
    sparse_indices: NDArray[np.int64]
    ls_solution: FloatArray
    tsvd_solution: FloatArray
    tikhonov_solution: FloatArray
    oracle_solution: FloatArray
    alpha_grid: FloatArray
    alpha_gcv: float
    alpha_oracle: float
    gcv_scores: FloatArray
    truth_errors: FloatArray
    prior_sd_modes: FloatArray
    posterior_mean: FloatArray
    posterior_sd: FloatArray
    pointwise_coverage: FloatArray
    sensor_designs: dict[str, NDArray[np.int64]]
    design_trace: dict[str, FloatArray]
    sensor_trial_errors: dict[str, tuple[FloatArray, FloatArray]]
    mc_method_errors: dict[str, FloatArray]
    summary: dict[str, float]
    result_rows: dict[str, list[dict[str, Any]]]


def _check_size(n: int) -> None:
    """Reject invalid dimensions, including bool masquerading as an int."""

    if type(n) is not int or n < 4:
        raise ValueError("n must be an integer of at least four")


def _check_positive(value: float, name: str) -> None:
    """Check scalar model parameters without silent NaN propagation."""

    if not np.isscalar(value) or not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def cosine_basis(n: int) -> FloatArray:
    """Return an explicitly orthonormal DCT-II matrix (columns are modes)."""

    _check_size(n)
    j = np.arange(n, dtype=np.float64)[:, None]
    k = np.arange(n, dtype=np.float64)[None, :]
    basis = np.sqrt(2.0 / n) * np.cos(np.pi * (j + 0.5) * k / n)
    basis[:, 0] /= np.sqrt(2.0)
    return basis


def smoothing_operator(n: int, decay: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Construct a symmetric diffusion-like blur with known singular spectrum.

    This is a dimensionless spectral surrogate for a smoothing operator. It
    should not be described as a calibrated physical heat-transfer model.
    """

    _check_size(n)
    _check_positive(decay, "decay")
    modes = cosine_basis(n)
    frequency = np.arange(n, dtype=np.float64)
    singular = np.exp(-0.5 * np.square(frequency / decay))
    operator = (modes * singular[None, :]) @ modes.T
    return operator, modes, singular


def signal_truth(n: int) -> FloatArray:
    """Generate smooth spatial structure with two localized features."""

    _check_size(n)
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    return (
        0.55 * np.sin(2 * np.pi * grid)
        + 0.34 * np.cos(4 * np.pi * grid)
        + 0.75 * np.exp(-np.square((grid - 0.30) / 0.065))
        - 0.53 * np.exp(-np.square((grid - 0.70) / 0.05))
    )


def truncated_svd(y: FloatArray, modes: FloatArray, singular: FloatArray, rank: int) -> FloatArray:
    """Invert only the first rank ordered modes; exclude the remaining modes."""

    n = len(singular)
    if type(rank) is not int or not 1 <= rank <= n:
        raise ValueError("rank must be a positive integer no larger than n")
    if y.shape != (n,) or modes.shape != (n, n):
        raise ValueError("incompatible measurement, modes, and singular spectrum")
    coefficients = modes.T @ y
    inverse = np.zeros(n, dtype=np.float64)
    inverse[:rank] = coefficients[:rank] / singular[:rank]
    return modes @ inverse


def tikhonov(y: FloatArray, modes: FloatArray, singular: FloatArray, alpha: float) -> FloatArray:
    """Solve min_x ||Ax-y||² + alpha ||x||² in the supplied singular basis."""

    _check_positive(alpha, "alpha")
    n = len(singular)
    if y.shape != (n,) or modes.shape != (n, n):
        raise ValueError("incompatible measurement, modes, and singular spectrum")
    filter_weights = singular / (np.square(singular) + alpha)
    return modes @ (filter_weights * (modes.T @ y))


def regularization_path(
    y: FloatArray, truth: FloatArray, modes: FloatArray, singular: FloatArray, alphas: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return GCV, latent errors, effective degrees of freedom and residuals.

    The true-error path is an *oracle diagnostic* possible only in simulation.
    The selected practical estimate uses GCV without access to latent truth.
    """

    n = len(singular)
    if alphas.ndim != 1 or alphas.size < 2 or not np.all(np.isfinite(alphas)) or np.any(alphas <= 0):
        raise ValueError("alphas must be a non-empty positive one-dimensional grid")
    if y.shape != (n,) or truth.shape != (n,):
        raise ValueError("y and truth must have n elements")
    z = modes.T @ y
    factors = singular[:, None] ** 2 / (singular[:, None] ** 2 + alphas[None, :])
    filtered_coeffs = singular[:, None] / (singular[:, None] ** 2 + alphas[None, :]) * z[:, None]
    errors = np.sqrt(np.mean(np.square(modes @ filtered_coeffs - truth[:, None]), axis=0))
    residual_sq = np.sum(np.square((1 - factors) * z[:, None]), axis=0)
    df = np.sum(factors, axis=0)
    gcv = (residual_sq / n) / np.square(1 - df / n)
    return gcv, errors, df, np.sqrt(residual_sq)


def normal_prior_sd(n: int) -> FloatArray:
    """Specify a deliberately quite smooth zero-mean DCT prior.

    The fixed illustrative truth is not sampled from this prior. Its
    localized features make the fixed-truth coverage exercise informative.
    """

    _check_size(n)
    frequency = np.arange(n, dtype=np.float64)
    return 0.7 / (1.0 + np.square(frequency / 4.0))


def bayesian_full(
    y: FloatArray, modes: FloatArray, singular: FloatArray, noise_sd: float, prior_sd: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return Gaussian posterior mean, spatial SD and mode variances.

    Prior: independent zero-mean normal DCT coefficients. Noise: iid normal
    with known SD. The posterior is exact *under these assumed models*.
    """

    _check_positive(noise_sd, "noise_sd")
    n = len(singular)
    if y.shape != (n,) or prior_sd.shape != (n,) or np.any(prior_sd <= 0):
        raise ValueError("invalid y or prior standard deviations")
    variances = 1.0 / (1.0 / np.square(prior_sd) + np.square(singular) / noise_sd**2)
    posterior_modes = variances * singular * (modes.T @ y) / noise_sd**2
    posterior_mean = modes @ posterior_modes
    spatial_sd = np.sqrt(np.square(modes) @ variances)
    return posterior_mean, spatial_sd, variances


def posterior_sensor(
    operator: FloatArray, indices: NDArray[np.int64], prior_cov: FloatArray, noise_sd: float
) -> tuple[FloatArray, FloatArray]:
    """Return linear gain and posterior covariance for a fixed sensor design."""

    _check_positive(noise_sd, "noise_sd")
    n = operator.shape[1]
    if operator.shape != (n, n) or prior_cov.shape != (n, n):
        raise ValueError("operator and covariance must be square and compatible")
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("sensor indices must be a nonempty one-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("sensor indices must be integer-valued")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("sensor indices must be within the observation grid")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("sensor indices must be unique")
    rows = operator[indices]
    gram = rows @ prior_cov @ rows.T + noise_sd**2 * np.eye(len(indices))
    gain = np.linalg.solve(gram, rows @ prior_cov).T
    post = prior_cov - gain @ rows @ prior_cov
    return gain, (post + post.T) / 2.0


def greedy_a_optimal(
    operator: FloatArray, prior_cov: FloatArray, noise_sd: float, count: int
) -> tuple[NDArray[np.int64], FloatArray]:
    """Select distinct sensors greedily by one-step posterior-trace decrease.

    This is an A-optimal *heuristic*, not a proof of a globally optimal subset.
    """

    _check_positive(noise_sd, "noise_sd")
    n = operator.shape[0]
    if type(count) is not int or not 1 <= count <= n:
        raise ValueError("count must be an integer in [1, n]")
    if operator.shape != (n, n) or prior_cov.shape != (n, n):
        raise ValueError("incompatible operator and covariance")
    posterior = prior_cov.copy()
    indices: list[int] = []
    traces: list[float] = []
    available = np.ones(n, dtype=np.bool_)
    for _ in range(count):
        # Sherman–Morrison update for one scalar observation.
        cross = posterior @ operator.T
        denom = np.sum(operator.T * cross, axis=0) + noise_sd**2
        reduction = np.sum(np.square(cross), axis=0) / denom
        reduction[~available] = -np.inf
        chosen = int(np.argmax(reduction))
        vector = cross[:, chosen]
        posterior -= np.outer(vector, vector) / denom[chosen]
        posterior = (posterior + posterior.T) / 2.0
        available[chosen] = False
        indices.append(chosen)
        traces.append(float(np.trace(posterior)))
    return np.asarray(indices, dtype=np.int64), np.asarray(traces, dtype=np.float64)


def posterior_trace_sequence(
    operator: FloatArray, prior_cov: FloatArray, noise_sd: float, indices: NDArray[np.int64]
) -> FloatArray:
    """Track posterior trace as a prespecified sequence of sensors is added."""

    posterior = prior_cov.copy()
    traces: list[float] = []
    for selected in indices:
        row = operator[int(selected)]
        vector = posterior @ row
        denom = float(row @ vector + noise_sd**2)
        posterior -= np.outer(vector, vector) / denom
        posterior = (posterior + posterior.T) / 2.0
        traces.append(float(np.trace(posterior)))
    return np.asarray(traces, dtype=np.float64)


def spread_sensor_order(n: int) -> NDArray[np.int64]:
    """Deterministic multiscale spatial coverage for power-of-two grids."""

    _check_size(n)
    if n & (n - 1):
        raise ValueError("bit-reversed spread order requires a power of two")
    width = n.bit_length() - 1
    return np.asarray([int(f"{i:0{width}b}"[::-1], 2) for i in range(n)], dtype=np.int64)


def _rmse(predicted: FloatArray, actual: FloatArray) -> float:
    """Evaluate latent or forward-space errors on a fixed coordinate grid."""

    return float(np.sqrt(np.mean(np.square(predicted - actual))))


def run_study(*, quick: bool = False, seed: int = SEED) -> Study:
    """Run a fully reproducible constructed experiment, including Monte Carlo.

    The quick variant uses the same initial observations and smaller Monte
    Carlo samples; it never overwrites the full publication outputs.
    """

    if type(quick) is not bool or type(seed) is not int:
        raise TypeError("quick must be bool and seed must be int")
    rng_single = np.random.default_rng(seed)
    rng_method = np.random.default_rng(seed + 1)
    rng_bayes = np.random.default_rng(seed + 2)
    rng_sensor = np.random.default_rng(seed + 3)
    mc_size = 64 if quick else 480

    n = N
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    truth = signal_truth(n)
    operator, basis, singular = smoothing_operator(n, DECAY)
    noiseless = operator @ truth
    observed = noiseless + rng_single.normal(0, NOISE_SD, size=n)

    # Sparse sensing creates an *exact* non-trivial null space, unlike small
    # singular values in the fully observed square operator.
    sparse = np.linspace(0, n - 1, SENSOR_BUDGET, dtype=np.int64)
    _, _, vh = np.linalg.svd(operator[sparse], full_matrices=True)
    null_vector = vh[-1] / np.linalg.norm(vh[-1])
    alternate_truth = truth + 5.0 * null_vector

    ls = basis @ ((basis.T @ observed) / singular)
    truncated = truncated_svd(observed, basis, singular, TRUNCATION_RANK)
    gcv, true_error, df, residual = regularization_path(
        observed, truth, basis, singular, REFERENCE_ALPHA
    )
    gcv_idx, oracle_idx = int(np.argmin(gcv)), int(np.argmin(true_error))
    alpha_gcv, alpha_oracle = float(REFERENCE_ALPHA[gcv_idx]), float(REFERENCE_ALPHA[oracle_idx])
    regularized = tikhonov(observed, basis, singular, alpha_gcv)
    oracle = tikhonov(observed, basis, singular, alpha_oracle)

    # Fixed-truth Monte Carlo: a calibration study under repeated new noise.
    # It is not a Bayesian prior-predictive experiment.
    noise_batch = rng_method.normal(0, NOISE_SD, size=(n, mc_size))
    z = basis.T @ (noiseless[:, None] + noise_batch)
    factors = singular[:, None] ** 2 / (
        singular[:, None] ** 2 + REFERENCE_ALPHA[None, :]
    )
    residual_sq = np.sum(np.square((1.0 - factors[:, :, None]) * z[:, None, :]), axis=0)
    gcv_batch = (residual_sq / n) / (
        np.square(1.0 - np.sum(factors, axis=0) / n)[:, None]
    )
    alpha_by_trial = REFERENCE_ALPHA[np.argmin(gcv_batch, axis=0)]
    full_recovery_coeff = z / singular[:, None]
    truncated_coeff = np.zeros_like(z)
    truncated_coeff[:TRUNCATION_RANK] = z[:TRUNCATION_RANK] / singular[:TRUNCATION_RANK, None]
    gcv_coeff = singular[:, None] * z / (
        singular[:, None] ** 2 + alpha_by_trial[None, :]
    )
    mc_method_coeff = {
        "Unregularized inverse": full_recovery_coeff,
        f"TSVD rank {TRUNCATION_RANK}": truncated_coeff,
        "Tikhonov (GCV)": gcv_coeff,
    }
    mc_method_errors = {
        method: np.sqrt(np.mean(np.square(basis @ coefs - truth[:, None]), axis=0))
        for method, coefs in mc_method_coeff.items()
    }

    prior_sd = normal_prior_sd(n)
    prior_cov = (basis * prior_sd[None, :] ** 2) @ basis.T
    post_mean, post_sd, _ = bayesian_full(observed, basis, singular, NOISE_SD, prior_sd)
    bayes_noise = rng_bayes.normal(0, NOISE_SD, size=(n, mc_size))
    bayes_z = basis.T @ (noiseless[:, None] + bayes_noise)
    post_variance = 1.0 / (1.0 / prior_sd**2 + singular**2 / NOISE_SD**2)
    repeated_means = basis @ (
        (post_variance * singular / NOISE_SD**2)[:, None] * bayes_z
    )
    pointwise_coverage = np.mean(
        np.abs(repeated_means - truth[:, None]) <= 1.96 * post_sd[:, None],
        axis=1,
    )

    spread = spread_sensor_order(n)[:SENSOR_BUDGET]
    cluster = np.arange(SENSOR_BUDGET, dtype=np.int64)
    chosen, greedy_traces = greedy_a_optimal(operator, prior_cov, NOISE_SD, SENSOR_BUDGET)
    designs = {
        "Clustered": cluster,
        "Spread": spread,
        "Greedy A-opt": chosen,
    }
    traces = {
        "Clustered": posterior_trace_sequence(operator, prior_cov, NOISE_SD, cluster),
        "Spread": posterior_trace_sequence(operator, prior_cov, NOISE_SD, spread),
        "Greedy A-opt": greedy_traces,
    }
    # Common full-grid noise across designs makes per-trial comparisons paired.
    sensor_noise = rng_sensor.normal(0, NOISE_SD, size=(n, mc_size))
    sensor_trial_errors: dict[str, tuple[FloatArray, FloatArray]] = {}
    for name, sensor_indices in designs.items():
        gain, _ = posterior_sensor(operator, sensor_indices, prior_cov, NOISE_SD)
        measured = noiseless[sensor_indices, None] + sensor_noise[sensor_indices]
        estimated = gain @ measured
        latent_error = np.sqrt(np.mean(np.square(estimated - truth[:, None]), axis=0))
        forward_error = np.sqrt(
            np.mean(np.square(operator @ estimated - noiseless[:, None]), axis=0)
        )
        sensor_trial_errors[name] = (latent_error, forward_error)

    known_noise_floor = np.sqrt(n) * NOISE_SD
    ratios = {
        "nullspace_residual": float(
            np.linalg.norm(operator[sparse] @ (alternate_truth - truth))
        ),
        "nullspace_latent_distance": _rmse(alternate_truth, truth),
        "condition_number": float(singular[0] / singular[-1]),
        "ls_latent_rmse": _rmse(ls, truth),
        "ls_forward_rmse": _rmse(operator @ ls, noiseless),
        "tsvd_latent_rmse": _rmse(truncated, truth),
        "tikhonov_latent_rmse": _rmse(regularized, truth),
        "tikhonov_forward_rmse": _rmse(operator @ regularized, noiseless),
        "oracle_latent_rmse": _rmse(oracle, truth),
        "alpha_gcv": alpha_gcv,
        "alpha_oracle": alpha_oracle,
        "posterior_mean_rmse": _rmse(post_mean, truth),
        "mean_pointwise_coverage": float(np.mean(pointwise_coverage)),
        "prior_trace": float(np.trace(prior_cov)),
        "noise_euclidean_expected": known_noise_floor,
        "mc_size": float(mc_size),
    }
    for name, trace in traces.items():
        ratios[f"{name.lower().replace(' ', '_').replace('-', '_')}_posterior_trace"] = float(
            trace[-1]
        )

    rows: dict[str, list[dict[str, Any]]] = {}
    rows["summary.csv"] = [{"metric": key, "value": value} for key, value in ratios.items()]
    c_true = basis.T @ truth
    c_obs = basis.T @ observed
    rows["singular_modes.csv"] = [
        {
            "mode": k,
            "singular_value": singular[k],
            "latent_coefficient": c_true[k],
            "measured_coefficient": c_obs[k],
            "noise_coefficient": c_obs[k] - singular[k] * c_true[k],
            "inverse_amplification": 1.0 / singular[k],
        }
        for k in range(n)
    ]
    rows["regularization_path.csv"] = [
        {
            "alpha": alpha,
            "gcv": score,
            "oracle_latent_rmse": error,
            "effective_degrees_of_freedom": effective,
            "residual_l2": residual_l2,
        }
        for alpha, score, error, effective, residual_l2 in zip(
            REFERENCE_ALPHA, gcv, true_error, df, residual, strict=True
        )
    ]
    solutions = {
        "Unregularized inverse": ls,
        f"TSVD rank {TRUNCATION_RANK}": truncated,
        "Tikhonov (GCV)": regularized,
        "Tikhonov (oracle)": oracle,
        "Bayesian posterior mean": post_mean,
    }
    rows["method_comparison.csv"] = [
        {
            "method": label,
            "latent_rmse": _rmse(solution, truth),
            "forward_rmse_against_noise_free_observation": _rmse(operator @ solution, noiseless),
            "observed_residual_rmse": _rmse(operator @ solution, observed),
        }
        for label, solution in solutions.items()
    ]
    rows["bayesian_intervals.csv"] = [
        {
            "position": grid[k],
            "true_signal": truth[k],
            "posterior_mean": post_mean[k],
            "posterior_sd": post_sd[k],
            "interval_lower_95": post_mean[k] - 1.96 * post_sd[k],
            "interval_upper_95": post_mean[k] + 1.96 * post_sd[k],
            "fixed_truth_repetition_coverage": pointwise_coverage[k],
        }
        for k in range(n)
    ]
    rows["sensor_designs.csv"] = [
        {
            "design": name,
            "selection_order": order + 1,
            "sensor_index_zero_based": sensor_idx,
            "position": grid[sensor_idx],
        }
        for name, indices in designs.items()
        for order, sensor_idx in enumerate(indices)
    ]
    rows["design_budget.csv"] = [
        {"design": name, "sensor_count": k + 1, "posterior_trace": value}
        for name, values in traces.items()
        for k, value in enumerate(values)
    ]
    rows["monte_carlo_methods.csv"] = [
        {"method": name, "replicate": k + 1, "latent_rmse": value}
        for name, values in mc_method_errors.items()
        for k, value in enumerate(values)
    ]
    rows["sensor_trials.csv"] = [
        {
            "design": name,
            "replicate": k + 1,
            "latent_rmse": latent[k],
            "full_grid_forward_rmse": forward[k],
        }
        for name, (latent, forward) in sensor_trial_errors.items()
        for k in range(mc_size)
    ]
    rows["observed_signal.csv"] = [
        {
            "position": grid[k],
            "true_signal": truth[k],
            "noiseless_observation": noiseless[k],
            "observed_value": observed[k],
            "measurement_noise": observed[k] - noiseless[k],
        }
        for k in range(n)
    ]

    return Study(
        grid=grid,
        truth=truth,
        operator=operator,
        basis=basis,
        singular_values=singular,
        observed=observed,
        noiseless_observed=noiseless,
        exact_null_vector=null_vector,
        alternate_truth=alternate_truth,
        sparse_indices=sparse,
        ls_solution=ls,
        tsvd_solution=truncated,
        tikhonov_solution=regularized,
        oracle_solution=oracle,
        alpha_grid=REFERENCE_ALPHA,
        alpha_gcv=alpha_gcv,
        alpha_oracle=alpha_oracle,
        gcv_scores=gcv,
        truth_errors=true_error,
        prior_sd_modes=prior_sd,
        posterior_mean=post_mean,
        posterior_sd=post_sd,
        pointwise_coverage=pointwise_coverage,
        sensor_designs=designs,
        design_trace=traces,
        sensor_trial_errors=sensor_trial_errors,
        mc_method_errors=mc_method_errors,
        summary=ratios,
        result_rows=rows,
    )


def _style(ax: Any, xlabel: str, ylabel: str) -> None:
    """Apply shared, Medium-friendly figure styling."""

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20)


def plot_figures(study: Study, directory: Path, *, quick: bool = False) -> None:
    """Generate eight publication figures; quick outputs have distinct names."""

    if type(quick) is not bool:
        raise TypeError("quick must be bool")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""

    def save(fig: Any, name: str) -> None:
        target = directory / name.replace(".png", f"{suffix}.png")
        fig.savefig(target, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Figure 1: exact non-identifiability under fewer sensors than unknowns.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].plot(study.grid, study.truth, lw=2.2, label="Original latent truth")
    axs[0].plot(study.grid, study.alternate_truth, lw=1.7, label="Alternative latent truth")
    axs[0].scatter(
        study.grid[study.sparse_indices],
        study.truth[study.sparse_indices],
        c="black",
        s=11,
        label="Sensor locations (not direct measurements)",
        zorder=5,
    )
    axs[0].legend(fontsize=8)
    axs[0].set_title("Two different signals, same sparse forward data")
    _style(axs[0], "Position", "Latent signal")
    original = study.operator[study.sparse_indices] @ study.truth
    altered = study.operator[study.sparse_indices] @ study.alternate_truth
    axs[1].scatter(study.grid[study.sparse_indices], original, s=45, label="Original predictions")
    axs[1].scatter(
        study.grid[study.sparse_indices], altered, s=17, marker="x", color="#ad4c4c",
        label="Alternative predictions", zorder=5,
    )
    axs[1].legend(fontsize=9)
    axs[1].set_title("Predicted sensor measurements coincide")
    _style(axs[1], "Sensor position", "Noiseless measured value")
    fig.suptitle("Exact non-identifiability is a property of the observation map")
    save(fig, FIGURE_NAMES[0])

    # Figure 2: singular decay and amplified noise in direct inversion.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    k = np.arange(N)
    axs[0].semilogy(k, study.singular_values, "-o", ms=2.6, lw=1.6)
    axs[0].axhline(NOISE_SD, ls=":", color="#aa5533", label="Noise SD (illustrative scale)")
    axs[0].legend(fontsize=8)
    axs[0].set_title("High-frequency modes are strongly attenuated")
    _style(axs[0], "DCT mode index", "Forward singular value (log scale)")
    mode_noise = study.basis.T @ (study.observed - study.noiseless_observed)
    axs[1].semilogy(k, np.abs(mode_noise), label="Magnitude of measurement noise", lw=1.5)
    axs[1].semilogy(k, np.abs(mode_noise / study.singular_values), label="After inverse division", lw=1.5)
    axs[1].legend(fontsize=8)
    axs[1].set_title("Tiny singular values turn noise into large coefficients")
    _style(axs[1], "DCT mode index", "Absolute coefficient (log scale)")
    fig.suptitle("Algebraic inversion can be stable numerically yet unstable statistically")
    save(fig, FIGURE_NAMES[1])

    # Figure 3: raw latent reconstruction versus forward-space data fit.
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.5))
    axs[0].plot(study.grid, study.truth, color="black", lw=2.2, label="Known truth")
    axs[0].plot(study.grid, study.tsvd_solution, lw=1.7, label=f"TSVD rank {TRUNCATION_RANK}")
    axs[0].plot(study.grid, study.tikhonov_solution, lw=1.7, label="Tikhonov selected by GCV")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Regularized estimates recover broad structure")
    _style(axs[0], "Position", "Latent signal")
    methods = [study.ls_solution, study.tsvd_solution, study.tikhonov_solution, study.posterior_mean]
    names = ["Direct inverse", "TSVD", "Tikhonov GCV", "Bayes mean"]
    axs[1].semilogy(
        names,
        [_rmse(solution, study.truth) for solution in methods],
        "o-", color="#b55a47", lw=1.9, label="Latent reconstruction error",
    )
    axs[1].semilogy(
        names,
        [max(_rmse(study.operator @ solution, study.observed), 1e-14) for solution in methods],
        "s--", color="#367c90", lw=1.6, label="Observed-data residual",
    )
    axs[1].legend(fontsize=8, loc="center right")
    axs[1].set_title("Fitting observations is not recovering the signal")
    axs[1].tick_params(axis="x", labelrotation=16)
    _style(axs[1], "Estimator", "RMSE on different spaces (log scale)")
    fig.suptitle("A near-perfect fit to observations does not guarantee latent recovery")
    save(fig, FIGURE_NAMES[2])

    # Figure 4: observed coefficients and regularization filter factors.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    c = np.abs(study.basis.T @ study.observed)
    true_forward_c = np.abs(study.singular_values * (study.basis.T @ study.truth))
    axs[0].semilogy(k, np.maximum(c, 1e-15), label="Observed coefficient", lw=1.6)
    axs[0].semilogy(k, np.maximum(true_forward_c, 1e-15), label="Noise-free coefficient", lw=1.6)
    axs[0].axhline(NOISE_SD, ls=":", color="#a54a40", label="Noise SD reference")
    axs[0].legend(fontsize=8)
    axs[0].set_title("The noisy Picard coefficients stop decaying")
    _style(axs[0], "DCT mode", "Absolute data coefficient (log scale)")
    factor_gcv = study.singular_values**2 / (study.singular_values**2 + study.alpha_gcv)
    factor_oracle = study.singular_values**2 / (study.singular_values**2 + study.alpha_oracle)
    axs[1].plot(k, factor_gcv, label="Tikhonov GCV")
    axs[1].plot(k, factor_oracle, ls="--", label="Tikhonov oracle")
    axs[1].step(k, (k < TRUNCATION_RANK).astype(float), where="mid", label="TSVD cutoff")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Regularization controls amplification, not information")
    _style(axs[1], "DCT mode", "Spectral filter factor")
    fig.suptitle("The discrete Picard perspective explains why inversion is unstable")
    save(fig, FIGURE_NAMES[3])

    # Figure 5: inaccessible oracle error versus observable GCV selection.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].loglog(study.alpha_grid, study.truth_errors, label="Truth RMSE (oracle)", lw=1.8)
    axs[0].axvline(study.alpha_gcv, ls="--", label="GCV selection")
    axs[0].axvline(study.alpha_oracle, ls=":", color="#ae452f", label="Oracle selection")
    axs[0].set_title("The best latent reconstruction parameter is hidden")
    axs[0].legend(fontsize=8)
    _style(axs[0], "Penalty strength α", "Known-truth RMSE (log scale)")
    axs[1].loglog(study.alpha_grid, study.gcv_scores, label="GCV prediction score")
    axs[1].axvline(study.alpha_gcv, ls="--", label="Selected minimum")
    axs[1].legend(fontsize=8)
    axs[1].set_title("GCV uses the data without revealing truth")
    _style(axs[1], "Penalty strength α", "Generalized cross-validation score")
    fig.suptitle("Parameter choice is a statistical decision, not a numerical afterthought")
    save(fig, FIGURE_NAMES[4])

    # Figure 6: posterior intervals and fixed-truth repeated-noise coverage.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].fill_between(
        study.grid,
        study.posterior_mean - 1.96 * study.posterior_sd,
        study.posterior_mean + 1.96 * study.posterior_sd,
        color="#7cb1ba", alpha=0.4, label="Nominal 95% Bayesian intervals",
    )
    axs[0].plot(study.grid, study.posterior_mean, label="Posterior mean", lw=1.7)
    axs[0].plot(study.grid, study.truth, color="black", lw=1.5, label="Fixed synthetic truth")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Prior and likelihood determine posterior uncertainty")
    _style(axs[0], "Position", "Signal")
    axs[1].plot(study.grid, study.pointwise_coverage, "-o", markersize=2.2)
    axs[1].axhline(0.95, color="#ac4f3d", ls="--", label="Nominal 95% level")
    axs[1].set_ylim(-0.02, 1.04)
    axs[1].legend(fontsize=8)
    axs[1].set_title("Fixed-truth repeated-noise coverage need not be 95%")
    _style(axs[1], "Position", "Empirical interval coverage")
    fig.suptitle("Posterior probability and frequentist coverage are different claims")
    save(fig, FIGURE_NAMES[5])

    # Figure 7: exactly equal budgets, different spatial designs.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    palette = {"Clustered": "#b94c49", "Spread": "#3a7793", "Greedy A-opt": "#3e8776"}
    for level, (name, indices) in enumerate(study.sensor_designs.items()):
        axs[0].scatter(study.grid[indices], np.full(len(indices), level), color=palette[name], s=35)
    axs[0].set_yticks([0, 1, 2], list(study.sensor_designs.keys()))
    axs[0].set_ylim(-0.5, 2.5)
    axs[0].set_title("Twenty measured locations per design")
    _style(axs[0], "Position", "Design")
    budgets = np.arange(1, SENSOR_BUDGET + 1)
    for name, trace in study.design_trace.items():
        axs[1].plot(budgets, trace, "-o", markersize=2.4, label=name, color=palette[name])
    axs[1].legend(fontsize=8)
    axs[1].set_title("Posterior uncertainty depends on where we observe")
    _style(axs[1], "Cumulative number of sensors", "Posterior covariance trace")
    fig.suptitle("Sensor placement modifies the inverse problem before estimation")
    save(fig, FIGURE_NAMES[6])

    # Figure 8: same 20-sensor budget, paired noise draws, two kinds of risk.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    design_names = list(study.sensor_trial_errors)
    latent = [study.sensor_trial_errors[name][0] for name in design_names]
    forward = [study.sensor_trial_errors[name][1] for name in design_names]
    for ax, values, title, label in (
        (axs[0], latent, "Reconstruction uncertainty varies by design", "Latent RMSE"),
        (axs[1], forward, "Forward predictions can look better than recovery", "Full-grid forward RMSE"),
    ):
        bp = ax.boxplot(values, tick_labels=design_names, patch_artist=True, showfliers=False)
        for box, color in zip(bp["boxes"], ["#c98581", "#7facc1", "#78b6a1"], strict=True):
            box.set_facecolor(color)
            box.set_alpha(0.75)
        ax.set_title(title)
        _style(ax, "Twenty-sensor design", label)
    fig.suptitle("Repeated measurements quantify uncertainty beyond one successful realization")
    save(fig, FIGURE_NAMES[7])


def write_results(study: Study, directory: Path, *, quick: bool = False) -> None:
    """Write ten compact CSVs, keeping quick and publication results separate."""

    if type(quick) is not bool:
        raise TypeError("quick must be bool")
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in study.result_rows.items():
        if name not in RESULT_NAMES or not data:
            raise ValueError("unexpected or empty result table")
        target = directory / name.replace(".csv", "_quick.csv" if quick else ".csv")
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)


def main() -> None:
    """Run full figures and CSV generation from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Fewer Monte Carlo replicates")
    args = parser.parse_args()
    study = run_study(quick=bool(args.quick))
    folder = Path(__file__).resolve().parent
    plot_figures(study, folder / "figures", quick=bool(args.quick))
    write_results(study, folder / "results", quick=bool(args.quick))
    print(f"seed: {SEED}")
    print(f"Monte Carlo replications: {int(study.summary['mc_size'])}")
    for name, value in study.summary.items():
        if name != "mc_size":
            print(f"{name}: {value:.8g}")


if __name__ == "__main__":
    main()
