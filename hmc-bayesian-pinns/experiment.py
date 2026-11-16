"""Reproducible Bayesian inverse-diffusion benchmark for an HMC/PINN essay.

The forward solution is analytic, not a trained PINN.  Keeping forward error
zero isolates posterior geometry and sampler behaviour.  The essay explicitly
distinguishes this controlled benchmark from full neural-weight B-PINNs.

Requires Python >=3.10, NumPy, SciPy and Matplotlib.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]
Design = Literal["narrow", "wide"]

TRUE_AMPLITUDE = 1.15
TRUE_DIFFUSIVITY = 0.26
OBSERVATION_SIGMA = 0.035
PRIOR_MEAN = np.log(np.array([1.0, 0.22], dtype=np.float64))
PRIOR_SD = np.array([0.48, 0.70], dtype=np.float64)
PARAMETER_LABELS = ("log amplitude", "log diffusivity")


@dataclass(frozen=True)
class Observations:
    """Noisy measurements at space/time coordinates under a known noise law."""

    x: FloatArray
    t: FloatArray
    y: FloatArray
    sigma: float = OBSERVATION_SIGMA

    def __post_init__(self) -> None:
        if self.x.shape != self.t.shape or self.t.shape != self.y.shape:
            raise ValueError("x, t and y must have identical shapes")
        if self.x.ndim != 1 or len(self.x) < 2 or self.sigma <= 0:
            raise ValueError("Require at least two 1D observations and sigma > 0")


def solution(x: FloatArray, t: FloatArray, amplitude: FloatArray, diffusivity: FloatArray) -> FloatArray:
    """Exact first eigenmode of u_t = kappa u_xx with zero Dirichlet BCs."""
    return amplitude * np.sin(np.pi * x) * np.exp(-diffusivity * np.pi**2 * t)


def simulate(seed: int, design: Design) -> Observations:
    """Generate the same spatial locations with either weak or strong time spread."""
    if design not in ("narrow", "wide"):
        raise ValueError("design must be 'narrow' or 'wide'")
    rng = np.random.default_rng(seed)
    n: int = 24
    x = np.linspace(0.18, 0.82, n, dtype=np.float64)
    if design == "narrow":
        t = np.linspace(0.218, 0.262, n, dtype=np.float64)
    else:
        t = np.linspace(0.06, 0.57, n, dtype=np.float64)
    # Avoid a spurious space/time ordering by permuting the time design.
    t = rng.permutation(t)
    y_true = solution(x, t, TRUE_AMPLITUDE, TRUE_DIFFUSIVITY)
    y = y_true + rng.normal(0.0, OBSERVATION_SIGMA, size=n)
    return Observations(x=x, t=t, y=y)


class Posterior:
    """Gaussian-observation posterior in (log amplitude, log diffusivity)."""

    def __init__(self, data: Observations):
        if not isinstance(data, Observations):
            raise TypeError("data must be Observations")
        self.data = data
        self.sin_x: FloatArray = np.sin(np.pi * data.x)

    def potential(self, q: FloatArray) -> FloatArray:
        """Negative log posterior, excluding parameter-independent constants."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape[-1] != 2:
            raise ValueError("Last dimension of q must equal 2")
        amp = np.exp(q[..., 0])[..., None]
        kap = np.exp(q[..., 1])[..., None]
        pred = amp * self.sin_x * np.exp(-kap * np.pi**2 * self.data.t)
        residual = (pred - self.data.y) / self.data.sigma
        prior = (q - PRIOR_MEAN) / PRIOR_SD
        return 0.5 * (np.sum(residual**2, axis=-1) + np.sum(prior**2, axis=-1))

    def gradient(self, q: FloatArray) -> FloatArray:
        """Analytic gradient of the negative log posterior (batch aware)."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape[-1] != 2:
            raise ValueError("Last dimension of q must equal 2")
        kap = np.exp(q[..., 1])[..., None]
        pred = np.exp(q[..., 0])[..., None] * self.sin_x * np.exp(-kap * np.pi**2 * self.data.t)
        scaled_error = (pred - self.data.y) / self.data.sigma**2
        d_amp = np.sum(scaled_error * pred, axis=-1)
        d_kap = np.sum(scaled_error * (-kap * np.pi**2 * self.data.t * pred), axis=-1)
        return np.stack((d_amp, d_kap), axis=-1) + (q - PRIOR_MEAN) / PRIOR_SD**2

    def map_estimate(self) -> FloatArray:
        """Return a posterior mode in log coordinates."""
        opt = minimize(
            fun=lambda q: float(self.potential(q)),
            x0=PRIOR_MEAN.copy(),
            jac=self.gradient,
            method="BFGS",
            options={"gtol": 1e-9, "maxiter": 1200},
        )
        if not np.isfinite(opt.fun) or np.linalg.norm(self.gradient(opt.x)) > 1e-4:
            raise RuntimeError("MAP optimisation did not reach a stationary point")
        return np.asarray(opt.x, dtype=np.float64)

    def local_hessian(self, q: FloatArray, step: float = 1e-4) -> FloatArray:
        """Central-difference Hessian of the analytic gradient."""
        h = np.empty((2, 2), dtype=np.float64)
        for axis in range(2):
            shift = np.zeros(2, dtype=np.float64)
            shift[axis] = step
            h[:, axis] = (self.gradient(q + shift) - self.gradient(q - shift)) / (2 * step)
        return (h + h.T) / 2


@dataclass
class GridPosterior:
    """Deterministic two-dimensional numerical integration reference."""

    points: FloatArray
    probabilities: FloatArray
    shape: tuple[int, int]
    axis_amplitude: FloatArray
    axis_diffusivity: FloatArray
    boundary_mass: float

    def expectation(self, values: FloatArray) -> FloatArray:
        """Calculate posterior expectation on the quadrature grid."""
        return np.einsum("i,i...->...", self.probabilities, values)

    def summary(self) -> dict[str, object]:
        """Return moments, correlation and central quantiles on physical scale."""
        actual = np.exp(self.points)
        mean = self.expectation(actual)
        delta = actual - mean
        cov = np.einsum("i,ij,ik->jk", self.probabilities, delta, delta)
        return {
            "mean_amplitude": float(mean[0]),
            "mean_diffusivity": float(mean[1]),
            "sd_diffusivity": float(np.sqrt(cov[1, 1])),
            "correlation": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
            "interval90_diffusivity": [float(v) for v in self.quantiles(actual[:, 1], (0.05, 0.95))],
        }

    def quantiles(self, values: FloatArray, levels: tuple[float, ...]) -> FloatArray:
        """Posterior weighted quantiles, requiring an equal-length values vector."""
        if values.shape != self.probabilities.shape:
            raise ValueError("values must have one entry per grid point")
        order = np.argsort(values)
        cum = np.cumsum(self.probabilities[order])
        return np.interp(np.asarray(levels), cum, values[order])

    def draws(self, rng: np.random.Generator, number: int) -> FloatArray:
        """Sample from the quadrature reference for plotting only."""
        return self.points[rng.choice(len(self.probabilities), number, p=self.probabilities)]


def posterior_grid(posterior: Posterior, resolution: int = 480) -> GridPosterior:
    """Integrate over a fixed ±5 prior SD domain without stochastic quadrature."""
    if resolution < 40:
        raise ValueError("resolution must be at least 40")
    axes = [
        np.linspace(PRIOR_MEAN[j] - 5 * PRIOR_SD[j], PRIOR_MEAN[j] + 5 * PRIOR_SD[j], resolution)
        for j in range(2)
    ]
    a, k = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((a.ravel(), k.ravel()))
    potential = posterior.potential(points)
    weights = np.exp(-potential - logsumexp(-potential))
    weights = weights / np.sum(weights)
    grid_w = weights.reshape(resolution, resolution)
    mass_at_edges = float(
        np.sum(grid_w[0, :]) + np.sum(grid_w[-1, :])
        + np.sum(grid_w[:, 0]) + np.sum(grid_w[:, -1])
    )
    return GridPosterior(
        points=points,
        probabilities=weights,
        shape=(resolution, resolution),
        axis_amplitude=axes[0],
        axis_diffusivity=axes[1],
        boundary_mass=mass_at_edges,
    )


@dataclass
class MCMCResult:
    """Four-chain samples and chain-level HMC diagnostics."""

    draws: FloatArray  # (chains, draws, 2), log parameters
    acceptance: FloatArray
    step_sizes: FloatArray
    gradients: int
    seconds: float

    def summary(self) -> dict[str, object]:
        """Report effective sample size, Rhat, and computational work."""
        k_samples = np.exp(self.draws[..., 1])
        ess_k = effective_sample_size(k_samples)
        ess_a = effective_sample_size(np.exp(self.draws[..., 0]))
        return {
            "acceptance": float(np.mean(self.acceptance)),
            "step_size_mean": float(np.mean(self.step_sizes)),
            "ess_diffusivity": float(ess_k),
            "ess_amplitude": float(ess_a),
            "ess_per_1000_gradients": float(1000 * ess_k / self.gradients),
            "rhat_diffusivity": float(rhat(k_samples)),
            "draws": int(k_samples.size),
            "gradient_evaluations": int(self.gradients),
            "seconds": float(self.seconds),
            "mean_diffusivity": float(np.mean(k_samples)),
            "interval90_diffusivity": [float(v) for v in np.quantile(k_samples, (0.05, 0.95))],
        }


def hmc(
    posterior: Posterior,
    q_start: FloatArray,
    *,
    mass: FloatArray,
    seed: int,
    warmup: int = 1100,
    draws: int = 1500,
    chains: int = 4,
    target_accept: float = 0.78,
) -> MCMCResult:
    """Fixed-mass leapfrog HMC with Robbins–Monro step-size warm-up.

    Path lengths are jittered to reduce resonance. Mass is fixed for the
    entire run; tuning the step size is limited to warm-up, never sampling.
    """
    mass = np.asarray(mass, dtype=np.float64)
    if mass.shape != (2, 2) or np.min(np.linalg.eigvalsh(mass)) <= 0:
        raise ValueError("mass must be a symmetric positive definite 2x2 matrix")
    inverse_mass = np.linalg.inv(mass)
    chol = np.linalg.cholesky(mass)
    rng = np.random.default_rng(seed)
    samples = np.empty((chains, draws, 2), dtype=np.float64)
    accept_rates = np.empty(chains, dtype=np.float64)
    step_sizes = np.empty(chains, dtype=np.float64)
    gradient_count = 0
    start_clock = time.perf_counter()

    for chain in range(chains):
        q = q_start + rng.normal(size=2) * np.array([0.08, 0.13])
        initial_eps = 0.026 if np.allclose(mass, np.eye(2)) else 0.14
        eps = initial_eps
        accept_total = 0.0
        for iteration in range(warmup + draws):
            momentum = chol @ rng.standard_normal(2)
            original_energy = float(posterior.potential(q) + 0.5 * momentum @ inverse_mass @ momentum)
            q_next = q.copy()
            p_next = momentum.copy()
            leapfrogs = int(rng.integers(9, 22))
            grad = posterior.gradient(q_next)
            gradient_count += 1
            p_next -= 0.5 * eps * grad
            for step in range(leapfrogs):
                q_next += eps * (inverse_mass @ p_next)
                grad = posterior.gradient(q_next)
                gradient_count += 1
                if step + 1 < leapfrogs:
                    p_next -= eps * grad
            p_next -= 0.5 * eps * grad
            new_energy = float(posterior.potential(q_next) + 0.5 * p_next @ inverse_mass @ p_next)
            delta = original_energy - new_energy
            probability = min(1.0, float(np.exp(min(0.0, delta)))) if np.isfinite(delta) else 0.0
            accepted = float(rng.random() < probability)
            if accepted:
                q = q_next
            if iteration < warmup:
                gain = 0.11 / (1 + iteration / 120) ** 0.55
                eps = float(np.clip(eps * np.exp(gain * (probability - target_accept)), 0.0003, 0.9))
            else:
                samples[chain, iteration - warmup] = q
                accept_total += accepted
        accept_rates[chain] = accept_total / draws
        step_sizes[chain] = eps

    return MCMCResult(
        draws=samples,
        acceptance=accept_rates,
        step_sizes=step_sizes,
        gradients=gradient_count,
        seconds=time.perf_counter() - start_clock,
    )


def rhat(chains: FloatArray) -> float:
    """Classical split-Rhat, used together with ESS for a basic chain check."""
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2 or chains.shape[1] < 20:
        raise ValueError("chains must have shape (chains, at least 20 draws)")
    length = chains.shape[1] // 2
    split = np.concatenate((chains[:, :length], chains[:, -length:]), axis=0)
    between = length * np.var(split.mean(axis=1), ddof=1)
    within = np.mean(np.var(split, axis=1, ddof=1))
    return float(np.sqrt(((length - 1) * within / length + between / length) / within))


def effective_sample_size(chains: FloatArray) -> float:
    """FFT autocorrelations with the Geyer initial-positive-pair cutoff."""
    arr = np.asarray(chains, dtype=np.float64)
    n_chains, n = arr.shape
    centered = arr - arr.mean(axis=1, keepdims=True)
    length = 1 << (2 * n - 1).bit_length()
    transforms = np.fft.rfft(centered, n=length, axis=1)
    correlation = np.fft.irfft(transforms * np.conj(transforms), n=length, axis=1)[:, :n]
    normalized = correlation / np.maximum(correlation[:, 0, None], 1e-14)
    rho = np.mean(normalized, axis=0)
    tau = 1.0
    for lag in range(1, n - 1, 2):
        pair_sum = rho[lag] + rho[lag + 1]
        if pair_sum <= 0:
            break
        tau += 2 * pair_sum
    return float(min(n_chains * n, n_chains * n / max(tau, 1)))


@dataclass
class VariationalResult:
    """Mean-field Gaussian approximation in unconstrained log coordinates."""

    mean: FloatArray
    sd: FloatArray
    objective: float

    def summary(self, rng: np.random.Generator) -> dict[str, object]:
        """Monte Carlo summaries on physical scale."""
        z = rng.normal(size=(100_000, 2)) * self.sd + self.mean
        k = np.exp(z[:, 1])
        return {
            "mean_diffusivity": float(np.mean(k)),
            "interval90_diffusivity": [float(v) for v in np.quantile(k, (0.05, 0.95))],
            "sd_log_amplitude": float(self.sd[0]),
            "sd_log_diffusivity": float(self.sd[1]),
            "negative_elbo_without_constant": float(self.objective),
        }


def mean_field_variational(posterior: Posterior, mode: FloatArray) -> VariationalResult:
    """Optimise reverse KL(q || p) with deterministic Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    yy, xx = np.meshgrid(nodes, nodes, indexing="ij")
    quadrature = np.column_stack((yy.ravel(), xx.ravel())) * np.sqrt(2)
    integration_weights = np.outer(weights, weights).ravel() / np.pi

    def objective_and_grad(params: FloatArray) -> tuple[float, FloatArray]:
        mean = params[:2]
        sd = np.exp(params[2:])
        deviations = quadrature * sd
        points = mean + deviations
        potentials = posterior.potential(points)
        gradients = posterior.gradient(points)
        val = float(integration_weights @ potentials - np.sum(params[2:]))
        grad_mean = np.einsum("i,ij->j", integration_weights, gradients)
        grad_scale = np.einsum("i,ij,ij->j", integration_weights, gradients, deviations) - 1
        return val, np.concatenate((grad_mean, grad_scale))

    hess = posterior.local_hessian(mode)
    initial_sd = 1 / np.sqrt(np.maximum(np.diag(hess), 1e-4))
    params = np.concatenate((mode, np.log(initial_sd)))
    result = minimize(
        objective_and_grad,
        params,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0), (-5.5, 1.5), (-8.0, 0.8), (-8.0, 0.8)],
        options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success and np.linalg.norm(result.jac) > 5e-5:
        raise RuntimeError(f"VI minimisation failed: {result.message}")
    return VariationalResult(result.x[:2], np.exp(result.x[2:]), float(result.fun))


def coverages(seed: int, designs: tuple[Design, ...], replications: int = 100) -> dict[str, object]:
    """Frequentist repeated-data coverage of exact-grid Bayesian intervals.

    This checks only the reduced known-forward model. It does not validate
    uncertainty calibration for an approximate neural surrogate.
    """
    answer: dict[str, object] = {}
    for design in designs:
        count_k = 0
        count_u = 0
        for replicate in range(replications):
            observations = simulate(seed + replicate * 1009, design)
            grid = posterior_grid(Posterior(observations), resolution=230)
            kappas = np.exp(grid.points[:, 1])
            bounds_k = grid.quantiles(kappas, (0.05, 0.95))
            count_k += bool(bounds_k[0] <= TRUE_DIFFUSIVITY <= bounds_k[1])
            predicted = solution(np.array(0.5), np.array(0.55), np.exp(grid.points[:, 0]), kappas)
            bounds_u = grid.quantiles(predicted, (0.05, 0.95))
            target = float(solution(np.array(0.5), np.array(0.55), TRUE_AMPLITUDE, TRUE_DIFFUSIVITY))
            count_u += bool(bounds_u[0] <= target <= bounds_u[1])
        answer[design] = {
            "replications": replications,
            "nominal_coverage": 0.9,
            "parameter_coverage": count_k / replications,
            "solution_coverage": count_u / replications,
            "binomial_standard_error_at_nominal": float(np.sqrt(0.9 * 0.1 / replications)),
        }
    return answer


def plot_geometry(grid: GridPosterior, hmc_result: MCMCResult, vi: VariationalResult, mode: FloatArray, path: Path) -> None:
    """Posterior density with sampled trajectory positions and VI ellipse."""
    log_density = np.log(np.maximum(grid.probabilities.reshape(grid.shape), 1e-300))
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    ax.contourf(np.exp(grid.axis_amplitude), np.exp(grid.axis_diffusivity),
                np.exp(log_density.T - np.max(log_density)), 22, cmap="Blues")
    xx = np.exp(hmc_result.draws[0, ::22, 0])
    yy = np.exp(hmc_result.draws[0, ::22, 1])
    ax.scatter(xx, yy, s=9, c="#f3a43b", alpha=0.50, label="HMC retained positions")
    ax.scatter([np.exp(mode[0])], [np.exp(mode[1])], marker="x", s=100,
               color="#cc4b35", linewidth=2, label="MAP")
    ax.scatter([TRUE_AMPLITUDE], [TRUE_DIFFUSIVITY], marker="*", s=180,
               color="#19191d", label="Data-generating parameters")
    angles = np.linspace(0, 2 * np.pi, 300)
    xvi = np.exp(vi.mean[0] + 2 * vi.sd[0] * np.cos(angles))
    yvi = np.exp(vi.mean[1] + 2 * vi.sd[1] * np.sin(angles))
    ax.plot(xvi, yvi, color="#de6333", linewidth=2, label="Mean-field VI: 2-SD contour")
    ax.set(xlabel="Initial amplitude A", ylabel="Diffusivity κ",
           title="Sparse times produce an amplitude–diffusivity ridge")
    ax.set_xlim(0.65, 1.9)
    ax.set_ylim(0.055, 0.53)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_prediction(grid: GridPosterior, vi: VariationalResult, data: Observations, path: Path) -> None:
    """Posterior prediction bands with exact-known-forward model uncertainty."""
    rng = np.random.default_rng(103)
    t = np.linspace(0, 0.8, 140)
    x = np.full_like(t, 0.5)
    samples = grid.draws(rng, 12000)
    pred = solution(x[None, :], t[None, :], np.exp(samples[:, 0, None]), np.exp(samples[:, 1, None]))
    vi_draws = rng.normal(size=(12000, 2)) * vi.sd + vi.mean
    vi_pred = solution(x[None, :], t[None, :], np.exp(vi_draws[:, 0, None]), np.exp(vi_draws[:, 1, None]))
    low, median, high = np.quantile(pred, (0.05, 0.5, 0.95), axis=0)
    vi_lo, vi_hi = np.quantile(vi_pred, (0.05, 0.95), axis=0)
    true = solution(x, t, TRUE_AMPLITUDE, TRUE_DIFFUSIVITY)
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    ax.fill_between(t, low, high, color="#3279ad", alpha=0.26, label="Exact posterior 90% band")
    ax.plot(t, median, color="#25628c", linewidth=2, label="Posterior median")
    ax.plot(t, true, color="#171c22", ls="--", linewidth=2, label="True solution")
    ax.plot(t, vi_lo, color="#c75536", ls=":", linewidth=1.5, label="Mean-field VI 90% bounds")
    ax.plot(t, vi_hi, color="#c75536", ls=":", linewidth=1.5)
    y_scaled = data.y / np.sin(np.pi * data.x)
    ax.scatter(data.t, y_scaled, color="#555", alpha=0.6, s=24,
               label="Observed amplitudes y / sin(πx)")
    ax.axvspan(float(min(data.t)), float(max(data.t)), color="#d5b389", alpha=0.14)
    ax.set(xlabel="Time t", ylabel="u(0.5, t)",
           title="Uncertainty expands away from the observed time window")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_designs(grids: dict[Design, GridPosterior], path: Path) -> None:
    """Show geometry as measurement times are spread apart."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, (design, grid) in zip(axes, grids.items(), strict=True):
        density = grid.probabilities.reshape(grid.shape)
        ax.contourf(np.exp(grid.axis_amplitude), np.exp(grid.axis_diffusivity),
                    density.T / density.max(), 18, cmap="Blues")
        ax.scatter([TRUE_AMPLITUDE], [TRUE_DIFFUSIVITY], c="black", marker="*", s=110)
        correlation = grid.summary()["correlation"]
        ax.set(title=f"{design.capitalize()} time design\nposterior corr = {correlation:+.2f}",
               xlabel="Amplitude A", xlim=(0.65, 1.9), ylim=(0.055, 0.53))
    axes[0].set_ylabel("Diffusivity κ")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_mcmc(plain: MCMCResult, conditioned: MCMCResult, path: Path) -> None:
    """Display first-chain traces and autocorrelations for two mass matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), constrained_layout=True)
    for j, (name, result, color) in enumerate(
        (("Identity mass", plain, "#b2632f"), ("MAP-Hessian mass", conditioned, "#24618c"))
    ):
        series = np.exp(result.draws[0, :, 1])
        axes[0, j].plot(series, color=color, linewidth=0.65)
        axes[0, j].axhline(TRUE_DIFFUSIVITY, color="black", ls="--")
        axes[0, j].set(title=name, xlabel="Retained iteration", ylabel="κ trace")
        centered = series - np.mean(series)
        auto = np.correlate(centered, centered, mode="full")[len(series) - 1:]
        auto = auto[:70] / auto[0]
        axes[1, j].plot(np.arange(70), auto, color=color)
        axes[1, j].axhline(0, color="#444", linewidth=0.6)
        axes[1, j].set(xlabel="Lag", ylabel="Autocorrelation", ylim=(-0.2, 1.02))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_coverage(results: dict[str, object], path: Path) -> None:
    """Compare empirical coverage with the nominal target and sampling error."""
    names = ["Narrow: κ", "Narrow: u", "Wide: κ", "Wide: u"]
    values = [
        results["narrow"]["parameter_coverage"], results["narrow"]["solution_coverage"],
        results["wide"]["parameter_coverage"], results["wide"]["solution_coverage"],
    ]
    n = results["narrow"]["replications"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.bar(names, values, color=["#3279ad", "#6fa6be", "#3279ad", "#6fa6be"], width=0.62)
    ax.axhline(0.90, color="#af552e", ls="--", label="Nominal 90%")
    ax.axhspan(0.90 - 1.96 * np.sqrt(0.09 / n), 0.90 + 1.96 * np.sqrt(0.09 / n),
               color="#e9cc9e", alpha=0.30, label="Approx. ±1.96 binomial SE")
    ax.set(ylim=(0.65, 1.0), ylabel="Proportion of repeated data sets covered",
           title=f"Exact-reference interval coverage ({n} synthetic data sets per design)")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def run(output: Path, *, seed: int = 20261116, replications: int = 100) -> dict[str, object]:
    """Run reference integration, VI, two HMC variants, and coverage audit."""
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    observed = {d: simulate(seed, d) for d in ("narrow", "wide")}
    posteriors = {d: Posterior(observed[d]) for d in observed}
    grids = {d: posterior_grid(posteriors[d]) for d in observed}
    if any(g.boundary_mass > 1e-7 for g in grids.values()):
        raise RuntimeError("Reference grid truncates material posterior probability")

    narrow = posteriors["narrow"]
    mode = narrow.map_estimate()
    vi = mean_field_variational(narrow, mode)
    hess = narrow.local_hessian(mode)
    if np.min(np.linalg.eigvalsh(hess)) <= 0:
        raise RuntimeError("Local Hessian is not positive definite")

    plain = hmc(narrow, mode, mass=np.eye(2), seed=seed + 11)
    condition = hmc(narrow, mode, mass=hess, seed=seed + 21)
    coverage = coverages(seed + 30000, ("narrow", "wide"), replications=replications)

    plot_geometry(grids["narrow"], condition, vi, mode, figures / "posterior_geometry.png")
    plot_prediction(grids["narrow"], vi, observed["narrow"], figures / "predictive_uncertainty.png")
    plot_designs(grids, figures / "observation_geometry.png")
    plot_mcmc(plain, condition, figures / "sampler_diagnostics.png")
    plot_coverage(coverage, figures / "repeated_coverage.png")

    metrics: dict[str, object] = {
        "seed": seed,
        "model": "Exact one-mode heat equation; not a trained PINN",
        "true_parameters": {"amplitude": TRUE_AMPLITUDE, "diffusivity": TRUE_DIFFUSIVITY},
        "sample_size_each_design": 24,
        "noise_sigma": OBSERVATION_SIGMA,
        "prior_mean_log": PRIOR_MEAN.tolist(),
        "prior_sd_log": PRIOR_SD.tolist(),
        "reference": {k: v.summary() | {"boundary_mass": v.boundary_mass} for k, v in grids.items()},
        "map": {"amplitude": float(np.exp(mode[0])), "diffusivity": float(np.exp(mode[1]))},
        "vi": vi.summary(np.random.default_rng(seed + 37)),
        "hmc_identity": plain.summary(),
        "hmc_map_hessian": condition.summary(),
        "coverage": coverage,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--replications", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.output, seed=args.seed, replications=args.replications), indent=2))


if __name__ == "__main__":
    main()
