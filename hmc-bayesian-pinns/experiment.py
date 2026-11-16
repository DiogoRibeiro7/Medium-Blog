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

import diagnostics

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


def simulate(
    seed: int,
    design: Design,
    *,
    amplitude: float = TRUE_AMPLITUDE,
    diffusivity: float = TRUE_DIFFUSIVITY,
) -> Observations:
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
    y_true = solution(x, t, amplitude, diffusivity)
    y = y_true + rng.normal(0.0, OBSERVATION_SIGMA, size=n)
    return Observations(x=x, t=t, y=y)


def simulate_at(
    seed: int,
    x: FloatArray,
    t: FloatArray,
    *,
    amplitude: float = TRUE_AMPLITUDE,
    diffusivity: float = TRUE_DIFFUSIVITY,
) -> Observations:
    """Noisy observations at an explicit (x, t) design."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y = solution(x, t, amplitude, diffusivity) + rng.normal(0.0, OBSERVATION_SIGMA, size=x.size)
    return Observations(x=x, t=t, y=y)


class Posterior:
    """Gaussian-observation posterior in (log amplitude, log diffusivity)."""

    def __init__(
        self,
        data: Observations,
        prior_mean: FloatArray = PRIOR_MEAN,
        prior_sd: FloatArray = PRIOR_SD,
    ):
        if not isinstance(data, Observations):
            raise TypeError("data must be Observations")
        self.data = data
        self.sin_x: FloatArray = np.sin(np.pi * data.x)
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        self.prior_sd = np.asarray(prior_sd, dtype=np.float64)
        if self.prior_mean.shape != (2,) or self.prior_sd.shape != (2,) or np.any(self.prior_sd <= 0):
            raise ValueError("prior_mean and prior_sd must be length-2 with positive SDs")

    def potential(self, q: FloatArray) -> FloatArray:
        """Negative log posterior, excluding parameter-independent constants."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape[-1] != 2:
            raise ValueError("Last dimension of q must equal 2")
        amp = np.exp(q[..., 0])[..., None]
        kap = np.exp(q[..., 1])[..., None]
        pred = amp * self.sin_x * np.exp(-kap * np.pi**2 * self.data.t)
        residual = (pred - self.data.y) / self.data.sigma
        prior = (q - self.prior_mean) / self.prior_sd
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
        return np.stack((d_amp, d_kap), axis=-1) + (q - self.prior_mean) / self.prior_sd**2

    def map_estimate(self) -> FloatArray:
        """Return a posterior mode in log coordinates."""
        opt = minimize(
            fun=lambda q: float(self.potential(q)),
            x0=self.prior_mean.copy(),
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
            "interval90_diffusivity": [float(v) for v in self.kappa_quantiles((0.05, 0.95))],
        }

    def quantiles(self, values: FloatArray, levels: tuple[float, ...]) -> FloatArray:
        """Posterior weighted quantiles, requiring an equal-length values vector."""
        if values.shape != self.probabilities.shape:
            raise ValueError("values must have one entry per grid point")
        order = np.argsort(values)
        cum = np.cumsum(self.probabilities[order])
        return np.interp(np.asarray(levels), cum, values[order])

    def kappa_quantiles(self, levels: tuple[float, ...]) -> FloatArray:
        """Quantiles of kappa from the log-kappa marginal, linear CDF within each grid cell.

        Weighted quantiles of the raw grid values snap to cell centres; interpolating
        the marginal CDF removes that half-cell error.
        """
        axis, density = self.marginal_log_diffusivity()
        step = axis[1] - axis[0]
        edges = np.concatenate(([axis[0] - step / 2], axis + step / 2))
        cdf = np.concatenate(([0.0], np.cumsum(density * step)))
        return np.exp(np.interp(np.asarray(levels, dtype=np.float64), cdf, edges))

    def log_moments(self) -> tuple[FloatArray, FloatArray]:
        """Posterior mean and covariance of (log A, log kappa), the coordinates the samplers use."""
        mean = self.expectation(self.points)
        delta = self.points - mean
        return mean, np.einsum("i,ij,ik->jk", self.probabilities, delta, delta)

    def log_summary(self) -> dict[str, float]:
        """Moments in the log coordinates, where HMC and VI operate."""
        mean, cov = self.log_moments()
        return {
            "mean_log_amplitude": float(mean[0]),
            "mean_log_diffusivity": float(mean[1]),
            "sd_log_amplitude": float(np.sqrt(cov[0, 0])),
            "sd_log_diffusivity": float(np.sqrt(cov[1, 1])),
            "correlation_log": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
        }

    def marginal_log_diffusivity(self) -> tuple[FloatArray, FloatArray]:
        """Marginal density of log kappa on the grid axis (integrates to one)."""
        mass = self.probabilities.reshape(self.shape).sum(axis=0)
        step = self.axis_diffusivity[1] - self.axis_diffusivity[0]
        return self.axis_diffusivity, mass / step

    def draws(self, rng: np.random.Generator, number: int) -> FloatArray:
        """Sample from the quadrature reference for plotting only."""
        return self.points[rng.choice(len(self.probabilities), number, p=self.probabilities)]


def posterior_grid(
    posterior: Posterior,
    resolution: int | tuple[int, int] = 480,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> GridPosterior:
    """Integrate over a fixed domain (default ±5 prior SD) without stochastic quadrature."""
    sizes = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
    if min(sizes) < 40:
        raise ValueError("resolution must be at least 40")
    if bounds is None:
        bounds = tuple(
            (posterior.prior_mean[j] - 5 * posterior.prior_sd[j], posterior.prior_mean[j] + 5 * posterior.prior_sd[j])
            for j in range(2)
        )
    axes = [np.linspace(bounds[j][0], bounds[j][1], sizes[j]) for j in range(2)]
    a, k = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((a.ravel(), k.ravel()))
    potential = posterior.potential(points)
    weights = np.exp(-potential - logsumexp(-potential))
    weights = weights / np.sum(weights)
    grid_w = weights.reshape(sizes)
    mass_at_edges = float(
        np.sum(grid_w[0, :]) + np.sum(grid_w[-1, :])
        + np.sum(grid_w[:, 0]) + np.sum(grid_w[:, -1])
    )
    return GridPosterior(
        points=points,
        probabilities=weights,
        shape=(int(sizes[0]), int(sizes[1])),
        axis_amplitude=axes[0],
        axis_diffusivity=axes[1],
        boundary_mass=mass_at_edges,
    )


def adaptive_grid(posterior: Posterior, resolution: int = 230, width: float = 10.0) -> GridPosterior:
    """Grid centred on the MAP, ±width Laplace SDs per axis, widened until the edges hold < 1e-8.

    A domain fixed by the prior is far wider than a well-identified posterior,
    which then falls on only a few grid cells; centring on the posterior keeps
    the spacing near a tenth of a posterior SD whatever the design.
    """
    try:
        mode = posterior.map_estimate()
        sd = np.sqrt(np.diag(np.linalg.inv(posterior.local_hessian(mode))))
        if not np.all(np.isfinite(sd)):
            raise RuntimeError("non-finite Laplace scale")
    except (RuntimeError, np.linalg.LinAlgError):
        return posterior_grid(posterior, resolution=max(resolution, 480))
    for _ in range(6):
        bounds = tuple((mode[j] - width * sd[j], mode[j] + width * sd[j]) for j in range(2))
        grid = posterior_grid(posterior, resolution=resolution, bounds=bounds)
        if grid.boundary_mass < 1e-8:
            return grid
        width *= 1.6
    raise RuntimeError("adaptive grid could not contain the posterior")


@dataclass
class MCMCResult:
    """Four-chain samples and chain-level HMC diagnostics."""

    draws: FloatArray  # (chains, draws, 2), log parameters
    acceptance: FloatArray
    step_sizes: FloatArray
    gradients: int
    seconds: float
    accept_probability: FloatArray | None = None  # (chains, draws) Metropolis probability per transition
    start_states: FloatArray | None = None  # (chains, draws, 2) state at which each transition began
    divergences: int = 0  # sampling transitions with energy error > 1000 or non-finite

    def summary(self) -> dict[str, object]:
        """Report effective sample size, Rhat, and computational work."""
        k_samples = np.exp(self.draws[..., 1])
        ess_k = effective_sample_size(k_samples)
        ess_a = effective_sample_size(np.exp(self.draws[..., 0]))
        bulk = diagnostics.ess_bulk(k_samples)
        tail = diagnostics.ess_tail(k_samples)
        return {
            "acceptance": float(np.mean(self.acceptance)),
            "step_size_mean": float(np.mean(self.step_sizes)),
            "ess_diffusivity": float(ess_k),
            "ess_amplitude": float(ess_a),
            "ess_per_1000_gradients": float(1000 * ess_k / self.gradients),
            "rhat_diffusivity": float(rhat(k_samples)),
            "rhat_rank_diffusivity": float(diagnostics.rhat_rank(k_samples)),
            "ess_bulk_diffusivity": float(bulk),
            "ess_tail_diffusivity": float(tail),
            "ess_bulk_per_1000_gradients": float(1000 * bulk / self.gradients),
            "ess_tail_per_1000_gradients": float(1000 * tail / self.gradients),
            "mcse_q05_diffusivity": float(diagnostics.mcse_quantile(k_samples, 0.05)),
            "mcse_q95_diffusivity": float(diagnostics.mcse_quantile(k_samples, 0.95)),
            "divergences": int(self.divergences),
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
    leapfrog_range: tuple[int, int] = (9, 21),
) -> MCMCResult:
    """Fixed-mass leapfrog HMC with Robbins–Monro step-size warm-up.

    Path lengths are drawn uniformly from ``leapfrog_range`` (inclusive) to
    reduce resonance. Mass is fixed for the entire run; tuning the step size
    is limited to warm-up, never sampling. A sampling transition whose energy
    error exceeds 1000, or is not finite, is counted as divergent (the
    threshold Stan uses).
    """
    mass = np.asarray(mass, dtype=np.float64)
    if mass.shape != (2, 2) or np.min(np.linalg.eigvalsh(mass)) <= 0:
        raise ValueError("mass must be a symmetric positive definite 2x2 matrix")
    low_steps, high_steps = leapfrog_range
    if not 1 <= low_steps <= high_steps:
        raise ValueError("leapfrog_range must satisfy 1 <= low <= high")
    inverse_mass = np.linalg.inv(mass)
    chol = np.linalg.cholesky(mass)
    rng = np.random.default_rng(seed)
    samples = np.empty((chains, draws, 2), dtype=np.float64)
    starts = np.empty((chains, draws, 2), dtype=np.float64)
    probabilities = np.empty((chains, draws), dtype=np.float64)
    divergent = 0
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
            leapfrogs = int(rng.integers(low_steps, high_steps + 1))
            with np.errstate(over="ignore", invalid="ignore"):
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
            start = q
            if accepted:
                q = q_next
            if iteration < warmup:
                gain = 0.11 / (1 + iteration / 120) ** 0.55
                eps = float(np.clip(eps * np.exp(gain * (probability - target_accept)), 0.0003, 0.9))
            else:
                samples[chain, iteration - warmup] = q
                starts[chain, iteration - warmup] = start
                probabilities[chain, iteration - warmup] = probability
                divergent += int((not np.isfinite(delta)) or (-delta > 1000.0))
                accept_total += accepted
        accept_rates[chain] = accept_total / draws
        step_sizes[chain] = eps

    return MCMCResult(
        draws=samples,
        acceptance=accept_rates,
        step_sizes=step_sizes,
        gradients=gradient_count,
        seconds=time.perf_counter() - start_clock,
        accept_probability=probabilities,
        start_states=starts,
        divergences=divergent,
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


@dataclass
class FullRankResult:
    """Full-covariance Gaussian approximation q = N(mean, L L^T) in log coordinates."""

    mean: FloatArray
    chol: FloatArray
    objective: float

    @property
    def covariance(self) -> FloatArray:
        return self.chol @ self.chol.T

    def summary(self, rng: np.random.Generator) -> dict[str, object]:
        """Monte Carlo summaries on physical scale."""
        z = rng.normal(size=(100_000, 2)) @ self.chol.T + self.mean
        k = np.exp(z[:, 1])
        cov = self.covariance
        return {
            "mean_diffusivity": float(np.mean(k)),
            "interval90_diffusivity": [float(v) for v in np.quantile(k, (0.05, 0.95))],
            "sd_log_amplitude": float(np.sqrt(cov[0, 0])),
            "sd_log_diffusivity": float(np.sqrt(cov[1, 1])),
            "correlation_log": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
            "negative_elbo_without_constant": float(self.objective),
        }


GAUSS_HERMITE_24 = np.polynomial.hermite.hermgauss(24)


def _standard_normal_rule() -> tuple[FloatArray, FloatArray]:
    """24x24 Gauss–Hermite nodes and weights for a standard bivariate normal."""
    nodes, weights = GAUSS_HERMITE_24
    yy, xx = np.meshgrid(nodes, nodes, indexing="ij")
    z = np.column_stack((yy.ravel(), xx.ravel())) * np.sqrt(2)
    return z, np.outer(weights, weights).ravel() / np.pi


def _unpack_full_rank(params: FloatArray) -> tuple[FloatArray, FloatArray]:
    chol = np.array([[np.exp(params[2]), 0.0], [params[4], np.exp(params[3])]])
    return params[:2], chol


def full_rank_objective(posterior: Posterior, params: FloatArray) -> tuple[float, FloatArray]:
    """Negative ELBO (up to a constant) and its gradient for q = N(m, L L^T).

    params = (m1, m2, log L11, log L22, L21).  E_q[U] - log det L, with the
    expectation computed by the deterministic Gauss–Hermite rule.
    """
    z, w = _standard_normal_rule()
    mean, chol = _unpack_full_rank(params)
    points = mean + z @ chol.T
    with np.errstate(over="ignore", invalid="ignore"):
        potentials = posterior.potential(points)
        gradients = posterior.gradient(points)
    if not np.all(np.isfinite(potentials)) or not np.all(np.isfinite(gradients)):
        return float("inf"), np.zeros_like(params)
    value = float(w @ potentials - params[2] - params[3])
    g_mean = w @ gradients
    g_l = np.einsum("i,ij,ik->jk", w, gradients, z)  # d E[U] / d L_jk
    grad = np.array([
        g_mean[0], g_mean[1],
        g_l[0, 0] * chol[0, 0] - 1.0,
        g_l[1, 1] * chol[1, 1] - 1.0,
        g_l[1, 0],
    ])
    return value, grad


def full_rank_variational(posterior: Posterior, mode: FloatArray) -> FullRankResult:
    """Optimise reverse KL(q || p) over all bivariate Gaussians (Cholesky parameterisation).

    The expectation uses the same deterministic 24x24 Gauss–Hermite rule as the
    mean-field fit, so the two approximations differ only in their family.
    """
    hess = posterior.local_hessian(mode)
    start_chol = np.linalg.cholesky(np.linalg.inv(hess))
    params = np.array([mode[0], mode[1], np.log(start_chol[0, 0]), np.log(start_chol[1, 1]), start_chol[1, 0]])
    result = minimize(lambda p: full_rank_objective(posterior, p), params, jac=True, method="L-BFGS-B",
                      options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9})
    if not np.isfinite(result.fun):
        raise RuntimeError(f"Full-rank VI failed: {result.message}")
    mean, chol = _unpack_full_rank(result.x)
    return FullRankResult(mean, chol, float(result.fun))


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
            grid = adaptive_grid(Posterior(observations), resolution=230)
            kappas = np.exp(grid.points[:, 1])
            bounds_k = grid.kappa_quantiles((0.05, 0.95))
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


def _ellipse(mean: FloatArray, cov: FloatArray, radius: float = 2.0) -> FloatArray:
    angles = np.linspace(0, 2 * np.pi, 400)
    vals, vecs = np.linalg.eigh(cov)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    return mean + radius * circle @ (vecs * np.sqrt(vals)).T


def plot_geometry(grid: GridPosterior, hmc_result: MCMCResult, vi: VariationalResult, full: FullRankResult,
                  mode: FloatArray, path: Path) -> None:
    """Posterior density in physical and in log (sampler) coordinates, with both Gaussian approximations."""
    density = grid.probabilities.reshape(grid.shape)
    density = density / density.max()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), constrained_layout=True)
    mf_ellipse = _ellipse(vi.mean, np.diag(vi.sd**2))
    fr_ellipse = _ellipse(full.mean, full.covariance)
    pts = hmc_result.draws[0, ::22]
    for ax, transform, labels in (
        (axes[0], np.exp, ("Initial amplitude A", "Diffusivity κ")),
        (axes[1], lambda v: v, ("log A (sampler coordinate)", "log κ (sampler coordinate)")),
    ):
        ax.contourf(transform(grid.axis_amplitude), transform(grid.axis_diffusivity), density.T, 22, cmap="Blues")
        ax.scatter(transform(pts[:, 0]), transform(pts[:, 1]), s=8, c="#f3a43b", alpha=0.55,
                   label="Identity-mass HMC draws (every 22nd)")
        ax.plot(transform(fr_ellipse[:, 0]), transform(fr_ellipse[:, 1]), color="#5b8c3a", lw=1.8,
                label="Full-covariance VI: 2-SD contour")
        ax.plot(transform(mf_ellipse[:, 0]), transform(mf_ellipse[:, 1]), color="#de6333", lw=2,
                label="Mean-field VI: 2-SD contour")
        ax.scatter(transform(mode[0]), transform(mode[1]), marker="x", s=90, color="#cc4b35", lw=2, label="MAP")
        ax.scatter(transform(np.log(TRUE_AMPLITUDE)), transform(np.log(TRUE_DIFFUSIVITY)), marker="*", s=170,
                   color="#19191d", label="Data-generating parameters")
        ax.set(xlabel=labels[0], ylabel=labels[1])
    axes[0].set(xlim=(0.65, 1.9), ylim=(0.04, 0.53), title="Physical coordinates")
    axes[1].set(xlim=(-0.55, 0.65), ylim=(-3.3, -0.6), title="Log coordinates, where HMC and VI operate")
    axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
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


def plot_designs(grids: dict[str, GridPosterior], titles: dict[str, str], path: Path) -> None:
    """Posterior geometry under three time designs with the same 24 locations and noise."""
    fig, axes = plt.subplots(1, len(grids), figsize=(4.6 * len(grids), 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (name, grid) in zip(axes, grids.items(), strict=True):
        density = grid.probabilities.reshape(grid.shape)
        ax.contourf(np.exp(grid.axis_amplitude), np.exp(grid.axis_diffusivity),
                    density.T / density.max(), 18, cmap="Blues")
        ax.scatter([TRUE_AMPLITUDE], [TRUE_DIFFUSIVITY], c="black", marker="*", s=110)
        summary = grid.summary()
        lo, hi = summary["interval90_diffusivity"]
        ax.set(title=f"{titles[name]}\nκ 90%: [{lo:.3f}, {hi:.3f}], corr {summary['correlation']:+.2f}",
               xlabel="Amplitude A", xlim=(0.65, 1.9), ylim=(0.04, 0.53))
    axes[0].set_ylabel("Diffusivity κ")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def run(output: Path, *, seed: int = 20261116, replications: int = 100, quick: bool = False) -> dict[str, object]:
    """Run the reference integration, VI, HMC, identification, sampler, design and coverage studies.

    ``quick`` shrinks replications, seeds and grid sizes for CI; the committed
    results come from the full run.
    """
    import design
    import studies

    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if quick:
        replications = min(replications, 12)
    observed = {d: simulate(seed, d) for d in ("narrow", "wide")}
    posteriors = {d: Posterior(observed[d]) for d in observed}
    grids = {"narrow": posterior_grid(posteriors["narrow"]), "wide": adaptive_grid(posteriors["wide"], resolution=480)}
    if any(g.boundary_mass > 1e-7 for g in grids.values()):
        raise RuntimeError("Reference grid truncates material posterior probability")
    fine = posterior_grid(posteriors["narrow"], resolution=960)
    fine_wide = adaptive_grid(posteriors["wide"], resolution=960)
    refinement = {
        "interval90_diffusivity_480": grids["narrow"].summary()["interval90_diffusivity"],
        "interval90_diffusivity_960": fine.summary()["interval90_diffusivity"],
        "sd_diffusivity_480": grids["narrow"].summary()["sd_diffusivity"],
        "sd_diffusivity_960": fine.summary()["sd_diffusivity"],
        "wide_interval90_diffusivity_480": grids["wide"].summary()["interval90_diffusivity"],
        "wide_interval90_diffusivity_960": fine_wide.summary()["interval90_diffusivity"],
    }

    narrow = posteriors["narrow"]
    mode = narrow.map_estimate()
    vi = mean_field_variational(narrow, mode)
    full = full_rank_variational(narrow, mode)
    hess = narrow.local_hessian(mode)
    if np.min(np.linalg.eigvalsh(hess)) <= 0:
        raise RuntimeError("Local Hessian is not positive definite")
    _, log_cov = grids["narrow"].log_moments()
    masses = {"identity": np.eye(2), "map_hessian": hess, "posterior_covariance": np.linalg.inv(log_cov)}

    plain = hmc(narrow, mode, mass=masses["identity"], seed=seed + 11)
    condition = hmc(narrow, mode, mass=masses["map_hessian"], seed=seed + 21)
    covariance = hmc(narrow, mode, mass=masses["posterior_covariance"], seed=seed + 31)
    runs = {"identity": plain, "map_hessian": condition, "posterior_covariance": covariance}
    coverage = coverages(seed + 30000, ("narrow", "wide"), replications=replications)

    ident = {d: studies.identification(observed[d], grids[d], posteriors[d]) for d in observed}
    sensitivity = {d: studies.prior_sensitivity(observed[d]) for d in observed}
    ridge = studies.ridge_points(narrow, grids["narrow"], (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    margins = {name: studies.ridge_in_metric(ridge, m) for name, m in masses.items()}
    steps = {name: float(np.mean(r.step_sizes)) for name, r in runs.items()}
    sweep = studies.sampler_study(
        observed["narrow"], masses, mode,
        seeds=2 if quick else 8, path_means=(5, 15) if quick else (4, 8, 15, 30), base_seed=seed + 50000,
    )
    design_results = studies.design_study(observed["narrow"].x, replications=replications, base_seed=seed + 70000,
                                          resolution=160 if quick else 230)
    truth_kappas = (0.03, 0.05, 0.10, 0.18, 0.26, 0.40, 0.60)
    by_truth = studies.coverage_by_truth(truth_kappas, replications=replications, base_seed=seed + 90000,
                                         resolution=160 if quick else 230)
    prior_drawn = studies.coverage_prior_drawn(replications=2 * replications, base_seed=seed + 110000,
                                               resolution=160 if quick else 230)

    targeted = design_results["designs"]["kappa_targeted_at_truth"]
    plan = design.TwoTimeDesign(targeted["early_time"], targeted["late_time"], targeted["n_early"], 24)
    x = observed["narrow"].x
    targeted_data = simulate_at(seed, x, design.assign_times(x, plan))
    targeted_grid = adaptive_grid(Posterior(targeted_data), resolution=480)

    plot_geometry(grids["narrow"], plain, vi, full, mode, figures / "posterior_geometry.png")
    plot_prediction(grids["narrow"], vi, observed["narrow"], figures / "predictive_uncertainty.png")
    plot_designs(
        {"narrow": grids["narrow"], "wide": grids["wide"], "two_time": targeted_grid},
        {"narrow": "Narrow window (0.218–0.262)", "wide": "Wide window (0.06–0.57)",
         "two_time": f"{plan.n_early} at t = {plan.early:.2f}, {24 - plan.n_early} at t = {plan.late:.2f}"},
        figures / "observation_geometry.png",
    )
    studies.plot_identification(observed, figures / "identification.png")
    studies.plot_sampler(runs, grids["narrow"], margins, steps, sweep, figures / "sampler_diagnostics.png")
    studies.plot_design(x, design_results, figures / "design_tradeoff.png")
    studies.plot_coverage_by_truth(by_truth, figures / "coverage_by_truth.png")

    metrics: dict[str, object] = {
        "seed": seed,
        "quick": quick,
        "model": "Exact one-mode heat equation; not a trained PINN",
        "true_parameters": {"amplitude": TRUE_AMPLITUDE, "diffusivity": TRUE_DIFFUSIVITY},
        "sample_size_each_design": 24,
        "noise_sigma": OBSERVATION_SIGMA,
        "prior_mean_log": PRIOR_MEAN.tolist(),
        "prior_sd_log": PRIOR_SD.tolist(),
        "reference": {k: v.summary() | v.log_summary() | {"boundary_mass": v.boundary_mass} for k, v in grids.items()},
        "reference_two_time_design": targeted_grid.summary() | targeted_grid.log_summary(),
        "grid_refinement": refinement,
        "map": {"amplitude": float(np.exp(mode[0])), "diffusivity": float(np.exp(mode[1]))},
        "vi": vi.summary(np.random.default_rng(seed + 37)),
        "vi_full_rank": full.summary(np.random.default_rng(seed + 38)),
        "hmc_identity": plain.summary(),
        "hmc_map_hessian": condition.summary(),
        "hmc_posterior_covariance": covariance.summary(),
        "hmc_stability_margin": {
            name: [{**row, "step_over_thin_sd": steps[name] / row["thin_sd"]} for row in rows]
            for name, rows in margins.items()
        },
        "hmc_replicates": sweep,
        "identification": ident,
        "prior_sensitivity": sensitivity,
        "design": design_results,
        "coverage": coverage,
        "coverage_by_true_diffusivity": by_truth,
        "coverage_parameters_drawn_from_prior": prior_drawn,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="small run for CI; not the committed results")
    args = parser.parse_args()
    metrics = run(args.output, seed=args.seed, replications=args.replications, quick=args.quick)
    print(json.dumps({k: metrics[k] for k in ("reference", "vi", "vi_full_rank", "hmc_identity", "hmc_map_hessian",
                                              "hmc_posterior_covariance")}, indent=2))


if __name__ == "__main__":
    # studies.py imports this file as the module ``experiment``; running through that
    # module keeps a single copy of the classes (isinstance checks would fail otherwise).
    import experiment

    experiment.main()
