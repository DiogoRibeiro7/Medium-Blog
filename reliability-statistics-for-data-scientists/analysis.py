"""Synthetic industrial reliability, censoring, and maintenance decisions.

This self-contained research implementation demonstrates:

* Weibull survival, hazard and conditional failure probabilities;
* Kaplan–Meier and Nelson–Aalen estimation with right-censoring;
* maximum likelihood with right-censored versus complete-case data;
* age-replacement cost per renewal operating hour;
* Aalen–Johansen cumulative incidence under competing risks.

All input data are synthetic. The latent failure time saved to the evaluation
CSV is never used for estimation and is not observable in operational data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma, gammainc


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

SEED = 20261030
TRUE_SHAPE = 2.30
TRUE_SCALE = 1400.0
CENSOR_LOW = 550.0
CENSOR_HIGH = 2100.0
PREVENTIVE_COST = 220.0
FAILURE_COST = 1150.0
CONDITIONAL_HORIZON = 300.0
COMPETING_SHAPES = (2.2, 1.3)
COMPETING_SCALES = (1450.0, 2150.0)

FIGURE_NAMES = (
    "fig1_survival_and_hazards.png",
    "fig2_censoring_and_km.png",
    "fig3_likelihood_and_hazard.png",
    "fig4_censoring_sensitivity.png",
    "fig5_conditional_failure_risk.png",
    "fig6_age_replacement_cost.png",
    "fig7_renewal_validation.png",
    "fig8_competing_risks.png",
)
RESULT_NAMES = (
    "summary.csv",
    "observed_lifetimes.csv",
    "survival_estimates.csv",
    "hazard_shapes.csv",
    "likelihood_surface.csv",
    "censor_sensitivity.csv",
    "conditional_risk.csv",
    "maintenance_curve.csv",
    "maintenance_policies.csv",
    "competing_risks.csv",
)


def _positive_float(value: float, name: str) -> float:
    """Validate a positive finite real input and return its float value."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _positive_integer(value: int, name: str) -> int:
    """Validate a positive integral count without silently truncating floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class CensoredSample:
    """Observed right-censored lifetimes and synthetic oracle-only quantities."""

    observed: FloatArray
    event: BoolArray
    latent_lifetime: FloatArray
    censor_time: FloatArray

    def __post_init__(self) -> None:
        n = self.observed.size
        if n == 0 or any(x.ndim != 1 or x.size != n for x in (
            self.observed, self.event, self.latent_lifetime, self.censor_time
        )):
            raise ValueError("Censored sample arrays must be nonempty and aligned")
        if self.event.dtype.kind != "b":
            raise TypeError("event must be a boolean array")
        if any(not np.all(np.isfinite(x) & (x > 0)) for x in (
            self.observed, self.latent_lifetime, self.censor_time
        )):
            raise ValueError("Times must be positive and finite")
        if not np.allclose(self.observed, np.minimum(self.latent_lifetime, self.censor_time)):
            raise ValueError("Observed times must be minima of lifetime and censor time")
        if not np.array_equal(self.event, self.latent_lifetime <= self.censor_time):
            raise ValueError("The event indicator does not match the observed times")


@dataclass(frozen=True)
class WeibullFit:
    """Fitted Weibull shape and scale with observed-data log likelihood."""

    shape: float
    scale: float
    log_likelihood: float
    n_events: int


@dataclass(frozen=True)
class ProductLimit:
    """Product-limit survival and Nelson–Aalen cumulative hazard at event times."""

    times: FloatArray
    survival: FloatArray
    cumulative_hazard: FloatArray
    greenwood_se: FloatArray
    risk_set: IntArray
    failures: IntArray


@dataclass(frozen=True)
class CompetingSample:
    """Synthetic competing-risks data, event code 0, 1, or 2."""

    observed: FloatArray
    cause: IntArray
    latent_cause1: FloatArray
    latent_cause2: FloatArray
    censor_time: FloatArray

    def __post_init__(self) -> None:
        n = self.observed.size
        if n == 0 or any(x.ndim != 1 or x.size != n for x in (
            self.observed, self.cause, self.latent_cause1,
            self.latent_cause2, self.censor_time,
        )):
            raise ValueError("Competing sample arrays must be aligned")
        if self.cause.dtype.kind not in "iu" or not np.all(np.isin(self.cause, (0, 1, 2))):
            raise ValueError("cause must be integer 0, 1, or 2")
        if np.any(self.observed <= 0) or not np.all(np.isfinite(self.observed)):
            raise ValueError("Observed times must be positive and finite")


@dataclass(frozen=True)
class CumulativeIncidence:
    """Nonparametric all-cause survival and both cause-specific CIF estimates."""

    times: FloatArray
    survival: FloatArray
    cause1: FloatArray
    cause2: FloatArray
    risk_set: IntArray


def weibull_survival(time: FloatArray | float, shape: float, scale: float) -> FloatArray:
    """Return P(T > t) for a two-parameter Weibull lifetime distribution."""
    _positive_float(shape, "shape")
    _positive_float(scale, "scale")
    t = np.asarray(time, dtype=np.float64)
    if np.any(~np.isfinite(t) | (t < 0)):
        raise ValueError("time must be nonnegative and finite")
    # Clipping affects only magnitudes whose Weibull survival already rounds
    # to zero, and prevents overflow in the power for very large finite ages.
    return np.exp(-np.power(np.minimum(t / scale, 1e12), shape))


def weibull_hazard(time: FloatArray | float, shape: float, scale: float) -> FloatArray:
    """Return instantaneous failure hazard for strictly positive time."""
    _positive_float(shape, "shape")
    _positive_float(scale, "scale")
    t = np.asarray(time, dtype=np.float64)
    if np.any(~np.isfinite(t) | (t <= 0)):
        raise ValueError("hazard time must be strictly positive and finite")
    return (shape / scale) * np.power(t / scale, shape - 1)


def weibull_mean(shape: float, scale: float) -> float:
    """Return the expected Weibull lifetime."""
    return _positive_float(scale, "scale") * float(gamma(1 + 1 / _positive_float(shape, "shape")))


def conditional_failure_probability(
    age: FloatArray | float, horizon: float, shape: float, scale: float
) -> FloatArray:
    """Compute P(T <= age+horizon | T > age) without subtractive tail cancellation."""
    _positive_float(horizon, "horizon")
    t = np.asarray(age, dtype=np.float64)
    if np.any(~np.isfinite(t) | (t < 0)):
        raise ValueError("age must be nonnegative and finite")
    _positive_float(shape, "shape")
    _positive_float(scale, "scale")
    integrated_hazard = np.power((t + horizon) / scale, shape) - np.power(t / scale, shape)
    return -np.expm1(-integrated_hazard)


def simulate_censored(
    n: int, rng: np.random.Generator, *, shape: float = TRUE_SHAPE,
    scale: float = TRUE_SCALE, censor_low: float = CENSOR_LOW,
    censor_high: float = CENSOR_HIGH,
) -> CensoredSample:
    """Draw independent Weibull lifetimes and administrative censoring times."""
    _positive_integer(n, "n")
    _positive_float(shape, "shape")
    _positive_float(scale, "scale")
    _positive_float(censor_low, "censor_low")
    if not np.isfinite(censor_high) or censor_high <= censor_low:
        raise ValueError("censor_high must exceed censor_low")
    lifetime = scale * rng.weibull(shape, size=n)
    censor = rng.uniform(censor_low, censor_high, size=n)
    return CensoredSample(
        observed=np.minimum(lifetime, censor),
        event=np.asarray(lifetime <= censor, dtype=np.bool_),
        latent_lifetime=lifetime,
        censor_time=censor,
    )


def censored_log_likelihood(
    shape: float, scale: float, observed: FloatArray, event: BoolArray
) -> float:
    """Evaluate sum[delta log f(t)+(1-delta)log S(t)] for independent censoring."""
    _positive_float(shape, "shape")
    _positive_float(scale, "scale")
    if observed.ndim != 1 or event.ndim != 1 or observed.size != event.size:
        raise ValueError("observed and event arrays must be one-dimensional and aligned")
    if np.any(~np.isfinite(observed) | (observed <= 0)) or event.dtype.kind != "b":
        raise ValueError("invalid observed times or event indicators")
    log_hazard = np.log(shape) - shape * np.log(scale) + (shape - 1) * np.log(observed)
    cum_hazard = np.power(observed / scale, shape)
    return float(np.sum(event.astype(np.float64) * log_hazard - cum_hazard))


def fit_weibull(observed: FloatArray, event: BoolArray) -> WeibullFit:
    """Maximize the properly censored Weibull likelihood on log-parameter scale."""
    if observed.ndim != 1 or event.ndim != 1 or observed.size != event.size:
        raise ValueError("observed and event must be aligned vectors")
    n_events = int(np.count_nonzero(event))
    if n_events < 5:
        raise ValueError("at least five observed failures are required")
    initial_scale = float(max(np.median(observed), 150.0))

    log_times = np.log(observed)
    indicators = event.astype(np.float64)
    n_events_float = float(np.sum(indicators))

    def objective(log_parameters: FloatArray) -> tuple[float, FloatArray]:
        """Return negative log-likelihood and its exact log-scale gradient."""
        # Positive parameters are enforced by exponentiating their logarithms.
        log_shape, log_scale = map(float, log_parameters)
        shape = float(np.exp(log_shape))
        q = log_times - log_scale
        cumulative_hazard = np.exp(shape * q)
        log_likelihood = float(np.sum(
            indicators * (log_shape - log_scale + (shape - 1) * q)
            - cumulative_hazard
        ))
        # An analytic Jacobian avoids fragile finite differences when many
        # simulated lifetimes are censored.
        grad_shape = -(n_events_float + shape * np.sum(
            (indicators - cumulative_hazard) * q
        ))
        grad_scale = shape * (n_events_float - np.sum(cumulative_hazard))
        return -log_likelihood, np.array([grad_shape, grad_scale], dtype=np.float64)

    result = minimize(
        objective,
        x0=np.log(np.array([1.6, initial_scale], dtype=np.float64)),
        method="L-BFGS-B",
        jac=True,
        bounds=[(np.log(0.20), np.log(8.0)), (np.log(80.0), np.log(20000.0))],
        options={"maxiter": 350, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Weibull fit failed: {result.message}")
    shape, scale = np.exp(result.x)
    return WeibullFit(float(shape), float(scale), -float(result.fun), n_events)


def kaplan_meier(observed: FloatArray, event: BoolArray) -> ProductLimit:
    """Compute tied-time product-limit and Nelson–Aalen estimates from first principles."""
    if observed.ndim != 1 or event.ndim != 1 or observed.size != event.size:
        raise ValueError("observed and event must be aligned")
    if observed.size == 0 or np.any(~np.isfinite(observed) | (observed <= 0)):
        raise ValueError("observed times must be positive and finite")
    if event.dtype.kind != "b":
        raise TypeError("event indicators must be boolean")
    times = np.unique(observed[event])
    sorted_times = np.sort(observed)
    n_risk = observed.size - np.searchsorted(sorted_times, times, side="left")
    failure_times, failure_counts = np.unique(observed[event], return_counts=True)
    if not np.array_equal(times, failure_times):
        raise AssertionError("Failure time index is inconsistent")
    if np.any(failure_counts > n_risk):
        raise ValueError("failures exceed the risk set")
    survival = np.cumprod(1.0 - failure_counts / n_risk)
    cum_hazard = np.cumsum(failure_counts / n_risk)
    # Greenwood's variance uses contributions at times where survivors remain.
    greenwood_term = np.divide(
        failure_counts,
        n_risk * (n_risk - failure_counts),
        out=np.zeros_like(survival),
        where=(n_risk > failure_counts),
    )
    greenwood_se = survival * np.sqrt(np.cumsum(greenwood_term))
    return ProductLimit(
        times=times, survival=survival, cumulative_hazard=cum_hazard,
        greenwood_se=greenwood_se, risk_set=n_risk.astype(np.int64),
        failures=failure_counts.astype(np.int64),
    )


def step_values(times: FloatArray, values: FloatArray, grid: FloatArray, initial: float) -> FloatArray:
    """Evaluate a right-continuous event-time estimator on arbitrary time points."""
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        raise ValueError("step function vectors must be aligned")
    if np.any(np.diff(times) <= 0) or np.any(grid < 0):
        raise ValueError("event times must increase and grid must be nonnegative")
    index = np.searchsorted(times, grid, side="right") - 1
    result = np.full(grid.shape, initial, dtype=np.float64)
    keep = index >= 0
    result[keep] = values[index[keep]]
    return result


def expected_age_replacement_rate(
    replacement_age: FloatArray | float, shape: float, scale: float,
    preventive_cost: float = PREVENTIVE_COST, failure_cost: float = FAILURE_COST,
) -> FloatArray:
    """Return long-run cost/operating hour by the renewal–reward theorem.

    The policy replaces on failure, or preventively at age tau if still alive.
    Each cycle starts with a new independently distributed component. It has
    no downtime, setup costs, or imperfect repairs; those require a richer model.
    """
    _positive_float(shape, "shape")
    _positive_float(scale, "scale")
    _positive_float(preventive_cost, "preventive_cost")
    _positive_float(failure_cost, "failure_cost")
    tau = np.asarray(replacement_age, dtype=np.float64)
    if np.any((tau <= 0) | np.isnan(tau)):
        raise ValueError("replacement_age must be positive (infinity allowed)")
    reliability = weibull_survival(np.minimum(tau, scale * 1e12), shape, scale)
    z = np.power(np.minimum(tau / scale, 1e12), shape)
    mean_cycle = weibull_mean(shape, scale) * gammainc(1.0 / shape, z)
    mean_cost = preventive_cost * reliability + failure_cost * (1.0 - reliability)
    return mean_cost / mean_cycle


def optimize_replacement_age(
    shape: float, scale: float, preventive_cost: float = PREVENTIVE_COST,
    failure_cost: float = FAILURE_COST,
) -> float:
    """Optimize preventive age, returning infinity if run-to-failure is better."""
    _positive_float(shape, "shape")
    _positive_float(scale, "scale")
    _positive_float(preventive_cost, "preventive_cost")
    _positive_float(failure_cost, "failure_cost")
    if shape <= 1.0:
        return float("inf")
    objective = lambda age: float(expected_age_replacement_rate(
        age, shape, scale, preventive_cost, failure_cost
    ))
    result = minimize_scalar(
        objective, bounds=(max(scale * .03, 1.0), scale * 8),
        method="bounded", options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError("Age-replacement optimization did not converge")
    if result.fun >= objective(float("inf")):
        return float("inf")
    return float(result.x)


def simulate_renewal_policy(
    replacement_age: float, shape: float, scale: float,
    rng: np.random.Generator, n_cycles: int,
    preventive_cost: float = PREVENTIVE_COST, failure_cost: float = FAILURE_COST,
) -> tuple[float, float]:
    """Estimate average renewal cost rate and realized corrective-repair fraction."""
    _positive_integer(n_cycles, "n_cycles")
    if np.isnan(replacement_age) or replacement_age <= 0:
        raise ValueError("replacement_age must be positive")
    lifetime = scale * rng.weibull(shape, size=n_cycles)
    failed = lifetime <= replacement_age
    operating_time = np.minimum(lifetime, replacement_age)
    cost = np.where(failed, failure_cost, preventive_cost)
    return float(np.sum(cost) / np.sum(operating_time)), float(np.mean(failed))


def simulate_competing(
    n: int, rng: np.random.Generator,
    shapes: tuple[float, float] = COMPETING_SHAPES,
    scales: tuple[float, float] = COMPETING_SCALES,
    censor_low: float = 700.0, censor_high: float = 2600.0,
) -> CompetingSample:
    """Generate two independent latent causes with independent administrative censoring."""
    _positive_integer(n, "n")
    for shape, scale in zip(shapes, scales, strict=True):
        _positive_float(shape, "shape")
        _positive_float(scale, "scale")
    if censor_low <= 0 or censor_high <= censor_low:
        raise ValueError("invalid competing-risk censor window")
    t1 = scales[0] * rng.weibull(shapes[0], size=n)
    t2 = scales[1] * rng.weibull(shapes[1], size=n)
    censor = rng.uniform(censor_low, censor_high, size=n)
    observed = np.minimum.reduce((t1, t2, censor))
    cause = np.where(
        observed == censor, 0, np.where(t1 < t2, 1, 2)
    ).astype(np.int64)
    return CompetingSample(observed, cause, t1, t2, censor)


def aalen_johansen(observed: FloatArray, cause: IntArray) -> CumulativeIncidence:
    """Estimate two cumulative incidence curves without treating competing events as censoring."""
    if observed.ndim != 1 or cause.ndim != 1 or observed.size != cause.size:
        raise ValueError("observed and cause must be aligned")
    if np.any(~np.isfinite(observed) | (observed <= 0)):
        raise ValueError("times must be positive and finite")
    if cause.dtype.kind not in "iu" or not np.all(np.isin(cause, (0, 1, 2))):
        raise ValueError("cause must be an integer in 0,1,2")
    times = np.unique(observed[cause != 0])
    sorted_observed = np.sort(observed)
    risk = observed.size - np.searchsorted(sorted_observed, times, side="left")
    d1 = np.array([np.count_nonzero((observed == t) & (cause == 1)) for t in times])
    d2 = np.array([np.count_nonzero((observed == t) & (cause == 2)) for t in times])
    survival = np.empty(times.size, dtype=np.float64)
    cif1 = np.empty_like(survival)
    cif2 = np.empty_like(survival)
    s_prev = 1.0
    v1, v2 = 0.0, 0.0
    for i, n_risk in enumerate(risk):
        # The all-cause survival just before the event multiplies each CIF jump.
        v1 += s_prev * d1[i] / n_risk
        v2 += s_prev * d2[i] / n_risk
        s_prev *= 1 - (d1[i] + d2[i]) / n_risk
        survival[i], cif1[i], cif2[i] = s_prev, v1, v2
    return CumulativeIncidence(times, survival, cif1, cif2, risk.astype(np.int64))


def competing_truth(grid: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Integrate cause hazards against survival on a fine internal mesh.

    A high-resolution quadrature mesh avoids artifacts near time zero where
    fractional Weibull powers change sharply, even on a sparse output grid.
    """
    if grid.ndim != 1 or grid.size < 2 or grid[0] != 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("competing_truth grid must be increasing and start at zero")
    fine = np.linspace(0, float(grid[-1]), max(2, int(np.ceil(grid[-1]/.20)) + 1))
    s1 = weibull_survival(fine, COMPETING_SHAPES[0], COMPETING_SCALES[0])
    s2 = weibull_survival(fine, COMPETING_SHAPES[1], COMPETING_SCALES[1])
    safe_time = np.maximum(fine, np.finfo(float).tiny)
    hz1 = weibull_hazard(safe_time, COMPETING_SHAPES[0], COMPETING_SCALES[0])
    hz2 = weibull_hazard(safe_time, COMPETING_SHAPES[1], COMPETING_SCALES[1])
    cif1 = cumulative_trapezoid(s1 * s2 * hz1, fine, initial=0)
    cif2 = cumulative_trapezoid(s1 * s2 * hz2, fine, initial=0)
    return np.interp(grid, fine, cif1), np.interp(grid, fine, cif2)


def _rows_to_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write a deterministic UTF-8 CSV with an explicit header and column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _styled_axes(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent, publication-friendly axis styling."""
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)


def _save_figure(fig: Any, destination: Path) -> None:
    """Save a Matplotlib figure reproducibly and release its resources."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)


def run_experiment(output_directory: Path, quick: bool = False) -> dict[str, float]:
    """Run the seeded study and save original figures plus source data tables.

    Quick runs write their own suffix and never replace the full study.
    """
    if not isinstance(quick, bool) or not isinstance(output_directory, Path):
        raise TypeError("quick must be boolean and output_directory must be a Path")
    output_directory.mkdir(parents=True, exist_ok=True)
    fig_dir, results_dir = output_directory / "figures", output_directory / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""

    def result_path(name: str) -> Path:
        return results_dir / name.replace(".csv", f"{suffix}.csv")

    def figure_path(name: str) -> Path:
        return fig_dir / name.replace(".png", f"{suffix}.png")

    n = 340 if quick else 1200
    sample = simulate_censored(n, np.random.default_rng(SEED + 10))
    proper = fit_weibull(sample.observed, sample.event)
    naive = fit_weibull(
        sample.observed[sample.event],
        np.ones(int(np.count_nonzero(sample.event)), dtype=np.bool_),
    )
    product = kaplan_meier(sample.observed, sample.event)
    n_events = int(np.count_nonzero(sample.event))
    time_grid = np.linspace(0.0, 1950.0, 131)
    true_s = weibull_survival(time_grid, TRUE_SHAPE, TRUE_SCALE)
    censored_s = weibull_survival(time_grid, proper.shape, proper.scale)
    naive_s = weibull_survival(time_grid, naive.shape, naive.scale)
    km_s = step_values(product.times, product.survival, time_grid, 1.0)
    km_se = step_values(product.times, product.greenwood_se, time_grid, 0.0)

    # Figure 1 shows that the average life cannot determine failure hazard.
    hazard_times = np.linspace(50.0, 2700.0, 155)
    hazard_shapes = (0.7, 1.0, TRUE_SHAPE)
    hazard_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for shape in hazard_shapes:
        reliability = weibull_survival(hazard_times, shape, TRUE_SCALE)
        hazard = weibull_hazard(hazard_times, shape, TRUE_SCALE)
        axes[0].plot(hazard_times, reliability, label=f"shape = {shape:.1f}")
        axes[1].plot(hazard_times, 1000 * hazard, label=f"shape = {shape:.1f}")
        for t, surv, rate in zip(hazard_times, reliability, hazard, strict=True):
            hazard_rows.append({"time": t, "shape": shape, "scale": TRUE_SCALE,
                                "survival": surv, "hazard_per_hour": rate})
    _styled_axes(axes[0], "Survival at a shared characteristic life", "Operating hours", "P(T > t)")
    _styled_axes(axes[1], "Hazard shapes", "Operating hours", "Hazard per 1,000 hours")
    for ax in axes:
        ax.legend()
    _save_figure(fig, figure_path(FIGURE_NAMES[0]))
    _rows_to_csv(result_path("hazard_shapes.csv"), hazard_rows,
                 ("time", "shape", "scale", "survival", "hazard_per_hour"))

    # Figure 2 contrasts the product-limit estimator with throwing censored
    # components away as if they supplied no survival information.
    event_times = np.sort(sample.observed[sample.event])
    empirical_failures_only = 1 - np.searchsorted(
        event_times, time_grid, side="right"
    ) / n_events
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(sample.observed[sample.event], bins=23, alpha=.75,
                 color="#d46a45", label=f"Failures (n={n_events})")
    axes[0].hist(sample.observed[~sample.event], bins=23, alpha=.60,
                 color="#407a99", label=f"Right-censored (n={n-n_events})")
    _styled_axes(axes[0], "Observed ages at exit", "Operating hours", "Units")
    axes[0].legend()
    axes[1].plot(time_grid, true_s, color="#242933", linewidth=2, label="True survival")
    axes[1].step(time_grid, km_s, where="post", color="#13877b", label="Kaplan–Meier")
    axes[1].fill_between(time_grid, np.clip(km_s-1.96*km_se,0,1),
                         np.clip(km_s+1.96*km_se,0,1),
                         color="#13877b", alpha=.13, label="Approx. ±1.96 Greenwood SE")
    axes[1].plot(time_grid, empirical_failures_only, "--",
                 color="#c95e48", label="Failures-only empirical")
    _styled_axes(axes[1], "Censoring changes the risk set", "Operating hours", "Survival")
    axes[1].set_ylim(-.02, 1.03)
    axes[1].legend(fontsize=8)
    _save_figure(fig, figure_path(FIGURE_NAMES[1]))
    survival_rows = [
        {"time": t, "km": km, "km_greenwood_se": se, "true": truth,
         "weibull_censored_mle": fitted, "weibull_failures_only": naive_value,
         "empirical_failures_only": empirical}
        for t, km, se, truth, fitted, naive_value, empirical in zip(
            time_grid, km_s, km_se, true_s, censored_s, naive_s,
            empirical_failures_only, strict=True)
    ]
    _rows_to_csv(result_path("survival_estimates.csv"), survival_rows,
                 ("time", "km", "km_greenwood_se", "true", "weibull_censored_mle",
                  "weibull_failures_only", "empirical_failures_only"))
    observation_rows = [
        {"unit": i + 1, "observed_time": float(t), "event": int(e),
         "latent_true_lifetime_oracle_only": float(latent), "censor_time": float(c)}
        for i, (t, e, latent, c) in enumerate(zip(
            sample.observed[:500], sample.event[:500], sample.latent_lifetime[:500],
            sample.censor_time[:500], strict=True))
    ]
    _rows_to_csv(result_path("observed_lifetimes.csv"), observation_rows,
                 ("unit", "observed_time", "event",
                  "latent_true_lifetime_oracle_only", "censor_time"))

    # Figure 3: likelihood contours and the effect of biased complete-case
    # fitting on estimated conditional hazard.
    shape_grid = np.linspace(1.35, 3.35, 15 if quick else 35)
    scale_grid = np.linspace(1050, 1850, 15 if quick else 35)
    nll = np.array([
        [-censored_log_likelihood(float(sh), float(sc), sample.observed, sample.event)
         for sc in scale_grid] for sh in shape_grid
    ])
    nll_delta = nll + proper.log_likelihood
    likelihood_rows = [
        {"shape": sh, "scale": sc, "delta_negative_log_likelihood": float(nll_delta[i,j])}
        for i, sh in enumerate(shape_grid)
        for j, sc in enumerate(scale_grid)
    ]
    _rows_to_csv(result_path("likelihood_surface.csv"), likelihood_rows,
                 ("shape", "scale", "delta_negative_log_likelihood"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    cc = axes[0].contour(scale_grid, shape_grid, nll_delta,
                         levels=[1,2.30,4.61,10,20,40,100],
                         colors="#3c7392", linewidths=1.05)
    axes[0].clabel(cc, inline=True, fontsize=8)
    axes[0].scatter([TRUE_SCALE, proper.scale, naive.scale],
                    [TRUE_SHAPE, proper.shape, naive.shape],
                    c=["black","#148277","#c46141"], s=42, zorder=5)
    _styled_axes(axes[0], "Censored negative log-likelihood contours",
                 "Weibull scale", "Weibull shape")
    hz_grid = np.linspace(100, 2250, 125)
    for title, sh, sc, style, color in (
        ("True", TRUE_SHAPE, TRUE_SCALE, "-", "#242933"),
        ("Censored MLE", proper.shape, proper.scale, "--", "#148277"),
        ("Failures only", naive.shape, naive.scale, ":", "#c46141"),
    ):
        axes[1].plot(hz_grid, 1000*weibull_hazard(hz_grid, sh, sc),
                     linestyle=style, color=color, label=title)
    _styled_axes(axes[1], "Different fits imply different hazard",
                 "Operating hours", "Hazard per 1,000 hours")
    axes[1].legend()
    _save_figure(fig, figure_path(FIGURE_NAMES[2]))

    # Figure 4: vary independent censoring times, then repeat sampling.
    censor_caps = (950.,1250.,1650.,2100.,2700.,3500.)
    sensitivity_rng = np.random.default_rng(SEED+20)
    repetitions = 4 if quick else 28
    sensitivity_rows: list[dict[str, Any]] = []
    for cap in censor_caps:
        fitted_shapes: list[float] = []
        naive_shapes: list[float] = []
        fitted_scales: list[float] = []
        naive_scales: list[float] = []
        event_fractions: list[float] = []
        for _ in range(repetitions):
            one = simulate_censored(300 if quick else 540, sensitivity_rng,
                                    censor_low=250.0, censor_high=cap)
            mle = fit_weibull(one.observed, one.event)
            bad = fit_weibull(
                one.observed[one.event],
                np.ones(int(np.count_nonzero(one.event)), dtype=np.bool_))
            event_fractions.append(float(np.mean(one.event)))
            fitted_shapes.append(mle.shape)
            naive_shapes.append(bad.shape)
            fitted_scales.append(mle.scale)
            naive_scales.append(bad.scale)
        sensitivity_rows.append({
            "censor_upper_hour": cap, "n_replicates": repetitions,
            "average_event_fraction": float(np.mean(event_fractions)),
            "censored_shape_mean": float(np.mean(fitted_shapes)),
            "censored_shape_se": float(np.std(fitted_shapes,ddof=1)/np.sqrt(repetitions)),
            "failures_only_shape_mean": float(np.mean(naive_shapes)),
            "failures_only_shape_se": float(np.std(naive_shapes,ddof=1)/np.sqrt(repetitions)),
            "censored_scale_mean": float(np.mean(fitted_scales)),
            "censored_scale_se": float(np.std(fitted_scales,ddof=1)/np.sqrt(repetitions)),
            "failures_only_scale_mean": float(np.mean(naive_scales)),
            "failures_only_scale_se": float(np.std(naive_scales,ddof=1)/np.sqrt(repetitions)),
        })
    _rows_to_csv(result_path("censor_sensitivity.csv"), sensitivity_rows,
                 ("censor_upper_hour","n_replicates","average_event_fraction",
                  "censored_shape_mean","censored_shape_se","failures_only_shape_mean",
                  "failures_only_shape_se","censored_scale_mean","censored_scale_se",
                  "failures_only_scale_mean","failures_only_scale_se"))
    frac = np.array([r["average_event_fraction"] for r in sensitivity_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, prop, target, label in (
        (axes[0], "shape", TRUE_SHAPE, "Weibull shape"),
        (axes[1], "scale", TRUE_SCALE, "Weibull scale (hours)"),
    ):
        ax.plot(frac, [r[f"censored_{prop}_mean"] for r in sensitivity_rows],
                "o-", color="#148277", label="Censor-aware MLE")
        ax.plot(frac, [r[f"failures_only_{prop}_mean"] for r in sensitivity_rows],
                "s--", color="#c46141", label="Failures-only fit")
        ax.axhline(target, color="#242933", linestyle=":", label="Generating value")
        _styled_axes(ax, f"Parameter estimate under different censor windows",
                     "Share observed to fail", label)
        ax.legend(fontsize=8)
    _save_figure(fig, figure_path(FIGURE_NAMES[3]))

    # Figure 5: instantaneous hazard is distinct from the probability of a
    # failure over a fixed future interval conditional on survival to age a.
    ages = np.linspace(0.0, 2000.0, 101)
    p_true = conditional_failure_probability(
        ages, CONDITIONAL_HORIZON, TRUE_SHAPE, TRUE_SCALE)
    p_fit = conditional_failure_probability(
        ages, CONDITIONAL_HORIZON, proper.shape, proper.scale)
    p_naive = conditional_failure_probability(
        ages, CONDITIONAL_HORIZON, naive.shape, naive.scale)
    conditional_rows = [
        {"age": a, "horizon": CONDITIONAL_HORIZON,
         "true_probability": truth, "censored_fit_probability": fit,
         "failures_only_probability": bad}
        for a, truth, fit, bad in zip(ages,p_true,p_fit,p_naive,strict=True)
    ]
    _rows_to_csv(result_path("conditional_risk.csv"), conditional_rows,
                 ("age","horizon","true_probability",
                  "censored_fit_probability","failures_only_probability"))
    fig, axes = plt.subplots(1, 2, figsize=(11,4), constrained_layout=True)
    for label, probability, sh, sc, style, color in (
        ("Generating law",p_true,TRUE_SHAPE,TRUE_SCALE,"-","#242933"),
        ("Censored MLE",p_fit,proper.shape,proper.scale,"--","#148277"),
        ("Failures only",p_naive,naive.shape,naive.scale,":","#c46141"),
    ):
        axes[0].plot(ages, probability, linestyle=style, color=color, label=label)
        axes[1].plot(np.maximum(ages,1),
                     1000*weibull_hazard(np.maximum(ages,1),sh,sc),
                     linestyle=style,color=color,label=label)
    _styled_axes(axes[0],f"Failure within the next {CONDITIONAL_HORIZON:.0f} hours",
                 "Current age (hours)","Conditional probability")
    _styled_axes(axes[1],"Hazard is not a probability",
                 "Current age (hours)","Hazard per 1,000 hours")
    for ax in axes:
        ax.legend()
    _save_figure(fig,figure_path(FIGURE_NAMES[4]))

    # Figure 6: optimize expected cycle cost divided by expected cycle length,
    # not expected one-cycle cost or a statistical goodness-of-fit score.
    true_optimal = optimize_replacement_age(TRUE_SHAPE,TRUE_SCALE)
    fitted_optimal = optimize_replacement_age(proper.shape,proper.scale)
    naive_optimal = optimize_replacement_age(naive.shape,naive.scale)
    mean_lifetime = weibull_mean(TRUE_SHAPE,TRUE_SCALE)
    age_grid = np.linspace(200,3700,161)
    cost_true = expected_age_replacement_rate(age_grid,TRUE_SHAPE,TRUE_SCALE)
    cost_fit = expected_age_replacement_rate(age_grid,proper.shape,proper.scale)
    cost_naive = expected_age_replacement_rate(age_grid,naive.shape,naive.scale)
    maintenance_rows = [
        {"replacement_age": age,"true_cost_per_hour": tr,
         "censored_fit_model_cost_per_hour": f,
         "failures_only_model_cost_per_hour": bad}
        for age,tr,f,bad in zip(age_grid,cost_true,cost_fit,cost_naive,strict=True)
    ]
    _rows_to_csv(result_path("maintenance_curve.csv"),maintenance_rows,
                 ("replacement_age","true_cost_per_hour",
                  "censored_fit_model_cost_per_hour","failures_only_model_cost_per_hour"))
    fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot(age_grid,cost_true,color="#242933",label="True cost rate")
    axes[0].plot(age_grid,cost_fit,"--",color="#148277",label="Censored-model cost rate")
    axes[0].plot(age_grid,cost_naive,":",color="#c46141",label="Failures-only model cost rate")
    axes[0].axvline(true_optimal,color="#242933",linestyle="--",linewidth=1)
    axes[0].axvline(fitted_optimal,color="#148277",linestyle=":",linewidth=1)
    axes[0].axvline(mean_lifetime,color="#8b8b8b",linestyle="-.",linewidth=1)
    axes[0].axhline(float(expected_age_replacement_rate(
        float("inf"),TRUE_SHAPE,TRUE_SCALE)),
        color="#8b628b",linestyle=":",label="Run to failure")
    _styled_axes(axes[0],"Renewal–reward age-replacement objective",
                 "Preventive replacement age (hours)","Cost per operating hour")
    axes[0].legend(fontsize=7.5)
    candidate_ages = [float("inf"),mean_lifetime,true_optimal,
                      fitted_optimal,naive_optimal]
    candidate_names = ["Run to\nfailure","At mean\nlifetime",
                       "True\noptimum","Censored\nMLE","Failures\nonly"]
    candidate_rates = [
        float(expected_age_replacement_rate(age,TRUE_SHAPE,TRUE_SCALE))
        for age in candidate_ages
    ]
    axes[1].bar(candidate_names,candidate_rates,
                color=["#8b628b","#8398ad","#424b58","#148277","#c46141"])
    _styled_axes(axes[1],"True economic cost of each frozen decision",
                 "Replacement rule","Cost per operating hour")
    _save_figure(fig,figure_path(FIGURE_NAMES[5]))

    # Figure 7: independent renewal-cycle Monte Carlo checks the analytic
    # renewal rate. Error bars are descriptive 1.96 batch standard errors.
    policy_names = ("Run to failure","Replace at mean lifetime",
                    "Oracle optimal age","Censored MLE age","Failures-only age")
    renewal_rows: list[dict[str,Any]] = []
    batch_count,batch_size = (5,700) if quick else (40,3300)
    for j,(name,age) in enumerate(zip(policy_names,candidate_ages,strict=True)):
        rng = np.random.default_rng(SEED+30+j)
        rates: list[float] = []
        fail_fractions: list[float] = []
        for _ in range(batch_count):
            rate,frac_failed = simulate_renewal_policy(
                age,TRUE_SHAPE,TRUE_SCALE,rng,batch_size)
            rates.append(rate)
            fail_fractions.append(frac_failed)
        renewal_rows.append({
            "policy":name,"replacement_age":age,
            "theoretical_cost_rate":float(expected_age_replacement_rate(
                age,TRUE_SHAPE,TRUE_SCALE)),
            "simulated_cost_rate":float(np.mean(rates)),
            "simulated_batch_mean_se":float(np.std(rates,ddof=1)/np.sqrt(batch_count)),
            "corrective_fraction":float(np.mean(fail_fractions)),
            "total_simulated_cycles":batch_count*batch_size,
        })
    _rows_to_csv(result_path("maintenance_policies.csv"),renewal_rows,
                 ("policy","replacement_age","theoretical_cost_rate",
                  "simulated_cost_rate","simulated_batch_mean_se",
                  "corrective_fraction","total_simulated_cycles"))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    labels=["Run to failure","Mean life","True optimal","Censored MLE","Failures only"]
    xx=np.arange(len(labels))
    theory=np.array([r["theoretical_cost_rate"] for r in renewal_rows])
    estimate=np.array([r["simulated_cost_rate"] for r in renewal_rows])
    se=np.array([r["simulated_batch_mean_se"] for r in renewal_rows])
    axes[0].bar(xx-.17,theory,width=.34,color="#425a71",label="Renewal theory")
    axes[0].bar(xx+.17,estimate,width=.34,color="#159085",
                yerr=1.96*se,error_kw={"capsize":2,"elinewidth":1},label="Monte Carlo")
    axes[0].set_xticks(xx,labels,rotation=20,ha="right",fontsize=8)
    _styled_axes(axes[0],"Theory versus repeated renewal simulation",
                 "Fixed policy","Cost per operating hour")
    axes[0].legend()
    axes[1].bar(xx,[r["corrective_fraction"] for r in renewal_rows],color="#c27a45")
    axes[1].set_xticks(xx,labels,rotation=20,ha="right",fontsize=8)
    axes[1].set_ylim(0,1.05)
    _styled_axes(axes[1],"Cycle fraction ending in an actual failure",
                 "Fixed policy","Corrective replacements / cycles")
    _save_figure(fig,figure_path(FIGURE_NAMES[6]))

    # Figure 8: competing-risks cumulative incidence. 1-KM with the other
    # cause censored is a net-risk function, not observed first-event risk.
    competing=simulate_competing(650 if quick else 2900,
                                np.random.default_rng(SEED+40))
    aj=aalen_johansen(competing.observed,competing.cause)
    net1=kaplan_meier(competing.observed,competing.cause==1)
    competing_grid=np.linspace(0,2200,151)
    cif_true1,cif_true2=competing_truth(competing_grid)
    cif_aj1=step_values(aj.times,aj.cause1,competing_grid,0)
    cif_aj2=step_values(aj.times,aj.cause2,competing_grid,0)
    net_risk1=1-step_values(net1.times,net1.survival,competing_grid,1)
    competing_rows=[
        {"time":t,"true_incidence_cause1":tr1,
         "aj_incidence_cause1":a1,"naive_one_minus_km_cause1":net,
         "true_incidence_cause2":tr2,"aj_incidence_cause2":a2}
        for t,tr1,a1,net,tr2,a2 in zip(
            competing_grid,cif_true1,cif_aj1,net_risk1,
            cif_true2,cif_aj2,strict=True)
    ]
    _rows_to_csv(result_path("competing_risks.csv"),competing_rows,
                 ("time","true_incidence_cause1","aj_incidence_cause1",
                  "naive_one_minus_km_cause1","true_incidence_cause2",
                  "aj_incidence_cause2"))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot(competing_grid,cif_true1,color="#242933",label="Cause 1 true incidence")
    axes[0].step(competing_grid,cif_aj1,where="post",color="#13877b",
                 label="Aalen–Johansen estimate")
    axes[0].step(competing_grid,net_risk1,where="post",color="#c46141",
                 linestyle="--",label="1 − KM, competing cause censored")
    _styled_axes(axes[0],"Actual incidence and hypothetical net risk",
                 "Operating hours","Probability")
    axes[0].legend(fontsize=8)
    axes[1].plot(competing_grid,cif_true2,color="#242933",label="Cause 2 true incidence")
    axes[1].step(competing_grid,cif_aj2,where="post",color="#4777a2",
                 label="Aalen–Johansen estimate")
    axes[1].plot(competing_grid,1-cif_true1-cif_true2,":",
                 color="#8f678e",label="True all-cause survival")
    _styled_axes(axes[1],"Failure modes compete for the first event",
                 "Operating hours","Probability")
    axes[1].legend(fontsize=8)
    _save_figure(fig,figure_path(FIGURE_NAMES[7]))

    at_age=1000.
    cf_idx=int(np.argmin(np.abs(competing_grid-1800)))
    summary: dict[str,float]={
        "n_train":float(n),"n_observed_failures":float(n_events),
        "fraction_observed_failed":n_events/n,
        "true_weibull_shape":TRUE_SHAPE,"true_weibull_scale":TRUE_SCALE,
        "censored_fit_shape":proper.shape,"censored_fit_scale":proper.scale,
        "failures_only_fit_shape":naive.shape,"failures_only_fit_scale":naive.scale,
        "true_mean_lifetime":mean_lifetime,
        "conditional_horizon":CONDITIONAL_HORIZON,"failure_risk_age":at_age,
        "true_conditional_risk":float(conditional_failure_probability(
            at_age,CONDITIONAL_HORIZON,TRUE_SHAPE,TRUE_SCALE)),
        "censored_fit_conditional_risk":float(conditional_failure_probability(
            at_age,CONDITIONAL_HORIZON,proper.shape,proper.scale)),
        "failures_only_conditional_risk":float(conditional_failure_probability(
            at_age,CONDITIONAL_HORIZON,naive.shape,naive.scale)),
        "preventive_replacement_cost":PREVENTIVE_COST,
        "corrective_failure_cost":FAILURE_COST,
        "true_optimal_replacement_age":true_optimal,
        "censored_fit_optimal_replacement_age":fitted_optimal,
        "failures_only_optimal_replacement_age":naive_optimal,
        "true_optimal_cost_per_hour":float(expected_age_replacement_rate(
            true_optimal,TRUE_SHAPE,TRUE_SCALE)),
        "run_to_failure_cost_per_hour":float(expected_age_replacement_rate(
            float("inf"),TRUE_SHAPE,TRUE_SCALE)),
        "censored_policy_true_cost_per_hour":float(expected_age_replacement_rate(
            fitted_optimal,TRUE_SHAPE,TRUE_SCALE)),
        "failures_only_policy_true_cost_per_hour":float(expected_age_replacement_rate(
            naive_optimal,TRUE_SHAPE,TRUE_SCALE)),
        "n_competing":float(competing.observed.size),
        "n_competing_cause1":float(np.count_nonzero(competing.cause==1)),
        "n_competing_cause2":float(np.count_nonzero(competing.cause==2)),
        "cif_age_near_1800":float(competing_grid[cf_idx]),
        "true_cause1_cif_at_1800":float(cif_true1[cf_idx]),
        "aj_cause1_cif_at_1800":float(cif_aj1[cf_idx]),
        "naive_net_risk_cause1_at_1800":float(net_risk1[cf_idx]),
    }
    _rows_to_csv(result_path("summary.csv"),
                 [{"metric":k,"value":v} for k,v in summary.items()],
                 ("metric","value"))
    return summary


def main() -> None:
    """Run the CLI study with independent output destinations."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="Small CI smoke run with separate files")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent,
                        help="Directory containing figures/ and results/")
    arguments=parser.parse_args()
    summary=run_experiment(arguments.output,quick=arguments.quick)
    for key in (
        "n_observed_failures","censored_fit_shape","censored_fit_scale",
        "failures_only_fit_shape","failures_only_fit_scale",
        "true_optimal_replacement_age","censored_fit_optimal_replacement_age",
        "failures_only_optimal_replacement_age","true_optimal_cost_per_hour",
        "censored_policy_true_cost_per_hour",
        "failures_only_policy_true_cost_per_hour",
        "true_cause1_cif_at_1800","naive_net_risk_cause1_at_1800",
    ):
        print(f"{key}: {summary[key]:.6f}")
    print(f"Wrote {len(FIGURE_NAMES)} figures and {len(RESULT_NAMES)} tables")


if __name__ == "__main__":
    main()

