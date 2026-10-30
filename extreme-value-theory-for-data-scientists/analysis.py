"""Synthetic extreme-value study accompanying the Medium article.

All observations are simulated. The data-generating distribution is specified
exactly, allowing model-based quantiles to be checked against known truth.
Run with --quick to produce smoke-test artifacts with a _quick suffix.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray: TypeAlias = NDArray[np.float64]
Row: TypeAlias = dict[str, str | int | float]

SEED = 20261029
TRUE_THRESHOLD = 10.0
TRUE_EXCEEDANCE_PROBABILITY = 0.10
TRUE_SHAPE = 0.30
TRUE_SCALE = 3.0
BODY_A = 2.0
BODY_B = 5.0
QUANTILES = (0.99, 0.995, 0.999, 0.9995, 0.9999)
THRESHOLD_PROBABILITIES = (0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.965, 0.975)
BLOCK_SIZES = (10, 25, 50, 100, 200)


@dataclass(frozen=True)
class POTFit:
    """Parameters of an iid generalized Pareto excess model."""

    threshold: float
    shape: float
    scale: float
    p_exceed: float
    n_exceed: int
    sample_size: int


@dataclass(frozen=True)
class Study:
    """Model estimates, synthetic samples and publication tables."""

    training: FloatArray
    holdout: FloatArray
    pot: POTFit
    normal_mean: float
    normal_std: float
    lognormal_params: tuple[float, float, float]
    threshold_rows: list[Row]
    quantile_rows: list[Row]
    bootstrap_levels: FloatArray
    block_maxima: FloatArray
    gev_params: tuple[float, float, float]
    iid_extremes: FloatArray
    dependent_extremes: FloatArray
    dependence_rows: list[Row]
    summary_rows: list[Row]


def _check_probability(probability: float) -> None:
    """Reject probabilities outside the strict interior of (0, 1)."""
    if not isinstance(probability, (int, float, np.floating)):
        raise TypeError("probability must be a real number")
    if not 0.0 < float(probability) < 1.0:
        raise ValueError("probability must be strictly between zero and one")


def _check_sample(sample: FloatArray, minimum: int = 30) -> FloatArray:
    """Require finite, positive, one-dimensional real observations."""
    array = np.asarray(sample, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(f"sample must be one-dimensional with >= {minimum} values")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("all observations must be positive and finite")
    return array


def _generator(namespace: int) -> np.random.Generator:
    """Use independent, stable streams for every experimental purpose."""
    return np.random.default_rng(np.random.SeedSequence((SEED, namespace)))


def sample_population(rng: np.random.Generator, size: int) -> FloatArray:
    """Draw the specified beta-body / exact-GPD-tail mixture."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("size must be a positive integer")

    values = TRUE_THRESHOLD * rng.beta(BODY_A, BODY_B, size=size)
    in_tail = rng.random(size) < TRUE_EXCEEDANCE_PROBABILITY
    count = int(in_tail.sum())
    # The body is strictly below 10; exceedances come from an exact GPD.
    values[in_tail] = TRUE_THRESHOLD + stats.genpareto.rvs(
        c=TRUE_SHAPE, loc=0.0, scale=TRUE_SCALE, size=count, random_state=rng
    )
    return np.asarray(values, dtype=np.float64)


def population_survival(x: FloatArray | float) -> FloatArray:
    """Return the exact unconditional survival of the synthetic population."""
    values = np.asarray(x, dtype=np.float64)
    result = np.ones_like(values, dtype=np.float64)
    middle = (values > 0.0) & (values < TRUE_THRESHOLD)
    result[middle] = TRUE_EXCEEDANCE_PROBABILITY + (
        1.0 - TRUE_EXCEEDANCE_PROBABILITY
    ) * stats.beta.sf(values[middle] / TRUE_THRESHOLD, BODY_A, BODY_B)
    tail = values >= TRUE_THRESHOLD
    result[tail] = TRUE_EXCEEDANCE_PROBABILITY * stats.genpareto.sf(
        values[tail] - TRUE_THRESHOLD, c=TRUE_SHAPE, loc=0.0, scale=TRUE_SCALE
    )
    return result


def population_quantile(probability: float) -> float:
    """Return the exact marginal quantile, including the tail transition."""
    _check_probability(probability)
    if probability < 1.0 - TRUE_EXCEEDANCE_PROBABILITY:
        return float(
            TRUE_THRESHOLD
            * stats.beta.ppf(
                probability / (1.0 - TRUE_EXCEEDANCE_PROBABILITY), BODY_A, BODY_B
            )
        )
    return float(
        TRUE_THRESHOLD
        + stats.genpareto.isf(
            (1.0 - probability) / TRUE_EXCEEDANCE_PROBABILITY,
            c=TRUE_SHAPE,
            loc=0.0,
            scale=TRUE_SCALE,
        )
    )


def fit_pot(sample: FloatArray, threshold: float = TRUE_THRESHOLD) -> POTFit:
    """Fit excesses above a fixed threshold with the location fixed at zero."""
    values = _check_sample(sample)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    excesses = values[values > threshold] - threshold
    if excesses.size < 50:
        raise ValueError("at least 50 exceedances are required for the fit")
    shape, location, scale = stats.genpareto.fit(excesses, floc=0.0)
    if location != 0.0 or not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("GPD optimization did not yield admissible parameters")
    return POTFit(
        threshold=float(threshold),
        shape=float(shape),
        scale=float(scale),
        p_exceed=float(excesses.size / values.size),
        n_exceed=int(excesses.size),
        sample_size=int(values.size),
    )


def pot_quantile(probability: float, fit: POTFit) -> float:
    """Estimate an unconditional high quantile using the fitted GPD tail."""
    _check_probability(probability)
    if not isinstance(fit, POTFit):
        raise TypeError("fit must be a POTFit")
    if probability <= 1.0 - fit.p_exceed:
        raise ValueError("quantile must lie above the fitted tail threshold")
    # The isf avoids cancellation when the tail probability is very small.
    quantile = fit.threshold + stats.genpareto.isf(
        (1.0 - probability) / fit.p_exceed,
        c=fit.shape,
        loc=0.0,
        scale=fit.scale,
    )
    if not np.isfinite(quantile):
        raise RuntimeError("extrapolated quantile is not finite")
    return float(quantile)


def pot_survival(x: FloatArray, fit: POTFit) -> FloatArray:
    """Return the unconditional survival where x is above the POT threshold."""
    values = np.asarray(x, dtype=np.float64)
    if np.any(values < fit.threshold):
        raise ValueError("POT survival is defined here only for x >= threshold")
    return np.asarray(
        fit.p_exceed
        * stats.genpareto.sf(
            values - fit.threshold, c=fit.shape, loc=0.0, scale=fit.scale
        ),
        dtype=np.float64,
    )


def extremal_index_proxy(values: FloatArray, threshold: float) -> float:
    """Calculate clusters / exceedances under consecutive-runs declustering.

    This finite-threshold ratio is descriptive, not a consistent extremal-index
    estimator without additional assumptions and a suitable threshold limit.
    """
    sample = _check_sample(values)
    exceed = sample > threshold
    n_exceed = int(exceed.sum())
    if n_exceed == 0:
        raise ValueError("cannot calculate a cluster ratio with zero exceedances")
    n_clusters = int(np.count_nonzero(exceed & np.r_[True, ~exceed[:-1]]))
    return n_clusters / n_exceed


def _dependence_samples(
    size: int,
) -> tuple[FloatArray, FloatArray]:
    """Create independent and 1-dependent sequences with the same marginal.

    If Z is Pareto(alpha=3), both max(Z1, Z2) and max(Z_t, Z_{t-1})
    have CDF (1-x**(-3))**2 for x >= 1. Only the latter clusters extremes.
    """
    if not isinstance(size, int) or size < 1_000:
        raise ValueError("size must be an integer of at least 1000")
    independent_rng = _generator(41)
    dependent_rng = _generator(42)
    iid_latent = 1.0 + independent_rng.pareto(3.0, size=2 * size)
    dep_latent = 1.0 + dependent_rng.pareto(3.0, size=size + 1)
    independent = np.maximum(iid_latent[::2], iid_latent[1::2])
    dependent = np.maximum(dep_latent[:-1], dep_latent[1:])
    return independent, dependent


def _dependence_diagnostics(iid: FloatArray, dependent: FloatArray) -> list[Row]:
    """Compare block-exceedance frequencies against known exact probabilities."""
    probability = 0.99
    threshold = float((1.0 - np.sqrt(probability)) ** (-1.0 / 3.0))
    iid_clusters = extremal_index_proxy(iid, threshold)
    dep_clusters = extremal_index_proxy(dependent, threshold)
    rows: list[Row] = []
    for size in BLOCK_SIZES:
        for label, sample, ratio in (
            ("iid", iid, iid_clusters),
            ("dependent", dependent, dep_clusters),
        ):
            n_blocks = sample.size // size
            maxima = sample[: n_blocks * size].reshape(n_blocks, size).max(axis=1)
            # A dependent block sees size+1 latent shocks, not 2*size.
            exact = (
                1.0 - probability**size
                if label == "iid"
                else 1.0 - probability ** ((size + 1.0) / 2.0)
            )
            rows.append(
                {
                    "process": label,
                    "block_size": size,
                    "block_count": n_blocks,
                    "threshold_at_marginal_p99": threshold,
                    "observed_block_exceedance_probability": float(np.mean(maxima > threshold)),
                    "exact_block_exceedance_probability": exact,
                    "observed_fraction_exceeding": float(np.mean(sample > threshold)),
                    "consecutive_run_clusters_per_exceedance": ratio,
                }
            )
    return rows


def run_study(quick: bool = False) -> Study:
    """Run all seeded numerical experiments without writing output files."""
    if not isinstance(quick, bool):
        raise TypeError("quick must be bool")
    train_n, holdout_n = (5_000, 16_000) if quick else (40_000, 120_000)
    # Keep enough blocks in the smoke run for a meaningful dependence plot.
    dependence_n = 30_000 if quick else 60_000
    bootstrap_count = 18 if quick else 180
    training = sample_population(_generator(10), train_n)
    holdout = sample_population(_generator(11), holdout_n)

    pot = fit_pot(training, TRUE_THRESHOLD)
    normal_mean = float(training.mean())
    normal_std = float(training.std(ddof=0))
    l_shape, l_loc, l_scale = stats.lognorm.fit(training, floc=0.0)
    lognormal_params = (float(l_shape), float(l_loc), float(l_scale))

    threshold_rows: list[Row] = []
    for probability in THRESHOLD_PROBABILITIES:
        threshold = float(np.quantile(training, probability))
        fitted = fit_pot(training, threshold)
        threshold_rows.append(
            {
                "empirical_threshold_quantile": probability,
                "threshold": threshold,
                "number_exceeding": fitted.n_exceed,
                "mean_excess": float((training[training > threshold] - threshold).mean()),
                "fitted_shape": fitted.shape,
                "fitted_scale": fitted.scale,
                "scale_back_to_true_threshold": fitted.scale
                - fitted.shape * (threshold - TRUE_THRESHOLD),
            }
        )

    quantile_rows: list[Row] = []
    for probability in QUANTILES:
        quantile_rows.append(
            {
                "probability": probability,
                "mean_events_between_exceedances_iid": 1.0 / (1.0 - probability),
                "exact_population": population_quantile(probability),
                "peaks_over_threshold": pot_quantile(probability, pot),
                "all_data_gaussian": float(stats.norm.ppf(probability, normal_mean, normal_std)),
                "all_data_lognormal": float(
                    stats.lognorm.ppf(probability, *lognormal_params)
                ),
            }
        )

    # The threshold and model family are fixed across bootstrap samples.
    bootstrap_rng = _generator(20)
    bootstrap_levels: list[float] = []
    target_probability = 0.9999
    for _ in range(bootstrap_count):
        resampled = bootstrap_rng.choice(training, size=training.size, replace=True)
        fitted = fit_pot(resampled, TRUE_THRESHOLD)
        bootstrap_levels.append(pot_quantile(target_probability, fitted))
    bootstrap_array = np.asarray(bootstrap_levels, dtype=np.float64)

    block_size = 100
    blocks = training[: (training.size // block_size) * block_size].reshape(-1, block_size)
    block_maxima = np.asarray(blocks.max(axis=1), dtype=np.float64)
    c, loc, scale = stats.genextreme.fit(block_maxima)
    if not np.all(np.isfinite((c, loc, scale))) or scale <= 0:
        raise RuntimeError("GEV optimization did not yield admissible parameters")

    iid_extremes, dependent_extremes = _dependence_samples(dependence_n)
    dependence_rows = _dependence_diagnostics(iid_extremes, dependent_extremes)
    lower, upper = np.quantile(bootstrap_array, (0.025, 0.975))
    q_target = next(row for row in quantile_rows if row["probability"] == target_probability)
    summary_rows: list[Row] = [
        {
            "seed": SEED,
            "training_sample_size": train_n,
            "holdout_sample_size": holdout_n,
            "true_threshold": TRUE_THRESHOLD,
            "true_shape": TRUE_SHAPE,
            "true_scale": TRUE_SCALE,
            "true_fraction_exceeding": TRUE_EXCEEDANCE_PROBABILITY,
            "fitted_shape": pot.shape,
            "fitted_scale": pot.scale,
            "fitted_fraction_exceeding": pot.p_exceed,
            "number_exceeding": pot.n_exceed,
            "target_probability": target_probability,
            "exact_target_quantile": float(q_target["exact_population"]),
            "pot_target_quantile": float(q_target["peaks_over_threshold"]),
            "gaussian_target_quantile": float(q_target["all_data_gaussian"]),
            "lognormal_target_quantile": float(q_target["all_data_lognormal"]),
            "bootstrap_repetitions": bootstrap_count,
            "bootstrap_lower_95": float(lower),
            "bootstrap_upper_95": float(upper),
            "block_size_for_gev": block_size,
            "fitted_gev_shape_xi": float(-c),
            "fitted_gev_location": float(loc),
            "fitted_gev_scale": float(scale),
            "iid_cluster_ratio": float(dependence_rows[0]["consecutive_run_clusters_per_exceedance"]),
            "dependent_cluster_ratio": float(
                dependence_rows[1]["consecutive_run_clusters_per_exceedance"]
            ),
        }
    ]
    return Study(
        training=training,
        holdout=holdout,
        pot=pot,
        normal_mean=normal_mean,
        normal_std=normal_std,
        lognormal_params=lognormal_params,
        threshold_rows=threshold_rows,
        quantile_rows=quantile_rows,
        bootstrap_levels=bootstrap_array,
        block_maxima=block_maxima,
        gev_params=(float(c), float(loc), float(scale)),
        iid_extremes=iid_extremes,
        dependent_extremes=dependent_extremes,
        dependence_rows=dependence_rows,
        summary_rows=summary_rows,
    )


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write deterministic UTF-8 CSV with an explicit column order."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not rows:
        raise ValueError("at least one row is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _empirical_survival(values: FloatArray, x: FloatArray) -> FloatArray:
    """Calculate a sample survival curve without quadratic broadcasting."""
    ordered = np.sort(values)
    counts = ordered.size - np.searchsorted(ordered, x, side="right")
    return np.asarray(counts / ordered.size, dtype=np.float64)


def make_figures(study: Study, output: Path, *, suffix: str = "") -> None:
    """Render eight labeled publication figures from actual computed estimates."""
    if not isinstance(study, Study):
        raise TypeError("study must be a Study")
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False}
    )
    blue, teal, orange, red = "#3b5878", "#3c8e92", "#cf9252", "#b9544d"

    # Figure 1: most observations are in the body, but the far tail drives risk.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    axes[0].hist(study.training, bins=np.linspace(0, 50, 80), color=blue, alpha=0.8)
    axes[0].axvline(TRUE_THRESHOLD, color=red, ls="--", label="Known tail boundary")
    axes[0].set_xlabel("Synthetic event magnitude")
    axes[0].set_ylabel("Observed events")
    axes[0].set_title("Most records look ordinary")
    axes[0].legend()
    grid = np.geomspace(2.0, 150.0, num=180)
    for values, label, color in (
        (study.training, "Training observations", blue),
        (study.holdout, "Independent holdout", teal),
    ):
        s = _empirical_survival(values, grid)
        axes[1].plot(grid, np.where(s > 0, s, np.nan), color=color, label=label)
    axes[1].plot(grid, population_survival(grid), color=red, ls="--", label="Known population")
    axes[1].axvline(TRUE_THRESHOLD, color="gray", ls=":")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-5, 1.0)
    axes[1].set_xlabel("Synthetic event magnitude")
    axes[1].set_ylabel("Fraction exceeding magnitude")
    axes[1].set_title("The tail needs a different scale")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Synthetic observations: a beta body and a GPD upper tail")
    fig.savefig(output / f"fig1_body_and_tail{suffix}.png", dpi=170)
    plt.close(fig)

    # Figure 2: compare models at extrapolation levels they were not tuned to.
    fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    rows = study.quantile_rows
    periods = [float(r["mean_events_between_exceedances_iid"]) for r in rows]
    for field, label, color, style in (
        ("exact_population", "Known population", red, "--"),
        ("peaks_over_threshold", "GPD above fixed threshold", teal, "-"),
        ("all_data_lognormal", "Lognormal fit to all data", orange, "-"),
        ("all_data_gaussian", "Gaussian fit to all data", blue, "-"),
    ):
        ax.plot(periods, [float(r[field]) for r in rows], marker="o", color=color, linestyle=style, label=label)
    ax.set_xscale("log")
    # Floating-point division can produce 99.999... for a 100-event level.
    ax.set_xticks(periods, [f"{round(p):,}" for p in periods])
    ax.set_xlabel("Mean event count per exceedance (marginal probability)")
    ax.set_ylabel("Estimated magnitude")
    ax.set_title("The far-quantile question changes the answer")
    ax.grid(alpha=0.24)
    ax.legend()
    fig.savefig(output / f"fig2_far_tail_quantiles{suffix}.png", dpi=170)
    plt.close(fig)

    # Figure 3: empirical mean excess becomes linear above the true tail boundary.
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    levels = study.threshold_rows
    thresholds = np.array([float(r["threshold"]) for r in levels])
    mean_excess = np.array([float(r["mean_excess"]) for r in levels])
    ax.plot(thresholds, mean_excess, "-o", color=blue, label="Sample mean excess")
    theory_x = np.linspace(TRUE_THRESHOLD, max(float(thresholds.max()), TRUE_THRESHOLD + 0.5), num=60)
    theory_y = (TRUE_SCALE + TRUE_SHAPE * (theory_x - TRUE_THRESHOLD)) / (1.0 - TRUE_SHAPE)
    ax.plot(theory_x, theory_y, color=teal, ls="--", label="Exact mean excess in the GPD region")
    ax.axvline(TRUE_THRESHOLD, color=red, ls=":", label="Known tail boundary")
    ax.set_xlabel("Candidate threshold")
    ax.set_ylabel("Mean excess above threshold")
    ax.set_title("Mean-excess diagnostic: a check, not a threshold oracle")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.savefig(output / f"fig3_mean_excess{suffix}.png", dpi=170)
    plt.close(fig)

    # Figure 4: show fit stability and the bias / sample-size trade-off.
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), constrained_layout=True)
    shape_estimates = np.array([float(r["fitted_shape"]) for r in levels])
    n_excess = np.array([int(r["number_exceeding"]) for r in levels])
    axes[0].plot(thresholds, shape_estimates, "-o", color=blue, label="GPD shape estimate")
    axes[0].axhline(TRUE_SHAPE, color=teal, ls="--", label="Known shape")
    axes[0].set_ylabel("Tail shape ξ")
    axes[0].set_title("Shape estimate across thresholds")
    axes[0].legend()
    axes[1].plot(thresholds, n_excess, "-o", color=orange)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Observations above threshold")
    axes[1].set_title("Higher thresholds have less information")
    for ax in axes:
        ax.axvline(TRUE_THRESHOLD, color=red, ls=":")
        ax.set_xlabel("Candidate threshold")
        ax.grid(alpha=0.22)
    fig.suptitle("Threshold choice trades approximation bias against estimation variance")
    fig.savefig(output / f"fig4_threshold_stability{suffix}.png", dpi=170)
    plt.close(fig)

    # Figure 5: the fitted model must also pass within-tail checks.
    excess = np.sort(study.training[study.training > study.pot.threshold] - study.pot.threshold)
    probabilities = (np.arange(excess.size) + 0.5) / excess.size
    theoretical = stats.genpareto.ppf(probabilities, c=study.pot.shape, scale=study.pot.scale)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    limit = float(max(np.quantile(excess, 0.995), np.quantile(theoretical, 0.995)))
    axes[0].scatter(theoretical, excess, color=blue, alpha=0.18, s=8)
    axes[0].plot([0, limit], [0, limit], ls="--", color=red, label="45° reference")
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("Model exceedance quantiles")
    axes[0].set_ylabel("Observed exceedances")
    axes[0].set_title("Q-Q comparison within the fitted tail")
    axes[0].legend()
    excess_grid = np.geomspace(0.1, max(45, np.quantile(excess, 0.999)), 150)
    axes[1].plot(excess_grid, _empirical_survival(excess, excess_grid), color=blue, label="Observed excesses")
    axes[1].plot(
        excess_grid,
        stats.genpareto.sf(excess_grid, c=study.pot.shape, scale=study.pot.scale),
        color=red,
        ls="--",
        label="Fitted GPD",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-4, 1.0)
    axes[1].set_xlabel("Excess above the fixed threshold")
    axes[1].set_ylabel("Conditional fraction exceeding")
    axes[1].set_title("Conditional excess-survival check")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("A small parameter standard error does not replace model diagnostics")
    fig.savefig(output / f"fig5_gpd_fit_diagnostics{suffix}.png", dpi=170)
    plt.close(fig)

    # Figure 6: nonparametric iid bootstrap is conditional on model and threshold.
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    ax.hist(study.bootstrap_levels, bins=28, color=blue, alpha=0.82)
    low, high = np.quantile(study.bootstrap_levels, (0.025, 0.975))
    estimated = pot_quantile(0.9999, study.pot)
    ax.axvline(estimated, color=teal, lw=2, label="Original fitted return level")
    ax.axvline(population_quantile(0.9999), color=red, lw=2, ls="--", label="Known population value")
    ax.axvspan(float(low), float(high), color=orange, alpha=0.18, label="Percentile bootstrap 95% interval")
    ax.set_xlabel("Estimated 99.99th-percentile magnitude")
    ax.set_ylabel("Bootstrap replications")
    ax.set_title("Uncertainty grows when extrapolating far beyond the data")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / f"fig6_return_level_uncertainty{suffix}.png", dpi=170)
    plt.close(fig)

    # Figure 7: use non-overlapping blocks and SciPy's opposite GEV shape sign.
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), constrained_layout=True)
    c, loc, scale = study.gev_params
    block_size = 100
    xgrid = np.linspace(
        float(np.quantile(study.block_maxima, 0.025)),
        float(np.quantile(study.block_maxima, 0.99)) * 1.25,
        num=180,
    )
    ordered_max = np.sort(study.block_maxima)
    axes[0].step(
        ordered_max,
        np.arange(1, ordered_max.size + 1) / ordered_max.size,
        where="post",
        color=blue,
        label="Observed block maxima",
    )
    axes[0].plot(
        xgrid,
        (1.0 - population_survival(xgrid)) ** block_size,
        color=red,
        ls="--",
        label="Exact block-maximum CDF",
    )
    axes[0].plot(xgrid, stats.genextreme.cdf(xgrid, c=c, loc=loc, scale=scale), color=teal, label="Fitted GEV CDF")
    axes[0].set_xlabel("Block maximum")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Maximum of 100 independent events")
    axes[0].legend()
    qgrid = np.array((0.5, 0.8, 0.9, 0.95, 0.99))
    axes[1].plot(
        qgrid,
        [population_quantile(float(q ** (1.0 / block_size))) for q in qgrid],
        "--o",
        color=red,
        label="Exact",
    )
    axes[1].plot(qgrid, stats.genextreme.ppf(qgrid, c=c, loc=loc, scale=scale), "-o", color=teal, label="Fitted GEV")
    axes[1].plot(
        qgrid,
        np.quantile(study.block_maxima, qgrid),
        "x",
        color=blue,
        markersize=8,
        label="Empirical maxima",
    )
    axes[1].set_xlabel("Block-maximum probability")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Block-maximum quantiles")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Block maxima and threshold exceedances are different routes to extremes")
    fig.savefig(output / f"fig7_block_maxima_gev{suffix}.png", dpi=170)
    plt.close(fig)

    # Figure 8: marginal tails match, but dependent extremes arrive in clusters.
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.9), constrained_layout=True)
    xgrid = np.geomspace(1.15, 17.0, num=160)
    for sample, label, color in (
        (study.iid_extremes, "Independent observations", blue),
        (study.dependent_extremes, "Moving-maximum process", teal),
    ):
        empirical = _empirical_survival(sample, xgrid)
        axes[0].plot(xgrid, np.where(empirical > 0.0, empirical, np.nan), color=color, label=label)
    axes[0].plot(xgrid, 1.0 - (1.0 - xgrid ** (-3.0)) ** 2, ls="--", color=red, label="Shared exact marginal tail")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-4, 1.0)
    axes[0].set_xlabel("Magnitude")
    axes[0].set_ylabel("Fraction exceeding")
    axes[0].set_title("The one-point distributions match")
    axes[0].legend()
    for label, color in (("iid", blue), ("dependent", teal)):
        rows = [r for r in study.dependence_rows if r["process"] == label]
        sizes = [int(r["block_size"]) for r in rows]
        axes[1].plot(
            sizes,
            [float(r["exact_block_exceedance_probability"]) for r in rows],
            color=color,
            ls="--",
            label=f"{label}: exact",
        )
        axes[1].scatter(
            sizes,
            [float(r["observed_block_exceedance_probability"]) for r in rows],
            color=color,
            s=42,
            label=f"{label}: sampled",
        )
    axes[1].set_xlabel("Consecutive observations per block")
    axes[1].set_ylabel("P(at least one event exceeds marginal p99)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Clustering changes block-level risk")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Equal marginal tails do not imply equal temporal risk")
    fig.savefig(output / f"fig8_dependence_clusters{suffix}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    """Generate full publication evidence, or separate smoke-test outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced smoke test, preserving full artifacts")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not isinstance(args.output_root, Path):
        raise TypeError("output-root must resolve to a Path")
    study = run_study(quick=bool(args.quick))
    suffix = "_quick" if args.quick else ""
    root = args.output_root
    write_csv(root / "results" / f"summary{suffix}.csv", study.summary_rows)
    write_csv(root / "results" / f"model_quantiles{suffix}.csv", study.quantile_rows)
    write_csv(root / "results" / f"threshold_diagnostics{suffix}.csv", study.threshold_rows)
    write_csv(
        root / "results" / f"bootstrap_return_levels{suffix}.csv",
        [
            {"replication": index + 1, "p9999_return_level": float(level)}
            for index, level in enumerate(study.bootstrap_levels)
        ],
    )
    write_csv(root / "results" / f"dependence_diagnostics{suffix}.csv", study.dependence_rows)
    make_figures(study, root / "figures", suffix=suffix)
    for key, value in study.summary_rows[0].items():
        print(f"{key:33} {value}")


if __name__ == "__main__":
    main()
