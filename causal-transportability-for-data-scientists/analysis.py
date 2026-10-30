"""Synthetic, reproducible study of transporting a randomized trial's ATE.

The source trial and target-only cohort are disjoint: S=1 versus S=0.
Treatment is randomized only inside the trial. Every population, outcome,
and numerical claim in the accompanying article is simulated by this file.

Example::

    python analysis.py
    python analysis.py --quick --output /tmp/transport-smoke
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
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

FloatArray = NDArray[np.float64]
BLUE, RED, TEAL, ORANGE, GRAY, PURPLE = (
    "#26629c", "#b6494f", "#188a7a", "#cb8138", "#596271", "#845c9c"
)


@dataclass(frozen=True)
class Design:
    """Explicit trial, target, treatment, and selection specifications."""

    seed: int = 20261030
    n_trial: int = 550
    n_target: int = 1800
    selection_slope: float = 0.8
    hidden_selection: float = 0.0
    hidden_effect: float = 0.0
    p_treatment: float = 0.5
    p_binary: float = 0.45
    p_hidden: float = 0.5
    noise_sd: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be a non-boolean integer")
        for name in ("n_trial", "n_target"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a positive integer")
            if value < 40:
                raise ValueError(f"{name} must be at least 40")
        for name in ("selection_slope", "hidden_selection", "hidden_effect",
                     "p_treatment", "p_binary", "p_hidden", "noise_sd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("p_treatment", "p_binary", "p_hidden"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if self.selection_slope < 0 or self.noise_sd <= 0:
            raise ValueError("selection_slope must be nonnegative and noise_sd positive")


@dataclass(frozen=True)
class Cohorts:
    """Observed trial units and observed target covariates (no target outcomes)."""

    trial_x: FloatArray
    trial_z: FloatArray
    trial_a: FloatArray
    trial_y: FloatArray
    target_x: FloatArray
    target_z: FloatArray


def treatment_effect(x: FloatArray, z: FloatArray, u: FloatArray | float = 0.0,
                     hidden_effect: float = 0.0) -> FloatArray:
    """Structural additive treatment effect; may depend on an unobserved U."""

    x_arr, z_arr = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    if x_arr.shape != z_arr.shape:
        raise ValueError("x and z must share a shape")
    return 0.4 + 0.9 * z_arr + 0.5 * x_arr + 0.35 * z_arr * x_arr + hidden_effect * u


def selection_probability(x: FloatArray, z: FloatArray, slope: float,
                          u: FloatArray | float = 0.0,
                          hidden_selection: float = 0.0) -> FloatArray:
    """Superpopulation probability of randomized-trial participation."""

    return np.asarray(expit(-0.85 - 1.1 * z - slope * x + hidden_selection * u),
                      dtype=np.float64)


def _conditional_sample(n: int, selected: bool, rng: np.random.Generator,
                        design: Design) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independent rejection sample from X,Z,U given S=selected."""

    x_parts: list[FloatArray] = []
    z_parts: list[FloatArray] = []
    u_parts: list[FloatArray] = []
    remaining = n
    while remaining > 0:
        batch = max(300, remaining * 5)
        x = rng.normal(size=batch)
        z = rng.binomial(1, design.p_binary, size=batch).astype(float)
        u = rng.binomial(1, design.p_hidden, size=batch).astype(float)
        p = selection_probability(x, z, design.selection_slope,
                                  u, design.hidden_selection)
        eligible = rng.random(batch) < p
        keep = np.flatnonzero(eligible == selected)[:remaining]
        if keep.size:
            x_parts.append(x[keep])
            z_parts.append(z[keep])
            u_parts.append(u[keep])
            remaining -= keep.size
    return np.concatenate(x_parts), np.concatenate(z_parts), np.concatenate(u_parts)


def sample_cohorts(design: Design, rng: np.random.Generator) -> Cohorts:
    """Draw disjoint source and target populations with randomized trial A."""

    tx, tz, tu = _conditional_sample(design.n_trial, True, rng, design)
    px, pz, _ = _conditional_sample(design.n_target, False, rng, design)
    a = rng.binomial(1, design.p_treatment, size=design.n_trial).astype(float)
    baseline = 1.0 + 0.5 * tz + 0.4 * tx + 0.15 * tx * tx
    y = (baseline + a * treatment_effect(tx, tz, tu, design.hidden_effect)
         + rng.normal(0.0, design.noise_sd, size=design.n_trial))
    return Cohorts(tx, tz, a, y, px, pz)


def oracle_effects(slope: float = 0.8, hidden_selection: float = 0.0,
                   hidden_effect: float = 0.0, p_binary: float = 0.45,
                   p_hidden: float = 0.5, n_nodes: int = 80) -> dict[str, float]:
    """Gaussian-Hermite integration of exact trial/target ATE and oracle bias.

    oracle_transport is what an infinitely large *observed-covariate-only*
    standardization would yield, even with perfectly fitted observed models.
    """

    if n_nodes < 16:
        raise ValueError("n_nodes must be at least 16")
    nodes, weights = hermgauss(n_nodes)
    xs = nodes * math.sqrt(2.0)
    density = weights / math.sqrt(math.pi)
    mass_trial = mass_target = num_trial = num_target = num_observed = 0.0
    for z, pz in ((0.0, 1 - p_binary), (1.0, p_binary)):
        zz = np.full_like(xs, z)
        p0 = selection_probability(xs, zz, slope, 0.0, hidden_selection)
        p1 = selection_probability(xs, zz, slope, 1.0, hidden_selection)
        p_source = (1 - p_hidden) * p0 + p_hidden * p1
        p_target = 1 - p_source
        # The latent probability among trial participants is generally not
        # the same as the probability among target nonparticipants.
        u_trial = p_hidden * p1 / p_source
        base_effect = treatment_effect(xs, zz)
        target_u_mass = p_hidden * (1 - p1)
        mass_trial += pz * float(np.dot(density, p_source))
        mass_target += pz * float(np.dot(density, p_target))
        num_trial += pz * float(np.dot(density,
            base_effect * p_source + hidden_effect * p_hidden * p1))
        num_target += pz * float(np.dot(density,
            base_effect * p_target + hidden_effect * target_u_mass))
        num_observed += pz * float(np.dot(density,
            p_target * (base_effect + hidden_effect * u_trial)))
    return {
        "trial_ate": num_trial / mass_trial,
        "target_ate": num_target / mass_target,
        "oracle_observed_transport": num_observed / mass_target,
        "oracle_transport_bias": (num_observed - num_target) / mass_target,
        "trial_participation_probability": mass_trial,
        "target_nonparticipation_probability": mass_target,
    }


def feature_matrix(x: FloatArray, z: FloatArray, rich: bool) -> FloatArray:
    """Outcome features: rich model is correct for E[Y|A,X,Z,S=1]."""

    if x.ndim != 1 or z.shape != x.shape:
        raise ValueError("x and z must be matching one-dimensional arrays")
    if rich:
        return np.column_stack((np.ones_like(x), z, x, z * x, x * x))
    return np.column_stack((np.ones_like(x), z))


def participation_matrix(x: FloatArray, z: FloatArray, rich: bool) -> FloatArray:
    """Selection features; omitting X makes the reduced model misspecified."""

    if x.ndim != 1 or z.shape != x.shape:
        raise ValueError("x and z must be matching one-dimensional arrays")
    return np.column_stack((np.ones_like(x), z, x)) if rich else feature_matrix(x, z, False)


def fit_logit(features: FloatArray, labels: FloatArray) -> FloatArray:
    """Stable unpenalized logistic MLE for pooled trial/target sample labels."""

    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("logistic inputs must have compatible shapes")
    if not np.all(np.isfinite(features)) or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("finite features and binary labels required")

    def objective(beta: FloatArray) -> tuple[float, FloatArray]:
        logits = features @ beta
        loss = np.logaddexp(0.0, logits).sum() - np.dot(labels, logits)
        grad = features.T @ (expit(logits) - labels)
        return float(loss), grad.astype(float)

    result = minimize(objective, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 100, "ftol": 1e-11})
    if not result.success and np.linalg.norm(result.jac) > 1e-3:
        raise RuntimeError(f"Logistic regression did not converge: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def trial_target_weights(cohorts: Cohorts, rich: bool) -> FloatArray:
    """Estimated inverse sampling odds for *disjoint* S=1 and S=0 cohorts.

    Fitting on concatenated case-control-like samples changes the logistic
    intercept, but its odds ratio remains proportional to f_target/f_trial.
    The constant cancels in self-normalized estimators.
    """

    x = np.concatenate((cohorts.trial_x, cohorts.target_x))
    z = np.concatenate((cohorts.trial_z, cohorts.target_z))
    s = np.r_[np.ones(cohorts.trial_x.size), np.zeros(cohorts.target_x.size)]
    beta = fit_logit(participation_matrix(x, z, rich), s)
    logits = participation_matrix(cohorts.trial_x, cohorts.trial_z, rich) @ beta
    # Relative inverse odds exp(-logit) can be rescaled without changing a
    # Hájek estimator. Subtracting a constant prevents numerical overflow.
    relative_logits = -logits
    relative_logits -= relative_logits.max()
    weights = np.exp(relative_logits)
    if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
        raise FloatingPointError("selection weights are not finite and positive")
    return weights


def effective_sample_size(weights: FloatArray) -> float:
    """Kish effective sample size for nonnegative unequal weights."""

    if weights.ndim != 1 or weights.size == 0 or np.any(weights < 0):
        raise ValueError("weights must be a nonempty nonnegative vector")
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("weights must be finite and have positive sum")
    return float(weights.sum() ** 2 / np.dot(weights, weights))


def outcome_predictions(cohorts: Cohorts, rich: bool) -> tuple[FloatArray, FloatArray,
                                                               FloatArray, FloatArray]:
    """Fit separate randomized treatment arms and predict on both cohorts."""

    source = feature_matrix(cohorts.trial_x, cohorts.trial_z, rich)
    target = feature_matrix(cohorts.target_x, cohorts.target_z, rich)
    if (cohorts.trial_a == 1).sum() < source.shape[1] or (cohorts.trial_a == 0).sum() < source.shape[1]:
        raise ValueError("insufficient treated or untreated trial participants")
    fitted: list[tuple[FloatArray, FloatArray]] = []
    for arm in (0.0, 1.0):
        mask = cohorts.trial_a == arm
        coeff, *_ = np.linalg.lstsq(source[mask], cohorts.trial_y[mask], rcond=None)
        fitted.append((source @ coeff, target @ coeff))
    return fitted[0][0], fitted[1][0], fitted[0][1], fitted[1][1]


def weighted_ate(cohorts: Cohorts, weights: FloatArray) -> float:
    """Separate normalized inverse-odds means in randomized trial arms."""

    yes, no = cohorts.trial_a == 1, cohorts.trial_a == 0
    if not weights[yes].sum() > 0 or not weights[no].sum() > 0:
        raise ValueError("both treatment arms require positive weight")
    return float(np.average(cohorts.trial_y[yes], weights=weights[yes])
                 - np.average(cohorts.trial_y[no], weights=weights[no]))


def augmented_ate(cohorts: Cohorts, weights: FloatArray, prediction: tuple[
    FloatArray, FloatArray, FloatArray, FloatArray], p_treatment: float = 0.5) -> float:
    """Target g-formula plus trial inverse-odds weighted residual correction.

    Consistent if the conditional outcome model OR selection odds model is
    correct (with randomized treatment, selection exchangeability, positivity,
    and sufficient regularity), but not necessarily when both are incorrect.
    """

    if not 0 < p_treatment < 1:
        raise ValueError("treatment probability must be in (0,1)")
    m0_trial, m1_trial, m0_target, m1_target = prediction
    a = cohorts.trial_a
    residual = (a / p_treatment * (cohorts.trial_y - m1_trial)
                - (1 - a) / (1 - p_treatment) * (cohorts.trial_y - m0_trial))
    return float(np.mean(m1_target - m0_target) + np.average(residual, weights=weights))


METHODS: tuple[str, ...] = (
    "naive_trial", "standardization_rich", "standardization_reduced",
    "inverse_odds_rich", "inverse_odds_reduced", "augmented_both_rich",
    "augmented_outcome_rich_selection_reduced",
    "augmented_outcome_reduced_selection_rich", "augmented_both_reduced",
)


def estimate_all(cohorts: Cohorts, p_treatment: float = 0.5) -> tuple[dict[str, float],
                                                                   dict[str, float]]:
    """Nine ATE estimators and weight concentration diagnostics."""

    prediction_rich = outcome_predictions(cohorts, rich=True)
    prediction_reduced = outcome_predictions(cohorts, rich=False)
    weight_rich = trial_target_weights(cohorts, rich=True)
    weight_reduced = trial_target_weights(cohorts, rich=False)
    a = cohorts.trial_a
    estimated = {
        "naive_trial": float(cohorts.trial_y[a == 1].mean() - cohorts.trial_y[a == 0].mean()),
        "standardization_rich": float(np.mean(prediction_rich[3] - prediction_rich[2])),
        "standardization_reduced": float(np.mean(prediction_reduced[3] - prediction_reduced[2])),
        "inverse_odds_rich": weighted_ate(cohorts, weight_rich),
        "inverse_odds_reduced": weighted_ate(cohorts, weight_reduced),
        "augmented_both_rich": augmented_ate(cohorts, weight_rich, prediction_rich, p_treatment),
        "augmented_outcome_rich_selection_reduced": augmented_ate(
            cohorts, weight_reduced, prediction_rich, p_treatment),
        "augmented_outcome_reduced_selection_rich": augmented_ate(
            cohorts, weight_rich, prediction_reduced, p_treatment),
        "augmented_both_reduced": augmented_ate(
            cohorts, weight_reduced, prediction_reduced, p_treatment),
    }
    diagnostics = {
        "rich_ess": effective_sample_size(weight_rich),
        "reduced_ess": effective_sample_size(weight_reduced),
        "rich_max_to_mean_weight": float(weight_rich.max() / weight_rich.mean()),
        "rich_p99_to_median_weight": float(np.quantile(weight_rich, 0.99) / np.median(weight_rich)),
    }
    return estimated, diagnostics


def _record_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save a schema-stable CSV; reject missing columns/nonfinite numbers."""

    if not rows:
        raise ValueError(f"cannot write empty evidence table: {path}")
    columns = list(rows[0])
    for row in rows:
        if list(row) != columns:
            raise ValueError(f"inconsistent table columns: {path}")
        for value in row.values():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise ValueError(f"nonfinite table evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary(records: list[dict[str, Any]], truth: float,
             group_key: str = "method") -> list[dict[str, Any]]:
    """Finite Monte Carlo bias, variance and RMSE using exact population truth."""

    output: list[dict[str, Any]] = []
    for name in sorted({str(r[group_key]) for r in records}):
        estimates = np.asarray([r["estimate"] for r in records if r[group_key] == name], dtype=float)
        output.append({group_key: name, "replicates": int(estimates.size),
                       "target_truth": truth, "mean_estimate": float(estimates.mean()),
                       "bias": float(estimates.mean() - truth),
                       "sd": float(estimates.std(ddof=1)),
                       "rmse": float(np.sqrt(np.mean((estimates - truth) ** 2)))})
    return output


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Consistent restrained publication styling."""

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.17)


def _save(fig: Any, output: Path, quick: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    name = output.with_name(f"{output.stem}{'_quick' if quick else ''}{output.suffix}")
    fig.savefig(name, dpi=125 if quick else 170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _table(result_dir: Path, name: str, rows: list[dict[str, Any]], quick: bool) -> None:
    _record_table(result_dir / (name + ("_quick" if quick else "") + ".csv"), rows)


def run_study(output_root: Path, quick: bool = False) -> dict[str, float]:
    """Generate all figures, evidence tables, and deterministic checks."""

    if not isinstance(output_root, Path):
        raise TypeError("output_root must be pathlib.Path")
    output_root.mkdir(parents=True, exist_ok=True)
    result_dir, figure_dir = output_root / "results", output_root / "figures"
    design = Design(n_trial=250 if quick else 550,
                    n_target=700 if quick else 1800)
    rng = np.random.default_rng(design.seed)
    source = sample_cohorts(design, rng)
    oracle = oracle_effects()
    observed, diagnostics = estimate_all(source)
    _table(result_dir, "population_truth", [oracle | {"source": "Gaussian-Hermite quadrature"}], quick)

    cov_rows: list[dict[str, Any]] = []
    for group, x, z in (("trial", source.trial_x, source.trial_z),
                        ("target_nonparticipants", source.target_x, source.target_z)):
        cov_rows.append({"group": group, "n": int(x.size), "x_mean": float(x.mean()),
                         "x_sd": float(x.std(ddof=1)), "z_prevalence": float(z.mean()),
                         "mean_structural_effect": float(treatment_effect(x, z).mean())})
    _table(result_dir, "cohort_diagnostics", cov_rows, quick)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    grid = np.linspace(-3.5, 3.5, 50)
    ax.hist(source.trial_x, bins=grid, density=True, alpha=0.46, color=BLUE, label="Randomized trial S=1")
    ax.hist(source.target_x, bins=grid, density=True, alpha=0.46, color=ORANGE, label="Target only S=0")
    ax.legend(frameon=False)
    _style(ax, "One source population, two different covariate distributions", "Effect modifier X", "Density")
    _save(fig, figure_dir / "fig1_trial_target_shift.png", quick)

    density_rows: list[dict[str, Any]] = []
    for group, x in (("trial", source.trial_x), ("target_nonparticipants", source.target_x)):
        hist, edges = np.histogram(x, bins=grid, density=True)
        density_rows.extend({"group": group, "left": float(lo), "right": float(hi),
                             "density": float(value)} for lo, hi, value in zip(edges[:-1], edges[1:], hist))
    _table(result_dir, "covariate_density", density_rows, quick)

    xgrid = np.linspace(-2.5, 2.5, 101)
    effect_rows = [{"x": float(x), "binary_modifier": z,
                    "effect": float(treatment_effect(np.array([x]), np.array([float(z)]))[0])}
                   for z in (0, 1) for x in xgrid]
    _table(result_dir, "conditional_effect_curves", effect_rows, quick)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for z, color in ((0, BLUE), (1, RED)):
        ey = np.array([r["effect"] for r in effect_rows if r["binary_modifier"] == z])
        ax.plot(xgrid, ey, color=color, label=f"Z={z}", linewidth=2.3)
    ax.axhline(oracle["trial_ate"], color=GRAY, linestyle="--", label="Trial ATE")
    ax.axhline(oracle["target_ate"], color=TEAL, linestyle="--", label="Target ATE")
    ax.legend(frameon=False, ncol=2)
    _style(ax, "Treatment-effect heterogeneity drives the transport gap", "Effect modifier X", "Individual treatment effect")
    _save(fig, figure_dir / "fig2_effect_heterogeneity.png", quick)

    main_n = 35 if quick else 300
    main_rows: list[dict[str, Any]] = []
    for replicate in range(main_n):
        cohort = sample_cohorts(Design(seed=replicate, n_trial=250 if quick else 500,
                                       n_target=650 if quick else 1400), rng)
        estimates, _ = estimate_all(cohort)
        main_rows.extend({"replicate": replicate, "method": name, "estimate": value,
                          "target_truth": oracle["target_ate"]} for name, value in estimates.items())
    _table(result_dir, "estimator_replicates", main_rows, quick)
    main_summary = _summary(main_rows, oracle["target_ate"])
    _table(result_dir, "estimator_summary", main_summary, quick)

    labels = ["Trial", "G: correct", "G: omitted X", "W: correct", "W: omitted X",
              "DR: both right", "DR: outcome right", "DR: weights right", "DR: both wrong"]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    distribution = [np.array([r["estimate"] for r in main_rows if r["method"] == method])
                    for method in METHODS]
    artists = ax.boxplot(distribution, tick_labels=labels, patch_artist=True, showfliers=False)
    for i, patch in enumerate(artists["boxes"]):
        patch.set(facecolor=TEAL if i in (1, 3, 5, 6, 7) else ORANGE, alpha=0.55)
    ax.axhline(oracle["target_ate"], color=RED, linewidth=2, label="Exact target ATE")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(frameon=False)
    _style(ax, "Trial internal validity does not remove external selection bias",
           "Estimator and nuisance-model assumptions", "Estimated target ATE")
    _save(fig, figure_dir / "fig3_estimators.png", quick)

    # Rows: selection right/wrong; columns: outcome right/wrong.
    bias_map = np.array([
        [next(r["bias"] for r in main_summary if r["method"] == "augmented_both_rich"),
         next(r["bias"] for r in main_summary if r["method"] == "augmented_outcome_reduced_selection_rich")],
        [next(r["bias"] for r in main_summary if r["method"] == "augmented_outcome_rich_selection_reduced"),
         next(r["bias"] for r in main_summary if r["method"] == "augmented_both_reduced")],
    ])
    bound = max(0.12, float(np.abs(bias_map).max()))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    img = ax.imshow(bias_map, vmin=-bound, vmax=bound, cmap="RdBu_r", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{bias_map[i,j]:+.3f}", ha="center", va="center",
                    fontsize=17, color="white" if abs(bias_map[i,j]) > bound * .5 else "black")
    ax.set(xticks=[0, 1], xticklabels=["Outcome model correct", "Outcome model reduced"],
           yticks=[0, 1], yticklabels=["Selection model correct", "Selection model reduced"],
           title="Augmentation has two chances, not unconditional immunity")
    fig.colorbar(img, ax=ax, label="Monte Carlo bias")
    _save(fig, figure_dir / "fig4_double_robustness.png", quick)

    severity_slopes = (0.3, 0.8, 1.4, 2.0)
    overlap_rows: list[dict[str, Any]] = []
    for slope in severity_slopes:
        truth = oracle_effects(slope=slope)["target_ate"]
        n_reps = 25 if quick else 180
        for rep in range(n_reps):
            cohort = sample_cohorts(Design(n_trial=250 if quick else 450,
                                           n_target=650 if quick else 1350,
                                           selection_slope=slope), rng)
            est, diag = estimate_all(cohort)
            overlap_rows.append({"slope": slope, "replicate": rep, "target_truth": truth,
                "naive_trial": est["naive_trial"],
                "standardization_rich": est["standardization_rich"],
                "inverse_odds_rich": est["inverse_odds_rich"],
                "augmented_both_rich": est["augmented_both_rich"],
                "weight_ess": diag["rich_ess"],
                "weight_max_over_mean": diag["rich_max_to_mean_weight"]})
    _table(result_dir, "overlap_replicates", overlap_rows, quick)
    overlap_summary = []
    for slope in severity_slopes:
        subset = [r for r in overlap_rows if r["slope"] == slope]
        truth = subset[0]["target_truth"]
        for method in ("naive_trial", "standardization_rich", "inverse_odds_rich", "augmented_both_rich"):
            values = np.array([r[method] for r in subset])
            overlap_summary.append({"slope": slope, "method": method,
                "target_truth": truth, "mean_bias": float(values.mean() - truth),
                "rmse": float(np.sqrt(np.mean((values - truth) ** 2))),
                "mean_weight_ess": float(np.mean([r["weight_ess"] for r in subset])),
                "p95_max_over_mean": float(np.quantile([r["weight_max_over_mean"] for r in subset], .95)),
                "replicates": len(subset)})
    _table(result_dir, "overlap_summary", overlap_summary, quick)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ess = [next(r["mean_weight_ess"] for r in overlap_summary if r["slope"] == s)
           for s in severity_slopes]
    ax.plot(severity_slopes, ess, marker="o", color=BLUE, linewidth=2.4)
    ax.set_ylim(bottom=0)
    _style(ax, "Loss of source–target overlap reduces weighted information",
           "Selection dependence on X (absolute log-odds slope)", "Mean weighted trial ESS")
    _save(fig, figure_dir / "fig5_overlap_ess.png", quick)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for method, label, color in (("naive_trial", "Trial", GRAY),
                                 ("standardization_rich", "G-formula", TEAL),
                                 ("inverse_odds_rich", "Inverse odds", ORANGE),
                                 ("augmented_both_rich", "Augmented", BLUE)):
        rmse = [next(r["rmse"] for r in overlap_summary if r["slope"] == s and r["method"] == method)
                for s in severity_slopes]
        ax.plot(severity_slopes, rmse, marker="o", linewidth=2.3, color=color, label=label)
    ax.legend(frameon=False)
    _style(ax, "Transport risk: bias and variance respond differently to overlap",
           "Selection dependence on X (absolute log-odds slope)", "RMSE versus exact target ATE")
    _save(fig, figure_dir / "fig6_overlap_rmse.png", quick)

    hidden_rows: list[dict[str, Any]] = []
    selection_grid = (-2.0, -1.0, 0.0, 1.0, 2.0)
    effect_grid = (-1.2, -0.6, 0.0, 0.6, 1.2)
    for gamma in selection_grid:
        for delta in effect_grid:
            o = oracle_effects(hidden_selection=gamma, hidden_effect=delta)
            hidden_rows.append({"hidden_selection_log_odds": gamma,
                "hidden_effect_modifier": delta, "target_truth": o["target_ate"],
                "observed_covariate_transport": o["oracle_observed_transport"],
                "bias": o["oracle_transport_bias"]})
    _table(result_dir, "hidden_modifier_sensitivity", hidden_rows, quick)
    hidden_mat = np.array([[next(r["bias"] for r in hidden_rows if
                                  r["hidden_selection_log_odds"] == g and
                                  r["hidden_effect_modifier"] == d) for d in effect_grid]
                           for g in selection_grid])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    limit = max(.1, float(np.abs(hidden_mat).max()))
    img = ax.imshow(hidden_mat, origin="lower", vmin=-limit, vmax=limit,
                    cmap="RdBu_r", aspect="auto")
    ax.set(xticks=range(5), xticklabels=effect_grid, yticks=range(5),
           yticklabels=selection_grid, xlabel="Unmeasured modifier's effect coefficient",
           ylabel="Unmeasured modifier's selection coefficient",
           title="Even infinite data cannot repair a missing effect modifier")
    fig.colorbar(img, ax=ax, label="Oracle observed-covariate transport bias")
    _save(fig, figure_dir / "fig7_hidden_effect_modifier.png", quick)

    # Independently resample trial and target to respect the disjoint design.
    bootstrap_rows: list[dict[str, Any]] = []
    n_boot = 60 if quick else 400
    for rep in range(n_boot):
        trial_indices = rng.integers(0, source.trial_x.size, size=source.trial_x.size)
        target_indices = rng.integers(0, source.target_x.size, size=source.target_x.size)
        c = Cohorts(source.trial_x[trial_indices], source.trial_z[trial_indices],
                    source.trial_a[trial_indices], source.trial_y[trial_indices],
                    source.target_x[target_indices], source.target_z[target_indices])
        est, _ = estimate_all(c)
        bootstrap_rows.extend({"replicate": rep, "method": name, "estimate": est[name]}
                              for name in ("naive_trial", "standardization_rich",
                                           "inverse_odds_rich", "augmented_both_rich"))
    _table(result_dir, "bootstrap_replicates", bootstrap_rows, quick)
    bootstrap_summary = []
    for name in ("naive_trial", "standardization_rich", "inverse_odds_rich", "augmented_both_rich"):
        values = np.array([r["estimate"] for r in bootstrap_rows if r["method"] == name])
        bootstrap_summary.append({"method": name, "point_estimate": observed[name],
            "ci_lower": float(np.quantile(values, .025)),
            "ci_upper": float(np.quantile(values, .975)),
            "bootstrap_sd": float(values.std(ddof=1)), "n_bootstrap": len(values),
            "target_truth": oracle["target_ate"]})
    _table(result_dir, "bootstrap_intervals", bootstrap_summary, quick)

    fig, ax = plt.subplots(figsize=(10.5, 5))
    for y, row in enumerate(bootstrap_summary):
        ax.plot([row["ci_lower"], row["ci_upper"]], [y, y], linewidth=4,
                color=[GRAY, TEAL, ORANGE, BLUE][y], solid_capstyle="round")
        ax.scatter([row["point_estimate"]], [y], marker="D", s=65,
                   color=[GRAY, TEAL, ORANGE, BLUE][y], zorder=5)
    ax.axvline(oracle["target_ate"], color=RED, linestyle="--", linewidth=2,
               label="Exact target ATE")
    ax.set(yticks=range(4), yticklabels=["Raw trial", "Standardization", "Inverse odds", "Augmented"],
           ylim=(-.7, 3.7))
    ax.legend(frameon=False)
    _style(ax, "Paired-cohort bootstrap quantifies uncertainty, not identifiability",
           "Estimate (dots) and percentile 95% interval (bars)", "Estimator")
    _save(fig, figure_dir / "fig8_bootstrap_intervals.png", quick)

    diagnostic_rows = [{"metric": key, "value": value} for key, value in (oracle | diagnostics | {
        "observed_naive_trial": observed["naive_trial"],
        "observed_augmented": observed["augmented_both_rich"],
        "main_replicates": float(main_n), "overlap_replicates_per_slope": float(25 if quick else 180),
        "bootstrap_replicates": float(n_boot),
    }).items()]
    _table(result_dir, "study_summary", diagnostic_rows, quick)
    return {**oracle, **diagnostics, "naive_trial": observed["naive_trial"],
            "augmented": observed["augmented_both_rich"], "n_main": float(main_n)}


def main() -> None:
    """Command-line entry point for full or reduced-cost seeded experiments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small, separately named CI evidence")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent,
                        help="Root directory for generated figures and results")
    arguments = parser.parse_args()
    result = run_study(arguments.output, quick=arguments.quick)
    for key, value in result.items():
        print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
