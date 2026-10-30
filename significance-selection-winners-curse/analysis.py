"""Seeded research experiments on significance filters and exaggerated effect sizes.

Every result is synthetic. Full runs generate eight publication PNGs and ten CSV
tables. The quick mode writes suffixed, smaller files for independent CI checks.
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
from scipy.stats import norm

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BLUE, RED, TEAL, ORANGE, GRAY = "#2463a0", "#bb4b51", "#218779", "#cf8238", "#656d77"


@dataclass(frozen=True)
class Design:
    """Explicit statistical assumptions and deterministic random seeds."""

    seed: int = 20261030
    alpha: float = 0.05
    true_effect: float = 0.30
    study_se: float = 0.30
    shrinkage_prior_sd: float = 0.20
    multiplicity_effect_z: float = 2.5

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        for field in ("alpha", "true_effect", "study_se", "shrinkage_prior_sd",
                      "multiplicity_effect_z"):
            value = getattr(self, field)
            if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
                raise TypeError(f"{field} must be a finite real number")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.true_effect <= 0 or self.study_se <= 0:
            raise ValueError("Effect and study standard error must be positive")
        if self.shrinkage_prior_sd <= 0 or self.multiplicity_effect_z <= 0:
            raise ValueError("Prior scale and alternative z effect must be positive")


def critical_value(alpha: float = 0.05) -> float:
    """Two-sided standard-Normal rejection boundary."""

    if not isinstance(alpha, (float, int)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite probability in (0,1)")
    return float(norm.ppf(1 - alpha / 2))


def two_sided_power(delta: float, alpha: float = 0.05) -> float:
    """Exact power for Z~N(delta,1), rejecting when |Z| exceeds a threshold."""

    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    c = critical_value(alpha)
    return float(norm.cdf(-c - delta) + norm.sf(c - delta))


def type_s_error(delta: float, alpha: float = 0.05) -> float:
    """Probability of the wrong reported sign, conditional on significance.

    Defined only for a nonzero true standardized effect.
    """

    if not np.isfinite(delta) or delta == 0:
        raise ValueError("Type S error requires a finite nonzero true effect")
    c = critical_value(alpha)
    wrong_tail = norm.cdf(-c - abs(delta))
    return float(wrong_tail / two_sided_power(abs(delta), alpha))


def type_m_error(delta: float, alpha: float = 0.05) -> float:
    """Exact E(|Z|/|delta| conditional on |Z|>c), for positive |delta|."""

    if not np.isfinite(delta) or delta == 0:
        raise ValueError("Type M exaggeration requires a nonzero true effect")
    value = abs(delta)
    c = critical_value(alpha)
    positive = value * norm.sf(c - value) + norm.pdf(c - value)
    negative = -value * norm.cdf(-c - value) + norm.pdf(c + value)
    return float((positive + negative) / (two_sided_power(value, alpha) * value))


def significant_interval_coverage(delta: float, alpha: float = 0.05) -> float:
    """Exact conditional coverage of a usual Normal CI given p<alpha.

    The usual interval has unconditional coverage 1-alpha. Conditioning on
    two-sided significance against zero changes coverage for the true effect.
    """

    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    c = critical_value(alpha)
    # By sign symmetry, a nonnegative standardized true effect suffices.
    value = abs(delta)
    lower_innovation = max(-c, c - value)
    covered_and_selected = max(0.0, float(norm.cdf(c) - norm.cdf(lower_innovation)))
    return float(covered_and_selected / two_sided_power(value, alpha))


def posterior_mean_normal(
    estimates: FloatArray, standard_error: float, prior_sd: float
) -> FloatArray:
    """Normal-prior posterior mean for a Gaussian observation model.

    The formula is valid only when the N(0,prior_sd²) effect prior is credible.
    """

    values = np.asarray(estimates, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("estimates must be a finite 1D array")
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("standard_error must be positive and finite")
    if not np.isfinite(prior_sd) or prior_sd <= 0:
        raise ValueError("prior_sd must be positive and finite")
    weight = prior_sd**2 / (prior_sd**2 + standard_error**2)
    return np.asarray(weight * values, dtype=np.float64)


def benjamini_hochberg(p_values: FloatArray, level: float = 0.05) -> BoolArray:
    """Benjamini–Hochberg step-up rule for one family of hypotheses."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("p-values must be a nonempty 1D array")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite probabilities")
    if not 0 < level < 1 or not np.isfinite(level):
        raise ValueError("level must be between zero and one")
    m = values.size
    sorted_p = np.sort(values)
    qualifying = np.flatnonzero(sorted_p <= level * np.arange(1, m + 1) / m)
    if not len(qualifying):
        return np.zeros(m, dtype=bool)
    threshold = sorted_p[qualifying[-1]]
    return np.asarray(values <= threshold, dtype=bool)


def rejection_summary(rejected: BoolArray, alternatives: BoolArray) -> dict[str, float]:
    """Family-level realized FDP, power and false-positive indicator."""

    r = np.asarray(rejected, dtype=bool)
    a = np.asarray(alternatives, dtype=bool)
    if r.ndim != 1 or r.shape != a.shape:
        raise ValueError("Aligned one-dimensional masks required")
    discoveries = int(r.sum())
    true = int(np.sum(r & a))
    false = int(np.sum(r & ~a))
    n_alts = int(a.sum())
    return {
        "discoveries": float(discoveries),
        "true_discoveries": float(true),
        "false_discoveries": float(false),
        "false_discovery_proportion": false / max(1, discoveries),
        "alternative_detection_fraction": true / max(1, n_alts),
        "any_false_discovery": float(false > 0),
    }


def one_multiple_testing_family(
    rng: np.random.Generator, *,
    m: int = 1000, pi0: float = 0.95, alternative_delta: float = 2.5,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Simulate independently sampled null/alternative tests within one family.

    FDR guarantees for BH apply under independent p-values in this experiment;
    they are not claimed for arbitrary correlation or adaptive hypothesis choice.
    """

    if not isinstance(m, int) or m < 1 or not np.isfinite(pi0) or not 0 <= pi0 <= 1:
        raise ValueError("Invalid hypothesis family size or null fraction")
    if not np.isfinite(alternative_delta):
        raise ValueError("alternative_delta must be finite")
    alternative = rng.random(m) >= pi0
    z_values = rng.standard_normal(m) + alternative_delta * alternative
    p_values = np.asarray(2 * norm.sf(np.abs(z_values)), dtype=np.float64)
    methods = {
        "uncorrected": p_values < alpha,
        "bonferroni": p_values < alpha / m,
        "benjamini_hochberg": benjamini_hochberg(p_values, alpha),
    }
    return {name: rejection_summary(rejected, alternative) for name, rejected in methods.items()}


def _write_rows(root: Path, name: str, rows: list[dict[str, Any]], suffix: str) -> None:
    """Write one schema-stable table without relying on pandas or hidden state."""

    if not rows:
        raise ValueError("Cannot write an empty result")
    target = root / "results" / f"{name}{suffix}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, root: Path, name: str, suffix: str) -> None:
    """Render publication figure with deterministic style and dimensions."""

    fig.tight_layout(pad=1.5)
    fig.savefig(root / "figures" / f"{name}{suffix}.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def _style() -> None:
    """Keep fonts and colors consistent across all eight figures."""

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.19, "axes.axisbelow": True,
        "figure.facecolor": "white",
    })


def run_study(
    config: Design | None = None, *, quick: bool = False, output_dir: Path | None = None
) -> dict[str, float]:
    """Generate eight independent figures and ten CSV tables from fixed assumptions."""

    cfg = config if config is not None else Design()
    if not isinstance(cfg, Design):
        raise TypeError("config must be a Design instance")
    root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    _style()

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        _write_rows(root, name, rows, suffix)

    c = critical_value(cfg.alpha)
    delta0 = cfg.true_effect / cfg.study_se
    rng = np.random.default_rng(cfg.seed)

    # Figure 1: same repeated sampling distribution, sharply different
    # probability laws before and after filtering on a p-value.
    x = np.linspace(-1.4, 1.6, 700)
    raw_density = norm.pdf(x, loc=cfg.true_effect, scale=cfg.study_se)
    selected = np.abs(x / cfg.study_se) > c
    selected_density = raw_density * selected / two_sided_power(delta0, cfg.alpha)
    fig, ax = plt.subplots(figsize=(10.1, 4.5))
    ax.plot(x, raw_density, color=BLUE, lw=2, label="Before significance filtering")
    ax.plot(x, selected_density, color=RED, lw=2, label="Among p < 0.05 reports")
    ax.axvline(cfg.true_effect, color=TEAL, ls="--", label="True effect")
    ax.axvspan(-c * cfg.study_se, c * cfg.study_se, alpha=0.09, color=GRAY, label="Unreported region")
    ax.set(title="Figure 1 · Filtering changes the distribution of reported estimates",
           xlabel="Effect estimate", ylabel="Density, conditional where indicated")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, root, "fig1_significance_filter", suffix)

    # Figure 2: sample size planning is a property of the alternative, not
    # the number of studies eventually reported as significant.
    n_grid = np.arange(10, 501, 5)
    planned_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(9.9, 4.4))
    for theta, color in ((0.15, GRAY), (0.25, BLUE), (0.35, TEAL), (0.50, ORANGE)):
        powers = [two_sided_power(theta * math.sqrt(int(n)), cfg.alpha) for n in n_grid]
        planned_rows.extend({
            "n": int(n), "true_effect": theta, "known_sigma": 1.0,
            "standardized_delta": theta * math.sqrt(int(n)),
            "exact_two_sided_power": float(power),
        } for n, power in zip(n_grid, powers))
        ax.plot(n_grid, powers, color=color, lw=2, label=f"True effect {theta:g} SD")
    ax.axhline(0.8, ls="--", color=GRAY, lw=1.1, label="80% planning reference")
    ax.set(title="Figure 2 · Power depends on effect, precision and the chosen threshold",
           xlabel="Independent observations per study (known sigma=1)",
           ylabel="Two-sided testing power", ylim=(0, 1.02))
    ax.legend(frameon=False, ncol=2, fontsize=8)
    write("power_design", planned_rows)
    _save(fig, root, "fig2_power_design", suffix)

    # Figures 3/4: exact normal-tail Type S/M quantities.
    deltas = np.linspace(0.1, 4.5, 140)
    selection_rows = [{
        "standardized_true_effect": float(d),
        "two_sided_power": two_sided_power(float(d), cfg.alpha),
        "type_s_wrong_sign_given_selected": type_s_error(float(d), cfg.alpha),
        "type_m_abs_exaggeration_given_selected": type_m_error(float(d), cfg.alpha),
        "mean_selected_absolute_z": float(d) * type_m_error(float(d), cfg.alpha),
        "selection_threshold_z": c,
    } for d in deltas]
    write("selection_errors", selection_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.3))
    axes[0].plot(deltas, [r["type_s_wrong_sign_given_selected"] for r in selection_rows],
                 color=RED, lw=2.2)
    axes[0].set(xlabel="True effect / standard error", ylabel="Conditional wrong-sign probability",
                title="Type S: direction errors among reports")
    axes[1].plot(deltas, [r["type_m_abs_exaggeration_given_selected"] for r in selection_rows],
                 color=ORANGE, lw=2.2)
    axes[1].axhline(1.0, color=GRAY, ls="--")
    axes[1].set(xlabel="True effect / standard error", ylabel="Absolute exaggeration factor",
                title="Type M: selected magnitudes / truth")
    _save(fig, root, "fig3_type_s_and_type_m", suffix)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot(deltas, [r["two_sided_power"] for r in selection_rows], color=BLUE, lw=2)
    axes[0].axhline(.8, color=GRAY, ls="--")
    axes[0].set(xlabel="True effect / standard error", ylabel="Probability of significance",
                title="Few low-signal studies pass the filter")
    axes[1].plot(deltas, [r["mean_selected_absolute_z"] for r in selection_rows],
                 color=RED, lw=2, label="Reported absolute estimate")
    axes[1].plot(deltas, deltas, ls="--", color=TEAL, label="True absolute effect")
    axes[1].axhline(c, color=GRAY, ls=":", label="Two-sided threshold")
    axes[1].set(xlabel="True effect / standard error", ylabel="Effect in SE units",
                title="Selected magnitude remains inflated")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig4_winners_curse", suffix)

    # Figure 5: discovery and genuinely independent replication. Replication
    # samples are drawn irrespective of their eventual significance.
    repetitions = 24000 if quick else 180000
    observed = np.asarray(
        cfg.true_effect + cfg.study_se * rng.standard_normal(repetitions), dtype=np.float64
    )
    report = np.abs(observed / cfg.study_se) > c
    replications = np.asarray(
        cfg.true_effect + cfg.study_se * rng.standard_normal(repetitions), dtype=np.float64
    )
    first = observed[report]
    second = replications[report]
    discovery_rows = [{
        "selected_study_index": int(original_index),
        "initial_effect": float(observed[original_index]),
        "independent_replication_effect": float(replications[original_index]),
        "initial_sign_was_wrong": int(observed[original_index] < 0),
        "replication_is_significant": int(abs(replications[original_index] / cfg.study_se) > c),
    } for original_index in np.flatnonzero(report)[:(350 if quick else 3000)]]
    write("discovery_replication", discovery_rows)
    same_sign = (np.sign(first) == np.sign(second))
    replication_sig = np.abs(second / cfg.study_se) > c
    replication_summary: dict[str, float] = {
        "candidate_studies": float(repetitions),
        "significant_studies": float(report.sum()),
        "selected_fraction": float(report.mean()),
        "selected_initial_mean": float(first.mean()),
        "selected_initial_absolute_mean": float(np.mean(np.abs(first))),
        "independent_replication_mean": float(second.mean()),
        "independent_replication_absolute_mean": float(np.mean(np.abs(second))),
        "wrong_sign_fraction_among_selected": float(np.mean(first < 0)),
        "replication_significant_fraction": float(np.mean(replication_sig)),
        "replication_significant_same_sign_fraction": float(np.mean(replication_sig & same_sign)),
        "true_effect": cfg.true_effect,
        "study_standard_error": cfg.study_se,
        "exact_discovery_power": two_sided_power(delta0, cfg.alpha),
        "exact_type_s_given_selected": type_s_error(delta0, cfg.alpha),
        "exact_type_m_given_selected": type_m_error(delta0, cfg.alpha),
    }
    write("replication_summary", [
        {"measure": key, "value": value} for key, value in replication_summary.items()
    ])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    subset = min(1400, len(first))
    axes[0].scatter(first[:subset], second[:subset], s=12, color=BLUE,
                    alpha=0.25, rasterized=True)
    axes[0].plot([-1.2, 1.3], [-1.2, 1.3], ls="--", color=GRAY, label="Equal estimates")
    axes[0].axhline(cfg.true_effect, color=TEAL, ls=":", label="True effect")
    axes[0].set(xlabel="Significant initial estimate", ylabel="Independent replication",
                title="Selected discovery does not constrain replication")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(first, bins=np.linspace(-1.5, 1.5, 70), density=True,
                 color=RED, alpha=0.42, label="Significant discoveries")
    axes[1].hist(second, bins=np.linspace(-1.5, 1.5, 70), density=True,
                 color=TEAL, alpha=0.48, label="Unfiltered replications")
    axes[1].axvline(cfg.true_effect, color=GRAY, ls="--", label="True effect")
    axes[1].set(xlabel="Effect estimate", ylabel="Density",
                title="Selection creates regression to the mean")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig5_discovery_vs_replication", suffix)

    # Figure 6: repeated families of independently sampled tests. This is an
    # explicit independence setting for the BH FDR guarantee.
    null_fractions = (0.80, 0.95, 0.99)
    methods = ("uncorrected", "bonferroni", "benjamini_hochberg")
    rounds = 32 if quick else 240
    family_size = 1000
    multiple_rows: list[dict[str, Any]] = []
    multiple_summary: list[dict[str, Any]] = []
    for pi0 in null_fractions:
        per_method = {name: [] for name in methods}
        for rep in range(rounds):
            one = one_multiple_testing_family(
                rng, m=family_size, pi0=pi0,
                alternative_delta=cfg.multiplicity_effect_z, alpha=cfg.alpha
            )
            for method in methods:
                row = {"null_fraction": pi0, "replicate": rep, "method": method,
                       **one[method]}
                multiple_rows.append(row)
                per_method[method].append(row)
        for method, rows in per_method.items():
            multiple_summary.append({
                "null_fraction": pi0, "method": method, "family_size": family_size,
                "repeated_families": rounds, "alternative_effect_in_se": cfg.multiplicity_effect_z,
                "fdr_mean_fdp": float(np.mean([r["false_discovery_proportion"] for r in rows])),
                "mean_false_discoveries": float(np.mean([r["false_discoveries"] for r in rows])),
                "mean_true_discoveries": float(np.mean([r["true_discoveries"] for r in rows])),
                "mean_alternative_detection_fraction": float(np.mean(
                    [r["alternative_detection_fraction"] for r in rows]
                )),
                "familywise_any_false_discovery_rate": float(np.mean(
                    [r["any_false_discovery"] for r in rows]
                )),
            })
    write("multiplicity_replicates", multiple_rows)
    write("multiplicity_summary", multiple_summary)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
    positions = np.arange(len(null_fractions))
    palette = (RED, GRAY, TEAL)
    for j, (method, color) in enumerate(zip(methods, palette)):
        rows = [r for r in multiple_summary if r["method"] == method]
        axes[0].bar(positions + .24 * (j-1),
                    [r["fdr_mean_fdp"] for r in rows],
                    width=.22, color=color, label=method.replace("_", " ").title())
        axes[1].bar(positions + .24 * (j-1),
                    [r["mean_alternative_detection_fraction"] for r in rows],
                    width=.22, color=color)
    axes[0].axhline(cfg.alpha, color=GRAY, ls=":", label="q = 0.05")
    axes[0].set(xticks=positions, xticklabels=[f"{x:.0%}" for x in null_fractions],
                ylim=(0, 1), xlabel="Fraction of true null hypotheses",
                ylabel="Mean false-discovery proportion",
                title="FDR: expectation includes zero-discovery families")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].set(xticks=positions, xticklabels=[f"{x:.0%}" for x in null_fractions],
                ylim=(0, 1), xlabel="Fraction of true null hypotheses",
                ylabel="Average fraction of alternatives detected",
                title="Error control comes with a discovery tradeoff")
    _save(fig, root, "fig6_many_tests", suffix)

    # Figure 7: an explicitly correct N(0,tau²) prior. Shrinkage is not
    # guaranteed to improve performance when this prior is wrong.
    n_effects = 28000 if quick else 180000
    effects = np.asarray(
        rng.normal(0.0, cfg.shrinkage_prior_sd, size=n_effects), dtype=np.float64
    )
    estimates = np.asarray(
        effects + cfg.study_se * rng.standard_normal(n_effects), dtype=np.float64
    )
    significant = np.abs(estimates / cfg.study_se) > c
    actual = effects[significant]
    published = estimates[significant]
    oracle_shrunk = posterior_mean_normal(
        published, cfg.study_se, cfg.shrinkage_prior_sd
    )
    shrink_rows = [{
        "effect_index": int(j),
        "true_effect": float(effects[j]),
        "reported_effect": float(estimates[j]),
        "posterior_mean_true_prior": float(posterior_mean_normal(
            np.asarray([estimates[j]]), cfg.study_se, cfg.shrinkage_prior_sd
        )[0]),
        "selection_threshold_abs_effect": float(c * cfg.study_se),
    } for j in np.flatnonzero(significant)[:(250 if quick else 3000)]]
    write("shrinkage_selected", shrink_rows)
    sensitivity_rows: list[dict[str, Any]] = []
    for assumed_sd in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 5.00):
        estimate = posterior_mean_normal(published, cfg.study_se, assumed_sd)
        sensitivity_rows.append({
            "assumed_prior_sd": assumed_sd, "true_prior_sd": cfg.shrinkage_prior_sd,
            "number_selected": int(len(published)),
            "conditional_mean_squared_error": float(np.mean((estimate - actual)**2)),
            "conditional_mean_absolute_error": float(np.mean(np.abs(estimate - actual))),
            "unshrunk_mse": float(np.mean((published - actual)**2)),
        })
    write("shrinkage_sensitivity", sensitivity_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    subset = min(900, len(published))
    axes[0].scatter(published[:subset], actual[:subset], s=10, color=BLUE,
                    alpha=.22, rasterized=True, label="Latent true effects")
    guide = np.linspace(-1.5, 1.5, 150)
    factor = cfg.shrinkage_prior_sd**2 / (
        cfg.shrinkage_prior_sd**2 + cfg.study_se**2
    )
    axes[0].plot(guide, guide, ls="--", color=GRAY, label="Unshrunk estimate")
    axes[0].plot(guide, factor * guide, color=TEAL, lw=2,
                 label="Posterior mean, true prior")
    axes[0].set(xlabel="Selected observed estimate", ylabel="Underlying true effect",
                title="A defined population distribution makes shrinkage evaluable")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot([r["assumed_prior_sd"] for r in sensitivity_rows],
                 [r["conditional_mean_squared_error"] for r in sensitivity_rows],
                 color=TEAL, lw=2, marker="o", label="Shrunk estimate")
    axes[1].axhline(sensitivity_rows[0]["unshrunk_mse"], color=RED, ls="--",
                    label="Unshrunk selected estimate")
    axes[1].axvline(cfg.shrinkage_prior_sd, color=GRAY, ls=":",
                    label="Correct prior scale")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Assumed prior standard deviation", ylabel="MSE among reported studies",
                title="Shrinkage depends on the assumed prior")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig7_selection_and_shrinkage", suffix)

    # Figure 8: usual Normal intervals are marginally 95% across all studies.
    # Restricting publication to p<0.05 changes their conditional coverage.
    grid = np.linspace(0, 5, 160)
    cov_rows = [{
        "standardized_true_effect": float(delta),
        "ordinary_unconditional_coverage": 1 - cfg.alpha,
        "coverage_given_significance": significant_interval_coverage(float(delta), cfg.alpha),
        "selection_probability": two_sided_power(float(delta), cfg.alpha),
        "two_sided_alpha": cfg.alpha,
    } for delta in grid]
    write("conditional_coverage", cov_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(grid, [r["coverage_given_significance"] for r in cov_rows],
                 color=RED, lw=2.2, label="Condition on significant result")
    axes[0].axhline(1-cfg.alpha, ls="--", color=TEAL, label="All results, unconditional")
    axes[0].set(xlabel="True effect / standard error", ylabel="Coverage of the true effect",
                ylim=(0, 1), title="The significance filter changes interval coverage")
    axes[0].legend(frameon=False, fontsize=8)
    ns = np.arange(5, 501, 5)
    for theta, color in ((0.15, ORANGE), (0.30, BLUE)):
        axes[1].plot(ns, [significant_interval_coverage(float(theta * math.sqrt(n)), cfg.alpha)
                          for n in ns],
                     lw=2, color=color, label=f"True effect {theta:g} SD")
    axes[1].axhline(1-cfg.alpha, ls="--", color=TEAL)
    axes[1].set(xlabel="Independent observations per study", ylabel="Coverage among p<0.05",
                ylim=(0, 1), title="Study size affects conditional interval calibration")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig8_post_selection_intervals", suffix)

    def multiple_metric(pi0: float, method: str, key: str) -> float:
        return next(float(row[key]) for row in multiple_summary
                    if row["null_fraction"] == pi0 and row["method"] == method)

    selected_naive_mse = float(np.mean((published-actual)**2))
    selected_oracle_shrink_mse = float(np.mean((oracle_shrunk-actual)**2))
    results = {
        "candidate_studies": float(repetitions),
        "alpha": cfg.alpha,
        "true_effect": cfg.true_effect,
        "standard_error": cfg.study_se,
        "standardized_effect_delta": delta0,
        "power_at_delta_1": two_sided_power(1.0, cfg.alpha),
        "type_s_at_delta_1": type_s_error(1.0, cfg.alpha),
        "type_m_at_delta_1": type_m_error(1.0, cfg.alpha),
        "selected_interval_coverage_delta_1": significant_interval_coverage(1.0, cfg.alpha),
        "selected_fraction_simulation": replication_summary["selected_fraction"],
        "selected_initial_absolute_mean": replication_summary["selected_initial_absolute_mean"],
        "replication_signed_mean": replication_summary["independent_replication_mean"],
        "replication_significant_fraction": replication_summary["replication_significant_fraction"],
        "null_fraction_multiple_testing": 0.95,
        "uncorrected_fdr_pi0_0_95": multiple_metric(.95, "uncorrected", "fdr_mean_fdp"),
        "bh_fdr_pi0_0_95": multiple_metric(.95, "benjamini_hochberg", "fdr_mean_fdp"),
        "bonferroni_fdr_pi0_0_95": multiple_metric(.95, "bonferroni", "fdr_mean_fdp"),
        "selected_naive_mse": selected_naive_mse,
        "selected_oracle_shrinkage_mse": selected_oracle_shrink_mse,
        "multiplicity_repeated_families_per_pi0": float(rounds),
    }
    write("summary", [{"measure": key, "value": value} for key, value in results.items()])
    return results


def main() -> None:
    """Command-line interface for reproducing the complete synthetic study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="CI-scale suffixed artifacts")
    parser.add_argument("--output", type=Path, default=None, help="Output root")
    args = parser.parse_args()
    results = run_study(quick=args.quick, output_dir=args.output)
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    print("Wrote 8 figures and 10 CSV tables")


if __name__ == "__main__":
    main()
