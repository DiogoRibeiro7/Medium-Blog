"""Synthetic rare-event detection, label-shift, and decision experiments.

The model is Z|Y=0 ~ Normal(0,1), Z|Y=1 ~ Normal(mu,1).
Run python analysis.py for publication results or --quick for CI smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BLUE, ORANGE, TEAL, RED, GRAY = "#2463a0", "#d07832", "#25877a", "#b64e50", "#636c76"


@dataclass(frozen=True)
class Scenario:
    """All population, outcome-selection, and economic assumptions."""

    seed: int = 20261030
    separation: float = 2.0
    source_prior: float = 0.20
    target_prior: float = 0.01
    false_alarm_cost: float = 8.0
    missed_event_cost: float = 250.0
    verification_probability: float = 0.04

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not np.isfinite(self.separation) or self.separation <= 0:
            raise ValueError("separation must be positive and finite")
        for key in ("source_prior", "target_prior", "verification_probability"):
            validate_prior(getattr(self, key))
        for key in ("false_alarm_cost", "missed_event_cost"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")


def validate_prior(prior: float) -> None:
    """Validate a nondegenerate Bernoulli probability."""

    if not np.isscalar(prior) or not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("Probability must be strictly between 0 and 1")


def posterior(score: FloatArray, prior: float, separation: float) -> FloatArray:
    """Exact calibrated posterior for two equal-variance Gaussian classes."""

    validate_prior(prior)
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Separation must be positive")
    z = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("Scores must be finite")
    return np.asarray(expit(logit(prior) + separation * z - separation**2 / 2))


def correct_prior(p_source: FloatArray, source_prior: float, target_prior: float) -> FloatArray:
    """Adjust posterior odds when P_target(z|y)=P_source(z|y)."""

    validate_prior(source_prior)
    validate_prior(target_prior)
    p = np.asarray(p_source, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Source probabilities must be in [0,1]")
    with np.errstate(divide="ignore"):
        return np.asarray(expit(logit(p) + logit(target_prior) - logit(source_prior)))


def operating_rates(threshold: float | FloatArray, separation: float) -> tuple[FloatArray, FloatArray]:
    """Return population true- and false-positive rates for score >= threshold."""

    z = np.asarray(threshold, dtype=float)
    if not np.all(np.isfinite(z)) or not np.isfinite(separation) or separation <= 0:
        raise ValueError("Thresholds must be finite and separation positive")
    return np.asarray(norm.sf(z - separation)), np.asarray(norm.sf(z))


def roc_auc(separation: float) -> float:
    """Gaussian ROC AUC, P(Z_positive > Z_negative)."""

    if separation <= 0 or not np.isfinite(separation):
        raise ValueError("Separation must be positive")
    return float(norm.cdf(separation / np.sqrt(2.0)))


def precision_at_prior(prior: float, tpr: FloatArray, fpr: FloatArray) -> FloatArray:
    """Bayes precision; NaN when the alarm has exactly zero probability."""

    validate_prior(prior)
    t, f = np.asarray(tpr, dtype=float), np.asarray(fpr, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(f)):
        raise ValueError("Rates must be finite")
    if np.any((t < 0) | (t > 1) | (f < 0) | (f > 1)):
        raise ValueError("Rates must be probabilities")
    num = prior * t
    den = num + (1 - prior) * f
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(den > 0, num / den, np.nan))


def population_sample(
    rng: np.random.Generator, n: int, prior: float, separation: float
) -> tuple[FloatArray, BoolArray]:
    """Simulate independent population labels and Gaussian conditional scores."""

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("Sample size must be a positive integer")
    validate_prior(prior)
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Separation must be positive")
    y = rng.random(n) < prior
    z = rng.standard_normal(n) + separation * y.astype(float)
    return np.asarray(z, dtype=float), np.asarray(y, dtype=bool)


def _check_probabilities(y: BoolArray, p: FloatArray) -> tuple[FloatArray, FloatArray]:
    yy, pp = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if yy.shape != pp.shape or yy.size == 0:
        raise ValueError("Labels and probabilities must share a nonempty shape")
    if np.any(~np.isfinite(pp)) or np.any((pp < 0) | (pp > 1)):
        raise ValueError("Invalid probabilities")
    if np.any((yy != 0) & (yy != 1)):
        raise ValueError("Expected binary labels")
    return yy, pp


def brier_score(y: BoolArray, p: FloatArray) -> float:
    """Strictly proper mean squared probability loss."""

    yy, pp = _check_probabilities(y, p)
    return float(np.mean((pp - yy) ** 2))


def binary_log_loss(y: BoolArray, p: FloatArray) -> float:
    """Strictly proper binary logarithmic loss; clip numerical endpoints."""

    yy, pp = _check_probabilities(y, p)
    q = np.clip(pp, 1e-12, 1 - 1e-12)
    return float(-np.mean(yy * np.log(q) + (1 - yy) * np.log1p(-q)))


def calibration_bins(y: BoolArray, p: FloatArray, n_bins: int = 12) -> list[dict[str, float]]:
    """Binned reliability data, omitting bins that have no observations."""

    yy, pp = _check_probabilities(y, p)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    bins = np.minimum((pp * n_bins).astype(int), n_bins - 1)
    rows: list[dict[str, float]] = []
    for j in range(n_bins):
        mask = bins == j
        if mask.any():
            rows.append({
                "bin": float(j),
                "n": float(mask.sum()),
                "mean_predicted": float(pp[mask].mean()),
                "observed_frequency": float(yy[mask].mean()),
            })
    return rows


def expected_calibration_error(rows: Iterable[Mapping[str, float]]) -> float:
    """Descriptive equal-width-bin ECE; not a proper score or a bound."""

    data = list(rows)
    count = sum(row["n"] for row in data)
    if count <= 0:
        raise ValueError("Nonempty bins required")
    return float(sum(
        row["n"] * abs(row["mean_predicted"] - row["observed_frequency"]) for row in data
    ) / count)


def label_shift_prior_mle(source_posteriors: FloatArray, source_prior: float) -> float:
    """Estimate unlabeled target prevalence using the label-shift likelihood."""

    validate_prior(source_prior)
    p = np.asarray(source_posteriors, dtype=float)
    if p.ndim != 1 or p.size < 10 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Supply at least ten valid calibrated source probabilities")

    def nll(pi: float) -> float:
        likelihood_ratio = (pi / source_prior) * p + (
            (1 - pi) / (1 - source_prior)
        ) * (1 - p)
        return -float(np.log(likelihood_ratio).mean())

    result = minimize_scalar(
        nll, bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-7}
    )
    if not result.success:
        raise RuntimeError("Target-prevalence optimization failed")
    return float(result.x)


def black_box_shift_prevalence(observed_alert_fraction: float, tpr: float, fpr: float) -> float:
    """Invert alert_rate=pi*TPR+(1-pi)*FPR, without hiding negative estimates."""

    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (observed_alert_fraction, tpr, fpr)):
        raise ValueError("Rates must be finite probabilities")
    if abs(tpr - fpr) < 1e-9:
        raise ValueError("Class-conditional alert rates are indistinguishable")
    return float((observed_alert_fraction - fpr) / (tpr - fpr))


def cost_probability_threshold(c_false_alarm: float, c_missed: float) -> float:
    """Optimal probability threshold when true alarms have zero extra loss."""

    if not np.isfinite(c_false_alarm) or not np.isfinite(c_missed) or min(c_false_alarm, c_missed) <= 0:
        raise ValueError("Costs must be positive and finite")
    return float(c_false_alarm / (c_false_alarm + c_missed))


def score_threshold(probability_threshold: float, prior: float, separation: float) -> float:
    """Invert the Gaussian posterior for a probability-based alarm rule."""

    validate_prior(probability_threshold)
    validate_prior(prior)
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Separation must be positive")
    return float((logit(probability_threshold) - logit(prior) + separation**2 / 2) / separation)


def decision_operating_point(
    threshold: float, prior: float, separation: float, c_false_alarm: float, c_missed: float
) -> dict[str, float]:
    """Population alarm fraction, precision, recall and expected loss per unit."""

    validate_prior(prior)
    if min(c_false_alarm, c_missed) <= 0:
        raise ValueError("Costs must be positive")
    tpr, fpr = operating_rates(threshold, separation)
    t, f = float(tpr), float(fpr)
    return {
        "threshold_z": float(threshold),
        "tpr": t,
        "fpr": f,
        "alert_rate": float(prior * t + (1 - prior) * f),
        "precision": float(precision_at_prior(prior, np.array(t), np.array(f))),
        "cost_per_case": float(c_false_alarm * (1 - prior) * f + c_missed * prior * (1 - t)),
    }


def budget_threshold(budget: float, prior: float, separation: float) -> float:
    """Unique score cutoff that fills an alert capacity of fraction budget."""

    validate_prior(budget)
    validate_prior(prior)
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError("Separation must be positive")

    def residual(z: float) -> float:
        t, f = operating_rates(z, separation)
        return float(prior * t + (1 - prior) * f - budget)

    return float(brentq(residual, -12.0, 12.0, xtol=1e-12))


def verification_estimates(
    labels: BoolArray, alarms: BoolArray, verified: BoolArray, nonalarm_probability: float
) -> tuple[float, float]:
    """Verified-only and Horvitz–Thompson prevalence assuming MAR given alarm."""

    validate_prior(nonalarm_probability)
    y, a, v = np.asarray(labels, dtype=bool), np.asarray(alarms, dtype=bool), np.asarray(verified, dtype=bool)
    if y.ndim != 1 or y.size == 0 or y.shape != a.shape or y.shape != v.shape:
        raise ValueError("Input arrays must be aligned and nonempty")
    if not v.any():
        raise ValueError("There must be some verified observations")
    propensity = np.where(a, 1.0, nonalarm_probability)
    return float(y[v].mean()), float(np.sum(y[v] / propensity[v]) / y.size)


def _write_csv(root: Path, filename: str, rows: list[dict[str, Any]], suffix: str) -> None:
    """Persist machine-readable results with deterministic column order."""

    if not rows:
        raise ValueError("No rows to write")
    location = root / "results" / f"{filename}{suffix}.csv"
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("w", encoding="utf-8", newline="") as handle:
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
    """Write eight deterministic figures and ten synthetic CSV tables."""

    scenario = config or Scenario()
    root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    _style()

    def csv_out(name: str, rows: list[dict[str, Any]]) -> None:
        _write_csv(root, name, rows, suffix)

    source, target, mu = scenario.source_prior, scenario.target_prior, scenario.separation
    z_grid = np.linspace(-3.5, 6.0, 300 if quick else 540)
    tpr, fpr = operating_rates(z_grid, mu)
    ps = precision_at_prior(source, tpr, fpr)
    pt = precision_at_prior(target, tpr, fpr)
    csv_out("operating_curves", [
        {
            "threshold_z": float(z), "pdf_negative": float(norm.pdf(z)),
            "pdf_positive": float(norm.pdf(z - mu)), "tpr": float(t),
            "fpr": float(f), "precision_source": float(p1), "precision_target": float(p2),
        }
        for z, t, f, p1, p2 in zip(z_grid, tpr, fpr, ps, pt)
    ])
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2))
    axes[0].plot(z_grid, norm.pdf(z_grid), color=BLUE, lw=2, label="No event")
    axes[0].plot(z_grid, norm.pdf(z_grid - mu), color=ORANGE, lw=2, label="Event")
    axes[0].set(xlabel="Detector score z", ylabel="Conditional density", title="Invariant class-conditional scores")
    axes[0].legend(frameon=False)
    axes[1].plot(fpr[::-1], tpr[::-1], color=BLUE, lw=2.4, label=f"ROC AUC = {roc_auc(mu):.3f}")
    axes[1].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC under pure prior shift")
    axes[1].legend(frameon=False)
    _save(fig, root, "fig1_scores_and_roc", suffix)

    t_ref, f_ref = operating_rates(1.5, mu)
    prevalences = np.geomspace(0.001, 0.5, 170)
    prev_rows = [
        {
            "prevalence": float(pi),
            "precision": float(precision_at_prior(pi, t_ref, f_ref)),
            "false_positives_per_100k": float(1e5 * (1 - pi) * f_ref),
            "true_positives_per_100k": float(1e5 * pi * t_ref),
        }
        for pi in prevalences
    ]
    csv_out("prevalence_sweep", prev_rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.semilogx(prevalences, [r["precision"] for r in prev_rows], color=BLUE, lw=2.4)
    for pi, label, color in [
        (target, "Deployment", ORANGE), (source, "Development", TEAL)
    ]:
        val = float(precision_at_prior(pi, t_ref, f_ref))
        ax.scatter([pi], [val], color=color, s=80, zorder=5)
        ax.annotate(f"{label}: {val:.1%}", (pi, val), xytext=(8, 15), textcoords="offset points")
    ax.set(xlabel="Event prevalence (log scale)", ylabel="Precision among alarms", ylim=(0, 1.03), title="Figure 2 · Precision changes even when TPR and FPR do not")
    _save(fig, root, "fig2_precision_and_prevalence", suffix)

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.2))
    axes[0].plot(fpr[::-1], tpr[::-1], color=BLUE, lw=2.5)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Same ROC curve")
    axes[1].plot(tpr[::-1], ps[::-1], color=TEAL, lw=2.3, label=f"Source π={source:.0%}")
    axes[1].plot(tpr[::-1], pt[::-1], color=ORANGE, lw=2.3, label=f"Target π={target:.0%}")
    axes[1].axhline(target, color=ORANGE, ls="--", lw=1, alpha=0.5)
    axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02), title="Different precision–recall curve")
    axes[1].legend(frameon=False)
    _save(fig, root, "fig3_roc_vs_precision_recall", suffix)

    rng = np.random.default_rng(scenario.seed + 11)
    n_population = 20000 if quick else 120000
    score, observed = population_sample(rng, n_population, target, mu)
    p_source = posterior(score, source, mu)
    p_target = correct_prior(p_source, source, target)
    source_bins, adjusted_bins = calibration_bins(observed, p_source), calibration_bins(observed, p_target)
    csv_out("calibration_bins", [
        {"method": method, **row}
        for method, rows in (("source_frozen", source_bins), ("prior_adjusted", adjusted_bins))
        for row in rows
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", label="Ideal calibration")
    for rows, name, color in [
        (source_bins, "Source probabilities, frozen", ORANGE),
        (adjusted_bins, "Target-prior adjusted", BLUE),
    ]:
        usable = [row for row in rows if row["n"] >= 30]
        ax.plot(
            [row["mean_predicted"] for row in usable],
            [row["observed_frequency"] for row in usable],
            "o-", color=color, lw=2, markersize=5, label=name,
        )
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event fraction", xlim=(0, 1), ylim=(0, 1), title="Figure 4 · Calibration after deployment shift")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, root, "fig4_calibration_after_shift", suffix)

    estimation_rows: list[dict[str, Any]] = []
    experiments = [(2.0, 3000), (2.0, 12000), (0.8, 3000), (0.8, 12000)]
    for scenario_idx, (sep, n) in enumerate(experiments):
        t_bb, f_bb = operating_rates(1.0, sep)
        for rep in range(5 if quick else 23):
            rep_rng = np.random.default_rng(scenario.seed + 10000 * (scenario_idx + 1) + rep)
            z, _ = population_sample(rep_rng, n, target, sep)
            p = posterior(z, source, sep)
            estimation_rows.append({
                "separation": sep, "n_unlabeled": n, "replicate": rep,
                "true_prevalence": target,
                "mle_prevalence": label_shift_prior_mle(p, source),
                "bbse_prevalence_unclipped": black_box_shift_prevalence(float(np.mean(z > 1)), float(t_bb), float(f_bb)),
            })
    csv_out("prior_estimates", estimation_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for ax, sep, name in zip(axes, (2.0, 0.8), ("Strong discrimination: μ=2", "Weak discrimination: μ=0.8")):
        groups: list[list[float]] = []
        names: list[str] = []
        colors: list[str] = []
        for n in (3000, 12000):
            rows = [r for r in estimation_rows if r["separation"] == sep and r["n_unlabeled"] == n]
            for field, label, color in [
                ("mle_prevalence", "ML", BLUE), ("bbse_prevalence_unclipped", "BB", ORANGE)
            ]:
                groups.append([float(r[field]) for r in rows])
                names.append(f"{n // 1000}k {label}")
                colors.append(color)
        boxes = ax.boxplot(groups, patch_artist=True, tick_labels=names)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.38)
        ax.axhline(target, color=TEAL, ls="--", lw=1.8, label="Actual π")
        ax.set(title=name, xlabel="Unlabeled sample and estimator")
        ax.tick_params(axis="x", labelrotation=28)
    axes[0].set_ylabel("Estimated prevalence (BB left unclipped)")
    axes[0].legend(frameon=False)
    _save(fig, root, "fig5_estimating_target_prevalence", suffix)

    c_fp, c_fn = scenario.false_alarm_cost, scenario.missed_event_cost
    cost_threshold = cost_probability_threshold(c_fp, c_fn)
    frozen_z = score_threshold(cost_threshold, source, mu)
    optimal_z = score_threshold(cost_threshold, target, mu)
    grid = np.linspace(-1, 4.8, 340)
    costs = [decision_operating_point(float(z), target, mu, c_fp, c_fn) for z in grid]
    csv_out("cost_curves", costs)
    frozen = decision_operating_point(frozen_z, target, mu, c_fp, c_fn)
    optimal = decision_operating_point(optimal_z, target, mu, c_fp, c_fn)
    never = decision_operating_point(9.0, target, mu, c_fp, c_fn)
    csv_out("policy_comparison", [
        {"policy": label, **point, "cost_per_100k": 100000 * point["cost_per_case"]}
        for label, point in [
            ("frozen_source_prior", frozen), ("target_prior_adjusted", optimal),
            ("almost_never_alert", never),
        ]
    ])
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.3))
    axes[0].plot(grid, [r["cost_per_case"] for r in costs], color=BLUE, lw=2.4)
    axes[0].axvline(frozen_z, color=ORANGE, ls="--", label="Frozen source threshold")
    axes[0].axvline(optimal_z, color=TEAL, ls="--", label="Target Bayes threshold")
    axes[0].set(xlabel="Score threshold", ylabel="Expected loss per case", title="Loss, not ROC AUC, determines the threshold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(
        ["Frozen", "Adjusted", "Never"],
        [r["cost_per_case"] for r in (frozen, optimal, never)],
        color=[ORANGE, TEAL, GRAY], alpha=0.85,
    )
    axes[1].set(ylabel="Expected loss per case", title="Three frozen operating policies")
    _save(fig, root, "fig6_cost_of_the_wrong_prior", suffix)

    budget_rows: list[dict[str, float]] = []
    for budget in (0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30):
        capacity_cutoff = budget_threshold(budget, target, mu)
        selected_cutoff = max(capacity_cutoff, optimal_z)
        budget_rows.append({
            "maximum_alert_fraction": budget, "capacity_filling_cutoff": capacity_cutoff,
            **decision_operating_point(selected_cutoff, target, mu, c_fp, c_fn),
        })
    csv_out("budget_frontier", budget_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.1))
    axes[0].plot(
        [r["maximum_alert_fraction"] for r in budget_rows],
        [r["tpr"] for r in budget_rows], "o-", color=BLUE, lw=2,
    )
    axes[0].set(xlabel="Maximum review capacity", ylabel="Fraction of events detected", title="Event capture versus capacity")
    axes[1].plot(
        [r["maximum_alert_fraction"] for r in budget_rows],
        [r["cost_per_case"] for r in budget_rows], "o-", color=ORANGE, lw=2,
    )
    axes[1].axhline(optimal["cost_per_case"], color=TEAL, ls="--", label="Unconstrained Bayes loss")
    axes[1].set(xlabel="Maximum review capacity", ylabel="Minimum expected loss", title="Capacity stops helping after the economic optimum")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, root, "fig7_capacity_constrained_alerts", suffix)

    drift_rows: list[dict[str, float]] = []
    for i, actual_mu in enumerate((2.0, 1.8, 1.6, 1.4, 1.2)):
        drng = np.random.default_rng(scenario.seed + 40000 + i)
        z, _ = population_sample(drng, 6000 if quick else 50000, target, actual_mu)
        inferred = label_shift_prior_mle(posterior(z, source, mu), source)
        wrong = decision_operating_point(optimal_z, target, actual_mu, c_fp, c_fn)
        oracle_z = score_threshold(cost_threshold, target, actual_mu)
        oracle = decision_operating_point(oracle_z, target, actual_mu, c_fp, c_fn)
        drift_rows.append({
            "actual_positive_separation": actual_mu, "true_prevalence": target,
            "mle_prior_under_wrong_conditionals": inferred, "true_roc_auc": roc_auc(actual_mu),
            "misspecified_cost_per_case": wrong["cost_per_case"],
            "oracle_cost_per_case": oracle["cost_per_case"],
            "regret_per_case": wrong["cost_per_case"] - oracle["cost_per_case"],
        })
    csv_out("conditional_shift", drift_rows)

    verification_rows: list[dict[str, Any]] = []
    for rep in range(8 if quick else 55):
        vrng = np.random.default_rng(scenario.seed + 90000 + rep)
        z, y = population_sample(vrng, 10000 if quick else 22000, target, mu)
        alarms = z >= 2.2
        prop_mar = np.where(alarms, 1.0, scenario.verification_probability)
        verified_mar = vrng.random(len(y)) < prop_mar
        naive_mar, ht_mar = verification_estimates(y, alarms, verified_mar, scenario.verification_probability)
        # Under MNAR, positive nonalarms are systematically less likely to be verified.
        prop_mnar = np.where(alarms, 1.0, np.where(y, 0.01, scenario.verification_probability))
        verified_mnar = vrng.random(len(y)) < prop_mnar
        naive_mnar, ht_mnar = verification_estimates(y, alarms, verified_mnar, scenario.verification_probability)
        verification_rows.append({
            "replicate": rep, "actual_sample_prevalence": float(y.mean()),
            "verified_fraction_mar": float(verified_mar.mean()),
            "naive_mar": naive_mar, "ht_mar": ht_mar,
            "verified_fraction_mnar": float(verified_mnar.mean()),
            "naive_mnar": naive_mnar, "ht_mnar_with_wrong_propensities": ht_mnar,
        })
    csv_out("verification_bias", verification_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    axes[0].plot(
        [r["actual_positive_separation"] for r in drift_rows],
        [r["mle_prior_under_wrong_conditionals"] for r in drift_rows],
        "o-", color=ORANGE, lw=2, label="Assumed label-shift MLE",
    )
    axes[0].axhline(target, color=TEAL, ls="--", label="Actual prevalence")
    axes[0].set(xlabel="Actual positive-score separation", ylabel="Inferred target prevalence", title="Conditional drift violates label-shift assumptions")
    axes[0].legend(frameon=False, fontsize=8)
    fields = [
        ("naive_mar", "Verified\nonly", ORANGE),
        ("ht_mar", "IPW\nMAR", TEAL),
        ("ht_mnar_with_wrong_propensities", "IPW\nMNAR", RED),
    ]
    boxes = axes[1].boxplot(
        [[r[field] for r in verification_rows] for field, _, _ in fields],
        tick_labels=[label for _, label, _ in fields], patch_artist=True,
    )
    for patch, (_, _, color) in zip(boxes["boxes"], fields):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    axes[1].axhline(target, color=BLUE, ls="--", label="Actual prevalence")
    axes[1].set(ylabel="Estimated prevalence", title="Selective labels and inverse-probability weights")
    axes[1].legend(frameon=False)
    _save(fig, root, "fig8_when_assumptions_fail", suffix)

    summary = {
        "source_prevalence": source, "target_prevalence": target,
        "score_separation": mu, "analytic_roc_auc": roc_auc(mu),
        "reference_threshold": 1.5, "reference_tpr": float(t_ref), "reference_fpr": float(f_ref),
        "source_precision_at_reference": float(precision_at_prior(source, t_ref, f_ref)),
        "target_precision_at_reference": float(precision_at_prior(target, t_ref, f_ref)),
        "target_brier_frozen": brier_score(observed, p_source),
        "target_brier_adjusted": brier_score(observed, p_target),
        "target_logloss_frozen": binary_log_loss(observed, p_source),
        "target_logloss_adjusted": binary_log_loss(observed, p_target),
        "target_ece_frozen": expected_calibration_error(source_bins),
        "target_ece_adjusted": expected_calibration_error(adjusted_bins),
        "frozen_threshold_z": frozen_z, "optimal_threshold_z": optimal_z,
        "frozen_cost_per_case": frozen["cost_per_case"],
        "adjusted_cost_per_case": optimal["cost_per_case"],
        "frozen_cost_ratio": frozen["cost_per_case"] / optimal["cost_per_case"],
        "sample_n": float(n_population),
    }
    csv_out("summary", [{"measure": name, "value": value} for name, value in summary.items()])
    return summary


def main() -> None:
    """CLI entrypoint for publication and CI reproducibility."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Write smaller outputs with quick suffix")
    parser.add_argument("--output", type=Path, default=None, help="Results root directory")
    arguments = parser.parse_args()
    stats = run_study(quick=arguments.quick, output_dir=arguments.output)
    for name, value in stats.items():
        print(f"{name}: {value:.6f}")
    print("Wrote 8 figures and 10 CSV tables")


if __name__ == "__main__":
    main()
