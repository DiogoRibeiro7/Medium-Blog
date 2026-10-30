"""Reproduce eight synthetic experiments on conformal-prediction validity.

Training, calibration, and evaluation use disjoint seeded random streams.
Neither these constructed populations nor their oracle density ratios are
observational evidence about any real organization.
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

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
SEED = 20261030
ALPHA = 0.10
BASE_RATE = 0.20
SHIFT_RATE = 0.60
METHODS = ("Global", "Mondrian", "Normalized")
FIGURE_NAMES = (
    "fig1_heteroscedasticity.png",
    "fig2_finite_sample_quantile.png",
    "fig3_marginal_vs_group_coverage.png",
    "fig4_interval_geometry.png",
    "fig5_covariate_mixture_shift.png",
    "fig6_conditional_shift.png",
    "fig7_temporal_adaptation.png",
    "fig8_simultaneous_targets.png",
)
RESULT_NAMES = (
    "summary.csv", "training_groups.csv", "calibration_scores.csv",
    "replicate_coverage.csv", "baseline_coverage.csv", "interval_geometry.csv",
    "covariate_shift.csv", "concept_shift.csv", "temporal_stream.csv",
    "multitarget.csv",
)


@dataclass(frozen=True)
class Batch:
    """Iid synthetic observations and their known generating parameters."""

    x: Floats
    group: Ints
    y: Floats
    truth: Floats
    noise_sd: Floats


@dataclass(frozen=True)
class Model:
    """Least-squares mean and residual scales estimated only on training."""

    beta: Floats
    scale: Floats


@dataclass(frozen=True)
class Calibration:
    """Frozen thresholds from an independent calibration set."""

    global_q: float
    group_q: Floats
    normalized_q: float
    scores: Floats
    normalized_scores: Floats
    groups: Ints


@dataclass(frozen=True)
class Study:
    """All deterministic, synthetic study evidence."""

    model: Model
    calibration: Calibration
    training: Batch
    baseline: Batch
    mixture: Batch
    conditional: Batch
    replicates: Floats
    geometry: dict[str, Floats]
    coverage: dict[str, dict[str, dict[str, float]]]
    stream: dict[str, Floats]
    multitarget: dict[str, Floats | float]
    summary: dict[str, float]
    rows: dict[str, list[dict[str, Any]]]


def check_probability(value: float, label: str) -> None:
    """Reject bool, nonfinite and boundary values."""

    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{label} must be a scalar probability")
    if not np.isfinite(value) or not 0.0 < float(value) < 1.0:
        raise ValueError(f"{label} must belong to (0,1)")


def check_size(n: int, label: str, minimum: int = 1) -> None:
    """Reject fractional sizes and booleans."""

    if type(n) is not int or n < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def split_threshold(scores: Floats, alpha: float = ALPHA) -> float:
    """Conformal rank ceil((n+1)(1-alpha)), including +infinity edge case."""

    check_probability(alpha, "alpha")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 1 or np.any(scores < 0) or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite, nonnegative, nonempty vector")
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float("inf") if k > len(scores) else float(np.partition(scores, k - 1)[k - 1])


def weighted_threshold(
    scores: Floats, groups: Ints, p_cal: float, p_target: float,
    test_group: int, alpha: float = ALPHA,
) -> float:
    """Oracle weighted conformal quantile with the test atom at infinity.

    This valid-covariate-shift construction uses *known*, synthetic density
    ratios. It does not estimate a ratio or handle conditional distribution shift.
    """

    for v, name in ((p_cal, "p_cal"), (p_target, "p_target"), (alpha, "alpha")):
        check_probability(v, name)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=np.int64)
    if scores.ndim != 1 or len(scores) < 1 or groups.shape != scores.shape:
        raise ValueError("scores and groups must align")
    if test_group not in (0, 1) or np.any((groups != 0) & (groups != 1)):
        raise ValueError("groups must be binary")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("scores must be finite and nonnegative")
    ratios = np.array([(1 - p_target) / (1 - p_cal), p_target / p_cal])
    w = ratios[groups]
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(w[order])
    mass = (1 - alpha) * (float(w.sum()) + float(ratios[test_group]))
    if mass > float(cumulative[-1]) + 1e-12:
        return float("inf")
    return float(scores[order[np.searchsorted(cumulative, mass, side="left")]])


def design(x: Floats, group: Ints) -> Floats:
    """Mean-regression basis for one continuous and one binary predictor."""

    x = np.asarray(x, dtype=float)
    group = np.asarray(group, dtype=np.int64)
    if x.ndim != 1 or x.shape != group.shape or not np.all(np.isfinite(x)):
        raise ValueError("x and group must be finite and aligned")
    if np.any((group != 0) & (group != 1)):
        raise ValueError("binary groups expected")
    return np.column_stack((np.ones(len(x)), np.sin(2 * np.pi * x), np.cos(4 * np.pi * x), group))


def true_mean(x: Floats, group: Ints) -> Floats:
    """Synthetic regression surface known by construction."""

    return design(x, group) @ np.array([0, 1.25, 0.50, 0.30])


def true_sd(x: Floats, group: Ints, b_multiplier: float = 1.0) -> Floats:
    """Unequal conditional variance, with rare group B noisier."""

    x = np.asarray(x, dtype=float)
    group = np.asarray(group, dtype=np.int64)
    if not np.isfinite(b_multiplier) or b_multiplier <= 0:
        raise ValueError("b_multiplier must be finite and positive")
    if np.any((x < 0) | (x > 1)) or np.any((group != 0) & (group != 1)):
        raise ValueError("out-of-domain feature or group")
    base = 0.35 + 0.35 * x
    return np.where(group == 1, (base + 0.90 + 0.30 * x) * b_multiplier, base)


def sample_batch(rng: np.random.Generator, n: int, p_b: float = BASE_RATE, b_multiplier: float = 1.0) -> Batch:
    """Draw independent (X,G,Y) from a specified synthetic target."""

    check_size(n, "n")
    check_probability(p_b, "p_b")
    x = rng.uniform(size=n)
    groups = (rng.random(n) < p_b).astype(np.int64)
    mean = true_mean(x, groups)
    sd = true_sd(x, groups, b_multiplier)
    return Batch(x, groups, mean + sd * rng.normal(size=n), mean, sd)


def fit_model(training: Batch) -> Model:
    """Freeze both mean and group RMS scale using training, never calibration."""

    beta = np.linalg.lstsq(design(training.x, training.group), training.y, rcond=None)[0]
    err = training.y - design(training.x, training.group) @ beta
    scale = np.empty(2)
    for group in (0, 1):
        residuals = err[training.group == group]
        if len(residuals) < 2:
            raise ValueError("each training group needs >= 2 observations")
        scale[group] = float(np.sqrt(np.mean(residuals**2)))
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("invalid training scales")
    return Model(beta, scale)


def predict(model: Model, batch: Batch) -> Floats:
    """Predict without touching observed test outcomes."""

    return design(batch.x, batch.group) @ model.beta


def calibrate(model: Model, batch: Batch, alpha: float = ALPHA) -> Calibration:
    """Pooled, Mondrian group-specific, and normalized conformal thresholds."""

    scores = np.abs(batch.y - predict(model, batch))
    group_q = np.array([split_threshold(scores[batch.group == g], alpha) for g in (0, 1)])
    norm = scores / model.scale[batch.group]
    return Calibration(split_threshold(scores, alpha), group_q, split_threshold(norm, alpha), scores, norm, batch.group.copy())


def halfwidth(method: str, model: Model, cal: Calibration, groups: Ints, p_target: float | None = None) -> Floats:
    """Apply frozen quantiles; oracle weighting is permitted only when requested."""

    groups = np.asarray(groups, dtype=np.int64)
    if groups.ndim != 1 or np.any((groups != 0) & (groups != 1)):
        raise ValueError("groups must be a binary vector")
    if method == "Global":
        return np.full(len(groups), cal.global_q)
    if method == "Mondrian":
        return cal.group_q[groups]
    if method == "Normalized":
        return cal.normalized_q * model.scale[groups]
    if method == "Oracle weighted" and p_target is not None:
        values = np.array([weighted_threshold(cal.scores, cal.groups, BASE_RATE, p_target, g) for g in (0, 1)])
        return values[groups]
    raise ValueError(f"unknown or improperly configured method: {method}")


def assess(
    batch: Batch, model: Model, cal: Calibration, *, oracle_target: float | None = None
) -> dict[str, dict[str, float]]:
    """Observed test coverage is empirical, not itself a finite-sample theorem."""

    results = {}
    methods = METHODS + (("Oracle weighted",) if oracle_target is not None else ())
    for method in methods:
        widths = halfwidth(method, model, cal, batch.group, oracle_target)
        hit = np.abs(batch.y - predict(model, batch)) <= widths
        a, b = batch.group == 0, batch.group == 1
        results[method] = {
            "marginal": float(hit.mean()),
            "group_a": float(hit[a].mean()),
            "group_b": float(hit[b].mean()),
            "full_width": float((2 * widths).mean()),
            "full_width_a": float((2 * widths[a]).mean()),
            "full_width_b": float((2 * widths[b]).mean()),
            "n_a": float(a.sum()),
            "n_b": float(b.sum()),
        }
    return results


def repeated_coverage(model: Model, replicates: int = 400, n_cal: int = 240, n_test: int = 1300) -> Floats:
    """Fresh calibration/test splits illustrate the unconditional coverage claim."""

    for n, label in ((replicates, "replicates"), (n_cal, "n_cal"), (n_test, "n_test")):
        check_size(n, label, 15)
    output = np.empty((replicates, 3))
    for i, source in enumerate(np.random.SeedSequence([SEED, 411]).spawn(replicates)):
        a, b = (np.random.default_rng(s) for s in source.spawn(2))
        cal_batch, test_batch = sample_batch(a, n_cal), sample_batch(b, n_test)
        q = split_threshold(np.abs(cal_batch.y - predict(model, cal_batch)))
        covered = np.abs(test_batch.y - predict(model, test_batch)) <= q
        output[i] = [covered.mean(), covered[test_batch.group == 0].mean(), covered[test_batch.group == 1].mean()]
    return output


def temporal_stream(n: int = 1200, window: int = 120) -> dict[str, Floats]:
    """AR(1) stream and a past-only rolling recalibration heuristic.

    No exchangeability or exact online validity is claimed for this heuristic.
    """

    check_size(n, "n", 700)
    check_size(window, "window", 30)
    if window >= 600:
        raise ValueError("window must leave pre-change observations")
    cal_rng = np.random.default_rng(np.random.SeedSequence([SEED, 501]))
    seq_rng = np.random.default_rng(np.random.SeedSequence([SEED, 502]))
    cal_scores = np.abs(cal_rng.normal(scale=0.6, size=800))
    fixed_q = split_threshold(cal_scores)
    residuals = np.empty(n)
    previous = float(seq_rng.normal(scale=0.6))
    for t in range(n):
        sigma = 0.6 if t < 600 else 1.05
        previous = 0.5 * previous + np.sqrt(0.75) * sigma * float(seq_rng.normal())
        residuals[t] = previous
    scores = np.abs(residuals)
    rolling_q = np.empty(n)
    history = cal_scores[-window:].tolist()
    for t in range(n):
        rolling_q[t] = split_threshold(np.asarray(history[-window:]))
        history.append(float(scores[t]))  # append only after the prediction
    fixed_hit = (scores <= fixed_q).astype(float)
    rolling_hit = (scores <= rolling_q).astype(float)
    fixed_trailing = np.full(n, np.nan)
    rolling_trailing = np.full(n, np.nan)
    for t in range(99, n):
        fixed_trailing[t] = fixed_hit[t - 99 : t + 1].mean()
        rolling_trailing[t] = rolling_hit[t - 99 : t + 1].mean()
    return {
        "t": np.arange(n, dtype=float), "residual": residuals,
        "fixed_q": np.full(n, fixed_q), "rolling_q": rolling_q,
        "fixed_hit": fixed_hit, "rolling_hit": rolling_hit,
        "fixed_trailing": fixed_trailing, "rolling_trailing": rolling_trailing,
    }


def multitarget_experiment(n_cal: int = 700, n_test: int = 28000) -> dict[str, Floats | float]:
    """Compare individual 90% intervals with Bonferroni simultaneous coverage."""

    check_size(n_cal, "n_cal", 5)
    check_size(n_test, "n_test", 100)
    cal_rng = np.random.default_rng(np.random.SeedSequence([SEED, 601]))
    test_rng = np.random.default_rng(np.random.SeedSequence([SEED, 602]))
    sd = np.array([0.55, 0.80, 1.10])
    cal = np.abs(cal_rng.normal(size=(n_cal, 3)) * sd)
    test = np.abs(test_rng.normal(size=(n_test, 3)) * sd)
    plain_q = np.array([split_threshold(cal[:, j]) for j in range(3)])
    bonf_q = np.array([split_threshold(cal[:, j], ALPHA / 3) for j in range(3)])
    plain, bonf = test <= plain_q, test <= bonf_q
    return {
        "sd": sd, "plain_q": plain_q, "bonf_q": bonf_q,
        "plain_marginal": plain.mean(axis=0), "bonf_marginal": bonf.mean(axis=0),
        "plain_joint": float(np.all(plain, axis=1).mean()),
        "bonf_joint": float(np.all(bonf, axis=1).mean()),
        "plain_total_width": float(2 * plain_q.sum()),
        "bonf_total_width": float(2 * bonf_q.sum()),
    }


def _assessment_rows(scenario: str, outcomes: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Prepare comparable coverage rows from each scenario."""

    return [{"scenario": scenario, "method": k, **v} for k, v in outcomes.items()]


def run_study(quick: bool = False) -> Study:
    """Construct independent splits and every scientific result once."""

    if type(quick) is not bool:
        raise ValueError("quick must be boolean")
    rngs = [np.random.default_rng(np.random.SeedSequence([SEED, i])) for i in (101, 102, 103, 104, 105)]
    training = sample_batch(rngs[0], 4000)
    model = fit_model(training)
    calibration_batch = sample_batch(rngs[1], 800)
    cal = calibrate(model, calibration_batch)
    n_test = 4000 if quick else 24000
    baseline = sample_batch(rngs[2], n_test)
    mixture = sample_batch(rngs[3], n_test, p_b=SHIFT_RATE)
    conditional = sample_batch(rngs[4], n_test, b_multiplier=1.65)
    coverage = {
        "baseline": assess(baseline, model, cal),
        "mixture": assess(mixture, model, cal, oracle_target=SHIFT_RATE),
        "conditional": assess(conditional, model, cal),
    }
    rep = repeated_coverage(model, 65 if quick else 400, n_test=700 if quick else 1300)
    stream = temporal_stream(900 if quick else 1200)
    multi = multitarget_experiment(n_test=4500 if quick else 28000)
    grid = np.linspace(0, 1, 160)
    x = np.tile(grid, 2)
    g = np.repeat(np.array([0, 1], dtype=np.int64), len(grid))
    geometry = {"x": x, "group": g.astype(float), "prediction": design(x, g) @ model.beta}
    geometry.update({method: halfwidth(method, model, cal, g) for method in METHODS})
    summary = {
        "nominal_coverage": 1 - ALPHA,
        "n_train": float(len(training.y)),
        "n_cal": float(len(cal.scores)),
        "n_test_scenario": float(n_test),
        "cal_group_b": float(np.sum(cal.groups)),
        "training_sd_a": float(model.scale[0]), "training_sd_b": float(model.scale[1]),
        "global_q": cal.global_q, "mondrian_q_a": float(cal.group_q[0]),
        "mondrian_q_b": float(cal.group_q[1]), "normalized_q": cal.normalized_q,
        "replicate_global_coverage_mean": float(rep[:, 0].mean()),
        "replicate_global_coverage_sd": float(rep[:, 0].std(ddof=1)),
        "baseline_global_marginal": coverage["baseline"]["Global"]["marginal"],
        "baseline_global_group_b": coverage["baseline"]["Global"]["group_b"],
        "baseline_mondrian_group_b": coverage["baseline"]["Mondrian"]["group_b"],
        "mixture_global_marginal": coverage["mixture"]["Global"]["marginal"],
        "mixture_oracle_weighted_marginal": coverage["mixture"]["Oracle weighted"]["marginal"],
        "conditional_mondrian_group_b": coverage["conditional"]["Mondrian"]["group_b"],
        "temporal_fixed_post_change": float(stream["fixed_hit"][600:].mean()),
        "temporal_rolling_post_change": float(stream["rolling_hit"][600:].mean()),
        "multitarget_plain_joint": float(multi["plain_joint"]),
        "multitarget_bonferroni_joint": float(multi["bonf_joint"]),
    }
    rows: dict[str, list[dict[str, Any]]] = {
        "summary.csv": [{"metric": k, "value": v} for k, v in summary.items()],
        "training_groups.csv": [
            {"group": i, "n": int(np.sum(training.group == i)), "train_residual_sd": float(model.scale[i])}
            for i in (0, 1)
        ],
        "calibration_scores.csv": [
            {"i": i, "group": int(g_i), "score": float(s), "normalized_score": float(ns)}
            for i, (g_i, s, ns) in enumerate(zip(cal.groups, cal.scores, cal.normalized_scores))
        ],
        "replicate_coverage.csv": [
            {"replicate": i, "marginal": float(r[0]), "group_a": float(r[1]), "group_b": float(r[2])}
            for i, r in enumerate(rep)
        ],
        "baseline_coverage.csv": _assessment_rows("baseline", coverage["baseline"]),
        "covariate_shift.csv": _assessment_rows("p_b_0.60", coverage["mixture"]),
        "concept_shift.csv": _assessment_rows("B_sd_times_1.65", coverage["conditional"]),
        "interval_geometry.csv": [
            {
                "x": float(x_i), "group": int(g_i), "prediction": float(p_i),
                "global_halfwidth": float(q1), "mondrian_halfwidth": float(q2),
                "normalized_halfwidth": float(q3),
            }
            for x_i, g_i, p_i, q1, q2, q3 in zip(
                x, g, geometry["prediction"], geometry["Global"], geometry["Mondrian"], geometry["Normalized"]
            )
        ],
        "temporal_stream.csv": [
            {key: float(values[i]) for key, values in stream.items()}
            for i in range(len(stream["t"]))
        ],
        "multitarget.csv": [
            {
                "target": j + 1, "sd": float(multi["sd"][j]),
                "plain_halfwidth": float(multi["plain_q"][j]),
                "bonferroni_halfwidth": float(multi["bonf_q"][j]),
                "plain_marginal": float(multi["plain_marginal"][j]),
                "bonf_marginal": float(multi["bonf_marginal"][j]),
                "plain_joint": float(multi["plain_joint"]),
                "bonf_joint": float(multi["bonf_joint"]),
            }
            for j in range(3)
        ],
    }
    assert set(rows) == set(RESULT_NAMES)
    return Study(model, cal, training, baseline, mixture, conditional, rep, geometry, coverage, stream, multi, summary, rows)


def plot_figures(study: Study, directory: Path, quick: bool = False) -> list[Path]:
    """Produce eight independently meaningful publication-quality PNG figures."""

    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 160, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18})
    directory.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    colors = {"Global": "#405b76", "Mondrian": "#088d80", "Normalized": "#e49631"}

    def save(fig: Any, name: str) -> None:
        filename = name.replace(".png", "_quick.png") if quick else name
        path = directory / filename
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    residual = study.baseline.y - predict(study.model, study.baseline)
    for g, ax in enumerate(axes):
        m = study.baseline.group == g
        stride = max(1, int(m.sum() / 1200))
        ax.scatter(study.baseline.x[m][::stride], residual[m][::stride], s=7, alpha=0.22, color=("#2375b7" if g == 0 else "#c86b3d"))
        xx = np.linspace(0, 1, 120)
        sd = true_sd(xx, np.full(len(xx), g, dtype=np.int64))
        ax.plot(xx, 1.645 * sd, "k--", label="True conditional ±1.645 SD")
        ax.plot(xx, -1.645 * sd, "k--")
        ax.set(title=f"Group {'AB'[g]} (n = {int(m.sum())})", xlabel="Feature x", ylabel="Y − fitted mean")
    fig.suptitle("Heteroscedasticity is built into the generating process")
    save(fig, FIGURE_NAMES[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    sorted_scores = np.sort(study.calibration.scores)
    k = int(np.ceil((len(sorted_scores) + 1) * (1 - ALPHA)))
    axes[0].plot(np.arange(1, len(sorted_scores) + 1), sorted_scores, color="#2375b7")
    axes[0].axvline(k, ls="--", color="#c94b37", label=f"Rank k = {k}")
    axes[0].axhline(study.calibration.global_q, color="#c94b37", ls=":")
    axes[0].set(title="The finite rank is not interpolation", xlabel="Sorted calibration rank", ylabel="Absolute error")
    axes[0].legend(frameon=False)
    axes[1].hist(study.replicates[:, 0], bins=23, edgecolor="white", color="#6c9fc0")
    axes[1].axvline(0.9, ls="--", color="#c94b37", label="Nominal 90%")
    axes[1].axvline(study.replicates[:, 0].mean(), color="#405b76", label="Across-split mean")
    axes[1].set(title="Coverage varies by independent calibration set", xlabel="Empirical marginal test coverage", ylabel="Replicates")
    axes[1].legend(frameon=False)
    save(fig, FIGURE_NAMES[1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, method in enumerate(METHODS):
        values = [study.coverage["baseline"][method][key] for key in ("marginal", "group_a", "group_b")]
        ax.bar(np.arange(3) + (i - 1) * 0.26, values, 0.24, label=method, color=colors[method])
    ax.axhline(0.9, color="#c94b37", ls="--", label="Nominal 90%")
    ax.set(xticks=range(3), xticklabels=["Marginal", "Group A", "Group B"], ylabel="Test coverage", ylim=(0, 1.035), title="Good pooled coverage need not protect group B")
    ax.legend(ncol=4, frameon=False, fontsize=9)
    save(fig, FIGURE_NAMES[2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for g, ax in enumerate(axes):
        mask = study.geometry["group"] == g
        for method in METHODS:
            ax.plot(study.geometry["x"][mask], 2 * study.geometry[method][mask], lw=2.3, label=method, color=colors[method])
        ax.set(title=f"Group {'AB'[g]}", xlabel="Feature x", ylabel="Full interval width")
    axes[0].legend(frameon=False)
    fig.suptitle("One marginal interval, two conditional policies, different costs")
    save(fig, FIGURE_NAMES[3])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["Calibration", "Shifted target"], [BASE_RATE, SHIFT_RATE], color=["#6194b8", "#c48249"])
    axes[0].set(title="Only the group mixture changes", ylabel="Group B proportion", ylim=(0, 1))
    labels = METHODS + ("Oracle weighted",)
    xx = np.arange(len(labels))
    axes[1].bar(xx - 0.18, [study.coverage["baseline"][m]["marginal"] if m in METHODS else np.nan for m in labels], 0.34, label="Baseline", color="#6194b8")
    axes[1].bar(xx + 0.18, [study.coverage["mixture"][m]["marginal"] for m in labels], 0.34, label="Shift", color="#c48249")
    axes[1].axhline(0.9, color="#c94b37", ls="--")
    axes[1].set(xticks=xx, xticklabels=["Global", "Mondrian", "Normalized", "Weighted*"], title="*Known synthetic density ratios", ylabel="Marginal coverage", ylim=(0, 1.03))
    axes[1].legend(frameon=False)
    save(fig, FIGURE_NAMES[4])

    fig, ax = plt.subplots(figsize=(10.7, 4.8))
    for i, method in enumerate(METHODS):
        ax.bar(np.arange(3) + (i - 1) * 0.25, [study.coverage["conditional"][method][key] for key in ("marginal", "group_a", "group_b")], 0.23, label=method, color=colors[method])
    ax.axhline(0.9, color="#c94b37", ls="--")
    ax.set(xticks=range(3), xticklabels=["Marginal", "Group A unchanged", "Group B SD × 1.65"], ylabel="Test coverage", ylim=(0, 1.03), title="A conditional change defeats group-stratified calibration")
    ax.legend(frameon=False, ncol=3)
    save(fig, FIGURE_NAMES[5])

    stream = study.stream
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(stream["t"], np.abs(stream["residual"]), lw=0.9, color="#a9bac8", label="Absolute residual")
    axes[0].plot(stream["t"], stream["fixed_q"], color="#c6503c", lw=2, label="Fixed calibrated q")
    axes[0].plot(stream["t"], stream["rolling_q"], color="#078a7c", lw=1.8, label="Past-only rolling q")
    axes[0].set(title="AR(1) residuals; variance jump at t = 600", ylabel="Absolute score / threshold")
    axes[0].legend(ncol=3, frameon=False)
    axes[1].plot(stream["t"], stream["fixed_trailing"], color="#c6503c", label="Fixed, trailing 100")
    axes[1].plot(stream["t"], stream["rolling_trailing"], color="#078a7c", label="Rolling, trailing 100")
    axes[1].axhline(0.9, color="black", ls=":", label="Reference 90%")
    axes[1].set(xlabel="Stream time", ylabel="Realized trailing coverage", ylim=(0.43, 1.03))
    axes[1].legend(ncol=3, frameon=False)
    for ax in axes:
        ax.axvline(600, color="black", ls="--", alpha=0.6)
    save(fig, FIGURE_NAMES[6])

    multi = study.multitarget
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["Three separate 90%\nintervals", "Bonferroni-adjusted\nintervals"], [multi["plain_joint"], multi["bonf_joint"]], color=["#cf8650", "#188e7f"])
    axes[0].axhline(0.9, color="#c94b37", ls="--", label="Desired joint 90%")
    axes[0].set(title="Joint coverage is not single-target coverage", ylabel="P(all three covered)", ylim=(0, 1.03))
    axes[0].legend(frameon=False)
    axes[1].bar(["Individual 90%", "Bonferroni"], [multi["plain_total_width"], multi["bonf_total_width"]], color=["#cf8650", "#188e7f"])
    axes[1].set(title="The cost of a simultaneous familywise claim", ylabel="Sum of full interval widths")
    save(fig, FIGURE_NAMES[7])
    return output


def write_results(study: Study, directory: Path, quick: bool = False) -> list[Path]:
    """Export ten deterministic CSV tables, preserving full-study results in quick mode."""

    directory.mkdir(parents=True, exist_ok=True)
    output = []
    for filename in RESULT_NAMES:
        rows = study.rows[filename]
        if not rows:
            raise ValueError(f"no data for {filename}")
        output_name = filename.replace(".csv", "_quick.csv") if quick else filename
        path = directory / output_name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output.append(path)
    return output


def main() -> None:
    """Run a complete or a suffixed smoke-test study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    study = run_study(quick=args.quick)
    figures = plot_figures(study, root / "figures", quick=args.quick)
    outputs = write_results(study, root / "results", quick=args.quick)
    print(f"Generated {len(figures)} figures and {len(outputs)} result tables.")
    for metric, value in study.summary.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
