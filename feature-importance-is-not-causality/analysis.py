"""Nine seeded experiments on model reliance, attribution, and causal limits.

All observations are synthetic. The full run writes nine pairs of evidence CSVs
and nine figures. Quick runs use a suffix and cannot overwrite canonical files.
"""

from __future__ import annotations

import argparse
import csv
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
SEED = 20261101
EXPERIMENTS = (
    "conditional", "proxies", "impurity", "shapley", "interaction",
    "leakage", "shift", "subgroup", "selection",
)
FIGURES = (
    "fig1_conditional_permutation", "fig2_proxy_reliance",
    "fig3_impurity_bias", "fig4_shapley_games", "fig5_xor_interaction",
    "fig6_leakage_availability", "fig7_importance_shift",
    "fig8_subgroup_averaging", "fig9_selection_optimism",
)
REPETITIONS = {
    "conditional": 24, "proxies": 24, "impurity": 18,
    "shapley": 24, "interaction": 24, "leakage": 24,
    "shift": 24, "subgroup": 24, "selection": 24,
}
Row = dict[str, str | int | float | None]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class Design:
    """Configuration of the reproducibility run."""

    root: Path = ROOT
    quick: bool = False
    seed: int = SEED


def check_matrix(x: NDArray[np.floating]) -> FloatMatrix:
    """Require a finite nonempty (n,p) numerical matrix with n >= 3."""

    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 1:
        raise ValueError("matrix must have at least three rows and one column")
    if not np.isfinite(a).all():
        raise ValueError("matrix contains nonfinite values")
    return a


def check_vector(y: NDArray[np.floating], n: int) -> NDArray[np.float64]:
    """Require an n-vector without missing or nonfinite values."""

    a = np.asarray(y, dtype=float)
    if a.ndim != 1 or len(a) != n or not np.isfinite(a).all():
        raise ValueError("response must be a finite vector with n rows")
    return a


def stream(seed: int, study: str, replicate: int) -> np.random.Generator:
    """Create a stable stream independent of Python's randomized hash seed."""

    if seed < 0 or replicate < 0 or study not in EXPERIMENTS:
        raise ValueError("invalid seed, study, or replicate")
    tag = zlib.crc32(study.encode("ascii"))
    return np.random.default_rng(np.random.SeedSequence([seed, tag, replicate]))


def mse(y: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    """Compute squared-error risk on one specified evaluation sample."""

    y = check_vector(y, len(y))
    p = check_vector(pred, len(y))
    return float(mean_squared_error(y, p))


def permutation_delta(
    model: object,
    x: FloatMatrix,
    y: NDArray[np.float64],
    column: int,
    rng: np.random.Generator,
) -> float:
    """Held-out risk difference from independently permuting one column.

    The fitted prediction function remains fixed; only its evaluation input
    changes. This is not a retraining ablation or a causal intervention.
    """

    x = check_matrix(x)
    y = check_vector(y, len(x))
    if column < 0 or column >= x.shape[1]:
        raise IndexError("column out of range")
    predict = getattr(model, "predict", None)
    if predict is None:
        raise TypeError("model must implement predict")
    before = mse(y, np.asarray(predict(x), dtype=float))
    perm = x.copy()
    perm[:, column] = rng.permutation(perm[:, column])
    return mse(y, np.asarray(predict(perm), dtype=float)) - before


def observe(
    replicate: int, condition: str, metric: str, value: float,
    theory: float | None = None,
) -> Row:
    """Create one long-format Monte Carlo observation."""

    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be nonempty")
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be nonempty")
    if not math.isfinite(float(value)):
        raise ValueError("observed value must be finite")
    if theory is not None and not math.isfinite(theory):
        raise ValueError("theory must be finite when provided")
    return {
        "replicate": replicate, "condition": condition, "metric": metric,
        "value": float(value), "theory": theory,
    }


def summarize(rows: list[Row]) -> list[Row]:
    """Independent-replication mean, sample SD, MC SE, and normal 95% interval.

    Conditional on the simulated design, these SEs quantify repeat-to-repeat
    Monte Carlo variation; they are not population or causal confidence limits.
    """

    if not rows:
        raise ValueError("cannot summarize an empty result")
    grouped: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["metric"]))
        grouped.setdefault(key, []).append(row)
    out: list[Row] = []
    for (condition, metric), group in sorted(grouped.items()):
        vals = np.asarray([float(row["value"]) for row in group], dtype=float)
        theories = {row["theory"] for row in group}
        if len(theories) != 1:
            raise ValueError("theory must be constant within a condition/metric")
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        se = sd / math.sqrt(len(vals))
        mean = float(vals.mean())
        out.append({
            "condition": condition, "metric": metric, "n": len(vals),
            "mean": mean, "sd": sd, "se": se,
            "lower95": mean - 1.96 * se, "upper95": mean + 1.96 * se,
            "theory": group[0]["theory"],
        })
    return out


def gaussian_pair(
    rng: np.random.Generator, n: int, rho: float,
) -> FloatMatrix:
    """Standard normal pair with population correlation rho."""

    if n < 3 or abs(rho) >= 1:
        raise ValueError("n must be at least three; |rho| must be less than one")
    x2 = rng.normal(size=n)
    x1 = rho * x2 + math.sqrt(1.0 - rho**2) * rng.normal(size=n)
    return np.column_stack((x1, x2))


def conditional_resample(
    rng: np.random.Generator, x2: NDArray[np.float64], rho: float,
) -> NDArray[np.float64]:
    """New X1 drawn from N(rho*X2, 1-rho²), independently given X2."""

    if abs(rho) >= 1:
        raise ValueError("|rho| must be below one")
    return rho * np.asarray(x2) + math.sqrt(1 - rho**2) * rng.normal(size=len(x2))


def conditional_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Oracle contrast: marginal versus conditional permutation under correlation."""

    beta = 1.5
    n = 350 if quick else 1500
    rows = []
    for rho in (0.0, 0.3, 0.6, 0.9, 0.98):
        x = gaussian_pair(rng, n, rho)
        y = beta * x[:, 0] + rng.normal(scale=0.8, size=n)
        baseline = mse(y, beta * x[:, 0])
        marginal = mse(y, beta * rng.permutation(x[:, 0])) - baseline
        conditional = mse(y, beta * conditional_resample(rng, x[:, 1], rho)) - baseline
        condition = f"{rho:.2f}"
        rows.extend((
            observe(rep, condition, "marginal_delta", marginal, 2 * beta**2),
            observe(rep, condition, "conditional_delta", conditional,
                    2 * beta**2 * (1 - rho**2)),
        ))
    return rows


def proxy_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Fixed-model perturbation versus refitting after deleting a proxy."""

    n_train, n_test = (140, 350) if quick else (380, 1000)
    rows = []
    for noise in (0.1, 0.4, 1.0, 2.0):
        ztr, zte = rng.normal(size=n_train), rng.normal(size=n_test)
        xtr = np.column_stack((ztr + noise * rng.normal(size=n_train),
                               ztr + noise * rng.normal(size=n_train)))
        xte = np.column_stack((zte + noise * rng.normal(size=n_test),
                               zte + noise * rng.normal(size=n_test)))
        ytr = ztr + rng.normal(scale=0.8, size=n_train)
        yte = zte + rng.normal(scale=0.8, size=n_test)
        full = Ridge(alpha=2.0).fit(xtr, ytr)
        reduced = Ridge(alpha=2.0).fit(xtr[:, 1:], ytr)
        baseline = mse(yte, full.predict(xte))
        fixed_delta = permutation_delta(full, xte, yte, 0, rng)
        refit_delta = mse(yte, reduced.predict(xte[:, 1:])) - baseline
        condition = f"{noise:.1f}"
        rows.extend((
            observe(rep, condition, "fixed_permutation_delta", fixed_delta),
            observe(rep, condition, "refit_ablation_delta", refit_delta),
            observe(rep, condition, "proxy_correlation",
                    float(np.corrcoef(xte.T)[0, 1])),
        ))
    return rows


def impurity_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Compare in-sample RF split credit with held-out permutation reliance."""

    ntr, nte = (170, 270) if quick else (440, 850)
    signal = rng.integers(0, 2, size=ntr + nte).astype(float)
    noise_binary = rng.integers(0, 2, size=ntr + nte).astype(float)
    noise_continuous = rng.normal(size=ntr + nte)
    x = np.column_stack((signal, noise_binary, noise_continuous))
    y = 1.8 * signal + rng.normal(scale=0.9, size=len(signal))
    model = RandomForestRegressor(
        n_estimators=36 if quick else 90, max_depth=8,
        min_samples_leaf=4, max_features=1.0,
        random_state=int(rng.integers(0, 2**31 - 1)), n_jobs=1,
    ).fit(x[:ntr], y[:ntr])
    labels = ("signal_binary", "noise_binary", "noise_continuous")
    rows = []
    for j, label in enumerate(labels):
        rows.extend((
            observe(rep, label, "mdi", float(model.feature_importances_[j])),
            observe(rep, label, "heldout_delta",
                    permutation_delta(model, x[ntr:], y[ntr:], j, rng)),
        ))
    return rows


def exact_two_feature_shapley(x1: float, x2: float, rho: float,
                              game: str) -> tuple[float, float]:
    """Exact two-feature coalitional values for f(x1,x2)=x1, mean zero.

    Interventional game integrates missing coordinates over their marginal
    distribution. Observational game conditions on coordinates in a coalition.
    This function is *not* a generic third-party SHAP implementation.
    """

    if abs(rho) >= 1:
        raise ValueError("|rho| must be below one")
    if game not in {"interventional", "observational"}:
        raise ValueError("unknown attribution game")
    v0 = 0.0
    v1 = x1
    v2 = rho * x2 if game == "observational" else 0.0
    v12 = x1
    return (0.5 * (v1 - v0 + v12 - v2),
            0.5 * (v2 - v0 + v12 - v1))


def shapley_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Dependence gives the unused feature observational Shapley credit."""

    rows = []
    n = 250 if quick else 1100
    for rho in (0.0, 0.3, 0.6, 0.9):
        x = gaussian_pair(rng, n, rho)
        obs = np.array([exact_two_feature_shapley(a, b, rho, "observational")
                        for a, b in x])
        inter = np.array([exact_two_feature_shapley(a, b, rho, "interventional")
                          for a, b in x])
        condition = f"{rho:.1f}"
        rows.extend((
            observe(rep, condition, "observational_abs_phi2",
                    float(np.mean(abs(obs[:, 1]))), rho / math.sqrt(2 * math.pi)),
            observe(rep, condition, "interventional_abs_phi2",
                    float(np.mean(abs(inter[:, 1]))), 0.0),
            observe(rep, condition, "observational_efficiency_error",
                    float(np.max(abs(obs.sum(axis=1) - x[:, 0]))), 0.0),
        ))
    return rows


def interaction_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Pure XOR interaction is predictive although marginal correlations vanish."""

    ntr, nte = (300, 450) if quick else (850, 1200)
    x = rng.choice(np.array([-1.0, 1.0]), size=(ntr + nte, 2))
    y = x[:, 0] * x[:, 1] + rng.normal(scale=0.4, size=len(x))
    model = DecisionTreeRegressor(
        max_depth=2, min_samples_leaf=5,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).fit(x[:ntr], y[:ntr])
    return [
        observe(rep, "xor", "abs_corr_x1", abs(float(np.corrcoef(x[ntr:, 0], y[ntr:])[0, 1])), 0.0),
        observe(rep, "xor", "abs_corr_x2", abs(float(np.corrcoef(x[ntr:, 1], y[ntr:])[0, 1])), 0.0),
        observe(rep, "xor", "permutation_delta_x1",
                permutation_delta(model, x[ntr:], y[ntr:], 0, rng)),
        observe(rep, "xor", "permutation_delta_x2",
                permutation_delta(model, x[ntr:], y[ntr:], 1, rng)),
        observe(rep, "xor", "test_r2", float(r2_score(y[ntr:], model.predict(x[ntr:])))),
    ]


def leakage_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Offline outcome leakage ceases to exist at genuine deployment time."""

    ntr, nte = (200, 350) if quick else (500, 1000)
    x = rng.normal(size=ntr + nte)
    y = 1.8 * x + rng.normal(size=len(x))
    leak = y + rng.normal(scale=0.1, size=len(x))
    model = Ridge(alpha=1.0).fit(np.column_stack((x[:ntr], leak[:ntr])), y[:ntr])
    clean = Ridge(alpha=1.0).fit(x[:ntr, None], y[:ntr])
    offline_x = np.column_stack((x[ntr:], leak[ntr:]))
    deployed_x = np.column_stack((x[ntr:], rng.normal(scale=2.1, size=nte)))
    scores = {
        "leak_offline": r2_score(y[ntr:], model.predict(offline_x)),
        "leak_deployed": r2_score(y[ntr:], model.predict(deployed_x)),
        "clean_offline": r2_score(y[ntr:], clean.predict(x[ntr:, None])),
        "clean_deployed": r2_score(y[ntr:], clean.predict(x[ntr:, None])),
    }
    return [observe(rep, scenario, "r2", float(score)) for scenario, score in scores.items()]


def shift_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Same oracle coefficients, different evaluation variances and rankings."""

    n = 300 if quick else 1100
    z = rng.normal(size=(n, 2))
    eps = rng.normal(scale=0.7, size=n)
    rows = []
    for label, s1, s2 in (("low_x1", 0.4, 2.0), ("balanced", 1.0, 1.0),
                          ("high_x1", 2.0, 0.25)):
        x = z * [s1, s2]
        y = 1.5 * x[:, 0] + 0.7 * x[:, 1] + eps
        baseline = mse(y, 1.5 * x[:, 0] + 0.7 * x[:, 1])
        for j, beta, scale in ((0, 1.5, s1), (1, 0.7, s2)):
            xp = x.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            delta = mse(y, 1.5 * xp[:, 0] + 0.7 * xp[:, 1]) - baseline
            rows.append(observe(rep, label, f"x{j+1}_delta",
                                delta, 2 * (beta * scale)**2))
    return rows


def subgroup_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Minority-relevant feature can have small globally averaged reliance."""

    n = 900 if quick else 2500
    rows = []
    for frac in (0.05, 0.1, 0.2, 0.5):
        # A fixed-size subgroup prevents empty minority samples at small frac.
        g = np.zeros(n, dtype=bool)
        g[:round(n * frac)] = True
        rng.shuffle(g)
        x = rng.normal(size=(n, 2))
        f = np.where(g, 2.0 * x[:, 1], 2.0 * x[:, 0])
        y = f + rng.normal(scale=0.7, size=n)
        base = mse(y, f)
        x1, x2 = x[:, 0].copy(), x[:, 1].copy()
        global1 = mse(y, np.where(g, 2.0 * x2, 2.0 * rng.permutation(x1))) - base
        global2 = mse(y, np.where(g, 2.0 * rng.permutation(x2), 2.0 * x1)) - base
        minority_base = mse(y[g], f[g])
        minority_delta = mse(y[g], 2.0 * rng.permutation(x2[g])) - minority_base
        condition = f"{frac:.2f}"
        rows.extend((
            observe(rep, condition, "global_x1_delta", global1, 8.0 * (1 - frac)),
            observe(rep, condition, "global_x2_delta", global2, 8.0 * frac),
            observe(rep, condition, "minority_x2_delta", minority_delta, 8.0),
        ))
    return rows


def selection_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Choosing maximal holdout noise importance biases its reported score."""

    ntr, nsel, nverify = (100, 150, 300) if quick else (180, 250, 800)
    rows = []
    for p in (10, 40, 80):
        xtr, xs, xv = (rng.normal(size=(n, p))
                       for n in (ntr, nsel, nverify))
        ytr, ys, yv = (rng.normal(size=n) for n in (ntr, nsel, nverify))
        model = Ridge(alpha=20.0).fit(xtr, ytr)
        selection = [permutation_delta(model, xs, ys, j, rng) for j in range(p)]
        best = int(np.argmax(selection))
        independent = permutation_delta(model, xv, yv, best, rng)
        condition = str(p)
        rows.extend((
            observe(rep, condition, "selected_holdout_delta", max(selection)),
            observe(rep, condition, "independent_delta", independent, 0.0),
            observe(rep, condition, "mean_candidate_delta", float(np.mean(selection)), 0.0),
        ))
    return rows


STUDIES: dict[str, Callable[[np.random.Generator, int, bool], list[Row]]] = {
    "conditional": conditional_study,
    "proxies": proxy_study,
    "impurity": impurity_study,
    "shapley": shapley_study,
    "interaction": interaction_study,
    "leakage": leakage_study,
    "shift": shift_study,
    "subgroup": subgroup_study,
    "selection": selection_study,
}


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write one deterministic UTF-8, newline-normalized evidence table."""

    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("inconsistent output schema")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def lookup(summary: list[Row], condition: str, metric: str, key: str = "mean") -> float:
    """Extract a unique numeric summary field for plotting or audit."""

    candidates = [row for row in summary
                  if row["condition"] == condition and row["metric"] == metric]
    if len(candidates) != 1:
        raise KeyError((condition, metric))
    value = candidates[0][key]
    if value is None:
        raise ValueError("requested field is unavailable")
    return float(value)


def line(ax: plt.Axes, summary: list[Row], conditions: list[str],
         metric: str, label: str, color: str, x: list[float] | None = None,
         marker: str = "o") -> None:
    """Draw MC mean and 95% normal interval against ordered conditions."""

    xx = x if x is not None else list(range(len(conditions)))
    yy = [lookup(summary, c, metric) for c in conditions]
    ee = [1.96 * lookup(summary, c, metric, "se") for c in conditions]
    ax.errorbar(xx, yy, yerr=ee, label=label, color=color,
                marker=marker, linewidth=2.2, capsize=3)


def plot_study(name: str, summary: list[Row], path: Path) -> None:
    """Render the original publication figure for one evidence family."""

    plt.rcParams.update({
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "font.size": 11, "axes.titlesize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 170,
    })
    fig, ax = plt.subplots(figsize=(9.2, 5.5), layout="constrained")
    navy, orange, teal = "#153856", "#d66a35", "#008b87"
    if name == "conditional":
        cond = ["0.00", "0.30", "0.60", "0.90", "0.98"]
        xx = [float(c) for c in cond]
        line(ax, summary, cond, "marginal_delta", "Marginal permutation", navy, xx)
        line(ax, summary, cond, "conditional_delta", "Conditional resampling", orange, xx)
        ax.plot(xx, [4.5] * len(xx), "--", c=navy, alpha=0.55)
        ax.plot(xx, [4.5 * (1 - r*r) for r in xx], "--", c=orange, alpha=0.65)
        ax.set(xlabel="Correlation ρ between X₁ and X₂", ylabel="Increase in MSE",
               title="1  |  The question changes when dependence is respected")
    elif name == "proxies":
        cond = ["0.1", "0.4", "1.0", "2.0"]
        xx = [float(c) for c in cond]
        line(ax, summary, cond, "fixed_permutation_delta", "Fixed model: permute X₁", navy, xx)
        line(ax, summary, cond, "refit_ablation_delta", "Retrain without X₁", orange, xx)
        ax.axhline(0, c="#777777", lw=1)
        ax.set(xlabel="Standard deviation of independent proxy noise",
               ylabel="Increase in held-out MSE",
               title="2  |  Perturbing and retraining are different estimands")
    elif name == "impurity":
        plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.1), layout="constrained")
        cond = ["signal_binary", "noise_binary", "noise_continuous"]
        labels = ["True binary\nsignal", "Unrelated\nbinary", "Unrelated\ncontinuous"]
        for a, metric, ttl in zip(
            axs, ("mdi", "heldout_delta"),
            ("Training impurity reduction", "Held-out permutation ΔMSE"),
        ):
            vals = [lookup(summary, c, metric) for c in cond]
            errs = [1.96 * lookup(summary, c, metric, "se") for c in cond]
            a.bar(labels, vals, color=[teal, navy, orange], yerr=errs, capsize=3)
            a.axhline(0, c="#777777", lw=1)
            a.set_title(ttl)
        fig.suptitle("3  |  Split opportunity and held-out reliance disagree", fontsize=15)
    elif name == "shapley":
        cond = ["0.0", "0.3", "0.6", "0.9"]
        xx = [float(c) for c in cond]
        line(ax, summary, cond, "observational_abs_phi2", "Conditional-value game", navy, xx)
        line(ax, summary, cond, "interventional_abs_phi2", "Marginal-value game", orange, xx)
        ax.plot(xx, [r / math.sqrt(2*math.pi) for r in xx],
                "--", c=navy, alpha=.5, label="Exact E|φ₂|")
        ax.set(xlabel="Feature correlation ρ", ylabel="Mean absolute attribution to unused X₂",
               title="4  |  An unused feature can receive conditional-game credit")
    elif name == "interaction":
        cond = "xor"
        metrics = ["abs_corr_x1", "abs_corr_x2",
                   "permutation_delta_x1", "permutation_delta_x2", "test_r2"]
        vals = [lookup(summary, cond, m) for m in metrics]
        errs = [1.96 * lookup(summary, cond, m, "se") for m in metrics]
        ax.bar(["|corr(X₁,Y)|", "|corr(X₂,Y)|", "ΔMSE X₁", "ΔMSE X₂", "Test R²"],
               vals, yerr=errs, color=[orange, orange, navy, navy, teal], capsize=3)
        ax.set(ylabel="Magnitude (metric indicated below each bar)",
               title="5  |  Near-zero univariate association can hide an interaction")
    elif name == "leakage":
        cond = ["leak_offline", "leak_deployed", "clean_offline", "clean_deployed"]
        vals = [lookup(summary, c, "r2") for c in cond]
        errs = [1.96 * lookup(summary, c, "r2", "se") for c in cond]
        ax.bar(["Leaky\nretrospective", "Leaky\ndeployed",
                "Clean\nretrospective", "Clean\ndeployed"],
               vals, yerr=errs, color=[navy, orange, teal, teal], capsize=3)
        ax.axhline(0, c="#777777", lw=1)
        ax.set(ylabel="Out-of-sample R²",
               title="6  |  An unavailable variable dominates only the offline story")
    elif name == "shift":
        cond = ["low_x1", "balanced", "high_x1"]
        line(ax, summary, cond, "x1_delta", "X₁ importance; coefficient 1.5", navy)
        line(ax, summary, cond, "x2_delta", "X₂ importance; coefficient 0.7", orange)
        ax.set_xticks(range(3), ["Low Var(X₁)\nhigh Var(X₂)",
                                 "Balanced\nvariance", "High Var(X₁)\nlow Var(X₂)"])
        ax.set(ylabel="Marginal permutation ΔMSE",
               title="7  |  Importance ranking flips when the evaluation mix changes")
    elif name == "subgroup":
        cond = ["0.05", "0.10", "0.20", "0.50"]
        xx = [float(c) for c in cond]
        line(ax, summary, cond, "global_x1_delta", "Global X₁", navy, xx)
        line(ax, summary, cond, "global_x2_delta", "Global X₂", orange, xx)
        line(ax, summary, cond, "minority_x2_delta", "X₂ within minority", teal, xx)
        ax.set(xlabel="Fraction in subgroup needing X₂", ylabel="Permutation ΔMSE",
               title="8  |  Global averaging can hide subgroup-critical reliance")
    elif name == "selection":
        cond = ["10", "40", "80"]
        xx = [int(c) for c in cond]
        line(ax, summary, cond, "selected_holdout_delta",
             "Maximum on selection holdout", orange, xx)
        line(ax, summary, cond, "independent_delta",
             "Same winner on independent holdout", navy, xx)
        ax.axhline(0, c="#777777", lw=1)
        ax.set(xlabel="Number of unrelated candidate features", ylabel="Permutation ΔMSE",
               title="9  |  Selecting a winner on the evaluation data is optimistic")
    else:
        plt.close(fig)
        raise KeyError(name)
    for a in fig.axes:
        a.grid(axis="y", color="#e5e9ed", linewidth=.7)
        a.set_axisbelow(True)
        if len(fig.axes) == 1 and name not in {"impurity", "interaction", "leakage"}:
            a.legend(frameon=False, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(design: Design = Design()) -> dict[str, list[Row]]:
    """Produce all 18 tables and nine figures, using independent seed streams."""

    if not isinstance(design.root, Path):
        raise TypeError("root must be a pathlib.Path")
    if design.seed < 0 or not isinstance(design.seed, int):
        raise ValueError("seed must be a nonnegative integer")
    suffix = "_quick" if design.quick else ""
    all_summaries = {}
    for index, name in enumerate(EXPERIMENTS):
        rows = []
        for rep in range(3 if design.quick else REPETITIONS[name]):
            rows.extend(STUDIES[name](stream(design.seed, name, rep), rep, design.quick))
        summary = summarize(rows)
        write_csv(design.root / "results" / f"{name}_replicates{suffix}.csv", rows)
        write_csv(design.root / "results" / f"{name}_summary{suffix}.csv", summary)
        plot_study(name, summary, design.root / "figures" / f"{FIGURES[index]}{suffix}.png")
        all_summaries[name] = summary
    return all_summaries


def main() -> None:
    """CLI entrypoint for regenerating canonical or isolated quick artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run three replicates")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = run(Design(root=args.output, quick=args.quick, seed=args.seed))
    for name in EXPERIMENTS:
        print(f"{name}: {len(summary[name])} condition-metric combinations")


if __name__ == "__main__":
    main()
