"""Nine reproducible studies of cross-validation, leakage, and adaptive evaluation.

All data and outcomes are synthetic. Study-level RNG streams are independent,
archival CSVs contain every outer repetition, and quick runs never overwrite
the canonical publication evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
SEED = 20261102
EXPERIMENTS = (
    "iid", "search", "nested", "preprocess", "temporal",
    "lookahead", "groups", "uncertainty", "adaptive",
)
FIGURES = (
    "fig1_iid_validation_baseline",
    "fig2_model_search_optimism",
    "fig3_nested_cross_validation",
    "fig4_supervised_preprocessing_leak",
    "fig5_temporal_validation_drift",
    "fig6_lookahead_feature_leakage",
    "fig7_hierarchical_group_validation",
    "fig8_fold_standard_error",
    "fig9_adaptive_holdout_reconstruction",
)
REPETITIONS = {
    "iid": 30, "search": 36, "nested": 20,
    "preprocess": 25, "temporal": 30, "lookahead": 25,
    "groups": 25, "uncertainty": 70, "adaptive": 30,
}
CSV_FIELDS = ("replicate", "condition", "metric", "value", "theory")
SUMMARY_FIELDS = ("condition", "metric", "n", "mean", "sd", "se",
                  "lower95", "upper95", "theory")

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Row: TypeAlias = dict[str, int | float | str | None]


@dataclass(frozen=True)
class Design:
    """Settings for one isolated archive-generation run."""

    root: Path = ROOT
    seed: int = SEED
    quick: bool = False


def stream(seed: int, study: str, replicate: int) -> np.random.Generator:
    """Stable independent stream for a named experiment and repetition."""

    if not isinstance(seed, int) or seed < 0 or study not in EXPERIMENTS:
        raise ValueError("invalid seed or study key")
    if not isinstance(replicate, int) or replicate < 0:
        raise ValueError("replicate must be a nonnegative integer")
    key = zlib.crc32(study.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([seed, key, replicate]))


def observation(replicate: int, condition: str, metric: str,
                value: float, theory: float | None = None) -> Row:
    """Validate one finite measured observation with optional theory."""

    number = float(value)
    if (replicate < 0 or not condition or not metric or not math.isfinite(number)
            or theory is not None and not math.isfinite(float(theory))):
        raise ValueError("invalid observation")
    return {"replicate": replicate, "condition": condition, "metric": metric,
            "value": number, "theory": theory}


def summarize(rows: list[Row]) -> list[Row]:
    """Summarize independent outer repetitions, guarding all expected keys."""

    if not rows:
        raise ValueError("cannot summarize empty observations")
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"]), str(row["metric"])].append(row)
    output: list[Row] = []
    for condition, metric in sorted(groups):
        group = groups[(condition, metric)]
        indices = [int(r["replicate"]) for r in group]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate study/condition/metric/replicate")
        measured = np.asarray([float(r["value"]) for r in group], dtype=float)
        if not np.isfinite(measured).all():
            raise ValueError("nonfinite archive values")
        theories = {None if r["theory"] is None else float(r["theory"]) for r in group}
        if len(theories) > 1:
            raise ValueError("inconsistent theoretical references")
        n = len(measured)
        mean = float(np.mean(measured))
        sd = float(np.std(measured, ddof=1)) if n > 1 else 0.
        se = sd / math.sqrt(n)
        output.append({"condition": condition, "metric": metric, "n": n,
                       "mean": mean, "sd": sd, "se": se,
                       "lower95": mean-1.96*se, "upper95": mean+1.96*se,
                       "theory": next(iter(theories))})
    return output


def lookup(rows: list[Row], condition: str, metric: str,
           field: str = "mean") -> float:
    """Access a unique numeric summary cell by its stable explicit keys."""

    matches = [r for r in rows if r["condition"] == condition and r["metric"] == metric]
    if len(matches) != 1 or matches[0].get(field) is None:
        raise KeyError((condition, metric, field))
    return float(matches[0][field])


def write_csv(path: Path, rows: list[Row], fields: tuple[str, ...]) -> None:
    """Write sorted, quoted-as-needed, UTF-8 CSV evidence with exact columns."""

    if not rows or any(set(r) != set(fields) for r in rows):
        raise ValueError("missing observations or schema mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def logistic_sample(rng: np.random.Generator, n: int,
                    dimensions: int = 2) -> tuple[FloatArray, IntArray]:
    """Sparse logistic population with independent standard-normal features."""

    if n < 20 or dimensions < 2:
        raise ValueError("invalid population dimensions")
    x = rng.normal(size=(n, dimensions))
    p = expit(1.20*x[:, 0]-.65*x[:, 1]+.2)
    y = (rng.random(n) < p).astype(np.int64)
    return x, y


def logistic_model(c: float = 1.) -> LogisticRegression:
    """A fresh, deterministic regularized classifier for each fitting step."""

    if not math.isfinite(c) or c <= 0:
        raise ValueError("regularization parameter must be finite and positive")
    return LogisticRegression(C=c, solver="lbfgs", max_iter=300)


def brier(y: IntArray, probability: FloatArray) -> float:
    """Proper binary probability score, smaller being better."""

    p = np.asarray(probability, dtype=float)
    if (p.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError("invalid predictive probabilities")
    return float(brier_score_loss(y, p))


def iid_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """A reassuring iid reference: CV, fresh test, and resubstitution."""

    rows: list[Row] = []
    for n in (120, 400, 900):
        x, y = logistic_sample(rng, n if not quick else max(100, n//3))
        tx, ty = logistic_sample(rng, 3600 if not quick else 900)
        model = logistic_model(5.)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
        cv_loss = -float(np.mean(cross_val_score(
            model, x, y, cv=cv, scoring="neg_brier_score", n_jobs=1,
        )))
        model.fit(x, y)
        test_loss = brier(ty, model.predict_proba(tx)[:, 1])
        train_loss = brier(y, model.predict_proba(x)[:, 1])
        cond = str(n)
        rows.extend((
            observation(rep, cond, "cv_brier", cv_loss),
            observation(rep, cond, "fresh_test_brier", test_loss),
            observation(rep, cond, "training_brier", train_loss),
        ))
    return rows


def search_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Select among unrelated noise predictors using one reused validation set."""

    n_val = 220 if not quick else 90
    n_test = 2800 if not quick else 700
    y_val = (rng.random(n_val) < .5).astype(np.int64)
    y_test = (rng.random(n_test) < .5).astype(np.int64)
    # Scores have no information about the latent bits. All candidates have
    # the same expected generalization Brier; selection exploits finite noise.
    scores = np.clip(.5+.12*rng.normal(size=(500, n_val)), .01, .99)
    losses = np.mean((scores-y_val[None, :])**2, axis=1)
    rows: list[Row] = []
    for count in (1, 5, 20, 100, 500):
        chosen = int(np.argmin(losses[:count]))
        selected_loss = float(losses[chosen])
        # Candidate identity is irrelevant under this deliberately null model;
        # a fresh independent prediction vector is a legitimate test draw.
        fresh_score = np.clip(.5+.12*rng.normal(size=n_test), .01, .99)
        independent_loss = brier(y_test, fresh_score)
        cond = str(count)
        rows.extend((
            observation(rep, cond, "selected_validation_brier", selected_loss),
            observation(rep, cond, "independent_test_brier", independent_loss),
            observation(rep, cond, "selection_optimism",
                        independent_loss-selected_loss),
        ))
    return rows


def nested_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Tune on inner folds, evaluate the entire selection process on outer folds."""

    n = 175 if not quick else 130
    x, y = logistic_sample(rng, n, dimensions=16)
    tx, ty = logistic_sample(rng, 2800 if not quick else 750, dimensions=16)
    params = {"C": [.001, .01, .1, 1., 10., 100.]}
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=rep+5)
    search = GridSearchCV(
        logistic_model(), param_grid=params, cv=inner,
        scoring="neg_brier_score", refit=True, n_jobs=1,
    )
    search.fit(x, y)
    selected_inner = -float(search.best_score_)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=rep+503)
    nested_brier = -float(np.mean(cross_val_score(
        GridSearchCV(logistic_model(), param_grid=params, cv=3,
                     scoring="neg_brier_score", n_jobs=1),
        x, y, scoring="neg_brier_score", cv=outer, n_jobs=1,
    )))
    fresh_brier = brier(ty, search.predict_proba(tx)[:, 1])
    return [
        observation(rep, "tuned_model", "selected_inner_cv_brier", selected_inner),
        observation(rep, "tuned_model", "nested_outer_cv_brier", nested_brier),
        observation(rep, "tuned_model", "fresh_test_brier", fresh_brier),
    ]


def preprocess_study(rng: np.random.Generator, rep: int,
                     quick: bool) -> list[Row]:
    """Feature selection before folding leaks test-fold outcomes under a null."""

    rows: list[Row] = []
    n = 230 if not quick else 150
    test_n = 2400 if not quick else 650
    y = (rng.random(n) < .5).astype(np.int64)
    ty = (rng.random(test_n) < .5).astype(np.int64)
    for p in (30, 150, 600):
        x, tx = rng.normal(size=(n, p)), rng.normal(size=(test_n, p))
        folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=rep)
        # Deliberately wrong: use every outcome, including held-out folds,
        # to determine which variables enter cross-validation.
        global_selector = SelectKBest(score_func=f_classif, k=10).fit(x, y)
        globally_selected = global_selector.transform(x)
        leaky_brier = -float(np.mean(cross_val_score(
            logistic_model(.2), globally_selected, y, cv=folds,
            scoring="neg_brier_score", n_jobs=1,
        )))
        # Correct: every outer training fold selects its own variables.
        pipeline = Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=10)),
            ("classifier", logistic_model(.2)),
        ])
        honest_brier = -float(np.mean(cross_val_score(
            pipeline, x, y, cv=folds, scoring="neg_brier_score", n_jobs=1,
        )))
        fitted = pipeline.fit(x, y)
        fresh_brier = brier(ty, fitted.predict_proba(tx)[:, 1])
        rows.extend((
            observation(rep, str(p), "leaky_cv_brier", leaky_brier),
            observation(rep, str(p), "fold_safe_cv_brier", honest_brier),
            observation(rep, str(p), "fresh_test_brier", fresh_brier),
        ))
    return rows


def temporal_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Random validation estimates yesterday's mixture, not tomorrow's regime."""

    n = 1400 if not quick else 650
    time = np.linspace(0., 1., n)
    x = rng.normal(size=(n, 1))
    cutoff = int(.72*n)
    rows: list[Row] = []
    for drift in (0., 1.5, 3.):
        beta = 1.7-drift*time
        probability = expit(beta*x[:, 0])
        y = (rng.random(n) < probability).astype(np.int64)
        history_x, history_y = x[:cutoff], y[:cutoff]
        future_x, future_y = x[cutoff:], y[cutoff:]
        random_cv = -float(np.mean(cross_val_score(
            logistic_model(), history_x, history_y,
            cv=StratifiedKFold(5, shuffle=True, random_state=rep),
            scoring="neg_brier_score", n_jobs=1,
        )))
        forward_cv = -float(np.mean(cross_val_score(
            logistic_model(), history_x, history_y,
            cv=TimeSeriesSplit(n_splits=4),
            scoring="neg_brier_score", n_jobs=1,
        )))
        fitted = logistic_model().fit(history_x, history_y)
        future_brier = brier(future_y, fitted.predict_proba(future_x)[:, 1])
        condition = f"{drift:.1f}"
        rows.extend((
            observation(rep, condition, "random_history_cv_brier", random_cv),
            observation(rep, condition, "forward_history_cv_brier", forward_cv),
            observation(rep, condition, "future_period_brier", future_brier),
        ))
    return rows


def ar1_path(rng: np.random.Generator, n: int,
             coefficient: float = .90) -> FloatArray:
    """Simulate a stationary unit-variance AR(1) process from its invariant law."""

    if n < 20 or not 0 < coefficient < 1:
        raise ValueError("invalid AR(1) settings")
    x = np.empty(n, dtype=float)
    x[0] = rng.normal()
    residual = rng.normal(scale=math.sqrt(1-coefficient**2), size=n-1)
    for i in range(1, n):
        x[i] = coefficient*x[i-1]+residual[i-1]
    return x


def lookahead_study(rng: np.random.Generator, rep: int,
                    quick: bool) -> list[Row]:
    """A chronological split cannot cure future features already in the table."""

    n = 2100 if not quick else 900
    path = ar1_path(rng, n+15)
    rows: list[Row] = []
    for horizon in (1, 3, 7, 14):
        t = np.arange(1, n)
        target_probability = expit(1.8*path[t+horizon])
        y = (rng.random(len(t)) < target_probability).astype(np.int64)
        causal = np.column_stack((path[t], path[t-1]))
        # Invalid in a real forecast issued at time t: the next feature
        # accesses the future value defining the event horizon.
        leaked = np.column_stack((causal, path[t+horizon]))
        cutoff = int(.72*len(t))
        clean_model = logistic_model().fit(causal[:cutoff], y[:cutoff])
        leaky_model = logistic_model().fit(leaked[:cutoff], y[:cutoff])
        clean_loss = brier(y[cutoff:],
                           clean_model.predict_proba(causal[cutoff:])[:, 1])
        leaky_loss = brier(y[cutoff:],
                           leaky_model.predict_proba(leaked[cutoff:])[:, 1])
        rows.extend((
            observation(rep, str(horizon), "causal_feature_brier", clean_loss),
            observation(rep, str(horizon), "lookahead_feature_brier", leaky_loss),
            observation(rep, str(horizon), "optimism_from_lookahead",
                        clean_loss-leaky_loss),
        ))
    return rows


def groups_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Row CV recognizes site-specific baselines unavailable at a new site."""

    sites = 30 if not quick else 15
    each = 34 if not quick else 28
    groups = np.repeat(np.arange(sites), each)
    n = len(groups)
    x = rng.normal(size=n)
    # Known site categories are encoded without using outcomes or statistics.
    # Group CV removes a site's complete outcome history from the training fold.
    encoded = np.column_stack((x, np.eye(sites)[groups]))
    rows: list[Row] = []
    for sd in (0., .8, 1.8):
        intercept = rng.normal(scale=sd, size=sites)
        p = expit(.35*x+intercept[groups])
        y = (rng.random(n) < p).astype(np.int64)
        row_cv = -float(np.mean(cross_val_score(
            logistic_model(5.), encoded, y, cv=StratifiedKFold(
                n_splits=5, shuffle=True, random_state=rep,
            ), scoring="neg_brier_score", n_jobs=1,
        )))
        site_cv = -float(np.mean(cross_val_score(
            logistic_model(5.), encoded, y,
            cv=GroupKFold(n_splits=5).split(encoded, y, groups),
            scoring="neg_brier_score", n_jobs=1,
        )))
        condition = f"{sd:.1f}"
        rows.extend((
            observation(rep, condition, "random_row_cv_brier", row_cv),
            observation(rep, condition, "unseen_site_cv_brier", site_cv),
            observation(rep, condition, "site_generalization_gap",
                        site_cv-row_cv),
        ))
    return rows


def uncertainty_study(rng: np.random.Generator, rep: int,
                      quick: bool) -> list[Row]:
    """Fold variability is not an independent-replication standard error."""

    x, y = logistic_sample(rng, 320 if not quick else 170)
    tx, ty = logistic_sample(rng, 8500 if not quick else 1500)
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
    scores = -cross_val_score(
        logistic_model(), x, y, scoring="neg_brier_score", cv=cv, n_jobs=1,
    )
    cv_mean = float(np.mean(scores))
    naive_se = float(np.std(scores, ddof=1)/math.sqrt(len(scores)))
    fitted = logistic_model().fit(x, y)
    external_brier = brier(ty, fitted.predict_proba(tx)[:, 1])
    covered = float(abs(cv_mean-external_brier) <= 1.96*naive_se)
    return [
        observation(rep, "5fold", "cv_brier", cv_mean),
        observation(rep, "5fold", "external_brier", external_brier),
        observation(rep, "5fold", "naive_fold_se", naive_se),
        observation(rep, "5fold", "absolute_cv_external_gap",
                    abs(cv_mean-external_brier)),
        observation(rep, "5fold", "naive_interval_covers_external", covered),
    ]


def adaptive_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Exact public accuracy feedback can reveal a fixed binary answer key."""

    n = 120
    heldout = (rng.random(n) < .5).astype(np.int64)
    fresh = (rng.random(1800 if not quick else 600) < .5).astype(np.int64)
    baseline = np.zeros(n, dtype=np.int64)

    def public_accuracy(predicted: IntArray) -> float:
        """Simulated exact-answer feedback; no real leaderboard is queried."""

        return float(np.mean(predicted == heldout))

    base_score = public_accuracy(baseline)
    recovered = baseline.copy()
    rows: list[Row] = []
    previous = 0
    for queries in (0, 20, 60, 120):
        for i in range(previous, queries):
            probe = baseline.copy()
            probe[i] = 1
            delta = public_accuracy(probe)-base_score
            if abs(abs(delta)-1/n) > 1e-10:
                raise AssertionError("exact accuracy oracle did not resolve one label")
            recovered[i] = int(delta > 0)
        # The index-memory rule has no relationship to independent fresh labels.
        fresh_guess = np.resize(recovered, len(fresh))
        condition = str(queries)
        rows.extend((
            observation(rep, condition, "public_holdout_accuracy",
                        public_accuracy(recovered)),
            observation(rep, condition, "fresh_independent_accuracy",
                        float(np.mean(fresh_guess == fresh)), .5),
            observation(rep, condition, "recovered_label_fraction",
                        queries/n, queries/n),
            observation(rep, condition, "public_queries_used", queries+1,
                        queries+1),
        ))
        previous = queries
    return rows


STUDIES: dict[str, Callable[[np.random.Generator, int, bool], list[Row]]] = {
    "iid": iid_study, "search": search_study,
    "nested": nested_study, "preprocess": preprocess_study,
    "temporal": temporal_study, "lookahead": lookahead_study,
    "groups": groups_study, "uncertainty": uncertainty_study,
    "adaptive": adaptive_study,
}


def line(ax: plt.Axes, summary: list[Row], conditions: list[str],
         metric: str, caption: str, color: str,
         x: list[float], style: str = "-") -> None:
    """Plot one archived mean curve with 1.96-MCSE error bars."""

    values = [lookup(summary, c, metric) for c in conditions]
    errors = [1.96*lookup(summary, c, metric, "se") for c in conditions]
    ax.errorbar(x, values, yerr=errors, marker="o", linewidth=2.1,
                capsize=3, linestyle=style, color=color, label=caption)


def plot_study(study: str, summary: list[Row], destination: Path) -> None:
    """Render one publication PNG from that study's canonical summary."""

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 180,
    })
    navy, orange, teal, violet = "#173956", "#d26a36", "#008b85", "#805d9e"
    fig, ax = plt.subplots(figsize=(9.5, 5.3), layout="constrained")

    if study == "iid":
        conditions = ["120", "400", "900"]
        x = [float(c) for c in conditions]
        for metric, caption, color in (
            ("cv_brier", "Stratified 5-fold CV", navy),
            ("fresh_test_brier", "Independent test", teal),
            ("training_brier", "Training resubstitution", orange),
        ):
            line(ax, summary, conditions, metric, caption, color, x)
        ax.set(xlabel="Nominal training sample size", ylabel="Brier loss",
               title="1  |  An iid baseline makes CV and fresh testing comparable")
        ax.legend(frameon=False)
    elif study == "search":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), layout="constrained")
        conditions = ["1", "5", "20", "100", "500"]
        x = [float(c) for c in conditions]
        for metric, caption, color in (
            ("selected_validation_brier", "Best validation score", orange),
            ("independent_test_brier", "Independent test score", navy),
        ):
            line(axes[0], summary, conditions, metric, caption, color, x)
        axes[0].set(xlabel="Number of unrelated candidates", ylabel="Brier loss",
                    xscale="log", title="Selection improves the observed minimum")
        axes[0].legend(frameon=False, fontsize=9)
        line(axes[1], summary, conditions, "selection_optimism",
             "Test minus selected validation", violet, x)
        axes[1].set(xlabel="Number of candidates", ylabel="Optimism in Brier loss",
                    xscale="log", title="The gap is not genuine improvement")
        fig.suptitle("2  |  More independent attempts buy an optimistic minimum",
                     fontsize=14)
    elif study == "nested":
        conditions = ["tuned_model"]
        metrics = [
            ("selected_inner_cv_brier", "Reused inner CV", orange),
            ("nested_outer_cv_brier", "Nested outer CV", teal),
            ("fresh_test_brier", "Fresh test", navy),
        ]
        values: list[float] = []
        errors: list[float] = []
        for i, (metric, caption, color) in enumerate(metrics):
            value = lookup(summary, conditions[0], metric)
            err = 1.96*lookup(summary, conditions[0], metric, "se")
            values.append(value)
            errors.append(err)
            ax.errorbar(i, value, yerr=err, marker="o", markersize=9,
                        capsize=5, color=color, linewidth=2)
        ax.set_xticks(range(3), [p[1] for p in metrics])
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(min(v-e for v, e in zip(values, errors))-.012,
                    max(v+e for v, e in zip(values, errors))+.012)
        ax.grid(axis="y", alpha=.25)
        ax.set(ylabel="Brier loss", title="3  |  Outer evaluation contains the tuning step")
    elif study == "preprocess":
        conditions = ["30", "150", "600"]
        x = [float(c) for c in conditions]
        for metric, caption, color in (
            ("leaky_cv_brier", "Select features before CV", orange),
            ("fold_safe_cv_brier", "Select within each fold", teal),
            ("fresh_test_brier", "Independent test", navy),
        ):
            line(ax, summary, conditions, metric, caption, color, x)
        ax.set(xlabel="Number of independent noise features", ylabel="Brier loss",
               title="4  |  A supervised transform outside the fold leaks answers")
        ax.legend(frameon=False)
    elif study == "temporal":
        conditions = ["0.0", "1.5", "3.0"]
        x = [float(c) for c in conditions]
        for metric, caption, color in (
            ("random_history_cv_brier", "Random CV on history", orange),
            ("forward_history_cv_brier", "Forward CV on history", teal),
            ("future_period_brier", "Later period", navy),
        ):
            line(ax, summary, conditions, metric, caption, color, x)
        ax.set(xlabel="Linear strength of coefficient drift", ylabel="Brier loss",
               title="5  |  The next regime is not a random fold of the past")
        ax.legend(frameon=False)
    elif study == "lookahead":
        conditions = ["1", "3", "7", "14"]
        x = [float(c) for c in conditions]
        for metric, caption, color in (
            ("causal_feature_brier", "Available at forecast time", navy),
            ("lookahead_feature_brier", "Future state included (invalid)", orange),
        ):
            line(ax, summary, conditions, metric, caption, color, x)
        ax.set(xlabel="Forecast horizon (time steps)", ylabel="Future-period Brier loss",
               title="6  |  Even chronological testing cannot fix future features")
        ax.legend(frameon=False)
    elif study == "groups":
        conditions = ["0.0", "0.8", "1.8"]
        x = [float(c) for c in conditions]
        for metric, caption, color in (
            ("random_row_cv_brier", "Rows mixed across sites", orange),
            ("unseen_site_cv_brier", "Entire sites held out", navy),
        ):
            line(ax, summary, conditions, metric, caption, color, x)
        ax.set(xlabel="Site-specific intercept standard deviation",
               ylabel="Brier loss",
               title="7  |  Known-site performance is not new-site performance")
        ax.legend(frameon=False)
    elif study == "uncertainty":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), layout="constrained")
        true_spread = lookup(summary, "5fold", "cv_brier", "sd")
        naive = lookup(summary, "5fold", "naive_fold_se")
        axes[0].bar([0, 1], [true_spread, naive], color=[navy, orange], width=.6)
        axes[0].set_xticks([0, 1], ["Between-dataset SD", "Mean fold-based SE"])
        axes[0].set(ylabel="Brier-loss scale",
                    title="The two uncertainty quantities are different")
        covered = lookup(summary, "5fold", "naive_interval_covers_external")
        axes[1].bar([0], [covered], color=teal, width=.5)
        axes[1].axhline(.95, linestyle="--", color="#666666", label="Nominal 95%")
        axes[1].set_xticks([0], ["Empirical inclusion"])
        axes[1].set(ylim=(0, 1.02), ylabel="Fraction of repetitions",
                    title="Naive interval containing independent test loss")
        axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle("8  |  Five correlated folds are not five new datasets",
                     fontsize=14)
    elif study == "adaptive":
        conditions = ["0", "20", "60", "120"]
        x = [float(c) for c in conditions]
        for metric, caption, color, style in (
            ("public_holdout_accuracy", "Public answer-key accuracy", orange, "-"),
            ("fresh_independent_accuracy", "Independent new labels", navy, "-"),
            ("recovered_label_fraction", "Answer keys decoded", violet, "--"),
        ):
            line(ax, summary, conditions, metric, caption, color, x, style)
        ax.set(xlabel="Single-coordinate score queries after baseline",
               ylabel="Accuracy or recovered fraction", ylim=(0, 1.04),
               title="9  |  Exact reusable feedback can disclose the full key")
        ax.legend(frameon=False, fontsize=9)
    else:
        raise ValueError(f"unknown figure study: {study}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def run(design: Design) -> dict[str, list[Row]]:
    """Generate all nine datasets, nine graphs, and eighteen CSV summaries."""

    if not isinstance(design.root, Path):
        raise TypeError("output root must be pathlib.Path")
    if not isinstance(design.seed, int) or design.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if design.seed != SEED and not design.quick and design.root.resolve() == ROOT:
        raise ValueError("noncanonical full seed requires explicit --output")
    suffix = "_quick" if design.quick else ""
    out: dict[str, list[Row]] = {}
    for study, figure in zip(EXPERIMENTS, FIGURES):
        repetitions = 3 if design.quick else REPETITIONS[study]
        observations: list[Row] = []
        for replicate in range(repetitions):
            observations.extend(STUDIES[study](
                stream(design.seed, study, replicate), replicate, design.quick,
            ))
        summary = summarize(observations)
        write_csv(design.root/"results"/f"{study}_replicates{suffix}.csv",
                  observations, CSV_FIELDS)
        write_csv(design.root/"results"/f"{study}_summary{suffix}.csv",
                  summary, SUMMARY_FIELDS)
        plot_study(study, summary, design.root/"figures"/f"{figure}{suffix}.png")
        out[study] = summary
    return out


def main() -> None:
    """Command line entry point for canonical and safely isolated quick modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="three outer repetitions and suffixed results")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    reports = run(Design(root=args.output, seed=args.seed, quick=args.quick))
    for name in EXPERIMENTS:
        print(f"{name}: {len(reports[name])} summary cells")


if __name__ == "__main__":
    main()


