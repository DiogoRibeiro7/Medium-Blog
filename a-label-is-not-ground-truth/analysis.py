"""Nine seeded label-quality studies, with explicit synthetic latent truth.

Canonical full mode writes nine independent figures and eighteen CSV files.
Quick mode writes suffixed files without overwriting canonical evidence.
Observed labels, latent outcomes, annotator votes, and review-selection flags
are kept conceptually distinct. None of the simulations proves field validity.
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
from scipy.special import expit, logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(__file__).resolve().parent
SEED = 20261102
EXPERIMENTS = (
    "symmetric", "asymmetric", "instance", "annotators", "correlation",
    "selection", "benchmark", "leakage", "policy",
)
FIGURES = (
    "fig1_symmetric_label_noise", "fig2_asymmetric_prevalence",
    "fig3_instance_dependent_noise", "fig4_annotator_aggregation",
    "fig5_correlated_annotator_failures", "fig6_selective_labels",
    "fig7_noisy_test_ranking_reversal", "fig8_entity_leakage",
    "fig9_annotation_policy_drift",
)
REPETITIONS = {
    "symmetric": 30, "asymmetric": 30, "instance": 30,
    "annotators": 24, "correlation": 24, "selection": 36,
    "benchmark": 30, "leakage": 20, "policy": 30,
}
Row = dict[str, str | int | float | None]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Design:
    """Validated execution configuration, for either evidence or a smoke test."""

    root: Path = ROOT
    quick: bool = False
    seed: int = SEED


def binary_labels(y: NDArray[np.integer], minimum: int = 3) -> IntArray:
    """Require a one-dimensional finite 0/1 label vector with enough rows."""

    a = np.asarray(y)
    if (a.ndim != 1 or len(a) < minimum or
            not np.issubdtype(a.dtype, np.integer) or not np.isin(a, [0, 1]).all()):
        raise ValueError("expected a binary integer label vector")
    return a.astype(np.int64)


def probabilities(p: NDArray[np.floating], minimum: int = 3) -> FloatArray:
    """Validate a one-dimensional probability vector including endpoints."""

    a = np.asarray(p, dtype=float)
    if (a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all() or
            np.any((a < 0) | (a > 1))):
        raise ValueError("expected finite probabilities in [0,1]")
    return a


def stream(seed: int, name: str, rep: int) -> np.random.Generator:
    """Independent stable SeedSequence stream keyed by study and replication."""

    if not isinstance(seed, int) or seed < 0 or name not in EXPERIMENTS or rep < 0:
        raise ValueError("invalid seed, study, or replicate")
    return np.random.default_rng(np.random.SeedSequence(
        [seed, zlib.crc32(name.encode("ascii")), rep]))


def observation(rep: int, condition: str, metric: str, value: float,
                theory: float | None = None) -> Row:
    """Construct a validated long-format Monte Carlo result."""

    if rep < 0 or not condition or not metric:
        raise ValueError("invalid observation keys")
    if not math.isfinite(value) or (theory is not None and not math.isfinite(theory)):
        raise ValueError("nonfinite observation or theoretical reference")
    return {"replicate": rep, "condition": condition, "metric": metric,
            "value": float(value), "theory": theory}


def summarize(rows: list[Row]) -> list[Row]:
    """Mean ±1.96 Monte Carlo SE across independent outer replicates."""

    if not rows:
        raise ValueError("cannot summarize zero replications")
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), str(row["metric"])), []).append(row)
    result = []
    for (condition, metric), group in sorted(groups.items()):
        if len({r["theory"] for r in group}) != 1:
            raise ValueError("theoretical reference varies within a result key")
        ids = [int(r["replicate"]) for r in group]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate outer-replication identifier")
        values = np.asarray([r["value"] for r in group], dtype=float)
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        mean = float(np.mean(values))
        mcse = sd/math.sqrt(len(values))
        result.append({"condition": condition, "metric": metric, "n": len(values),
                       "mean": mean, "sd": sd, "se": mcse,
                       "lower95": mean-1.96*mcse, "upper95": mean+1.96*mcse,
                       "theory": group[0]["theory"]})
    return result


def lookup(rows: list[Row], condition: str, metric: str,
           field: str = "mean") -> float:
    """Look up exactly one summary value; fail on missing or ambiguous keys."""

    match = [r for r in rows if r["condition"] == condition and r["metric"] == metric]
    if len(match) != 1 or match[0].get(field) is None:
        raise KeyError((condition, metric, field))
    return float(match[0][field])


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write deterministic UTF-8 CSV evidence without platform-specific EOLs."""

    if not rows:
        raise ValueError("empty CSV")
    fields = list(rows[0])
    if any(set(r) != set(fields) for r in rows):
        raise ValueError("inconsistent CSV fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def logistic_population(rng: np.random.Generator, n: int,
                        strength: float = 1.5) -> tuple[FloatArray, IntArray, FloatArray]:
    """Return two-feature X, latent Bernoulli outcome and conditional P(Y=1|X)."""

    if n < 10 or strength <= 0 or not math.isfinite(strength):
        raise ValueError("invalid population shape or signal strength")
    x = rng.normal(size=(n, 2))
    p = expit(strength*x[:, 0] + .65*x[:, 1] - .35)
    y = (rng.random(n) < p).astype(np.int64)
    return x.astype(float), y, p.astype(float)


def flip_labels(rng: np.random.Generator, y: IntArray,
                rate_0: float, rate_1: float) -> IntArray:
    """Class-conditional label flips: eta_c=P(observed != c | latent=c)."""

    truth = binary_labels(y)
    if not (0 <= rate_0 < 1 and 0 <= rate_1 < 1):
        raise ValueError("flip probabilities must lie in [0,1)")
    rates = np.where(truth == 1, rate_1, rate_0)
    return np.bitwise_xor(truth, (rng.random(len(truth)) < rates).astype(np.int64))


def invert_noisy_prevalence(observed: float, eta0: float, eta1: float) -> float:
    """Identify latent prevalence if class-conditional transition rates are known.

    q=P(tilde Y=1)=eta0+(1-eta0-eta1)*pi. This returns the linear
    unbiased correction without clipping; out-of-range finite-sample values
    are meaningful diagnostics of sampling uncertainty/model incompatibility.
    """

    if (not all(math.isfinite(v) for v in (observed, eta0, eta1)) or
            not 0 <= observed <= 1 or not 0 <= eta0 < 1 or
            not 0 <= eta1 < 1 or eta0+eta1 >= 1):
        raise ValueError("invalid prevalence correction or nonidentifiable channel")
    return (observed-eta0)/(1-eta0-eta1)


def noisy_probabilities(p: FloatArray, eta0: float, eta1: float) -> FloatArray:
    """Map latent posterior probabilities through a class-conditional channel."""

    values = probabilities(p)
    if not 0 <= eta0 < 1 or not 0 <= eta1 < 1:
        raise ValueError("invalid flip channel")
    return np.asarray(eta0+(1-eta0-eta1)*values, dtype=float)


def dawid_skene_binary(votes: NDArray[np.integer], iterations: int = 75,
                       tolerance: float = 1e-8) -> FloatArray:
    """Binary Dawid–Skene EM under conditionally independent annotators.

    Columns are complete annotators; there is no missing-vote mechanism.
    Beta(1,1)-style add-one pseudo-counts regularize confusion rates, and a
    majority-vote initialization chooses one latent label orientation.
    Identifiability of latent truth is *assumed*, not proven by this routine.
    """

    a = np.asarray(votes)
    if (a.ndim != 2 or a.shape[0] < 20 or a.shape[1] < 3 or
            not np.issubdtype(a.dtype, np.integer) or not np.isin(a, [0, 1]).all()):
        raise ValueError("expected an n-by-k 0/1 integer vote matrix, k>=3")
    if iterations < 1 or not 0 < tolerance < 1:
        raise ValueError("invalid EM stopping parameters")
    n, k = a.shape
    q = np.clip(.05+.90*a.mean(axis=1), .02, .98)
    for _ in range(iterations):
        # E step follows the M step estimates of class prior and worker rates.
        prior = float(np.clip(q.mean(), 1e-5, 1-1e-5))
        weights = np.stack((1-q, q), axis=1)
        probs = np.empty((2, k), dtype=float)
        for c in (0, 1):
            # Two pseudo-observations, one positive and one negative,
            # prevent zero likelihoods for nearly perfect workers.
            probs[c] = (weights[:, c] @ a + 1.)/(weights[:, c].sum()+2.)
        logscore = np.empty((n, 2), dtype=float)
        for c in (0, 1):
            logscore[:, c] = (math.log(prior if c else 1-prior) +
                              (a*np.log(probs[c]) +
                               (1-a)*np.log1p(-probs[c])).sum(axis=1))
        revised = np.exp(logscore[:, 1]-logsumexp(logscore, axis=1))
        if np.max(np.abs(revised-q)) < tolerance:
            q = revised
            break
        q = revised
    return probabilities(q)


def weighted_review_prevalence(y: IntArray, reviewed: NDArray[np.bool_],
                               audit: NDArray[np.bool_], audit_probability: float) -> float:
    """Horvitz–Thompson prevalence over reviewed plus randomized audits.

    Reviewed cases have inclusion probability 1; unreviewed cases have
    randomized audit probability a > 0. Unreviewed, unaudited latent labels
    are never read from y in the estimator. This assumes audit randomization.
    """

    labels = binary_labels(y)
    r, a = np.asarray(reviewed), np.asarray(audit)
    if (r.shape != labels.shape or a.shape != labels.shape or
            r.dtype != bool or a.dtype != bool or not 0 < audit_probability <= 1 or
            np.any(r & a)):
        raise ValueError("invalid disjoint review/audit indicators")
    return float((labels[r].sum()+labels[a].sum()/audit_probability)/len(labels))


def symmetric_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Symmetric train noise degrades probabilities, often preserves rank."""

    n = 650 if quick else 2100
    train_x, train_y, _ = logistic_population(rng, n)
    test_x, test_y, _ = logistic_population(rng, 1600 if quick else 4800)
    rows = []
    for eta in (0., .05, .15, .30):
        labels = flip_labels(rng, train_y, eta, eta)
        model = LogisticRegression(C=20., max_iter=300).fit(train_x, labels)
        prob = model.predict_proba(test_x)[:, 1]
        cond = f"{eta:.2f}"
        rows.extend((
            observation(rep, cond, "clean_test_auc", float(roc_auc_score(test_y, prob))),
            observation(rep, cond, "clean_test_brier", float(brier_score_loss(test_y, prob))),
            observation(rep, cond, "empirical_train_flip_rate", float((labels!=train_y).mean()), eta),
        ))
    return rows


def asymmetric_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Observed positive prevalence is a measurement-channel functional."""

    latent_pi = .25
    n = 900 if quick else 4500
    y = (rng.random(n) < latent_pi).astype(np.int64)
    rows = []
    for eta0, eta1 in ((0., 0.), (.05, .10), (.05, .30), (.15, .40)):
        noisy = flip_labels(rng, y, eta0, eta1)
        naive = float(noisy.mean())
        corrected = invert_noisy_prevalence(naive, eta0, eta1)
        cond = f"{eta0:.2f}:{eta1:.2f}"
        reference = eta0+(1-eta0-eta1)*latent_pi
        rows.extend((
            observation(rep, cond, "observed_positive_fraction", naive, reference),
            observation(rep, cond, "oracle_corrected_prevalence", corrected, latent_pi),
            observation(rep, cond, "latent_sample_prevalence", float(y.mean()), latent_pi),
        ))
    return rows


def instance_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Same number of flips; location in feature space changes harm."""

    x, y, p = logistic_population(rng, 1100 if quick else 3100)
    tx, ty, _ = logistic_population(rng, 1700 if quick else 5400)
    nbad = round(.12*len(y))
    difficulty = np.abs(p-.5)
    methods = {
        "uniform": rng.choice(len(y), size=nbad, replace=False),
        "boundary": np.argsort(difficulty)[:nbad],
        "high_margin": np.argsort(difficulty)[-nbad:],
    }
    rows = []
    for name, indices in methods.items():
        corrupted = y.copy()
        corrupted[indices] = 1-corrupted[indices]
        classifier = LogisticRegression(C=20., max_iter=300).fit(x, corrupted)
        prob = classifier.predict_proba(tx)[:, 1]
        rows.extend((
            observation(rep, name, "clean_test_auc", float(roc_auc_score(ty, prob))),
            observation(rep, name, "clean_test_brier", float(brier_score_loss(ty, prob))),
            observation(rep, name, "observed_flip_fraction", float(np.mean(corrupted!=y)), .12),
        ))
    return rows


def worker_votes(rng: np.random.Generator, truth: IntArray,
                 sensitivity: FloatArray, specificity: FloatArray) -> NDArray[np.int64]:
    """Generate conditionally independent votes with worker-specific errors."""

    y = binary_labels(truth)
    sens, spec = np.asarray(sensitivity, dtype=float), np.asarray(specificity, dtype=float)
    if (sens.ndim != 1 or spec.shape != sens.shape or len(sens) < 3 or
            not np.isfinite(sens).all() or not np.isfinite(spec).all() or
            np.any((sens <= 0) | (sens >= 1)) or np.any((spec <= 0) | (spec >= 1))):
        raise ValueError("worker sensitivities/specificities must be proper vectors")
    positive_vote = np.where(y[:, None] == 1, sens[None, :], 1-spec[None, :])
    return (rng.random(positive_vote.shape) < positive_vote).astype(np.int64)


def annotators_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Independent, heterogeneous annotators: majority versus latent EM."""

    n = 400 if quick else 1300
    y = (rng.random(n) < .30).astype(np.int64)
    worker_sets = {
        "3_similar": ([.80, .82, .80], [.85, .85, .83]),
        "5_heterogeneous": ([.95, .82, .70, .62, .90], [.95, .84, .77, .61, .60]),
        "5_weak_majority": ([.95, .62, .62, .62, .62], [.96, .66, .66, .66, .66]),
    }
    rows = []
    for name, (sens, spec) in worker_sets.items():
        votes = worker_votes(rng, y, np.array(sens), np.array(spec))
        majority = (votes.mean(axis=1) > .5).astype(int)
        em_prob = dawid_skene_binary(votes, iterations=60)
        em_predict = (em_prob >= .5).astype(int)
        rows.extend((
            observation(rep, name, "majority_accuracy", float(accuracy_score(y, majority))),
            observation(rep, name, "em_accuracy", float(accuracy_score(y, em_predict))),
            observation(rep, name, "em_brier", float(brier_score_loss(y, em_prob))),
        ))
    return rows


def correlation_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Shared item-level mistakes violate conditional vote independence.

    Marginal per-worker XOR flip rate stays exactly .2 theoretically, while
    the shared fraction moves from 0 to .2. EM is still intentionally fit
    under its false conditional-independence assumption.
    """

    n = 500 if quick else 1800
    truth = (rng.random(n) < .5).astype(np.int64)
    rows = []
    for common in (0., .05, .10, .20):
        independent = (.20-common)/(1-2*common)
        shared = (rng.random(n) < common).astype(np.int64)
        residual = (rng.random((n, 5)) < independent).astype(np.int64)
        votes = np.bitwise_xor(truth[:, None], np.bitwise_xor(shared[:, None], residual))
        majority = (votes.mean(axis=1) > .5).astype(int)
        em_prob = dawid_skene_binary(votes, iterations=60)
        em_label = (em_prob >= .5).astype(int)
        wrong = em_label != truth
        rows.extend((
            observation(rep, f"{common:.2f}", "majority_error_rate",
                        float(np.mean(majority!=truth))),
            observation(rep, f"{common:.2f}", "em_error_rate", float(wrong.mean())),
            observation(rep, f"{common:.2f}", "confident_wrong_fraction",
                        float(np.mean(wrong & (np.maximum(em_prob, 1-em_prob) >= .9)))),
            observation(rep, f"{common:.2f}", "worker_flip_fraction",
                        float(np.mean(votes != truth[:, None])), .2),
        ))
    return rows


def selection_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Review-selected outcomes are not an unbiased sample of prevalence."""

    n = 2500 if quick else 9500
    score = rng.normal(size=n)
    latent = (rng.random(n) < expit(1.4*score-1.3)).astype(np.int64)
    truth = float(latent.mean())
    rows = []
    audit_p = .10
    for share in (.10, .20, .40):
        threshold = float(np.quantile(score, 1-share))
        reviewed = score > threshold
        audit = (~reviewed) & (rng.random(n) < audit_p)
        observed_naive = float(latent[reviewed].mean())
        repaired = weighted_review_prevalence(latent, reviewed, audit, audit_p)
        rows.extend((
            observation(rep, f"{share:.2f}", "review_only_prevalence", observed_naive),
            observation(rep, f"{share:.2f}", "review_and_random_audit_ht", repaired),
            observation(rep, f"{share:.2f}", "latent_population_prevalence", truth),
            observation(rep, f"{share:.2f}", "audit_fraction", float(audit.mean())),
        ))
    return rows


def benchmark_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Even sparse B-favoring benchmark labels can reverse model ordering.

    Model A is better by design on the true population. For selected
    minority cases, the observed test label copies model B's prediction.
    This is deliberate model-correlated label bias, not independent noise.
    """

    n = 3500 if quick else 10500
    minority = rng.random(n) < .20
    truth = (rng.random(n) < .5).astype(np.int64)
    a = np.bitwise_xor(truth, (rng.random(n) < .10).astype(np.int64))
    b_error = np.where(minority, .30, .07)
    b = np.bitwise_xor(truth, (rng.random(n) < b_error).astype(np.int64))
    rows = []
    for copy_fraction in (0., .10, .25, .50, .75):
        observed = truth.copy()
        replace = minority & (rng.random(n) < copy_fraction)
        observed[replace] = b[replace]
        cond = f"{copy_fraction:.2f}"
        rows.extend((
            observation(rep, cond, "true_accuracy_A", float(np.mean(a==truth))),
            observation(rep, cond, "true_accuracy_B", float(np.mean(b==truth))),
            observation(rep, cond, "observed_accuracy_A", float(np.mean(a==observed))),
            observation(rep, cond, "observed_accuracy_B", float(np.mean(b==observed))),
            observation(rep, cond, "label_error_fraction", float(np.mean(observed!=truth))),
        ))
    return rows


def leakage_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Repeated entity fingerprints make row splitting optimistic."""

    entities = 130 if quick else 280
    repeats, dimensions = 4, 14
    fingerprint = rng.normal(size=(entities, dimensions))
    labels = (rng.random(entities) < .5).astype(np.int64)
    x = np.repeat(fingerprint, repeats, axis=0)
    x += rng.normal(scale=.02, size=x.shape)
    y = np.repeat(labels, repeats)
    groups = np.repeat(np.arange(entities), repeats)
    row_train, row_test = train_test_split(np.arange(len(y)), test_size=.3,
                                          random_state=rep, stratify=y)
    split = GroupShuffleSplit(n_splits=1, test_size=.3, random_state=rep)
    group_train, group_test = next(split.split(x, y, groups))
    rows = []
    for mode, tr, te in (("row_split", row_train, row_test),
                         ("entity_holdout", group_train, group_test)):
        forest = RandomForestClassifier(n_estimators=40 if quick else 75,
                                        min_samples_leaf=1, max_features="sqrt",
                                        random_state=rep, n_jobs=1).fit(x[tr], y[tr])
        accuracy = float(accuracy_score(y[te], forest.predict(x[te])))
        overlap = len(set(groups[tr]) & set(groups[te]))/len(set(groups[te]))
        rows.append(observation(rep, mode, "test_accuracy", accuracy))
        rows.append(observation(rep, mode, "test_entity_overlap", overlap,
                                0. if mode == "entity_holdout" else None))
    return rows


def policy_study(rng: np.random.Generator, rep: int, quick: bool) -> list[Row]:
    """Changed labeling policy creates metric drift with fixed true outcomes."""

    x, truth, p = logistic_population(rng, 2600 if quick else 9000)
    predicted = (p >= .5).astype(np.int64)
    rows = []
    for name, eta0, eta1 in (
        ("baseline", .02, .04), ("lenient", .25, .04), ("strict", .02, .30)
    ):
        observed = flip_labels(rng, truth, eta0, eta1)
        rows.extend((
            observation(rep, name, "latent_outcome_prevalence", float(truth.mean())),
            observation(rep, name, "recorded_positive_fraction", float(observed.mean())),
            observation(rep, name, "true_model_accuracy", float(np.mean(predicted==truth))),
            observation(rep, name, "recorded_label_accuracy", float(np.mean(predicted==observed))),
        ))
    return rows


STUDIES: dict[str, Callable[[np.random.Generator, int, bool], list[Row]]] = {
    "symmetric": symmetric_study, "asymmetric": asymmetric_study,
    "instance": instance_study, "annotators": annotators_study,
    "correlation": correlation_study, "selection": selection_study,
    "benchmark": benchmark_study, "leakage": leakage_study,
    "policy": policy_study,
}


def series(ax: plt.Axes, summary: list[Row], cond: list[str], metric: str,
           label: str, color: str, x: list[float] | None = None,
           linestyle: str = "-") -> None:
    """One curve with independent-outer-replication ±1.96-MCSE error bars."""

    xx = x if x is not None else list(range(len(cond)))
    values = [lookup(summary, c, metric) for c in cond]
    errors = [1.96*lookup(summary, c, metric, "se") for c in cond]
    ax.errorbar(xx, values, yerr=errors, marker="o", capsize=3,
                linewidth=2.1, color=color, label=label, linestyle=linestyle)


def plot_study(name: str, summary: list[Row], output: Path) -> None:
    """Produce original publication chart from full experimental summary."""

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.size": 11, "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    navy, orange, teal, violet = "#173956", "#d26a36", "#008b85", "#805d9e"
    fig, ax = plt.subplots(figsize=(9.4, 5.3), layout="constrained")
    if name == "symmetric":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), layout="constrained")
        c = ["0.00", "0.05", "0.15", "0.30"]
        for a, metric, label in zip(axes, ("clean_test_auc", "clean_test_brier"),
                                    ("Clean-test ROC AUC", "Clean-test Brier loss")):
            series(a, summary, c, metric, label, navy, [float(z) for z in c])
            a.set(xlabel="Symmetric label flip probability", ylabel=label)
        fig.suptitle("1  |  Ranking and probability calibration react differently to noisy labels",
                     fontsize=14)
    elif name == "asymmetric":
        c = ["0.00:0.00", "0.05:0.10", "0.05:0.30", "0.15:0.40"]
        for metric, label, color in (
            ("observed_positive_fraction", "Recorded positive fraction", orange),
            ("oracle_corrected_prevalence", "Oracle channel correction", teal),
            ("latent_sample_prevalence", "Latent sample prevalence", navy),
        ):
            series(ax, summary, c, metric, label, color)
        ax.axhline(.25, color="#555555", ls="--", label="Latent population prevalence")
        ax.set_xticks(range(4), ["0 / 0", ".05 / .10", ".05 / .30", ".15 / .40"])
        ax.set(xlabel="False-positive / false-negative label channel rates",
               ylabel="Positive fraction", title="2  |  Label frequency depends on the annotation channel")
        ax.legend(frameon=False, fontsize=9)
    elif name == "instance":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained")
        c = ["uniform", "boundary", "high_margin"]
        for a, metric, label in zip(axes, ("clean_test_auc", "clean_test_brier"),
                                    ("Clean-test ROC AUC", "Clean-test Brier loss")):
            values = [lookup(summary, k, metric) for k in c]
            errors = [1.96*lookup(summary, k, metric, "se") for k in c]
            a.bar(range(3), values, yerr=errors, color=[navy, teal, orange], capsize=3)
            a.set_xticks(range(3), ["Uniform", "Near boundary", "High margin"])
            a.set(ylabel=label, title=label)
        fig.suptitle("3  |  Twelve percent flips are not a complete noise model", fontsize=15)
    elif name == "annotators":
        c = ["3_similar", "5_heterogeneous", "5_weak_majority"]
        for metric, label, color in (
            ("majority_accuracy", "Majority vote", orange),
            ("em_accuracy", "Dawid–Skene EM", navy),
        ):
            series(ax, summary, c, metric, label, color)
        ax.set_xticks(range(3), ["3 similar", "5 varied skills", "5, four weak"])
        ax.set(ylabel="Accuracy against synthetic latent outcome", ylim=(.45, 1.02),
               title="4  |  Annotator reliability matters more than raw consensus")
        ax.legend(frameon=False)
    elif name == "correlation":
        c = ["0.00", "0.05", "0.10", "0.20"]
        x = [float(z) for z in c]
        series(ax, summary, c, "majority_error_rate", "Majority error", orange, x)
        series(ax, summary, c, "em_error_rate", "Dawid–Skene error", navy, x)
        series(ax, summary, c, "confident_wrong_fraction", "Confidently wrong EM", violet, x)
        ax.set(xlabel="Shared item-level flip probability", ylabel="Fraction of rows",
               title="5  |  Five matching votes may share one underlying mistake")
        ax.legend(frameon=False)
    elif name == "selection":
        c = ["0.10", "0.20", "0.40"]
        x = [float(z) for z in c]
        for metric, label, color in (
            ("review_only_prevalence", "Only reviewed cases", orange),
            ("review_and_random_audit_ht", "Review + randomized audit (HT)", teal),
            ("latent_population_prevalence", "Full latent population (simulated)", navy),
        ):
            series(ax, summary, c, metric, label, color, x)
        ax.set(xlabel="High-score fraction automatically reviewed",
               ylabel="Estimated positive fraction",
               title="6  |  Selected labels distort observed prevalence")
        ax.legend(frameon=False, fontsize=9)
    elif name == "benchmark":
        c = ["0.00", "0.10", "0.25", "0.50", "0.75"]
        x = [float(z) for z in c]
        for metric, label, color, style in (
            ("true_accuracy_A", "A vs latent outcome", navy, "-"),
            ("true_accuracy_B", "B vs latent outcome", teal, "-"),
            ("observed_accuracy_A", "A vs recorded test label", orange, "--"),
            ("observed_accuracy_B", "B vs recorded test label", violet, "--"),
        ):
            series(ax, summary, c, metric, label, color, x, style)
        ax.set(xlabel="Minority-label replacement probability favoring B",
               ylabel="Test accuracy", title="7  |  An observed test ranking can reverse")
        ax.legend(frameon=False, fontsize=9)
    elif name == "leakage":
        c = ["row_split", "entity_holdout"]
        values = [lookup(summary, k, "test_accuracy") for k in c]
        errors = [1.96*lookup(summary, k, "test_accuracy", "se") for k in c]
        ax.bar(range(2), values, yerr=errors, color=[orange, navy], capsize=3,
               width=.6)
        ax.axhline(.5, color="#666666", ls="--", label="Unseen random-label chance")
        ax.set_xticks(range(2), ["Random row split", "Entity-held-out split"])
        ax.set(ylabel="Accuracy on test records", ylim=(0, 1.03),
               title="8  |  Memorizing familiar entities is not generalization")
        ax.legend(frameon=False)
    elif name == "policy":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.), layout="constrained")
        c = ["baseline", "lenient", "strict"]
        names = ["Original policy", "Lenient (more +)", "Strict (more −)"]
        for a, metrics, heading in zip(axes,
            (("latent_outcome_prevalence", "recorded_positive_fraction"),
             ("true_model_accuracy", "recorded_label_accuracy")),
            ("Measured positive prevalence", "Measured model accuracy"),
        ):
            for off, metric, color in ((-.18, metrics[0], navy),
                                       (.18, metrics[1], orange)):
                values = [lookup(summary, k, metric) for k in c]
                errs = [1.96*lookup(summary, k, metric, "se") for k in c]
                a.bar(np.arange(3)+off, values, width=.34, yerr=errs,
                      color=color, capsize=3, label=metric.replace("_", " "))
            a.set_xticks(range(3), names, rotation=12, ha="right")
            a.set(title=heading, ylabel="Fraction", ylim=(0, 1.02))
            a.legend(frameon=False, fontsize=8)
        fig.suptitle("9  |  A labeling-policy change can masquerade as model drift",
                     fontsize=15)
    else:
        plt.close(fig)
        raise KeyError(name)
    for a in fig.axes:
        a.grid(axis="y", color="#e5e9ed", linewidth=.75)
        a.set_axisbelow(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(design: Design = Design()) -> dict[str, list[Row]]:
    """Regenerate full versioned evidence or safe quick-mode smoke-test files."""

    if not isinstance(design.root, Path):
        raise TypeError("output root must be pathlib.Path")
    if not isinstance(design.seed, int) or design.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not design.quick and design.seed != SEED and design.root.resolve() == ROOT.resolve():
        raise ValueError("alternate full seed requires a separate --output directory")
    suffix = "_quick" if design.quick else ""
    results: dict[str, list[Row]] = {}
    for i, name in enumerate(EXPERIMENTS):
        count = 3 if design.quick else REPETITIONS[name]
        rows: list[Row] = []
        for rep in range(count):
            rows.extend(STUDIES[name](stream(design.seed, name, rep), rep, design.quick))
        summary = summarize(rows)
        write_csv(design.root/"results"/f"{name}_replicates{suffix}.csv", rows)
        write_csv(design.root/"results"/f"{name}_summary{suffix}.csv", summary)
        plot_study(name, summary, design.root/"figures"/f"{FIGURES[i]}{suffix}.png")
        results[name] = summary
    return results


def main() -> None:
    """Run from the CLI: --quick / --output PATH / --seed INTEGER."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run(Design(root=args.output, quick=args.quick, seed=args.seed))
    for name in EXPERIMENTS:
        print(f"{name}: {len(result[name])} summary cells")


if __name__ == "__main__":
    main()
