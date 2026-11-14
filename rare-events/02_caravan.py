"""Evidence 2 - who buys caravan insurance (ISLR::Caravan, 348 buyers in 5,822).

Ridge-penalised logistic regression on the 85 regressors (standardised within
each training fold), with the penalty chosen by 5-fold cross-validated log loss
on the original training data, from {1, 3, 10, 30, 100, 300, 1000}. Each model is then
trained five ways (as is; balanced class weights; undersampling; oversampling;
SMOTE) with that penalty, and its probabilities are also prior-corrected.

Repeated stratified 10-fold cross-validation (5 repeats): out-of-fold AUC,
calibration intercept and slope, Brier score, log loss, mean predicted
probability; expected cost (a wasted offer costs 1, a missed buyer R = 5, 10,
16 or 30) at 0.5 and at 1 / (1 + R); and net benefit (Vickers and Elkin 2006)
at decision thresholds from 0.02 to 0.30.

Figure: fig3_caravan_net_benefit.png.

Run:  python 02_caravan.py   (about 3 minutes)
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, VIOLET, style, write
from data import caravan
from rare import (auc, balanced_weights, brier, calibration, cost_threshold, expected_cost, logistic, logloss,
                  net_benefit, oversample, prior_correct, smote, undersample)

warnings.filterwarnings("ignore")
style()
y, X, names = caravan()
n, n1 = len(y), int(y.sum())
METHODS = ["Original", "Class weights", "Undersampling", "Oversampling", "SMOTE"]
LAMBDAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
RATIOS = [5, 10, 16, 30]
THRESH = np.round(np.arange(0.02, 0.301, 0.02), 2)
REPEATS, FOLDS, INNER = 5, 10, 5


def standardise(Xtr, Xte):
    m, s = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0)
    s[s == 0] = 1.0
    f = lambda A: np.column_stack([A[:, 0], (A[:, 1:] - m) / s])
    return f(Xtr), f(Xte)


def strat_folds(yv, k, rng):
    fold = np.empty(len(yv), int)
    for cls in (0, 1):
        idx = np.flatnonzero(yv == cls)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % k
    return fold


def choose_lambda(Xtr, ytr, rng):
    inner = strat_folds(ytr, INNER, rng)
    loss = []
    for lam in LAMBDAS:
        ll = 0.0
        for j in range(INNER):
            a, b = standardise(Xtr[inner != j], Xtr[inner == j])
            f = logistic(a, ytr[inner != j], ridge=lam)
            ll += logloss(ytr[inner == j], f.predict(b)) * np.sum(inner == j)
        loss.append(ll / len(ytr))
    return LAMBDAS[int(np.argmin(loss))]


def fit(method, Xtr, ytr, lam, seed):
    rng = np.random.default_rng(seed)
    if method == "Original":
        return logistic(Xtr, ytr, ridge=lam)
    if method == "Class weights":
        return logistic(Xtr, ytr, w=balanced_weights(ytr), ridge=lam)
    if method == "Undersampling":
        return logistic(*undersample(Xtr, ytr, rng), ridge=lam)
    if method == "Oversampling":
        return logistic(*oversample(Xtr, ytr, rng), ridge=lam)
    Xs, ys = smote(Xtr[:, 1:], ytr, np.random.RandomState(seed), k=5)
    return logistic(np.column_stack([np.ones(len(ys)), Xs]), ys, ridge=lam)


KEYS = ("auc", "cal_a", "cal_s", "brier", "logloss", "mean")
rec = {(m, c): {k: [] for k in KEYS} for m in METHODS for c in ("raw", "corr")}
costs = {(m, c, R, r): [] for m in METHODS for c in ("raw", "corr") for R in RATIOS for r in ("0.5", "cost")}
flagged = {(m, c, R): [] for m in METHODS for c in ("raw", "corr") for R in RATIOS}
nb = {(m, c): [] for m in METHODS for c in ("raw", "corr")}
nb_all, lams = [], []
for rep in range(REPEATS):
    rng = np.random.default_rng([23, rep])
    fold = strat_folds(y, FOLDS, rng)
    oof = {(m, c): np.zeros(n) for m in METHODS for c in ("raw", "corr")}
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        lam = choose_lambda(X[tr], y[tr], rng)
        lams.append(lam)
        Xtr, Xte = standardise(X[tr], X[te])
        prev = y[tr].mean()
        for j, m in enumerate(METHODS):
            p = fit(m, Xtr, y[tr], lam, seed=rep * 100 + k * 10 + j).predict(Xte)
            oof[(m, "raw")][te] = p
            oof[(m, "corr")][te] = p if m == "Original" else prior_correct(p, 0.5, prev)
    for key, p in oof.items():
        a, s = calibration(y, p)
        for kk, v in zip(KEYS, (auc(y, p), a, s, brier(y, p), logloss(y, p), p.mean())):
            rec[key][kk].append(v)
        for R in RATIOS:
            costs[key + (R, "0.5")].append(expected_cost(y, p, 0.5, 1, R))
            costs[key + (R, "cost")].append(expected_cost(y, p, cost_threshold(1, R), 1, R))
            flagged[key + (R,)].append(np.mean(p >= cost_threshold(1, R)))
        nb[key].append([net_benefit(y, p, t) for t in THRESH])
    nb_all.append([y.mean() - (1 - y.mean()) * t / (1 - t) for t in THRESH])

vals, cnt = np.unique(lams, return_counts=True)
out = [f"Caravan: {n} customers, {n1} buyers ({n1 / n:.4f}); non-buyers per buyer {(n - n1) / n1:.2f}; "
       f"{X.shape[1] - 1} regressors",
       "Ridge penalty chosen in the outer training folds: " + "; ".join(f"{v:g} in {c}" for v, c in zip(vals, cnt)) +
       f" of {len(lams)} folds",
       f"Repeated stratified {FOLDS}-fold cross-validation ({REPEATS} repeats), out-of-fold predictions, means over repeats"]
for c in ("raw", "corr"):
    for m in METHODS:
        if c == "corr" and m == "Original":
            continue
        r = {k: np.mean(v) for k, v in rec[(m, c)].items()}
        out.append(f"   {m:14s} {'as fitted' if c == 'raw' else 'corrected'}: AUC {r['auc']:.4f}; calibration intercept "
                   f"{r['cal_a']:+.4f}, slope {r['cal_s']:.4f}; Brier {r['brier']:.5f}; log loss {r['logloss']:.5f}; mean "
                   f"predicted {r['mean']:.4f}")
out.append("Expected cost per customer (a wasted offer costs 1, a missed buyer R), cutoff 0.5 or 1 / (1 + R)")
for R in RATIOS:
    out.append(f"   R = {R} (cost-based cutoff {cost_threshold(1, R):.4f}; offering nobody {R * n1 / n:.4f}; offering "
               f"everybody {(n - n1) / n:.4f}):")
    for c in ("raw", "corr"):
        for m in METHODS:
            if c == "corr" and m == "Original":
                continue
            out.append(f"      {m:14s} {'as fitted' if c == 'raw' else 'corrected'}: cutoff 0.5 "
                       f"{np.mean(costs[(m, c, R, '0.5')]):.4f}; cost-based cutoff {np.mean(costs[(m, c, R, 'cost')]):.4f} "
                       f"(offers to {np.mean(flagged[(m, c, R)]):.4f} of customers)")
out.append("Net benefit (buyers reached per customer, net of wasted offers weighted by t / (1 - t)) at decision threshold t, "
           "treating each model's probabilities as risks")
for key in [("Original", "raw")] + [(m, c) for c in ("raw", "corr") for m in METHODS[1:]]:
    v = np.mean(nb[key], axis=0)
    out.append(f"   {key[0]:14s} {'as fitted' if key[1] == 'raw' else 'corrected'}: " +
               "; ".join(f"{t:.2f} {x:+.5f}" for t, x in zip(THRESH, v) if t in (0.04, 0.06, 0.08, 0.1, 0.16, 0.2, 0.3)))
v = np.mean(nb_all, axis=0)
out.append("   offer to everybody: " + "; ".join(f"{t:.2f} {x:+.5f}" for t, x in zip(THRESH, v) if t in (0.04, 0.06, 0.08, 0.1, 0.16, 0.2, 0.3)))
write("02_caravan.txt", "\n".join(out))

# --------------------------------------------------------------- figure 3
COL = {"Original": INK, "Class weights": BLUE, "Undersampling": ORANGE, "Oversampling": AQUA, "SMOTE": VIOLET}
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, c, title in zip(axes, ("raw", "corr"), ("Probabilities as fitted", "Probabilities prior-corrected")):
    ax.plot(THRESH, np.mean(nb_all, axis=0), color=INK2, lw=1.3, ls="--", label="Offer to everybody", zorder=6)
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    for m in METHODS:
        key = (m, "raw") if m == "Original" else (m, c)
        ax.plot(THRESH, np.mean(nb[key], axis=0), "o-" if m != "Original" else "s-", color=COL[m], ms=3.5,
                lw=2.2 if m == "Original" else 1.5, label=m)
    ax.set_xlabel("Decision threshold t (offer when the predicted probability ≥ t)")
    ax.set_title(title)
    ax.set_ylim(-0.02, 0.045)
    ax.set_xlim(0.015, 0.305)
axes[0].set_ylabel("Net benefit (buyers reached per customer)")
axes[1].legend(fontsize=8.3, loc="upper right")
fig.savefig(FIG / "fig3_caravan_net_benefit.png")
plt.close(fig)
