"""Evidence 1 - credit-card default (ISLR::Default, 333 defaults in 10,000).

Logistic regression of default on balance, income and student status, trained
five ways: on the data as they are; with balanced class weights; after random
undersampling of the non-defaulters to a 50:50 balance; after random
oversampling of the defaulters to 50:50; after SMOTE to 50:50. Each rebalanced
model's probabilities are also prior-corrected back to the training prevalence.

1. Full-data fits: coefficients, mean predicted probability, and for each
   rebalanced model the share of customers it flags at a 0.5 cutoff and the
   cutoff on the original model's probabilities that flags the same share.
2. Repeated stratified 10-fold cross-validation (20 repeats): out-of-fold AUC,
   calibration intercept and slope, Brier score, log loss, mean predicted
   probability; and the expected cost per customer, with a false alarm costing 1
   and a missed default R = 5, 10, 29 or 50, when customers are flagged at 0.5 or
   at the cost-based cutoff 1 / (1 + R).

Figures: fig1_default_probabilities.png, fig2_default_roc.png.

Run:  python 01_default.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, VIOLET, YELLOW, style, write
from data import default
from rare import (auc, balanced_weights, brier, calibration, confusion, cost_threshold, expected_cost, logistic,
                  logloss, oversample, prior_correct, smote, undersample)

warnings.filterwarnings("ignore")
style()
y, X, names = default()
n, n1 = len(y), int(y.sum())
METHODS = ["Original", "Class weights", "Undersampling", "Oversampling", "SMOTE"]
RATIOS = [5, 10, 29, 50]
REPEATS, FOLDS = 20, 10


def fit(method, Xtr, ytr, seed):
    rng = np.random.default_rng(seed)
    if method == "Original":
        return logistic(Xtr, ytr)
    if method == "Class weights":
        return logistic(Xtr, ytr, w=balanced_weights(ytr))
    if method == "Undersampling":
        return logistic(*undersample(Xtr, ytr, rng))
    if method == "Oversampling":
        return logistic(*oversample(Xtr, ytr, rng))
    if method == "SMOTE":
        Xs, ys = smote(Xtr[:, 1:], ytr, np.random.RandomState(seed), k=5)
        return logistic(np.column_stack([np.ones(len(ys)), Xs]), ys)
    raise ValueError(method)


out = [f"Default: {n} customers, {n1} defaults ({n1 / n:.4f}); non-defaulters per defaulter {(n - n1) / n1:.2f}"]
# 1. full-data fits
full = {m: fit(m, X, y, 1) for m in METHODS}
p_orig = full["Original"].predict(X)
out.append("Full-data fits: coefficients (intercept, balance, income in thousands, student); mean predicted probability")
for m, f in full.items():
    out.append(f"   {m:14s} {np.array2string(f.coef, precision=6, floatmode='fixed', separator=', ')}; mean "
               f"{f.predict(X).mean():.4f}; intercept minus the original's {f.coef[0] - full['Original'].coef[0]:+.4f}")
out.append(f"   log of non-defaulters per defaulter {np.log((n - n1) / n1):.4f}")
out.append(f"   training rows: original {n} ({n - n1} non-defaulters); undersampling keeps {n1} non-defaulters ({2 * n1} rows); "
           f"oversampling adds {n - 2 * n1} defaulters drawn with replacement ({2 * (n - n1)} rows); SMOTE adds {n - 2 * n1} synthetic defaulters")
med_inc = np.median(X[:, 2])
for m, f in full.items():
    b = f.coef
    out.append(f"   {m:14s} expected number of defaults (sum of probabilities) {f.predict(X).sum():.1f}; balance at which a "
               f"non-student with the median income ({1000 * med_inc:.0f} dollars) reaches 0.5: "
               f"{-(b[0] + b[2] * med_inc) / b[1]:.0f} dollars")
for m in METHODS[1:]:
    d = full[m].linear(X) - full["Original"].linear(X)
    out.append(f"   {m:14s} log-odds minus the original's, over the 10,000 customers: mean {d.mean():+.4f}, "
               f"5th to 95th percentile {np.quantile(d, 0.05):+.4f} to {np.quantile(d, 0.95):+.4f}; mean probability "
               f"after prior correction {prior_correct(full[m].predict(X), 0.5, n1 / n).mean():.4f}")
tp, fp, fn, tn = confusion(y, p_orig, 0.5)
out.append(f"   Original       flags {np.mean(p_orig >= 0.5):.4f} of customers at 0.5 (catches {tp} of {n1} defaults, "
           f"{fp} false alarms)")
for m in METHODS[1:]:
    p = full[m].predict(X)
    share = np.mean(p >= 0.5)
    t_eq = np.quantile(p_orig, 1 - share)
    tp, fp, fn, tn = confusion(y, p, 0.5)
    out.append(f"   {m:14s} flags {share:.4f} of customers at 0.5 (catches {tp} of {n1} defaults, {fp} false alarms); the "
               f"original model flags the same share at a cutoff of {t_eq:.4f}, the cost-based cutoff for R = "
               f"{(1 - t_eq) / t_eq:.1f}; agreement with that rule on who is flagged {np.mean((p >= 0.5) == (p_orig >= t_eq)):.4f}")

# 2. repeated cross-validation
rec = {(m, c): {k: [] for k in ("auc", "cal_a", "cal_s", "brier", "logloss", "mean")} for m in METHODS for c in ("raw", "corr")}
costs = {(m, c, R, rule): [] for m in METHODS for c in ("raw", "corr") for R in RATIOS for rule in ("0.5", "cost")}
oof_first = {}
for rep in range(REPEATS):
    rng = np.random.default_rng([11, rep])
    fold = np.empty(n, int)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % FOLDS
    oof = {(m, c): np.zeros(n) for m in METHODS for c in ("raw", "corr")}
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        prev = y[tr].mean()
        for j, m in enumerate(METHODS):
            f = fit(m, X[tr], y[tr], seed=rep * 100 + k * 10 + j)
            p = f.predict(X[te])
            oof[(m, "raw")][te] = p
            oof[(m, "corr")][te] = p if m == "Original" else prior_correct(p, 0.5 if m != "Original" else prev, prev)
    if rep == 0:
        oof_first = oof
    for (m, c), p in oof.items():
        a, s = calibration(y, p)
        for key, v in (("auc", auc(y, p)), ("cal_a", a), ("cal_s", s), ("brier", brier(y, p)), ("logloss", logloss(y, p)),
                       ("mean", p.mean())):
            rec[(m, c)][key].append(v)
        for R in RATIOS:
            costs[(m, c, R, "0.5")].append(expected_cost(y, p, 0.5, 1, R))
            costs[(m, c, R, "cost")].append(expected_cost(y, p, cost_threshold(1, R), 1, R))
out.append(f"Repeated stratified {FOLDS}-fold cross-validation ({REPEATS} repeats), out-of-fold predictions, means over "
           f"repeats (probabilities as fitted, and prior-corrected to the training prevalence)")
for c in ("raw", "corr"):
    for m in METHODS:
        if c == "corr" and m == "Original":
            continue
        r = {k: np.mean(v) for k, v in rec[(m, c)].items()}
        out.append(f"   {m:14s} {'as fitted' if c == 'raw' else 'corrected'}: AUC {r['auc']:.4f}; calibration intercept "
                   f"{r['cal_a']:+.4f}, slope {r['cal_s']:.4f}; Brier {r['brier']:.5f}; log loss {r['logloss']:.5f}; mean "
                   f"predicted {r['mean']:.4f}")
out.append("First repetition, top tenth of each model's out-of-fold predictions (figure 1, right): mean predicted, observed "
           "default rate")
for m in METHODS:
    p = oof_first[(m, "raw")]
    top = p >= np.quantile(p, 0.9)
    out.append(f"   {m:14s} {p[top].mean():.4f}, {y[top].mean():.4f}")
out.append(f"Expected cost per customer (a false alarm costs 1, a missed default R), cutoff 0.5 or 1 / (1 + R); flagging "
           f"nobody costs R x {n1 / n:.4f}")
for R in RATIOS:
    out.append(f"   R = {R} (cost-based cutoff {cost_threshold(1, R):.4f}; flagging nobody {R * n1 / n:.4f}):")
    for c in ("raw", "corr"):
        for m in METHODS:
            if c == "corr" and m == "Original":
                continue
            out.append(f"      {m:14s} {'as fitted' if c == 'raw' else 'corrected'}: cutoff 0.5 "
                       f"{np.mean(costs[(m, c, R, '0.5')]):.4f}; cost-based cutoff {np.mean(costs[(m, c, R, 'cost')]):.4f}")
write("01_default.txt", "\n".join(out))

# --------------------------------------------------------------- figure 1
COL = {"Original": INK, "Class weights": BLUE, "Undersampling": ORANGE, "Oversampling": AQUA, "SMOTE": VIOLET}
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), gridspec_kw={"wspace": 0.27})
ax = axes[0]
bal = np.linspace(0, 2700, 300)
G = np.column_stack([np.ones_like(bal), bal, np.full_like(bal, np.median(X[:, 2])), np.zeros_like(bal)])
for m in METHODS:
    ax.plot(bal, full[m].predict(G), color=COL[m], lw=2.2 if m == "Original" else 1.7,
            ls="-" if m == "Original" else "--", label=m)
ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("Credit-card balance (dollars)")
ax.set_ylabel("Predicted probability of default")
ax.set_title("Same customers, different probabilities")
ax.text(60, 0.53, "0.5 cutoff", color=INK2, fontsize=8.5)
ax.legend(fontsize=8.5, loc="upper left")
ax = axes[1]
for m in METHODS:
    p = oof_first[(m, "raw")]
    q = np.quantile(p, np.linspace(0, 1, 11))
    g = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    ax.plot([p[g == j].mean() for j in range(10)], [y[g == j].mean() for j in range(10)], "o-", color=COL[m], ms=4,
            lw=1.6, label=m)
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.5)
ax.set_xlabel("Mean predicted probability (deciles of each model's predictions)")
ax.set_ylabel("Observed default rate")
ax.set_title("Out-of-fold calibration")
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig(FIG / "fig1_default_probabilities.png")
plt.close(fig)

# --------------------------------------------------------------- figure 2
p = oof_first[("Original", "raw")]
order = np.argsort(-p)
tpr = np.r_[0, np.cumsum(y[order] == 1) / n1]
fpr = np.r_[0, np.cumsum(y[order] == 0) / (n - n1)]
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), gridspec_kw={"wspace": 0.27})
ax = axes[0]
ax.plot(fpr, tpr, color=INK, lw=1.8, label="Original model, out-of-fold")
for m in METHODS[1:]:
    tp, fp, fn, tn = confusion(y, oof_first[(m, "raw")], 0.5)
    ax.plot(fp / (n - n1), tp / n1, "o", color=COL[m], ms=5, label=f"{m}, cutoff 0.5", zorder=5)
for R, mk in zip(RATIOS, ("v", "^", "s", "D")):
    tp, fp, fn, tn = confusion(y, p, cost_threshold(1, R))
    ax.plot(fp / (n - n1), tp / n1, mk, mfc="none", ms=12, mec=YELLOW, mew=2.0, label=f"Original, cutoff 1/(1+{R})",
            zorder=4)
ax.set_xlim(0, 0.4)
ax.set_ylim(0, 1)
ax.set_xlabel("False-alarm rate (share of non-defaulters flagged)")
ax.set_ylabel("Share of defaults caught")
ax.set_title("Rebalancing moves along the same ROC curve")
ax.legend(fontsize=7.8, loc="lower right")
ax = axes[1]
Rgrid = np.array(RATIOS)
for m in METHODS:
    v = [np.mean(costs[(m, "raw", R, "0.5")]) for R in RATIOS]
    ax.plot(Rgrid, v, "o-" if m != "Original" else "s-", color=COL[m], ms=5, lw=1.6, label=f"{m}, cutoff 0.5")
v = [np.mean(costs[("Original", "raw", R, "cost")]) for R in RATIOS]
ax.plot(Rgrid, v, "D-", color=YELLOW, mec=INK, mew=0.6, ms=6, lw=2.2, label="Original, cutoff 1/(1+R)")
ax.plot(Rgrid, Rgrid * n1 / n, color=INK2, lw=1, ls=":", label="Flag nobody")
ax.set_xscale("log")
ax.set_xticks(RATIOS, [str(r) for r in RATIOS])
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Cost of a missed default relative to a false alarm (R)")
ax.set_ylabel("Expected cost per customer")
ax.set_title("Out-of-fold expected cost")
ax.legend(fontsize=7.8, loc="upper left")
fig.savefig(FIG / "fig2_default_roc.png")
plt.close(fig)
