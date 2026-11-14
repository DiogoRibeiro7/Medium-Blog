"""Simulation with known truth - what rebalancing does to probabilities and to decisions.

Eight correlated standard normal regressors (equicorrelation 0.3); the true
model is logistic with slopes (1, -0.8, 0.6, -0.5, 0.4, 0.3, 0, 0) and an
intercept set so that the event rate is 0.5%, 2% or 10%. Training samples of
5,000 or 20,000; 200 samples per cell; each model is evaluated on a test set of
100,000 cases with known true probabilities.

Methods: logistic regression on the data as they are; with balanced class
weights; after undersampling the non-events to 50:50; after oversampling the
events to 50:50; after SMOTE to 50:50. Each rebalanced model's probabilities
are also prior-corrected to the training event rate.

Per sample: AUC; calibration intercept and slope; mean absolute error of the
predicted probabilities against the true ones; log loss; and the excess cost
over the best possible decisions (true probabilities, cutoff 1 / (1 + R)), with
a false alarm costing 1 and a missed event R = 5, 20 or 100, for flagging at 0.5
and at 1 / (1 + R).

Outputs: results/03_simulation.txt, 03_simulation_digest.txt,
03_simulation_summary.csv, 03_simulation_replications.csv.gz (not committed);
figures fig4_simulation_calibration.png, fig5_simulation_cost.png.

Run:  python 03_simulation.py             (about 15 minutes on 2 cores)
      python 03_simulation.py --reps 20   (quick run)
      python 03_simulation.py --figures-only
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, style, write
from rare import (auc, balanced_weights, calibration, cost_threshold, expected_cost, logistic, logloss, oversample,
                  prior_correct, smote, undersample)

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--figures-only", action="store_true")
ARGS = ap.parse_args()

P = 8
BETA = np.array([1.0, -0.8, 0.6, -0.5, 0.4, 0.3, 0.0, 0.0])
RATES = [0.005, 0.02, 0.10]
NS = [5000, 20000]
NTEST = 100_000
METHODS = ["Original", "Class weights", "Undersampling", "Oversampling", "SMOTE"]
RATIOS = [5, 20, 100]
CHOL = np.linalg.cholesky(0.7 * np.eye(P) + 0.3)


def draw_x(n, rng):
    return rng.standard_normal((n, P)) @ CHOL.T


def intercept_for(rate):
    rng = np.random.default_rng(99)
    eta = draw_x(400_000, rng) @ BETA
    return optimize.brentq(lambda b0: expit(b0 + eta).mean() - rate, -15, 5)


B0 = {r: intercept_for(r) for r in RATES}


def test_set(rate):
    rng = np.random.default_rng([5, RATES.index(rate)])
    x = draw_x(NTEST, rng)
    p = expit(B0[rate] + x @ BETA)
    y = (rng.random(NTEST) < p).astype(int)
    return np.column_stack([np.ones(NTEST), x]), y, p


def one(task):
    rate, n, rep = task
    Xte, yte, pte = TEST[rate]
    rng = np.random.default_rng([3, RATES.index(rate), NS.index(n), rep])
    x = draw_x(n, rng)
    y = (rng.random(n) < expit(B0[rate] + x @ BETA)).astype(int)
    X = np.column_stack([np.ones(n), x])
    prev = y.mean()
    fits = {"Original": logistic(X, y), "Class weights": logistic(X, y, w=balanced_weights(y)),
            "Undersampling": logistic(*undersample(X, y, rng)), "Oversampling": logistic(*oversample(X, y, rng))}
    Xs, ys = smote(x, y, np.random.RandomState(rep + 1000 * NS.index(n) + 100000 * RATES.index(rate)), k=5)
    fits["SMOTE"] = logistic(np.column_stack([np.ones(len(ys)), Xs]), ys)
    oracle = {R: expected_cost(yte, pte, cost_threshold(1, R), 1, R) for R in RATIOS}
    row = {"rate": rate, "n": n, "rep": rep, "events": int(y.sum())}
    for m, f in fits.items():
        row[f"{m}|maxslope"] = float(np.max(np.abs(f.coef[1:])))
        p_raw = f.predict(Xte)
        for c, p in (("raw", p_raw), ("corr", p_raw if m == "Original" else prior_correct(p_raw, 0.5, prev))):
            if m == "Original" and c == "corr":
                continue
            k = f"{m}|{c}"
            a, s = calibration(yte, p)
            row[f"{k}|auc"] = auc(yte, p)
            row[f"{k}|cal_a"], row[f"{k}|cal_s"] = a, s
            row[f"{k}|mae"] = float(np.mean(np.abs(p - pte)))
            row[f"{k}|logloss"] = logloss(yte, p)
            for R in RATIOS:
                for rule, t in (("0.5", 0.5), ("cost", cost_threshold(1, R))):
                    row[f"{k}|excess_{R}_{rule}"] = expected_cost(yte, p, t, 1, R) / oracle[R] - 1
    return row


KEYS = [f"{m}|raw" for m in METHODS] + [f"{m}|corr" for m in METHODS[1:]]
DIVERGED = 15.0   # a fitted slope above 15 (true slopes are at most 1): the sample is (nearly) separated


def pct(x):
    s = f"{100 * x:.1f}%"
    return "0.0%" if s == "-0.0%" else s


def summarise(R_):
    rows = []
    for (rate, n), u in R_.groupby(["rate", "n"], sort=False):
        r = {"rate": rate, "n": n, "reps": len(u), "events_median": u.events.median(), "events_min": u.events.min()}
        for m in METHODS:
            r[f"{m}|diverged"] = int((u[f"{m}|maxslope"] > DIVERGED).sum())
        for k in KEYS:
            for q in ("auc", "mae", "logloss"):
                r[f"{k}|{q}"] = u[f"{k}|{q}"].mean()
            for q in ("cal_a", "cal_s"):
                r[f"{k}|{q}"] = u[f"{k}|{q}"].median()
            for Rr in RATIOS:
                for rule in ("0.5", "cost"):
                    r[f"{k}|excess_{Rr}_{rule}"] = u[f"{k}|excess_{Rr}_{rule}"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def lab(k):
    m, c = k.split("|")
    return m if m == "Original" else f"{m}, {'as fitted' if c == 'raw' else 'corrected'}"


def report(S):
    out = [f"True model: logistic, slopes {BETA.tolist()}, equicorrelated regressors (0.3); intercepts " +
           "; ".join(f"event rate {r}: {B0[r]:.4f}" for r in RATES) + f"; test sets of {NTEST:,} cases"]
    for r in S.to_dict("records"):
        out.append(f"-- event rate {r['rate']}, n = {r['n']} ({r['reps']} samples; events per sample: median "
                   f"{r['events_median']:g}, minimum {r['events_min']:.0f}); fits with a largest absolute slope above "
                   f"{DIVERGED:g} (separated or nearly separated samples): " + ", ".join(f"{m} {int(r[f'{m}|diverged'])}" for m in METHODS))
        for k in KEYS:
            out.append(f"   {lab(k):30s} AUC {r[f'{k}|auc']:.4f}; median calibration intercept {r[f'{k}|cal_a']:+.4f}, "
                       f"slope {r[f'{k}|cal_s']:.4f}; mean |p - true p| {r[f'{k}|mae']:.5f}; log loss {r[f'{k}|logloss']:.5f}")
            out.append("      excess cost over the best decisions, cutoff 0.5 / cutoff 1/(1+R): " + "; ".join(
                f"R = {Rr}: {pct(r[f'{k}|excess_{Rr}_0.5'])} / {pct(r[f'{k}|excess_{Rr}_cost'])}" for Rr in RATIOS))
    return "\n".join(out)


def digest(S):
    out = ["Digest"]
    reb = [f"{m}|raw" for m in METHODS[1:]]
    cor = [f"{m}|corr" for m in METHODS[1:]]
    for rate in RATES:
        u = S[S.rate == rate]
        da = [u[f"{k}|auc"] - u["Original|raw|auc"] for k in reb]
        out.append(f"event rate {rate} (non-events per event {(1 - rate) / rate:g}): AUC of the original model {u['Original|raw|auc'].min():.4f}-"
                   f"{u['Original|raw|auc'].max():.4f}; rebalanced minus original {min(d.min() for d in da):+.4f} to "
                   f"{max(d.max() for d in da):+.4f}; median calibration intercept as fitted, original "
                   f"{u['Original|raw|cal_a'].min():+.4f} to {u['Original|raw|cal_a'].max():+.4f}, rebalanced "
                   f"{min(u[f'{k}|cal_a'].min() for k in reb):+.4f} to {max(u[f'{k}|cal_a'].max() for k in reb):+.4f}, "
                   f"rebalanced and corrected {min(u[f'{k}|cal_a'].min() for k in cor):+.4f} to "
                   f"{max(u[f'{k}|cal_a'].max() for k in cor):+.4f}; mean |p - true p|, original "
                   f"{u['Original|raw|mae'].min():.5f}-{u['Original|raw|mae'].max():.5f}, rebalanced "
                   f"{min(u[f'{k}|mae'].min() for k in reb):.5f}-{max(u[f'{k}|mae'].max() for k in reb):.5f}, corrected "
                   f"{min(u[f'{k}|mae'].min() for k in cor):.5f}-{max(u[f'{k}|mae'].max() for k in cor):.5f}")
        for n in NS:
            r = u[u.n == n].iloc[0]
            out.append(f"   n = {n}: median calibration slope, original {r['Original|raw|cal_s']:.4f}; " + "; ".join(
                f"{m} {r[f'{m}|raw|cal_s']:.4f}" for m in METHODS[1:]) + f"; separated undersampled samples "
                f"{int(r['Undersampling|diverged'])} of {int(r['reps'])} (events per sample: median {r['events_median']:g})")
            for Rr in RATIOS:
                out.append(f"   n = {n}, R = {Rr}: excess cost, original at 1/(1+R) {pct(r[f'Original|raw|excess_{Rr}_cost'])}, "
                           f"at 0.5 {pct(r[f'Original|raw|excess_{Rr}_0.5'])}; rebalanced at 0.5 " +
                           ", ".join(pct(r[f'{k}|excess_{Rr}_0.5']) for k in reb) +
                           "; rebalanced as fitted at 1/(1+R) " +
                           ", ".join(pct(r[f'{k}|excess_{Rr}_cost']) for k in reb) +
                           "; corrected at 1/(1+R) " + ", ".join(pct(r[f'{k}|excess_{Rr}_cost']) for k in cor)
                           + " (order: " + ", ".join(METHODS[1:]) + ")")
    return "\n".join(out)


COL = {"Original": INK, "Class weights": BLUE, "Undersampling": ORANGE, "Oversampling": AQUA, "SMOTE": VIOLET}


def figures(S):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), gridspec_kw={"wspace": 0.3})
    xs = np.arange(len(RATES))
    for ax, (q, title) in zip(axes, (("cal_a", "Calibration intercept (0 is ideal)"), ("cal_s", "Calibration slope (1 is ideal)"),
                                     ("auc", "AUC"))):
        for j, m in enumerate(METHODS):
            k = f"{m}|raw"
            v = [S[(S.rate == r) & (S.n == 20000)][f"{k}|{q}"].iloc[0] for r in RATES]
            ax.plot(xs + (j - 2) * 0.09, v, "o", color=COL[m], ms=7, label=m)
            if m != "Original" and q == "cal_a":
                v = [S[(S.rate == r) & (S.n == 20000)][f"{m}|corr|{q}"].iloc[0] for r in RATES]
                ax.plot(xs + (j - 2) * 0.09, v, "D", color=COL[m], ms=5, mfc="white", mew=1.4,
                        label="Rebalanced, prior-corrected (open diamonds)" if m == "Class weights" else None)
        ax.set_xticks(xs, [f"{100 * r:g}%" for r in RATES])
        ax.set_xlabel("Event rate")
        ax.set_title(title, fontsize=10.5)
    axes[0].axhline(0, color=INK2, lw=0.8, ls=":")
    axes[1].axhline(1, color=INK2, lw=0.8, ls=":")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in METHODS] + [l.index("Rebalanced, prior-corrected (open diamonds)")]
    fig.legend([h[i] for i in order], [l[i] for i in order], fontsize=8.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.03), ncol=6)
    fig.savefig(FIG / "fig4_simulation_calibration.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, rate in zip(axes, RATES):
        r = S[(S.rate == rate) & (S.n == 20000)].iloc[0]
        ax.plot(RATIOS, [100 * r[f"Original|raw|excess_{R}_cost"] for R in RATIOS], "D-", color=YELLOW, mec=INK, mew=0.6,
                ms=7, lw=2.4, label="Original, cutoff 1/(1+R)")
        ax.plot(RATIOS, [100 * r[f"Original|raw|excess_{R}_0.5"] for R in RATIOS], "s--", color=INK, ms=5, lw=1.4,
                label="Original, cutoff 0.5")
        for m in METHODS[1:]:
            ax.plot(RATIOS, [100 * r[f"{m}|raw|excess_{R}_0.5"] for R in RATIOS], "o-", color=COL[m], ms=5, lw=1.5,
                    label=f"{m}, cutoff 0.5")
        ax.axvline((1 - rate) / rate, color=INK2, lw=0.8, ls=":")
        ax.text((1 - rate) / rate, 1.5, " R = (1 - rate) / rate", color=INK2, fontsize=8, rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xticks(RATIOS + [int(round((1 - rate) / rate))], [str(x) for x in RATIOS + [int(round((1 - rate) / rate))]])
        ax.set_xlim(4, 250)
        ax.set_title(f"Event rate {100 * rate:g}%, n = 20,000", fontsize=10.5)
    axes[0].set_ylabel("Excess cost over the best decisions (%)")
    fig.supxlabel("Cost of a missed event relative to a false alarm (R)", fontsize=10, y=0.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=6)
    fig.savefig(FIG / "fig5_simulation_cost.png")
    plt.close(fig)


TEST = {r: test_set(r) for r in RATES}

if __name__ == "__main__":
    if ARGS.figures_only:
        S = pd.read_csv(RES / "03_simulation_summary.csv")
    else:
        tasks = [(r, n, k) for r in RATES for n in NS for k in range(ARGS.reps)]
        with Pool(2) as pool:
            rows = pool.map(one, tasks, chunksize=5)
        R_ = pd.DataFrame(rows)
        R_.to_csv(RES / "03_simulation_replications.csv.gz", index=False)
        S = summarise(R_)
        S.to_csv(RES / "03_simulation_summary.csv", index=False, float_format="%.6g")
    write("03_simulation.txt", report(S))
    write("03_simulation_digest.txt", digest(S))
    figures(S)
