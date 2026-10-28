"""Rerun the MC dropout UCI regression benchmark and take its score apart.

Protocol of the reference code (github.com/yaringal/DropoutUncertaintyExps,
2018 grid-search version): for each published train/test split, the first 80%
of the training rows train one network per (dropout rate, tau) pair of the
data set's grid, the last 20% pick the pair with the best MC log-likelihood,
and a network with that pair is retrained on the whole training set and scored
on the test rows. One hidden layer (50 units, 100 for protein), dropout before
every dense layer, weight decay l^2 (1 - p) / (2 N tau) with l = 0.01, Adam,
minibatches of 128, and 100 times the data set's listed number of epochs:
the reference README describes the run as "10x epochs", but its committed
result files, whose numbers are the ones the README reports, are named
``*_100_xepochs_1_hidden_layers.txt``, and at 10x this implementation does not
reach the reported accuracy on the small data sets. Test predictions use
T = 1000 stochastic passes (the reference code used 10,000; the
log-likelihood estimate is stable at 1,000).

The default run covers the four small data sets (yacht, concrete, energy,
red wine) with all 20 published splits each, about 1.5 hours on one CPU core.
The large data sets can be added with --sets; one split takes 8 to 12 minutes
on one core for kin8nm, power plant and naval and about 35 for protein, and
MAX_SPLITS limits how many are run. The article reports the default run.

On top of the protocol, for each split this script records
- how much of the predictive variance comes from the dropout spread and how
  much from the constant 1/tau;
- the test log-likelihood of two Gaussians that throw the dropout spread away:
  N(MC mean, 1/tau) with the tuned tau, and N(MC mean, s2) with s2 the mean
  squared validation residual of the selected network;
- risk-coverage: the RMSE of the MC mean on the fraction of test points with
  the lowest score, for four scores - the MC dropout standard deviation, the
  spread of a 5-member deep ensemble trained alongside, the distance to the
  10 nearest training rows, and the realised error itself (the ceiling).

Usage: python 03_uci_benchmark.py [--sets yacht,concrete] [--workers 2]
       python 03_uci_benchmark.py --summary   (rebuild the text summary only)
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from mcd import (QUICK, RESULTS, UCI_LABEL, UCI_SETS, Nets, Report, gal_weight_decay,
                 gauss_loglik, knn_distance, load_uci, mc_loglik, risk_coverage,
                 steps_for)

COVERAGES = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
T_TEST = 1000
T_VAL = 500
N_ENS = 5
EPOCH_MULT = 100
MAX_SPLITS = {"kin8nm": 3, "power-plant": 3, "naval-propulsion-plant": 3,
              "protein-tertiary-structure": 1}
DEFAULT_SETS = ["yacht", "concrete", "energy", "wine-quality-red"]
if QUICK:
    EPOCH_MULT, T_TEST, T_VAL = 2, 50, 50
    MAX_SPLITS = {name: 2 for name in UCI_SETS}


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_split(D, k):
    X, y = D["X"], D["y"]
    tr, te = D["train"][k], D["test"][k]
    n_in = int(0.8 * len(tr))
    tr_in, val = tr[:n_in], tr[n_in:]
    d, h = X.shape[1], D["n_hidden"]
    epochs = D["n_epochs"] * EPOCH_MULT

    grid = [(p, t) for p in D["dropout_rates"] for t in D["tau_values"]]
    drop = np.array([p for p, _ in grid])
    tau = np.array([t for _, t in grid])
    seed = 7919 * (UCI_SETS.index(D["name"]) + 1) + k
    nets = Nets(d, [h], drop, gal_weight_decay(drop, n_in, tau), seed=seed)
    nets.fit(X, y, [tr_in] * len(grid), steps=steps_for(n_in, epochs))
    val_samples = nets.predict(X[val], T=T_VAL)
    val_ll = np.array([mc_loglik(y[val], val_samples[i], tau[i]).mean() for i in range(len(grid))])
    best = int(np.argmax(val_ll))
    p_best, tau_best = drop[best], tau[best]
    val_resid_var = float(np.mean((y[val] - val_samples[best].mean(0)) ** 2))

    fdrop = np.r_[p_best, np.zeros(N_ENS)]
    freg = np.r_[gal_weight_decay(p_best, len(tr), tau_best), np.zeros(N_ENS)]
    final = Nets(d, [h], fdrop, freg, seed=seed + 100000)
    final.fit(X, y, [tr] * (1 + N_ENS), steps=steps_for(len(tr), epochs))
    samples = final.predict(X[te], T=T_TEST, members=[0])[0]
    ens = final.predict(X[te], mc=False, members=range(1, 1 + N_ENS))[:, 0, :]
    yt = y[te]

    mc_mean = samples.mean(0)
    s2 = samples.var(0)
    err = np.abs(yt - mc_mean)
    ens_sd = ens.std(0, ddof=1)
    knn = knn_distance(X[tr], X[te])
    row = {
        "dataset": D["name"], "split": k, "n_train": len(tr), "n_test": len(te),
        "p": p_best, "tau": tau_best,
        "rmse_mc": float(np.sqrt(np.mean((yt - mc_mean) ** 2))),
        "rmse_ens": float(np.sqrt(np.mean((yt - ens.mean(0)) ** 2))),
        "ll_mc": float(mc_loglik(yt, samples, tau_best).mean()),
        "ll_tau_only": float(gauss_loglik(yt, mc_mean, 1.0 / tau_best).mean()),
        "ll_val_var": float(gauss_loglik(yt, mc_mean, val_resid_var).mean()),
        "noise_sd": float(np.sqrt(1.0 / tau_best)),
        "val_resid_sd": float(np.sqrt(val_resid_var)),
        "mc_sd_mean": float(np.sqrt(s2).mean()),
        "share_mc": float(np.mean(s2 / (s2 + 1.0 / tau_best))),
        "share_mc_max": float(np.max(s2 / (s2 + 1.0 / tau_best))),
        "rho_mc": spearman(np.sqrt(s2), err),
        "rho_ens": spearman(ens_sd, err),
        "rho_knn": spearman(knn, err),
    }
    full = np.sqrt(np.mean(err ** 2))
    for name, score in (("mc", np.sqrt(s2)), ("ens", ens_sd), ("knn", knn), ("oracle", err)):
        rc = risk_coverage(err, score, COVERAGES) / full
        for c, v in zip(COVERAGES, rc):
            row[f"rc_{name}_{int(round(c * 100))}"] = float(v)
    return row


def run_dataset(name):
    D = load_uci(name)
    rows = []
    for k in range(min(D["n_splits"], MAX_SPLITS.get(name, D["n_splits"]))):
        rows.append(run_split(D, k))
        r = rows[-1]
        print(f"{name} split {k}: p={r['p']} tau={r['tau']} rmse={r['rmse_mc']:.3f} "
              f"ll={r['ll_mc']:.3f} ll_tau_only={r['ll_tau_only']:.3f} share={r['share_mc']:.3f}",
              flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"03_splits_{name}.csv", index=False, float_format="%.6g")
    return rows


def mse(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(DEFAULT_SETS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if not args.summary:
        sets = args.sets.split(",")
        for s in sets:  # download in the parent process first
            load_uci(s)
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(run_dataset, sets))
    parts = [pd.read_csv(RESULTS / f"03_splits_{n}.csv") for n in UCI_SETS
             if (RESULTS / f"03_splits_{n}.csv").exists()]
    df = pd.concat(parts, ignore_index=True)
    sets = [n for n in UCI_SETS if n in set(df.dataset)]

    rep = Report("03_uci_benchmark")
    rep("MC dropout UCI benchmark, reference protocol (grid search, 100x epochs, one hidden layer)")
    rep("mean +- standard error over the published splits; Boston housing excluded")
    rep("")
    for name in sets:
        g = df[df.dataset == name]
        lab = UCI_LABEL[name]
        rep(f"{lab}: {len(g)} splits, n_train {int(g.n_train.iloc[0])}, n_test {int(g.n_test.iloc[0])}")
        ps = g.p.value_counts().sort_index()
        ts = g.tau.value_counts().sort_index()
        rep("  dropout rate chosen: " + ", ".join(f"{p:g} x{c}" for p, c in ps.items()))
        rep("  tau chosen: " + ", ".join(f"{t:g} x{c}" for t, c in ts.items()))
        for col, label in (("rmse_mc", "RMSE (MC mean)"), ("ll_mc", "test LL, MC dropout mixture"),
                           ("ll_tau_only", "test LL, N(MC mean, 1/tau)"),
                           ("ll_val_var", "test LL, N(MC mean, validation residual variance)"),
                           ("noise_sd", "noise sd 1/sqrt(tau)"), ("mc_sd_mean", "mean MC dropout sd"),
                           ("share_mc", "share of predictive variance from dropout spread"),
                           ("share_mc_max", "largest share at any test point")):
            m, se = mse(g[col])
            rep(f"  {label}: {m:.4f} +- {se:.4f}")
        m, se = mse(g.ll_mc - g.ll_tau_only)
        rep(f"  LL gain of the dropout spread over N(MC mean, 1/tau): {m:.4f} +- {se:.4f}")
        m, se = mse(g.ll_mc - g.ll_val_var)
        rep(f"  LL gain of MC dropout over N(MC mean, validation residual variance): {m:.4f} +- {se:.4f}")
        for sc in ("mc", "ens", "knn"):
            m, se = mse(g[f"rho_{sc}"])
            rep(f"  Spearman(score, |error|), {sc}: {m:.3f} +- {se:.3f}")
        for sc in ("mc", "ens", "knn", "oracle"):
            m80, se80 = mse(g[f"rc_{sc}_80"])
            m50, se50 = mse(g[f"rc_{sc}_50"])
            rep(f"  RMSE kept/all at 80% coverage, {sc}: {m80:.3f} +- {se80:.3f}; at 50%: {m50:.3f} +- {se50:.3f}")
        rep("")
    rep.save()


if __name__ == "__main__":
    main()
