"""Does the MC dropout spread shrink as the training set grows?

Protein tertiary structure data (UCI, 9 inputs), published split 0: training
subsets of n = 500 ... 32,000 rows are drawn from the 41,157 training rows and
every model is evaluated on the same 4,573 test rows. Two replicates per n
(different subsets and seeds).

Every network has the architecture of the reference MC dropout experiments
for this data set (one hidden layer of 100 ReLU units, dropout before both
dense layers, weight decay l^2 (1 - p) / (2 N tau) with l = 0.01 and
tau = 0.05, the value the reference grid search picked on every protein
split). Each subset of size n is split 80/20: the networks are fitted to the
first 80% and the last 20% is used for early stopping (each network keeps the
weights with the lowest validation error among 100 checkpoints over
max(400 epochs, 20,000) Adam steps) and for tuning. Trained side by side, per
replicate and n:

- MC dropout with fixed dropout rates 0.005, 0.01, 0.05, 0.1 and 0.5;
- MC dropout with the rate tuned as in the reference protocol: of the four
  networks with rates in the reference grid {0.005, 0.01, 0.05, 0.1}, the
  (rate, tau) pair with the best validation MC log-likelihood over the
  reference tau grid {0.025, 0.05, 0.075} is the one reported;
- a bootstrap ensemble: 10 networks trained with dropout rate 0.05, each on a
  resample of the fitted rows, evaluated with every unit kept: how much the
  fitted regression function of a dropout network changes from one data set
  to the next, which is the uncertainty the MC spread is meant to express;
- a deep ensemble: 5 such networks on the fitted rows themselves, differing
  only in initialisation and minibatch order;
- an exact Gaussian-process posterior on the same fitted rows (n <= 16,000)
  with ARD squared-exponential kernel, hyperparameters fitted once per
  replicate on 1,500 rows of the training pool and then held fixed, so that
  the amount of data is the only thing that changes.

The quantity compared is the standard deviation of the regression function at
a test point (epistemic spread, without the noise term), averaged over the
test rows, in the units of the target (Angstrom of RMSD).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from mcd import (QUICK, RESULTS, Nets, Report, fit_gp_hypers, gal_weight_decay,
                 gp_posterior, load_uci, mc_loglik, steps_for)

NS = [500, 1000, 2000, 4000, 8000, 16000, 32000]
REPS = 2
GP_MAX = 16000
FIXED_P = [0.005, 0.01, 0.05, 0.1, 0.5]
GRID_P = [0.005, 0.01, 0.05, 0.1]
TAU = 0.05
TAU_GRID = [0.025, 0.05, 0.075]
N_BOOT, N_ENS = 10, 5
BASE_P = 0.05  # dropout rate of the bootstrap and ensemble networks
T = 300
MIN_STEPS = 20000
if QUICK:
    NS, REPS, GP_MAX, T, MIN_STEPS = [500, 1000], 1, 1000, 50, 1500


def main():
    import sys
    if "--summary" in sys.argv:
        df = pd.read_csv(RESULTS / "02_concentration.csv")
        summarize(df[df.rep < REPS])
        return
    D = load_uci("protein-tertiary-structure")
    X, y = D["X"], D["y"]
    pool, test = D["train"][0], D["test"][0]
    yt = y[test]
    rows = []
    for r in range(REPS):
        rng = np.random.default_rng(1000 + r)
        hyp = fit_gp_hypers(X[pool], y[pool], n_sub=1500, seed=r)
        print(f"rep {r}: GP hypers amp {hyp['amp']:.3f} noise {hyp['noise']:.4f} "
              f"ls {np.round(hyp['ls'], 2)}", flush=True)
        perm = rng.permutation(pool)
        for n in NS:
            sub = perm[:n]  # nested subsets within a replicate
            n_fit = int(0.8 * n)
            fit_rows, val = sub[:n_fit], sub[n_fit:]
            pools = ([fit_rows] * len(FIXED_P)
                     + [rng.choice(fit_rows, n_fit, replace=True) for _ in range(N_BOOT)]
                     + [fit_rows] * N_ENS)
            drop = np.r_[FIXED_P, np.full(N_BOOT + N_ENS, BASE_P)]
            reg = gal_weight_decay(drop, n_fit, TAU)
            nets = Nets(X.shape[1], [100], drop, reg, seed=100000 * r + n)
            steps = max(steps_for(n_fit, 400 if not QUICK else 4), MIN_STEPS)
            nets.fit(X, y, pools, steps=steps, val=val)

            k1 = len(FIXED_P)
            mc = nets.predict(X[test], T=T, members=range(k1))
            vs = nets.predict(X[val], T=T, members=[FIXED_P.index(p) for p in GRID_P])
            best_ll, best_p, best_tau = -np.inf, None, None
            for i, p in enumerate(GRID_P):
                for tau in TAU_GRID:
                    ll = mc_loglik(y[val], vs[i], tau).mean()
                    if ll > best_ll:
                        best_ll, best_p, best_tau = ll, p, tau
            det = nets.predict(X[test], mc=False, members=range(k1, k1 + N_BOOT + N_ENS))[:, 0, :]
            boot, ens = det[:N_BOOT], det[N_BOOT:]
            stop = nets.best_step / steps

            def add(method, sd, mean, extra=None):
                row = {"rep": r, "n": n, "method": method, "sd": float(np.mean(sd)),
                       "rmse": float(np.sqrt(np.mean((yt - mean) ** 2)))}
                row.update(extra or {})
                rows.append(row)

            for i, p in enumerate(FIXED_P):
                add(f"mc_{p:g}", mc[i].std(0), mc[i].mean(0))
            i = FIXED_P.index(best_p)
            add("mc_tuned", mc[i].std(0), mc[i].mean(0), {"p": best_p, "tau": best_tau})
            add("bootstrap", boot.std(0, ddof=1), boot.mean(0))
            add("ensemble", ens.std(0, ddof=1), ens.mean(0))
            if n <= GP_MAX:
                gm, gsd, _ = gp_posterior(X[fit_rows], y[fit_rows], X[test], hyp)
                add("gp", gsd, gm)
            last = {row["method"]: row for row in rows if row["rep"] == r and row["n"] == n}
            print(f"rep {r} n {n} steps {steps} (best checkpoint at {np.round(stop, 2)}): " + "; ".join(
                f"{m} sd {v['sd']:.3f} rmse {v['rmse']:.3f}" for m, v in last.items())
                + f"; tuned p {best_p} tau {best_tau}", flush=True)
            RESULTS.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(RESULTS / "02_concentration.csv", index=False, float_format="%.6g")

    summarize(pd.DataFrame(rows))


def summarize(df):
    rep = Report("02_concentration")
    rep("Protein (UCI), split 0: epistemic sd of the regression function, mean over 4,573 test rows")
    rep("n = subset size; networks and GP fitted to 80% of it, the other 20% used for early stopping and tuning")
    rep(f"{REPS} replicates per n; mean over replicates (min-max in brackets)")
    rep("")
    order = [f"mc_{p:g}" for p in FIXED_P] + ["mc_tuned", "bootstrap", "ensemble", "gp"]
    for m in order:
        g = df[df.method == m].groupby("n")
        rep(m + ":")
        for n, gg in g:
            line = (f"  n={n:6d}: sd {gg.sd.mean():.4f} [{gg.sd.min():.4f}, {gg.sd.max():.4f}]"
                    f"  rmse {gg.rmse.mean():.4f}")
            if m == "mc_tuned":
                line += "  chosen p " + ",".join(f"{v:g}" for v in gg.p) + "  tau " + ",".join(f"{v:g}" for v in gg.tau)
            rep(line)
        mean_sd = g.sd.mean()
        if len(mean_sd) > 2:
            ns = np.array(mean_sd.index, float)
            slope = np.polyfit(np.log(ns), np.log(mean_sd.values), 1)[0]
            rep(f"  log-log slope of sd on n: {slope:.3f}")
            big = ns >= 4000
            if big.sum() >= 2:
                s2 = np.polyfit(np.log(ns[big]), np.log(mean_sd.values[big]), 1)[0]
                rep(f"  slope for n >= 4000: {s2:.3f}")
    rep.save()


if __name__ == "__main__":
    main()
