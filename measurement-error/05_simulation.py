"""Experiment 5 - nine corrections against six kinds of measurement error.

Population: NHANES 2009-2012 adults aged 20-79 with a complete blood-pressure
average, age, sex and BMI. The "true" exposure x is NHANES's own systolic
average (BPSysAve, in units of 10 mmHg); age, sex and BMI are the covariates
Z. Outcomes are generated from known models, so the target coefficient is
known:

  linear    y = 2 + 0.5 x + 0.02 age - 0.3 female + 0.05 bmi + N(0, 1)
  logistic  logit P(y = 1) = b0 + 0.3 x + 0.03 age - 0.2 female + 0.03 bmi
            (b0 set for a prevalence of about 20%)

Each replication draws n = 1000 people and gives every one two error-prone
readings w1, w2 of x and, for a random 15%, the true x (an internal
validation subsample). The error variance is set by the reliability
lambda = var(x) / (var(x) + var(u)) in {0.9, 0.7, 0.5}. Error structures:

  classical       w_j = x + u_j, u_j independent N(0, s2)
  correlated      the two errors share half their variance (as readings taken
                  minutes apart do), so the replicates understate the error
  heteroscedastic error SD proportional to x (average variance s2)
  berkson         x = w + u: w is an assigned value, the truth scatters around it
                  (both readings equal w)
  systematic      w_j = m + 0.8 (x - m) + u_j: reports flattened toward the mean
  differential    the error is correlated (0.3) with the outcome's own error,
                  shared by both readings

Run:  python 05_simulation.py            (about 35 minutes on 2 cores)
      python 05_simulation.py --reps 30  (quick run)
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import RES, load_nhanes, write
from me import (bootstrap_se, design, extrapolate, fit_model, iv_replicate, mime, ols,
                rc_replicates, rc_validation, simex)

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=300)
ARGS, _ = ap.parse_known_args()

P = load_nhanes().dropna(subset=["sbp_ave", "age", "bmi"]).reset_index(drop=True)
XPOP = P.sbp_ave.values / 10
ZPOP = P[["age", "female", "bmi"]].values
VX = XPOP.var()
N, VAL_SHARE, BOOT, SIMEX_BOOT, SIMEX_B = 1000, 0.15, 100, 50, 20
BETA_LIN = np.array([2.0, 0.5, 0.02, -0.3, 0.05])
BETA_LOG = np.array([0.0, 0.3, 0.03, -0.2, 0.03])
eta0 = design(XPOP, ZPOP) @ BETA_LOG
lo, hi = -15.0, 5.0
for _ in range(60):                                    # intercept for 20% prevalence
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if np.mean(1 / (1 + np.exp(-(eta0 + mid)))) < 0.2 else (lo, mid)
BETA_LOG[0] = (lo + hi) / 2
RELS = (0.9, 0.7, 0.5)
STRUCTS = ("classical", "correlated", "heteroscedastic", "berkson", "systematic", "differential")
CELLS = [("gaussian", s, r) for s in STRUCTS for r in RELS] + [("binomial", "classical", r) for r in RELS]
METHODS = ["Naive, one reading", "Naive, mean of two", "Regression calibration, replicates",
           "SIMEX, quadratic", "SIMEX, rational", "IV, second reading",
           "Regression calibration, validation", "Multiple imputation, validation",
           "Validation rows only"]


def make_data(rng, struct, rel, family):
    idx = rng.choice(len(XPOP), N, replace=False)
    x, Z = XPOP[idx].copy(), ZPOP[idx]
    s2 = VX * (1 - rel) / rel
    e_out = rng.standard_normal(N)
    if struct == "classical":
        W = x[:, None] + np.sqrt(s2) * rng.standard_normal((N, 2))
    elif struct == "correlated":
        shared = rng.standard_normal(N)[:, None]
        W = x[:, None] + np.sqrt(s2) * (np.sqrt(0.5) * shared + np.sqrt(0.5) * rng.standard_normal((N, 2)))
    elif struct == "heteroscedastic":
        scale = x / np.sqrt(np.mean(XPOP ** 2))
        W = x[:, None] + np.sqrt(s2) * scale[:, None] * rng.standard_normal((N, 2))
    elif struct == "berkson":
        W = np.repeat(x[:, None], 2, axis=1)
        x = x + np.sqrt(s2) * rng.standard_normal(N)
    elif struct == "systematic":
        m = XPOP.mean()
        W = (m + 0.8 * (x - m))[:, None] + np.sqrt(s2) * rng.standard_normal((N, 2))
    elif struct == "differential":
        shared = 0.3 * e_out + np.sqrt(1 - 0.3 ** 2) * rng.standard_normal(N)
        # half the error variance is shared by both readings and carries the correlation
        W = x[:, None] + np.sqrt(s2) * (np.sqrt(0.5) * shared[:, None]
                                        + np.sqrt(0.5) * rng.standard_normal((N, 2)))
    D = design(x, Z)
    if family == "gaussian":
        y = D @ BETA_LIN + e_out
    else:
        p = 1 / (1 + np.exp(-(D @ BETA_LOG)))
        y = (rng.random(N) < p).astype(float)
    val = rng.random(N) < VAL_SHARE
    return x, Z, W, y, val


def one(task):
    ci, rep = task
    family, struct, rel = CELLS[ci]
    rng = np.random.default_rng([2, ci, rep])
    x, Z, W, y, val = make_data(rng, struct, rel, family)
    s_u2 = np.mean(np.var(W, axis=1, ddof=1))
    wbar = W.mean(axis=1)
    out = []

    def rec(name, coef, se=None):
        out.append((family, struct, rel, rep, name, coef[1], np.nan if se is None else se[1], coef[2],
                    np.nan if se is None else se[2]))

    f = fit_model(design(W[:, 0], Z), y, family); rec(METHODS[0], f.coef, f.se)
    f = fit_model(design(wbar, Z), y, family); rec(METHODS[1], f.coef, f.se)
    # regression calibration from the replicates, bootstrap SE
    rc = rc_replicates(W, Z, y, family).coef
    se = bootstrap_se(lambda i: rc_replicates(W[i], Z[i], y[i], family).coef, N, BOOT, rng)
    rec(METHODS[2], rc, se)
    # SIMEX on the mean of the two readings, error variance s_u2 / 2
    sx, lam, path = simex(wbar, Z, y, s_u2 / 2, family, B=SIMEX_B, rng=rng, return_path=True)
    q, r = extrapolate(lam, path, "quadratic"), extrapolate(lam, path, "rational")
    if family == "gaussian":
        bq, br = [], []
        for _ in range(SIMEX_BOOT):
            i = rng.integers(0, N, N)
            s_i = np.mean(np.var(W[i], axis=1, ddof=1))
            _, lam_i, path_i = simex(wbar[i], Z[i], y[i], s_i / 2, family, B=SIMEX_B, rng=rng,
                                     return_path=True)
            bq.append(extrapolate(lam_i, path_i, "quadratic"))
            br.append(extrapolate(lam_i, path_i, "rational"))
        rec(METHODS[3], q, np.std(bq, axis=0, ddof=1))
        rec(METHODS[4], r, np.std(br, axis=0, ddof=1))
        f = iv_replicate(W, Z, y); rec(METHODS[5], f.coef, f.se)
    else:
        rec(METHODS[3], q); rec(METHODS[4], r)
    rcv = rc_validation(W, Z, y, x, val, family).coef
    se = bootstrap_se(lambda i: rc_validation(W[i], Z[i], y[i], x[i], val[i], family).coef, N, BOOT, rng)
    rec(METHODS[6], rcv, se)
    f = mime(W, Z, y, x, val, family, m=20, rng=rng); rec(METHODS[7], f.coef, f.se)
    f = fit_model(design(x[val], Z[val]), y[val], family); rec(METHODS[8], f.coef, f.se)

    # prediction: deploy the naive and the corrected model on a fresh sample,
    # measured with one reading (as in training) or measured exactly
    pred = []
    if family == "gaussian":
        xt, Zt, Wt, yt, _ = make_data(rng, struct, rel, family)
        naive = ols(design(W[:, 0], Z), y, se=False).coef
        corr = rc
        for model, b in (("naive", naive), ("corrected", corr)):
            for meas, xx in (("one reading", Wt[:, 0]), ("exact", xt)):
                yh = design(xx, Zt) @ b
                c = np.polyfit(yh, yt, 1)[0]
                pred.append((struct, rel, rep, model, meas, c, np.mean((yt - yh) ** 2)))
    return out, pred


def summarise(d):
    truth = {"gaussian": BETA_LIN, "binomial": BETA_LOG}
    d["truth"] = [truth[f][1] for f in d.family]
    d["truth_age"] = [truth[f][2] for f in d.family]
    d["err"] = d.est - d.truth
    d["cover"] = np.where(d.se.notna(), (np.abs(d.err) <= 1.959964 * d.se).astype(float), np.nan)
    g = d.groupby(["family", "struct", "rel", "method"], sort=False)
    t = pd.DataFrame({"mean": g.est.mean(), "median": g.est.median(), "sd": g.est.std(),
                      "rmse": np.sqrt(g.err.apply(lambda e: np.mean(e ** 2))),
                      "cover": g.cover.mean(), "meanse": g.se.mean(),
                      "age_mean": g.est_age.mean(), "truth": g.truth.first(),
                      "truth_age": g.truth_age.first()})
    t["relbias"] = t["mean"] / t.truth - 1
    t["age_relbias"] = t.age_mean / t.truth_age - 1
    t.to_csv(RES / "05_summary.csv")
    out = [f"Population: {len(XPOP)} NHANES adults; x = BPSysAve / 10, var(x) = {VX:.3f}; "
           f"n = {N} per sample, {VAL_SHARE:.0%} validated, {d.rep.nunique()} replications per cell",
           f"linear truth: x {BETA_LIN[1]}, age {BETA_LIN[2]}; logistic truth: x {BETA_LOG[1]}, "
           f"age {BETA_LOG[2]}, intercept {BETA_LOG[0]:.3f}; Monte Carlo SE of a 95% coverage "
           f"~ {np.sqrt(.95 * .05 / d.rep.nunique()):.3f}"]
    for (fam, st, rel), _ in d.groupby(["family", "struct", "rel"], sort=False):
        out.append(f"-- {fam}, {st}, reliability {rel}")
        for m in METHODS:
            if (fam, st, rel, m) not in t.index:
                continue
            r = t.loc[(fam, st, rel, m)]
            out.append(f"  {m:36s} mean {r['mean']:.4f} bias {r.relbias:+.1%} SD {r.sd:.4f} "
                       f"RMSE {r.rmse:.4f}" + (f" cover {r.cover:.3f} meanSE {r.meanse:.4f}"
                                               if not np.isnan(r.cover) else "")
                       + f" | age coef bias {r.age_relbias:+.1%}")
    return "\n".join(out)


def summarise_pred(p):
    g = p.groupby(["struct", "rel", "model", "measured"], sort=False)
    t = pd.DataFrame({"slope": g.slope.mean(), "mse": g.mse.mean()})
    t.to_csv(RES / "05_prediction.csv")
    out = ["\nPrediction on a fresh sample (linear outcome): mean calibration slope and MSE"]
    for (st, rel), s in t.groupby(level=[0, 1], sort=False):
        out.append(f"  {st}, reliability {rel}: " + "; ".join(
            f"{m} model on {me} x: slope {r.slope:.3f}, MSE {r.mse:.3f}"
            for (_, _, m, me), r in s.iterrows()))
    return "\n".join(out)


if __name__ == "__main__":
    tasks = [(ci, r) for ci in range(len(CELLS)) for r in range(ARGS.reps)]
    with Pool(2) as pool:
        res = pool.map(one, tasks, chunksize=4)
    d = pd.DataFrame([row for o, _ in res for row in o],
                     columns=["family", "struct", "rel", "rep", "method", "est", "se", "est_age", "se_age"])
    p = pd.DataFrame([row for _, pr in res for row in pr],
                     columns=["struct", "rel", "rep", "model", "measured", "slope", "mse"])
    d.to_csv(RES / "05_replications.csv.gz", index=False)
    write("05_simulation.txt", summarise(d) + "\n" + summarise_pred(p))
