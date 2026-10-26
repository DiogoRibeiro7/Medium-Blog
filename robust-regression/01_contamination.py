"""Experiment 1 - thirteen estimators under realistic data-entry errors.

Population: the 53,837 diamonds of ggplot2::diamonds whose implied density
(carat per cubic mm) lies between 0.0055 and 0.0068, with their real carat,
cut, colour and clarity. Log price is regenerated from the regression of real log price on
log carat and the three grades, with normal errors, so the regression model
is exactly right and its coefficients are known.

Each replication draws n = 500 diamonds and corrupts a share eps of them:

  price      a price typed with an extra zero: log price + log(10)
  carat      a carat with a slipped decimal point: log carat + log(10),
             price unchanged (a bad leverage point)
  tails      no corruption; errors from a t distribution with nu degrees of
             freedom, scaled to the same interquartile range as the normal
             errors (nu = 1 is the Cauchy distribution)

Run:  python 01_contamination.py      (about 40 minutes on 2 cores;
      --reps 50 for a quick run)
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

from common import RES, load_diamonds, write
from rob import flag_outliers, lts, m_estimate, mm_estimate, ols

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=300)
ARGS, _ = ap.parse_known_args()

D = load_diamonds()
D = D[D.density.between(0.0055, 0.0068)].reset_index(drop=True)
COLS = ["lc", "cut_n", "col_n", "cla_n"]
XP = np.column_stack([np.ones(len(D))] + [D[c] for c in COLS])
_b = np.linalg.lstsq(XP, D.lp.values, rcond=None)[0]
SIGMA = (D.lp.values - XP @ _b).std(ddof=5)
BETA = _b
N = 500
RATES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40)
NUS = (5, 3, 2, 1)
CELLS = [("clean", 0.0)] + [("price", r) for r in RATES] + [("carat", r) for r in RATES] + \
        [("tails", float(v)) for v in NUS]
METHODS = ["OLS", "OLS, drop |t| > 3", "OLS, drop |t| > 3 repeatedly", "OLS, drop Cook's D > 4/n",
           "LAD", "Huber M (statsmodels)", "Huber (scikit-learn)", "Bisquare M from OLS",
           "Theil-Sen", "RANSAC", "LTS", "S", "MM"]
SLOPE = 1  # coefficient on log carat


def cooks_and_t(X, y):
    n, p = X.shape
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    h = np.diag(H)
    e = y - H @ y
    s2 = e @ e / (n - p)
    r = e / np.sqrt(s2 * (1 - h))
    t = r * np.sqrt((n - p - 1) / (n - p - r ** 2))
    cook = r ** 2 * h / (p * (1 - h))
    return t, cook


def one(args):
    ci, rep = args
    kind, level = CELLS[ci]
    rng = np.random.default_rng([ci, rep])
    idx = rng.choice(len(D), N, replace=False)
    X = XP[idx].copy()
    if kind == "tails":
        nu = level
        # same interquartile range as the normal errors (variance is infinite for nu <= 2)
        eps = rng.standard_t(nu, N) * SIGMA * stats.norm.ppf(0.75) / stats.t.ppf(0.75, nu)
    else:
        eps = rng.normal(0, SIGMA, N)
    y = X @ BETA + eps
    bad = np.zeros(N, bool)
    if kind in ("price", "carat"):
        k = int(round(level * N))
        bad[rng.choice(N, k, replace=False)] = True
        if kind == "price":
            y[bad] += np.log(10)
        else:
            X[bad, 1] += np.log(10)
    rows = []

    def rec(name, coef, se=np.nan, flags=None):
        tp = np.nan if flags is None or not bad.any() else (flags & bad).sum() / bad.sum()
        fp = np.nan if flags is None else (flags & ~bad).sum() / (~bad).sum()
        rows.append((kind, level, rep, name, coef[SLOPE], se, tp, fp))

    o = ols(X, y)
    rec("OLS", o.coef, o.se[SLOPE])
    t, cook = cooks_and_t(X, y)
    keep = np.abs(t) <= 3
    f = ols(X[keep], y[keep]); rec("OLS, drop |t| > 3", f.coef, f.se[SLOPE], ~keep)
    keep = np.ones(N, bool)
    for _ in range(50):
        tt, _c = cooks_and_t(X[keep], y[keep])
        out = np.abs(tt) > 3
        if not out.any():
            break
        keep[np.flatnonzero(keep)[out]] = False
    f = ols(X[keep], y[keep]); rec("OLS, drop |t| > 3 repeatedly", f.coef, f.se[SLOPE], ~keep)
    keep = cook <= 4 / N
    f = ols(X[keep], y[keep]); rec("OLS, drop Cook's D > 4/n", f.coef, f.se[SLOPE], ~keep)
    q = sm.QuantReg(y, X).fit(q=0.5)
    rec("LAD", q.params, q.bse[SLOPE])
    h = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    rec("Huber M (statsmodels)", h.params, h.bse[SLOPE])
    hs = HuberRegressor(max_iter=1000).fit(X[:, 1:], y)
    rec("Huber (scikit-learn)", np.r_[hs.intercept_, hs.coef_])
    bq = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight()).fit()
    rec("Bisquare M from OLS", bq.params, bq.bse[SLOPE])
    ts = TheilSenRegressor(max_subpopulation=1000, random_state=rep).fit(X[:, 1:], y)
    rec("Theil-Sen", np.r_[ts.intercept_, ts.coef_])
    rs = RANSACRegressor(random_state=rep).fit(X[:, 1:], y)
    rec("RANSAC", np.r_[rs.estimator_.intercept_, rs.estimator_.coef_], flags=~rs.inlier_mask_)
    lt = lts(X, y, n_sub=500, seed=rep)
    rec("LTS", lt.coef, flags=flag_outliers(lt, 3))
    mm = mm_estimate(X, y, n_sub=500, seed=rep)
    rec("S", mm.init.coef, flags=flag_outliers(mm.init, 3))
    rec("MM", mm.coef, mm.se[SLOPE], flag_outliers(mm, 3))
    return rows


def summarise(d):
    tv = BETA[SLOPE]
    q = stats.norm.ppf(0.975)
    d["cover"] = np.where(d.se.notna(), np.abs(d.coef - tv) <= q * d.se, np.nan)
    d["err2"] = (d.coef - tv) ** 2
    g = d.groupby(["kind", "level", "method"], sort=False)
    t = pd.DataFrame({"mean": g.coef.mean(), "sd": g.coef.std(), "rmse": np.sqrt(g.err2.mean()),
                      "cover": g.cover.mean(), "meanse": g.se.mean(), "tp": g.tp.mean(),
                      "fp": g.fp.mean(), "median": g.coef.median()})
    t["bias"] = t["mean"] - tv
    t["relbias"] = t.bias / tv
    t.to_csv(RES / "01_summary.csv")
    out = [f"Population: {len(D)} diamonds with plausible dimensions; log price regenerated from "
           f"the fitted model, sigma = {SIGMA:.4f}; coefficients " +
           ", ".join(f"{k} {v:.4f}" for k, v in zip(["const"] + COLS, BETA)),
           f"n = {N} per sample, {d.rep.nunique()} replications per cell; target: coefficient on "
           f"log carat = {tv:.4f}; Monte Carlo SE of a 95% coverage ~ "
           f"{np.sqrt(.95 * .05 / d.rep.nunique()):.3f}"]
    for (kind, level) in CELLS:
        lab = {"clean": "clean data", "price": f"{level:.0%} prices x10",
               "carat": f"{level:.0%} carats x10", "tails": f"t errors, nu = {level:g}"}[kind]
        out.append(f"-- {lab}")
        for m in METHODS:
            r = t.loc[(kind, level, m)]
            out.append(f"  {m:30s} mean {r['mean']:.4f} bias {r.bias:+.4f} ({r.relbias:+.1%}) "
                       f"median {r['median']:.4f} SD {r.sd:.4f} RMSE {r.rmse:.4f}"
                       + (f" cover {r.cover:.3f} meanSE {r.meanse:.4f}" if not np.isnan(r.cover) else "")
                       + (f" flagged-bad {r.tp:.3f}" if not np.isnan(r.tp) else "")
                       + (f" flagged-clean {r.fp:.4f}" if not np.isnan(r.fp) else ""))
    # efficiency relative to OLS
    out.append("\nRMSE relative to OLS (clean and t errors)")
    for (kind, level) in [c for c in CELLS if c[0] in ("clean", "tails")]:
        base = t.loc[(kind, level, "OLS")].rmse
        out.append(f"  {kind} {level:g}: " + ", ".join(
            f"{m} {t.loc[(kind, level, m)].rmse / base:.2f}" for m in METHODS[4:]))
    return "\n".join(out)


if __name__ == "__main__":
    tasks = [(ci, r) for ci in range(len(CELLS)) for r in range(ARGS.reps)]
    with Pool(2) as p:
        rows = [row for chunk in p.imap_unordered(one, tasks, chunksize=5) for row in chunk]
    d = pd.DataFrame(rows, columns=["kind", "level", "rep", "method", "coef", "se", "tp", "fp"])
    d = d.sort_values(["kind", "level", "rep", "method"]).reset_index(drop=True)
    d.to_csv(RES / "01_replications.csv.gz", index=False)
    write("01_contamination.txt", summarise(d))
