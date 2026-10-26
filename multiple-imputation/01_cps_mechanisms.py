"""Experiment 1 - eleven ways to handle a missing covariate, four missingness mechanisms.

Population: the 28,155 men of the March 1988 CPS (no missing values).
Analysis model (Mincer): log wage ~ educ + exper + exper^2/100 + afam + smsa.
Each replication draws n = 1,000 men without replacement and deletes 40% of
`educ` under one of four mechanisms:

  MCAR         every value has the same chance of being deleted
  MAR-X        chance depends on (observed) experience
  MAR-Y        chance depends on (observed) log wage: low earners
  MNAR-X       chance depends on education itself: low education

Two populations are used. "real" keeps the observed log wages, so the Mincer
equation is only a linear approximation to the population regression. "model"
keeps the real covariates but regenerates log wage from the fitted Mincer
equation with normal errors, so the analysis model is correct by construction.
Targets are the population coefficients; standard errors are HC1.

Thirteen methods: complete data, complete cases, five single imputations (mean,
mean + indicator, regression, stochastic regression, kNN, random forest) and
five multiple imputations (Bayesian regression, PMM, PMM without the outcome,
PMM with wage interactions in the imputation model, random-forest MI).

Run:  python 01_cps_mechanisms.py --population real
      python 01_cps_mechanisms.py --population model
(about 13 minutes each on 2 cores; add `--reps 200` for a quick run)
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

from common import RES, load_cps, write
from mi import impute_one, mice, ols, pool

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=1000)
ap.add_argument("--population", choices=["real", "model"], default="real")
ARGS, _ = ap.parse_known_args()
POP = load_cps()
ANALYSIS = ["educ", "exper", "exper2", "afam", "smsa"]
AUX = ["parttime", "northeast", "midwest", "west"]
N, SHARE, GAMMA, M = 1000, 0.40, 1.5, 20
MECHS = ["MCAR", "MAR-X", "MAR-Y", "MNAR-X"]
METHODS = [
    "Complete data (no deletion)", "Complete cases", "Mean imputation",
    "Mean + missing indicator", "Regression imputation", "Stochastic regression (single)",
    "kNN imputation (k = 5)", "Random forest imputation", "MI, Bayesian regression",
    "MI, predictive mean matching", "MI, PMM without the outcome",
    "MI, PMM with interactions", "MI, random forest",
]
Xp = np.column_stack([np.ones(len(POP))] + [POP[c] for c in ANALYSIS])
_b, _ = ols(Xp, POP.logwage.values)
if ARGS.population == "model":
    _e = POP.logwage.values - Xp @ _b
    POP["logwage"] = Xp @ _b + np.random.default_rng(1988).normal(0, _e.std(ddof=6), len(POP))
    _b, _ = ols(Xp, POP.logwage.values)
TRUE = dict(zip(["const"] + ANALYSIS, _b))
TAG = ARGS.population


def z(v):
    return (v - v.mean()) / v.std()


def delete(df, mech, rng):
    if mech == "MCAR":
        p = np.full(len(df), SHARE)
    else:
        drv = {"MAR-X": z(df.exper.values), "MAR-Y": -z(df.logwage.values),
               "MNAR-X": -z(df.educ.values)}[mech]
        c = optimize.brentq(lambda c: expit(c + GAMMA * drv).mean() - SHARE, -10, 10)
        p = expit(c + GAMMA * drv)
    return rng.random(len(df)) < p


def fit(df_complete, educ, extra=None):
    """Analysis model with `educ` replaced.

    Returns (b_educ, se_educ, b_afam, se_afam, mean_educ, se_mean)."""
    cols = [np.ones(len(educ)), educ] + [df_complete[c].values for c in ANALYSIS[1:]]
    if extra is not None:
        cols.append(extra)
    b, v = ols(np.column_stack(cols), df_complete.logwage.values, robust=True)
    return (b[1], np.sqrt(v[1]), b[4], np.sqrt(v[4]),
            educ.mean(), educ.std(ddof=1) / np.sqrt(len(educ)))


COLS = ["mech", "rep", "method", "b_educ", "se_educ", "b_afam", "se_afam", "m_educ",
        "se_m", "rmse_imp", "df", "df_m"]


def one(args):
    mech, rep = args
    rng = np.random.default_rng([MECHS.index(mech), rep, 0 if TAG == "real" else 1])
    df = POP.sample(N, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    miss = delete(df, mech, rng)
    true_e = df.educ.values
    e = np.where(miss, np.nan, true_e)
    o = ~miss
    P = df[["logwage"] + ANALYSIS[1:] + AUX].values
    rows = []

    def rec(name, res, imputed=None, dfr=N - 6, dfm=N - 1):
        rmse = np.nan if imputed is None else np.sqrt(np.mean((imputed - true_e[miss]) ** 2))
        rows.append((mech, rep, name, *res, rmse, dfr, dfm))

    rec(METHODS[0], fit(df, true_e))
    rec(METHODS[1], fit(df[o], true_e[o]), dfr=o.sum() - 6, dfm=o.sum() - 1)
    mean_e = np.where(miss, np.nanmean(e), e)
    rec(METHODS[2], fit(df, mean_e), imputed=mean_e[miss])
    rec(METHODS[3], fit(df, mean_e, extra=miss.astype(float)), imputed=mean_e[miss], dfr=N - 7)
    X = np.column_stack([np.ones(N), P])
    b = np.linalg.lstsq(X[o], true_e[o], rcond=None)[0]
    reg_e = np.where(miss, X @ b, e)
    rec(METHODS[4], fit(df, reg_e), imputed=reg_e[miss])
    st = e.copy()
    st[miss] = impute_one(X[o], true_e[o], X[miss], "norm", rng)
    rec(METHODS[5], fit(df, st), imputed=st[miss])
    S = np.column_stack([e, P])
    mu, sd = np.nanmean(S, 0), np.nanstd(S, 0)
    knn = KNNImputer(n_neighbors=5).fit_transform((S - mu) / sd)[:, 0] * sd[0] + mu[0]
    knn_e = np.where(miss, knn, e)
    rec(METHODS[6], fit(df, knn_e), imputed=knn_e[miss])
    rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=rep,
                               n_jobs=1).fit(P[o], true_e[o])
    rf_e = np.where(miss, rf.predict(P), e)
    rec(METHODS[7], fit(df, rf_e), imputed=rf_e[miss])

    y, ex = df.logwage.values, df.exper.values
    rich = np.column_stack([P, y ** 2, y * ex / 10, y * ex ** 2 / 100, y * df.afam, y * df.smsa,
                            y * df.parttime, ex ** 3 / 1000])
    D = np.column_stack([e, P])          # column 0 = educ, 1 = logwage, ...
    Drich = np.column_stack([e, rich])
    for name, meth, preds, DD in ((METHODS[8], "norm", None, D), (METHODS[9], "pmm", None, D),
                                  (METHODS[10], "pmm", {0: list(range(2, D.shape[1]))}, D),
                                  (METHODS[11], "pmm", None, Drich), (METHODS[12], "rf", None, D)):
        imps = mice(DD, {0: meth}, predictors=preds, m=M, seed=[rep, MECHS.index(mech)])
        Q, U, rm = [], [], []
        for Zc in imps:
            r_ = fit(df, Zc[:, 0])
            Q.append([r_[0], r_[2], r_[4]]); U.append([r_[1] ** 2, r_[3] ** 2, r_[5] ** 2])
            rm.append(np.sqrt(np.mean((Zc[miss, 0] - true_e[miss]) ** 2)))
        pr = pool(Q, U, N - 6)
        pm = pool(np.array(Q)[:, 2:], np.array(U)[:, 2:], N - 1)
        rows.append((mech, rep, name, pr.est[0], pr.se[0], pr.est[1], pr.se[1],
                     pm.est[0], pm.se[0], float(np.mean(rm)), float(pr.df[0]), float(pm.df[0])))
    return rows


def summarise(d: pd.DataFrame) -> str:
    out = [f"Population '{TAG}' (N = {len(POP)}): educ {TRUE['educ']:.4f}, afam {TRUE['afam']:.4f}, "
           f"exper {TRUE['exper']:.4f}, exper2/100 {TRUE['exper2']:.4f}, smsa {TRUE['smsa']:.4f}; "
           f"mean educ {POP.educ.mean():.3f}",
           f"n = {N} per sample, {SHARE:.0%} of educ deleted, logistic slope {GAMMA} on the "
           f"standardised driver, m = {M} imputations, {d.rep.nunique()} replications per "
           f"mechanism; Monte Carlo SE of a 95% coverage ~ {np.sqrt(.95*.05/d.rep.nunique()):.3f}"]
    for par, tv, b, s, dfc in (("educ", TRUE["educ"], "b_educ", "se_educ", "df"),
                               ("afam", TRUE["afam"], "b_afam", "se_afam", "df"),
                               ("mean educ", POP.educ.mean(), "m_educ", "se_m", "df_m")):
        tq = stats.t.ppf(0.975, d[dfc].clip(upper=1e6))
        d["_cov"] = np.abs(d[b] - tv) <= tq * d[s]
        d["_w"] = 2 * tq * d[s]
        d["_se2"] = (d[b] - tv) ** 2
        g = d.groupby(["mech", "method"], sort=False)
        tab = pd.DataFrame({"mean": g[b].mean(), "empsd": g[b].std(), "meanse": g[s].mean(),
                            "cover": g["_cov"].mean(), "rmse_est": np.sqrt(g["_se2"].mean()),
                            "width": g["_w"].mean(), "rmse_imp": g["rmse_imp"].mean()})
        tab["bias"] = tab["mean"] - tv
        tab["relbias"] = tab.bias / abs(tv)
        out.append(f"\n=== {'coefficient on ' if par != 'mean educ' else ''}{par} "
                   f"(population {tv:.4f}) ===")
        for mech in MECHS:
            out.append(f"-- {mech}")
            for mth in METHODS:
                r = tab.loc[(mech, mth)]
                out.append(f"  {mth:32s} mean {r['mean']:.4f}  bias {r.bias:+.4f} "
                           f"({r.relbias:+.1%})  empSD {r.empsd:.4f}  meanSE {r.meanse:.4f}  "
                           f"SE/SD {r.meanse / r.empsd:.2f}  cover {r.cover:.3f}  "
                           f"RMSE {r.rmse_est:.4f}  CIwidth {r.width:.4f}"
                           + (f"  imputationRMSE {r.rmse_imp:.3f}" if par == "educ"
                              and not np.isnan(r.rmse_imp) else ""))
        tab.to_csv(RES / f"01_summary_{TAG}_{par.replace(' ', '_')}.csv")
    return "\n".join(out)


if __name__ == "__main__":
    tasks = [(mech, r) for mech in MECHS for r in range(ARGS.reps)]
    with Pool(2) as p:
        rows = [row for chunk in p.imap_unordered(one, tasks, chunksize=10) for row in chunk]
    d = pd.DataFrame(rows, columns=COLS).sort_values(["mech", "rep", "method"])
    d.to_csv(RES / f"01_replications_{TAG}.csv.gz", index=False)
    write(f"01_cps_mechanisms_{TAG}.txt", summarise(d))
