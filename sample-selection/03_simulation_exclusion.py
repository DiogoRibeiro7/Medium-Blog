"""Experiment 3 - what an exclusion restriction buys, and what normality alone buys.

Model: selection d = 1[a + 0.5 x + g z + v > 0]; outcome y = 1 + 0.5 x + u,
observed when d = 1; x and z independent N(0, 1); corr(u, v) = rho; var(u) =
var(v) = 1 (on average, for the heteroscedastic design). The target is the
slope 0.5.

Grid: four error distributions for (u, v),
  normal         bivariate normal (every Heckman assumption holds)
  t5             bivariate Student t with 5 degrees of freedom (heavy tails)
  skewed         v and the independent part of u standardised chi-square(3)
                 (right-skewed), u = rho v + sqrt(1 - rho^2) e
  heteroscedastic  bivariate normal, but the selection error's SD is
                 proportional to exp(0.5 x) (the probit is misspecified)
x rho in {0.5, 0.8} x exclusion strength g in {0, 0.25, 0.5, 1} x selected
share in {0.5, 0.85} (the intercept a is set by simulation), n = 1,000,
500 samples per cell.

Estimators of the slope: least squares on the selected; Heckman two-step,
ML and a series control function (cubic in the inverse Mills ratio) with z in
the selection equation; Heckman two-step and ML without z (selection on x
only: identification by functional form). ML is started from the two-step
estimates and from rho = -0.5 and 0.5; the highest maximum is kept.
Recorded: bias, SD, RMSE, median absolute error, coverage of 95% intervals
(two-step corrected and ML standard errors), how often ML reaches |rho| > 0.95,
how often the two-step start ends at a lower maximum, and the R^2 of the inverse
Mills ratio on (1, x) among the selected.

Run:  python 03_simulation_exclusion.py            (about 15 minutes on 2 cores)
      python 03_simulation_exclusion.py --reps 50  (quick run)
      python 03_simulation_exclusion.py --figures-only
"""

from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy import stats

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, YELLOW, style, write
from sel import heckit, heckman_ml, series_cf

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--figures-only", action="store_true")
ARGS, _ = ap.parse_known_args()

DISTS = ("normal", "t5", "skewed", "heteroscedastic")
RHOS = (0.5, 0.8)
GS = (0.0, 0.25, 0.5, 1.0)
SHARES = (0.5, 0.85)
N = 1000
EST = ["OLS on selected", "Two-step with z", "ML with z", "Series with z", "Two-step without z", "ML without z"]


def draw_errors(dist, rho, n, x, rng):
    if dist == "skewed":
        e1 = (rng.chisquare(3, n) - 3) / np.sqrt(6)
        e2 = (rng.chisquare(3, n) - 3) / np.sqrt(6)
        return rho * e1 + np.sqrt(1 - rho ** 2) * e2, e1
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    u, v = a, rho * a + np.sqrt(1 - rho ** 2) * b
    if dist == "t5":
        w = np.sqrt(rng.chisquare(5, n) / 5) / np.sqrt(3 / 5)
        u, v = u / w, v / w
    if dist == "heteroscedastic":
        v = v * np.exp(0.5 * x) / np.sqrt(np.exp(0.5))     # E exp(x) = exp(0.5) for x ~ N(0, 1)
    return u, v


def intercept(dist, g, share):
    """Intercept giving the target selected share, from 400,000 draws."""
    rng = np.random.default_rng(12345)
    m = 400_000
    x, z = rng.standard_normal(m), rng.standard_normal(m)
    _, v = draw_errors(dist, 0.5, m, x, rng)
    idx = 0.5 * x + g * z + v
    return -np.quantile(idx, 1 - share)


A = {(dd, g, s): intercept(dd, g, s) for dd in DISTS for g in GS for s in SHARES}


def best_ml(Z, d, X, y):
    f0 = heckman_ml(Z, d, X, y)
    best, lower = f0, False
    for r0 in (-0.5, 0.5):
        st = f0.theta.copy()
        st[-1] = np.arctanh(r0)
        f = heckman_ml(Z, d, X, y, start=st)
        if np.isfinite(f.loglik) and f.loglik > best.loglik + 1e-3:
            best, lower = f, True
    return best, lower


def one(task):
    dist, rho, g, share, rep = task
    rng = np.random.default_rng([31, DISTS.index(dist), RHOS.index(rho), GS.index(g), SHARES.index(share), rep])
    x, z = rng.standard_normal(N), rng.standard_normal(N)
    u, v = draw_errors(dist, rho, N, x, rng)
    d = (A[(dist, g, share)] + 0.5 * x + g * z + v) > 0
    y = 1 + 0.5 * x + u
    X = np.column_stack([np.ones(N), x])
    Zz = np.column_stack([np.ones(N), x, z])
    Z0 = X
    tc = stats.norm.ppf(0.975)
    rows = {}
    b = np.linalg.lstsq(X[d], y[d], rcond=None)[0]
    e = y[d] - X[d] @ b
    se = np.sqrt(e @ e / (d.sum() - 2) * np.linalg.inv(X[d].T @ X[d])[1, 1])
    rows["OLS on selected"] = (b[1], se, np.nan, False)
    for lab, Z in (("with z", Zz), ("without z", Z0)):
        h = heckit(Z, d, X, y)
        rows[f"Two-step {lab}"] = (h.beta[1], h.se()[1], h.rho, False)
        ml, lower = best_ml(Z, d, X, y)
        rows[f"ML {lab}"] = (ml.beta[1], ml.se()[Z.shape[1] + 1], ml.rho, lower)
        if lab == "without z":
            lam = h.lam
            fit = X[d] @ np.linalg.lstsq(X[d], lam, rcond=None)[0]
            r2 = 1 - np.sum((lam - fit) ** 2) / np.sum((lam - lam.mean()) ** 2)
        else:
            lam = h.lam
            fit = X[d] @ np.linalg.lstsq(X[d], lam, rcond=None)[0]
            r2z = 1 - np.sum((lam - fit) ** 2) / np.sum((lam - lam.mean()) ** 2)
    rows["Series with z"] = (series_cf(Zz, d, X, y, degree=3)[1], np.nan, np.nan, False)
    return [(dist, rho, g, share, rep, est, *rows[est], r2z, r2, d.mean(), tc) for est in EST]


def summarise(df, reps):
    out = [f"Slope (truth 0.5) by cell, {reps} samples of n = {N}; Monte Carlo SE of a 95% coverage ~ "
           f"{np.sqrt(.95 * .05 / reps):.4f}"]
    rows = []
    for (dist, rho, g, share), c in df.groupby(["dist", "rho", "g", "share"], sort=False):
        ols_rmse = np.sqrt(np.mean((c[c.est == "OLS on selected"].b - 0.5) ** 2))
        out.append(f"-- {dist}, rho {rho}, exclusion strength {g}, selected share {share} (realised "
                   f"{c.share_real.mean():.3f}); median R^2 of the inverse Mills ratio on (1, x): with z "
                   f"{c.r2z.median():.3f}, without z {c.r2.median():.3f}")
        for est in EST:
            t = c[c.est == est]
            err = t.b - 0.5
            rmse = np.sqrt(np.mean(err ** 2))
            cov = np.mean(np.abs(err) <= 1.96 * t.se) if t.se.notna().all() else np.nan
            r = {"dist": dist, "rho": rho, "g": g, "share": share, "est": est, "bias": err.mean(), "sd": t.b.std(ddof=1),
                 "rmse": rmse, "rmse_ratio": rmse / ols_rmse, "mae": np.median(np.abs(err)), "coverage": cov,
                 "rho_boundary": np.mean(np.abs(t.rho_hat) > 0.95) if est.startswith("ML") else np.nan,
                 "lower_max": t.lower.mean() if est.startswith("ML") else np.nan,
                 "r2z": c.r2z.median(), "r2": c.r2.median()}
            rows.append(r)
            extra = (f"; |rho| > 0.95 in {r['rho_boundary']:.3f}; two-step start at a lower maximum in "
                     f"{r['lower_max']:.3f}" if est.startswith("ML") else "")
            out.append(f"   {est:19s} bias {r['bias']:+.4f}; SD {r['sd']:.4f}; RMSE {rmse:.4f} ({r['rmse_ratio']:.3f} x OLS); "
                       f"median |error| {r['mae']:.4f}; coverage {cov:.3f}{extra}")
    S = pd.DataFrame(rows)
    S.to_csv(RES / "03_simulation_summary.csv", index=False)
    return "\n".join(out), S


def digest(S):
    out = ["Digest (RMSE relative to least squares on the selected, by error distribution)"]
    for dist in DISTS:
        t = S[S.dist == dist]
        for est in EST[1:]:
            u = t[t.est == est]
            g0, g1 = u[u.g == 0], u[u.g == 1.0]
            out.append(f"   {dist:15s} {est:19s} RMSE ratio: exclusion strength 0 {g0.rmse_ratio.min():.3f}-"
                       f"{g0.rmse_ratio.max():.3f}; strength 1 {g1.rmse_ratio.min():.3f}-{g1.rmse_ratio.max():.3f}; "
                       f"all strengths {u.rmse_ratio.min():.3f}-{u.rmse_ratio.max():.3f}; "
                       f"worse than OLS in {np.sum(u.rmse_ratio > 1)} of {len(u)} cells; bias {u.bias.min():+.4f} to "
                       f"{u.bias.max():+.4f}; coverage {u.coverage.min():.3f}-{u.coverage.max():.3f} over the "
                       f"{u.coverage.notna().sum()} cells with a standard error in every sample")
    out.append("All 64 designs: cells in which each estimator's RMSE exceeds least squares on the selected")
    for est in EST[1:]:
        u = S[S.est == est]
        line = f"   {est:19s} {np.sum(u.rmse_ratio > 1)} of {len(u)}"
        for g in GS:
            w = u[u.g == g]
            line += f"; strength {g:g}: {np.sum(w.rmse_ratio > 1)} of {len(w)}"
        out.append(line)
    for est in ("Two-step with z", "Series with z", "ML with z"):
        u = S[(S.est == est) & (S.g >= 0.5) & (S.dist != "heteroscedastic")]
        out.append(f"{est}, exclusion strength 0.5 or 1, errors normal, t5 or skewed: RMSE ratio {u.rmse_ratio.min():.3f}-"
                   f"{u.rmse_ratio.max():.3f}; bias {u.bias.min():+.4f} to {u.bias.max():+.4f}")
    u = S[(S.est == "Two-step with z") & (S.g == 0.25) & (S.dist != "heteroscedastic")]
    out.append(f"Two-step with z, exclusion strength 0.25, errors normal, t5 or skewed: RMSE ratio {u.rmse_ratio.min():.3f}-"
               f"{u.rmse_ratio.max():.3f}; worse than OLS in {np.sum(u.rmse_ratio > 1)} of {len(u)} cells")
    for dist in DISTS:
        u = S[(S.dist == dist) & S.est.str.startswith("ML")]
        out.append(f"ML ({dist}): |rho| > 0.95 in {u.rho_boundary.min():.3f}-{u.rho_boundary.max():.3f} of samples per "
                   f"cell; two-step start at a lower maximum in {u.lower_max.min():.3f}-{u.lower_max.max():.3f}")
        for est in ("ML with z", "ML without z"):
            w = u[u.est == est]
            out.append(f"   {est}: |rho| > 0.95 in {w.rho_boundary.min():.3f}-{w.rho_boundary.max():.3f}; coverage "
                       f"{w.coverage.min():.3f}-{w.coverage.max():.3f}; bias {w.bias.min():+.4f} to {w.bias.max():+.4f}")
    for est in ("ML without z", "ML with z"):
        u = S[S.est == est]
        out.append(f"{est}: |rho| > 0.95 in {u.rho_boundary.min():.3f}-{u.rho_boundary.max():.3f} of samples per cell; "
                   f"two-step start at a lower maximum in {u.lower_max.min():.3f}-{u.lower_max.max():.3f}")
    u = S[S.est == "OLS on selected"]
    out.append(f"Least squares on the selected: bias {u.bias.min():+.4f} to {u.bias.max():+.4f} over the 64 designs "
               f"(negative in {np.sum(u.bias < 0)}); coverage {u.coverage.min():.3f}-{u.coverage.max():.3f}")
    pos = u[u.bias > 0]
    out.append("   positive bias in: " + "; ".join(f"{r.dist}, rho {r.rho}, strength {r.g:g}, share {r.share} "
                                                  f"({r.bias:+.4f})" for r in pos.itertuples()))
    w = u[(u.g == 1) & (u.dist != "heteroscedastic")]
    out.append(f"Median R^2 of the inverse Mills ratio on (1, x) among the selected, without z: "
               f"{u.r2.min():.3f}-{u.r2.max():.3f}; with z at strength 1: {u[u.g == 1].r2z.min():.3f}-"
               f"{u[u.g == 1].r2z.max():.3f} (errors normal, t5 or skewed: {w.r2z.min():.3f}-{w.r2z.max():.3f})")
    write("03_simulation_digest.txt", "\n".join(out))


def figures(S):
    style()
    COL = {"Two-step with z": BLUE, "ML with z": AQUA, "Series with z": VIOLET, "Two-step without z": ORANGE,
           "ML without z": YELLOW}
    MK = {"Two-step with z": "o", "ML with z": "s", "Series with z": "^", "Two-step without z": "D", "ML without z": "v"}
    LS = {"Two-step with z": "-", "ML with z": "-", "Series with z": "-", "Two-step without z": "--", "ML without z": "--"}
    TIT = {"normal": "Normal errors", "t5": "Student t, 5 df", "skewed": "Skewed errors",
           "heteroscedastic": "Heteroscedastic selection error"}
    for name, metric, ylab in (("fig4_simulation_rmse", "rmse_ratio", "RMSE of the slope relative to\nleast squares on the selected (log scale)"),
                               ("fig5_simulation_coverage", "coverage", "Coverage of nominal 95% intervals")):
        fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.4), sharey=True, gridspec_kw={"hspace": 0.42, "wspace": 0.1})
        for r_, rho in enumerate(RHOS):
            for c_, dist in enumerate(DISTS):
                ax = axes[r_, c_]
                t = S[(S.dist == dist) & (S.rho == rho) & (S.share == 0.5)]
                for est in EST[1:]:
                    if metric == "coverage" and est == "Series with z":
                        continue
                    u = t[t.est == est].sort_values("g")
                    ax.plot(range(len(GS)), u[metric].clip(0.1, 30) if metric == "rmse_ratio" else u[metric],
                            marker=MK[est], ls=LS[est], color=COL[est], lw=1.6, ms=5, label=est)
                ax.set_xticks(range(len(GS)), [f"{g:g}" for g in GS])
                if metric == "rmse_ratio":
                    ax.set_yscale("log")
                    ax.axhline(1, color=INK, lw=0.8)
                    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20]))
                    ax.yaxis.set_minor_locator(NullLocator())
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                    ax.set_ylim(0.15, 32)
                else:
                    ax.axhline(0.95, color=INK, lw=0.8)
                    ax.set_ylim(0, 1.02)
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
                ax.set_title(f"{TIT[dist]}, ρ = {rho}", fontsize=10)
                if r_ == 1:
                    ax.set_xlabel("Exclusion strength")
                if c_ == 0:
                    ax.set_ylabel(ylab)
        fig.text(0.5, -0.005, "Exclusion strength: the coefficient of z in the selection equation (x has 0.5). "
                 f"Selected share 50%; {ARGS.reps} samples of 1,000 per point.", ha="center", fontsize=8.5, color=INK2)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.02), frameon=False)
        fig.savefig(FIG / f"{name}.png"); plt.close(fig)


if __name__ == "__main__":
    if ARGS.figures_only:
        S = pd.read_csv(RES / "03_simulation_summary.csv")
        digest(S)
        figures(S)
        raise SystemExit
    tasks = [(dd, r, g, s, k) for dd in DISTS for r in RHOS for g in GS for s in SHARES for k in range(ARGS.reps)]
    with Pool(2) as pool:
        res = pool.map(one, tasks, chunksize=20)
    df = pd.DataFrame([r for rr in res for r in rr],
                      columns=["dist", "rho", "g", "share", "rep", "est", "b", "se", "rho_hat", "lower", "r2z", "r2",
                               "share_real", "tc"])
    df.to_csv(RES / "03_simulation_replications.csv.gz", index=False)
    text, S = summarise(df, ARGS.reps)
    write("03_simulation_exclusion.txt", text)
    digest(S)
    figures(S)
