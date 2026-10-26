"""Experiment 3 - interval coverage when observations are clustered.

Data are simulated from a random-intercept, random-slope model whose
variance components are estimated on High School and Beyond
(mAch ~ catholic + cses + (1 + cses | school)). School sizes are resampled
from the real ones. Two quantities are estimated:

* gamma: a school-level binary 'treatment' assigned to half the schools (true 0)
* beta:  the slope of a student-level predictor whose slope varies by school

For each number of schools J, 2000 data sets; nominal 95% intervals from

* OLS, iid errors (normal)
* OLS, cluster-robust CR1 (t with J-1 df)
* random intercept only (normal; and t with J-2 df)
* random intercept + random slope (normal; and t with J-2 / J-1 df)

Run:  python 03_coverage_simulation.py      (about 5 minutes on 2 cores)
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import AQUA, BLUE, FIG, INK2, ORANGE, VIOLET, load_hsb, style, write
from lmm import LMM

warnings.filterwarnings("ignore")
h = load_hsb()
SIZES = h.groupby("school").size().values
SD_X = h.cses.std()
one = np.ones(len(h))
cal = LMM().fit(np.column_stack([one, h.catholic, h.cses]),
                np.column_stack([one, h.cses]), h.mAch, h.school)
D_TRUE, S2_TRUE, BETA = cal.D, cal.sigma2, cal.beta[2]
J_GRID = (10, 20, 40, 80, 160)
REPS = 2000


def ols_fit(X, y, g, J):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, p = X.shape
    V_iid = XtX_inv * (e @ e) / (n - p)
    S = np.zeros((J, p))
    np.add.at(S, g, X * e[:, None])
    meat = S.T @ S
    c = J / (J - 1) * (n - 1) / (n - p)
    V_cr = c * XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V_iid)), np.sqrt(np.diag(V_cr))


def one_rep(args):
    J, seed = args
    rng = np.random.default_rng(seed)
    n = rng.choice(SIZES, J)
    g = np.repeat(np.arange(J), n)
    N = n.sum()
    T = np.zeros(J); T[rng.permutation(J)[: J // 2]] = 1
    x = rng.normal(0, SD_X, N)
    x = x - np.bincount(g, x)[g] / n[g]            # centred within school
    b = rng.multivariate_normal([0, 0], D_TRUE, J)
    y = 12.0 + 0.0 * T[g] + (BETA + b[g, 1]) * x + b[g, 0] + rng.normal(0, S2_TRUE ** .5, N)
    X = np.column_stack([np.ones(N), T[g], x])
    res = {}
    est, se_iid, se_cr = ols_fit(X, y, g, J)
    res["OLS iid"] = (est, se_iid, (np.inf, np.inf))
    res["OLS cluster-robust"] = (est, se_cr, (J - 1, J - 1))
    f1 = LMM().fit(X, np.ones(N), y, g)
    res["RI, z"] = (f1.beta, f1.se, (np.inf, np.inf))
    res["RI, t"] = (f1.beta, f1.se, (J - 2, J - 2))
    f2 = LMM().fit(X, np.column_stack([np.ones(N), x]), y, g)
    res["RI + slope, z"] = (f2.beta, f2.se, (np.inf, np.inf))
    res["RI + slope, t"] = (f2.beta, f2.se, (J - 2, J - 1))
    rows = []
    for k, (bh, se, (df_g, df_b)) in res.items():
        for j, truth, df, par in ((1, 0.0, df_g, "gamma"), (2, BETA, df_b, "beta")):
            q = stats.norm.ppf(0.975) if np.isinf(df) else stats.t.ppf(0.975, df)
            rows.append((J, seed, k, par, bh[j], se[j], abs(bh[j] - truth) <= q * se[j]))
    return rows


if __name__ == "__main__":
    style()
    import sys
    if "--figure-only" in sys.argv:
        df = pd.read_csv("results/coverage_sim.csv.gz")
    else:
        tasks = [(J, 100000 * J + r) for J in J_GRID for r in range(REPS)]
        with Pool(2) as pool:
            rows = [row for chunk in pool.imap_unordered(one_rep, tasks, chunksize=20)
                    for row in chunk]
        df = pd.DataFrame(rows, columns=["J", "seed", "method", "par", "est", "se", "cover"])
        df.to_csv("results/coverage_sim.csv.gz", index=False)
    cov = df.groupby(["par", "method", "J"]).cover.mean().unstack()
    ratio = df.groupby(["par", "method", "J"]).apply(
        lambda d: d.se.mean() / d.est.std()).unstack()
    out = [f"Calibration from HSB (RI + random cses slope, catholic + cses): "
           f"intercept SD {D_TRUE[0,0]**.5:.2f}, slope SD {D_TRUE[1,1]**.5:.2f}, "
           f"corr {D_TRUE[0,1]/np.sqrt(D_TRUE[0,0]*D_TRUE[1,1]):.2f}, sigma "
           f"{S2_TRUE**.5:.2f}, beta {BETA:.2f}; SD of cses {SD_X:.2f}; {REPS} reps per J",
           f"Monte Carlo SE of a 95% coverage estimate ~ {np.sqrt(.95*.05/REPS):.3f}"]
    for par, lab in (("gamma", "School-level effect (gamma)"),
                     ("beta", "Student-level slope (beta)")):
        out.append(f"\n{lab}: coverage of nominal 95% intervals")
        for mth in cov.loc[par].index:
            out.append(f"  {mth:20s} " + "  ".join(
                f"J={J}: {cov.loc[(par, mth), J]:.3f}" for J in J_GRID))
        out.append(f"{lab}: mean SE / empirical SD of the estimate")
        for mth in ratio.loc[par].index:
            out.append(f"  {mth:20s} " + "  ".join(
                f"J={J}: {ratio.loc[(par, mth), J]:.2f}" for J in J_GRID))
    write("03_coverage.txt", "\n".join(out))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    show = {"OLS cluster-robust": ORANGE, "RI, z": VIOLET, "RI, t": INK2,
            "RI + slope, z": AQUA, "RI + slope, t": BLUE}
    band = 2 * np.sqrt(.95 * .05 / REPS)
    for ax, par, title in ((axes[0], "gamma", "School-level effect"),
                           (axes[1], "beta", "Student-level slope that varies by school")):
        ax.axhspan(0.95 - band, 0.95 + band, color="#e4e3df", zorder=0)
        ax.axhline(0.95, color=INK2, lw=0.8, ls=":")
        for mth, col in show.items():
            ax.plot(range(len(J_GRID)), cov.loc[(par, mth)].values, marker="o", ms=6,
                    color=col, label=mth, lw=1.8, ls="--" if mth == "RI, t" else "-")
        ax.set_xticks(range(len(J_GRID))); ax.set_xticklabels(J_GRID)
        ax.set_xlabel("Number of schools (J)")
        ax.set_title(title)
    axes[0].text(0.02, 0.03, f"OLS with iid errors: {cov.loc[('gamma', 'OLS iid')].min():.2f}-"
                 f"{cov.loc[('gamma', 'OLS iid')].max():.2f} at every J (off the scale)",
                 transform=axes[0].transAxes, color=INK2, fontsize=9)
    axes[1].plot(range(len(J_GRID)), cov.loc[("beta", "OLS iid")].values, marker="o", ms=6,
                 color="#9a9993", lw=1.4, label="OLS iid")
    axes[0].set_ylabel("Coverage of nominal 95% interval")
    axes[0].set_ylim(0.86, 0.99)
    axes[1].legend(loc="lower right", fontsize=8.5, ncol=2)
    fig.savefig(FIG / "fig5_coverage.png"); plt.close(fig)
