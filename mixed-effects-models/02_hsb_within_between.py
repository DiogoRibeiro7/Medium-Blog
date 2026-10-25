"""Experiment 2 - High School and Beyond: what a random intercept does to a slope
and to a standard error.

* ICC and design effect
* the SES slope: pooled OLS, within (school fixed effects), between (school
  means), random intercept, and the Mundlak correction
* standard errors for a school-level predictor (Catholic) and a student-level
  predictor (centred SES) under OLS, cluster-robust, RI and RI + random slope
* Raudenbush & Bryk's intercepts-and-slopes-as-outcomes model

Run:  python 02_hsb_within_between.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import AQUA, BLUE, FIG, INK2, ORANGE, load_hsb, style, write
from lmm import LMM

warnings.filterwarnings("ignore")
style()
h = load_hsb()
N, J = len(h), h.school.nunique()
m = h.groupby("school").size()
one = np.ones(N)
out = [f"HSB: N = {N} students in J = {J} schools ({(h.groupby('school').catholic.first()==1).sum()} Catholic); "
       f"school size mean {m.mean():.1f}, range {m.min()}-{m.max()}"]


def lmm(cols, Zcols=None, reml=True):
    X = np.column_stack([one] + [h[c] for c in cols])
    Z = one if Zcols is None else np.column_stack([one] + [h[c] for c in Zcols])
    f = LMM(reml=reml).fit(X, Z, h.mAch, h.school)
    return f


def ols(cols, cluster=False):
    X = sm.add_constant(h[cols])
    if cluster:
        return sm.OLS(h.mAch, X).fit(cov_type="cluster",
                                     cov_kwds={"groups": h.school})
    return sm.OLS(h.mAch, X).fit()


# ------------------------------------------------------------ 1. null model
f0 = lmm([])
t00, s2 = float(f0.D[0, 0]), f0.sigma2
icc = t00 / (t00 + s2)
deff = 1 + (m.mean() - 1) * icc
out.append(f"Null model: grand mean {f0.beta[0]:.2f}; tau00 = {t00:.2f} "
           f"(SD {t00**.5:.2f}), sigma^2 = {s2:.2f} (SD {s2**.5:.2f}); ICC = {icc:.3f}; "
           f"design effect 1 + (m-1)ICC = {deff:.1f}; effective N for a "
           f"school-level contrast ~ {N/deff:.0f} of {N}")
lam = m * t00 / (m * t00 + s2)
out.append(f"Reliability of school means lambda_j: mean {lam.mean():.2f}, "
           f"range {lam.min():.2f}-{lam.max():.2f}")

# ------------------------------------------------------- 2. the SES slope
b_ols = ols(["ses"]).params["ses"]
# within: school fixed effects == OLS on group-centred variables
yc = h.mAch - h.groupby("school").mAch.transform("mean")
b_w = (h.cses * yc).sum() / (h.cses ** 2).sum()
fe = sm.OLS(h.mAch, pd.concat([h[["cses"]], pd.get_dummies(h.school, dtype=float)],
                              axis=1)).fit(cov_type="cluster", cov_kwds={"groups": h.school})
sm_ = h.groupby("school")[["mAch", "meanses"]].mean()
btw = sm.OLS(sm_.mAch, sm.add_constant(sm_.meanses)).fit()
b_b = btw.params["meanses"]
ri = lmm(["ses"])
mund = lmm(["ses", "meanses"])
out.append(f"SES slope - pooled OLS: {b_ols:.3f}; within (school FE): {b_w:.3f} "
           f"(cluster SE {fe.bse['cses']:.3f}); between (school means): {b_b:.3f} "
           f"(SE {btw.bse['meanses']:.3f}); random intercept: {ri.beta[1]:.3f} "
           f"(SE {ri.se[1]:.3f})")
w = (ri.beta[1] - b_b) / (b_w - b_b)
out.append(f"RI slope as a blend: weight on within = {w:.3f}, on between = {1-w:.3f}")
out.append(f"Mundlak RI (ses + meanses): ses {mund.beta[1]:.3f} (SE {mund.se[1]:.3f}), "
           f"meanses (contextual) {mund.beta[2]:.3f} (SE {mund.se[2]:.3f}); "
           f"between - within = {b_b - b_w:.3f}; z for contextual effect "
           f"= {mund.beta[2]/mund.se[2]:.1f}")
out.append(f"Random intercept tau: without meanses {ri.D[0,0]**.5:.2f}, "
           f"with meanses {mund.D[0,0]**.5:.2f}")

# ----------------------------------------------------- 3. standard errors
o1, o1c = ols(["catholic", "cses"]), ols(["catholic", "cses"], cluster=True)
r1 = lmm(["catholic", "cses"])
r2 = lmm(["catholic", "cses"], Zcols=["cses"])
out.append("\nStandard errors, model mAch ~ catholic + cses")
for name, est, se_c, se_s in (
        ("OLS (iid)", o1.params, o1.bse["catholic"], o1.bse["cses"]),
        ("OLS cluster-robust (CR1)", o1c.params, o1c.bse["catholic"], o1c.bse["cses"]),
        ("Random intercept", pd.Series(r1.beta, ["const", "catholic", "cses"]),
         r1.se[1], r1.se[2]),
        ("RI + random SES slope", pd.Series(r2.beta, ["const", "catholic", "cses"]),
         r2.se[1], r2.se[2])):
    out.append(f"  {name:26s} catholic {est['catholic']:.3f} (SE {se_c:.3f}) | "
               f"cses {est['cses']:.3f} (SE {se_s:.3f})")
out.append(f"  ratio of SEs for catholic, CR1 / OLS iid = "
           f"{o1c.bse['catholic']/o1.bse['catholic']:.2f}; "
           f"for cses, CR1 / RI-only = {o1c.bse['cses']/r1.se[2]:.2f}, "
           f"RI+RS / RI-only = {r2.se[2]/r1.se[2]:.2f}")
out.append(f"  random SES slope SD = {r2.D[1,1]**.5:.2f}; corr(int, slope) = "
           f"{r2.D[0,1]/np.sqrt(r2.D[0,0]*r2.D[1,1]):.2f}")
lr = 2 * (r2.loglik - r1.loglik)
p_mix = 0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2)
out.append(f"  REML LRT for the random slope (variance + covariance): {lr:.1f}, "
           f"p (0.5 chi2_1 + 0.5 chi2_2 mixture) = {p_mix:.2g}, naive chi2_2 p = "
           f"{stats.chi2.sf(lr, 2):.2g}")

# ------------------------------------- 4. intercepts and slopes as outcomes
h["meanses_cses"] = h.meanses * h.cses
h["catholic_cses"] = h.catholic * h.cses
cols = ["meanses", "catholic", "cses", "meanses_cses", "catholic_cses"]
rb = lmm(cols, Zcols=["cses"])
names = ["intercept"] + cols
out.append("\nIntercepts-and-slopes-as-outcomes (REML, random intercept + cses slope)")
for k, b, s in zip(names, rb.beta, rb.se):
    out.append(f"  {k:14s} {b:7.3f}  (SE {s:.3f})")
out.append(f"  Tau: intercept SD {rb.D[0,0]**.5:.2f}, slope SD {rb.D[1,1]**.5:.2f}; "
           f"sigma {rb.sigma2**.5:.2f}")
out.append(f"  SES gradient at meanses = 0: public {rb.beta[3]:.2f}, Catholic "
           f"{rb.beta[3] + rb.beta[5]:.2f}")
out.append(f"  slope variance: {r2.D[1,1]:.3f} without school predictors of the "
           f"slope, {rb.D[1,1]:.3f} with them "
           f"({1 - rb.D[1,1]/r2.D[1,1]:.0%} explained); corr(int, slope) "
           f"{rb.D[0,1]/np.sqrt(rb.D[0,0]*rb.D[1,1]):.2f}")

# ------------------- 5. the random-effects structure is not neutral bookkeeping
dm = h.groupby("school").agg(mm=("mAch", "mean"), c=("catholic", "first"))
r3 = lmm(["catholic", "cses", "catholic_cses"], Zcols=["cses"])
out.append("\nCatholic coefficient (no meanses) under different random-effect structures")
out.append(f"  raw difference in school means: {dm[dm.c==1].mm.mean()-dm[dm.c==0].mm.mean():.3f}")
out.append(f"  RI only:                          {r1.beta[1]:.3f} (SE {r1.se[1]:.3f})")
out.append(f"  RI + slope, no catholic x cses:   {r2.beta[1]:.3f} (SE {r2.se[1]:.3f}); "
           f"corr(int, slope) {r2.D[0,1]/np.sqrt(r2.D[0,0]*r2.D[1,1]):.2f}")
out.append(f"  RI + slope, with catholic x cses: {r3.beta[1]:.3f} (SE {r3.se[1]:.3f}); "
           f"interaction {r3.beta[3]:.3f} (SE {r3.se[3]:.3f}); corr(int, slope) "
           f"{r3.D[0,1]/np.sqrt(r3.D[0,0]*r3.D[1,1]):.2f}, slope SD {r3.D[1,1]**.5:.2f}"
           f"  <- correlation on the boundary: singular fit")
write("02_hsb.txt", "\n".join(out))

# ----------------------------------------------------------------- figure
sch = h.groupby("school").agg(mses=("ses", "mean"), mach=("mAch", "mean"),
                              sdses=("ses", "std"), cat=("catholic", "first"))
fig, ax = plt.subplots(figsize=(8.4, 5.0))
for _, s in sch.iterrows():
    dx = 0.6 * s.sdses
    ax.plot([s.mses - dx, s.mses + dx], [s.mach - b_w * dx, s.mach + b_w * dx],
            color="#c9c8c3", lw=0.9, zorder=1)
ax.plot([], [], color="#c9c8c3", lw=0.9, label="Within-school slope, drawn through each school")
for cat, col, lab in ((0, ORANGE, "Public school"), (1, AQUA, "Catholic school")):
    d = sch[sch.cat == cat]
    ax.scatter(d.mses, d.mach, s=22, color=col, edgecolor="white", linewidth=0.5,
               zorder=3, label=lab)
xs = np.linspace(sch.mses.min() - 0.05, sch.mses.max() + 0.05, 50)
c0 = sch.mach.mean() - b_b * sch.mses.mean()
ax.plot(xs, c0 + b_b * xs, color=BLUE, lw=2.2, zorder=4,
        label=f"Between schools: {b_b:.1f} points per SES unit")
c1 = sch.mach.mean() - b_w * sch.mses.mean()
ax.plot(xs, c1 + b_w * xs, color=INK2, lw=1.6, ls="--", zorder=4,
        label=f"Within schools: {b_w:.1f} points per SES unit")
ax.set_xlabel("School mean SES")
ax.set_ylabel("School mean maths achievement")
ax.set_title(f"Between schools the SES gradient is {b_b:.1f} points; within them, {b_w:.1f}")
ax.legend(loc="upper left", fontsize=9)
fig.savefig(FIG / "fig4_hsb_within_between.png"); plt.close(fig)
