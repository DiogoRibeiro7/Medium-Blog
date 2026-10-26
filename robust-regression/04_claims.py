"""Experiment 4 - when the outliers are the signal: insurance claim severity.

insuranceData::dataCar: 67,856 one-year motor policies from 2004-05 (de Jong &
Heller 2008); the 4,624 with a claim are used to model the claim amount from
driver age band, sex, area, vehicle age and log vehicle value. 100 random
70/30 splits; each model predicts the expected claim on the test policies, and
the total of the predictions is compared with the total actually claimed.

Run:  python 04_claims.py      (about 6 minutes)
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import BLUE, FIG, INK2, ORANGE, AQUA, fetch, style, write
from rob import m_estimate, mm_estimate, ols

warnings.filterwarnings("ignore")
style()
c = fetch("insuranceData", "dataCar")
c = c[c.claimcst0 > 0].reset_index(drop=True)
y = c.claimcst0.values
dm = pd.get_dummies(c[["agecat", "area"]].astype(str), drop_first=True, dtype=float)
X = np.column_stack([np.ones(len(c)), dm.values, (c.gender == "M").astype(float),
                     c.veh_age.astype(float), np.log(c.veh_value + 0.01)])
ly = np.log(y)
REPS = 100
METHODS = ["Gamma GLM (log link)", "OLS on log claim, smearing", "OLS on log claim, exp(fit)",
           "MM on log claim, exp(fit)", "MM on log claim, smearing", "OLS on claim",
           "Huber M on claim", "MM on claim"]
rows, byage = [], []
for rep in range(REPS):
    rng = np.random.default_rng([4, rep])
    tr = rng.random(len(c)) < 0.7
    te = ~tr
    preds = {}
    g = sm.GLM(y[tr], X[tr], family=sm.families.Gamma(sm.families.links.Log())).fit()
    preds[METHODS[0]] = g.predict(X[te])
    o = ols(X[tr], ly[tr])
    preds[METHODS[1]] = np.exp(X[te] @ o.coef) * np.mean(np.exp(o.resid))
    preds[METHODS[2]] = np.exp(X[te] @ o.coef)
    m = mm_estimate(X[tr], ly[tr], n_sub=500, seed=rep)
    preds[METHODS[3]] = np.exp(X[te] @ m.coef)
    preds[METHODS[4]] = np.exp(X[te] @ m.coef) * np.mean(np.exp(m.resid))
    preds[METHODS[5]] = X[te] @ ols(X[tr], y[tr]).coef
    preds[METHODS[6]] = X[te] @ m_estimate(X[tr], y[tr], "huber").coef
    preds[METHODS[7]] = X[te] @ mm_estimate(X[tr], y[tr], n_sub=500, seed=rep).coef
    actual = y[te].sum()
    for k, p in preds.items():
        rows.append((rep, k, p.sum() / actual, np.sqrt(np.mean((y[te] - p) ** 2)),
                     np.mean(np.abs(y[te] - p))))
        for a in range(1, 7):
            s = (c.agecat.values[te] == a)
            byage.append((rep, k, a, p[s].mean(), y[te][s].mean()))
d = pd.DataFrame(rows, columns=["rep", "method", "ratio", "rmse", "mae"])
b = pd.DataFrame(byage, columns=["rep", "method", "agecat", "pred", "obs"])
d.to_csv("results/04_claims_reps.csv.gz", index=False)
gq = d.groupby("method", sort=False)
out = [f"dataCar: {len(c)} claims; mean {y.mean():.0f}, median {np.median(y):.0f}, "
       f"largest {y.max():.0f}; top 1% of claims = {np.sort(y)[-int(len(y) * 0.01):].sum() / y.sum():.1%} "
       f"of the total; top 10% = {np.sort(y)[-int(len(y) * 0.10):].sum() / y.sum():.1%}",
       f"{REPS} random 70/30 splits; predicted total / actual total on the test claims"]
for k in METHODS:
    x = gq.get_group(k)
    out.append(f"  {k:30s} ratio mean {x.ratio.mean():.3f} (5-95%: {x.ratio.quantile(.05):.3f}-"
               f"{x.ratio.quantile(.95):.3f}); RMSE {x.rmse.mean():.0f}; MAE {x.mae.mean():.0f}")
ga = b.groupby(["method", "agecat"], sort=False)[["pred", "obs"]].mean()
out.append("Mean predicted vs mean observed claim by driver age band (averaged over splits):")
for k in (METHODS[0], METHODS[3], METHODS[4]):
    out.append(f"  {k:30s} " + "; ".join(
        f"age {a}: {ga.loc[(k, a)].pred:.0f} vs {ga.loc[(k, a)].obs:.0f}" for a in range(1, 7)))
o_all, m_all = ols(X, ly), mm_estimate(X, ly, n_sub=500, seed=0)
out.append(f"Full-data fits on log claim: OLS scale {o_all.scale:.3f}, MM scale {m_all.scale:.3f}; "
           f"MM intercept - OLS intercept {m_all.coef[0] - o_all.coef[0]:+.3f}; "
           f"share of claims with MM weight < 0.5: {np.mean(m_all.weights < 0.5):.1%}; "
           f"|MM residual / scale| > 3: {np.mean(np.abs(m_all.resid / m_all.scale) > 3):.1%}")
write("04_claims.txt", "\n".join(out))

# figure
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
yy = np.arange(len(METHODS))[::-1]
for yi, k in zip(yy, METHODS):
    x = gq.get_group(k).ratio
    col = BLUE if k in METHODS[:2] else (ORANGE if "MM" in k or "Huber" in k else INK2)
    ax.plot([x.quantile(.05), x.quantile(.95)], [yi, yi], color=col, lw=2.2, alpha=0.5)
    ax.scatter(x.mean(), yi, color=col, s=55, zorder=3, edgecolor="white", linewidth=0.8)
ax.axvline(1, color=INK2, lw=0.9, ls="--")
ax.set_yticks(yy); ax.set_yticklabels(METHODS, fontsize=9)
ax.set_xlabel("Predicted total claims / actual total (test policies)")
ax.set_title("Robust fits price the typical claim, not the expected one", fontsize=10.5)
ax.grid(axis="y", visible=False)
ax = axes[1]
ages = np.arange(1, 7)
obs = [ga.loc[(METHODS[0], a)].obs for a in ages]
ax.plot(ages, obs, "o", color=INK2, ms=7, label="Observed mean claim", zorder=4)
for k, col, ls in ((METHODS[0], BLUE, "-"), (METHODS[3], ORANGE, "-"), (METHODS[4], AQUA, "--")):
    ax.plot(ages, [ga.loc[(k, a)].pred for a in ages], color=col, ls=ls, marker="o", ms=4, label=k)
ax.set_xticks(ages); ax.set_xticklabels(["youngest", "2", "3", "4", "5", "oldest"])
ax.set_xlabel("Driver age band"); ax.set_ylabel("Mean claim (AUD)")
ax.set_ylim(0, None)
ax.set_title("Mean claim by driver age, test policies", fontsize=10.5)
ax.legend(fontsize=8.5, loc="lower left")
fig.savefig(FIG / "fig7_claims.png"); plt.close(fig)
