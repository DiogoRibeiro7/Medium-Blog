"""Experiment 1 - Minnesota radon: no pooling, complete pooling, partial pooling.

* variance components and the shrinkage factor of every county
* what scikit-learn's TargetEncoder(smooth="auto") does to the same counties
* repeated hold-out: houses held out inside known counties, error by how many
  training houses the county had
* leave-counties-out: prediction for counties never seen in training

Run:  python 01_radon_pooling.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import TargetEncoder

from common import (AQUA, BLUE, FIG, GREEN, INK2, MAGENTA, ORANGE, VIOLET,
                    YELLOW, load_radon, style, write)
from lmm import LMM

warnings.filterwarnings("ignore")
style()
rng = np.random.default_rng(20261025)
r = load_radon()
N, J = len(r), r.county.nunique()
n_j = r.groupby("county").size()
out = []

# ---------------------------------------------------------------- 1. full data
X = np.column_stack([np.ones(N), r.floor])
pp = LMM().fit(X, np.ones(N), r.y, r.county)
tau2, s2 = float(pp.D[0, 0]), pp.sigma2
out.append(f"Radon: N = {N} houses, J = {J} counties; county size median "
           f"{n_j.median():.0f}, min {n_j.min()}, max {n_j.max()}; "
           f"{(n_j <= 2).sum()} counties have 1-2 houses, "
           f"{(n_j <= 5).sum()} have 5 or fewer")
out.append(f"Random intercept (REML): beta0 = {pp.beta[0]:.3f} (SE {pp.se[0]:.3f}), "
           f"floor = {pp.beta[1]:.3f} (SE {pp.se[1]:.3f}); tau = {tau2**.5:.3f}, "
           f"sigma = {s2**.5:.3f}; ICC = {tau2/(tau2+s2):.3f}; "
           f"n at which lambda = 0.5: sigma^2/tau^2 = {s2/tau2:.2f}")
lam = n_j * tau2 / (n_j * tau2 + s2)
out.append(f"Shrinkage factor lambda_j: median {lam.median():.2f}, "
           f"min {lam.min():.2f} (n=1), max {lam.max():.2f} (n={n_j.max()})")

Xu = np.column_stack([X, r.u])
ppu = LMM().fit(Xu, np.ones(N), r.y, r.county)
tau2u = float(ppu.D[0, 0])
out.append(f"With county log-uranium: beta = {np.round(ppu.beta, 3)}, "
           f"SE = {np.round(ppu.se, 3)}; tau = {tau2u**.5:.3f} "
           f"(was {tau2**.5:.3f}); sigma = {ppu.sigma2**.5:.3f}; "
           f"share of between-county variance explained by uranium "
           f"= {1 - tau2u/tau2:.2f}")

# no-pooling county intercepts: OLS with one dummy per county and a common floor slope
D = pd.get_dummies(r.county, prefix="c").astype(float)
np_fit = np.linalg.lstsq(np.column_stack([D.values, r.floor]), r.y, rcond=None)[0]
alpha_np = pd.Series(np_fit[:-1], index=[int(c[2:]) for c in D.columns])
alpha_pp = pd.Series({k: pp.beta[0] + v[0] for k, v in pp.blup.items()})
spread_np, spread_pp = alpha_np.std(), alpha_pp.std()
out.append(f"Spread of county intercepts: no pooling SD {spread_np:.3f}, "
           f"partial pooling SD {spread_pp:.3f}, model tau {tau2**.5:.3f}")

# ------------------------------------------- 2. TargetEncoder vs empirical Bayes
null = LMM().fit(np.ones((N, 1)), np.ones(N), r.y, r.county)
t0, s0 = float(null.D[0, 0]), null.sigma2
lam_eb = n_j * t0 / (n_j * t0 + s0)
grp = r.groupby("county").y
ssd = grp.apply(lambda v: ((v - v.mean()) ** 2).sum())
vy = r.y.var(ddof=0)
lam_te = vy * n_j / (vy * n_j + ssd / n_j)          # sklearn's formula
enc = TargetEncoder(smooth="auto", target_type="continuous").fit(
    r[["county"]], r.y)
te_vals = pd.Series(enc.encodings_[0], index=enc.categories_[0])
check = (lam_te * grp.mean() + (1 - lam_te) * r.y.mean() - te_vals).abs().max()
out.append(f"TargetEncoder(smooth='auto') reproduced by hand: max abs diff "
           f"{check:.2e}")
out.append(f"Null model: tau^2 = {t0:.4f}, sigma^2 = {s0:.4f}, Var(y) = {vy:.4f}; "
           f"EB smoothing constant m = sigma^2/tau^2 = {s0/t0:.2f}")
single = n_j[n_j == 1].index
out.append(f"Counties with one house: {len(single)}; TargetEncoder lambda for them "
           f"= {lam_te[single].round(3).tolist()}, empirical Bayes lambda "
           f"= {lam_eb[single].round(3).tolist()}")
for k in (2, 3, 4):
    idx = n_j[n_j == k].index
    out.append(f"n_j = {k}: TE lambda median {lam_te[idx].median():.2f} "
               f"(range {lam_te[idx].min():.2f}-{lam_te[idx].max():.2f}); "
               f"EB lambda {lam_eb[idx].iloc[0]:.2f}")
small = n_j[n_j <= 5].index
out.append(f"Counties with n_j <= 5: median TE lambda {lam_te[small].median():.2f} "
           f"vs EB {lam_eb[small].median():.2f}")
# the downstream regression partly compensates with one global slope
for name, sm in (("auto", "auto"), ("EB", s0 / t0)):
    enc = TargetEncoder(smooth=sm, target_type="continuous", cv=5, shuffle=True,
                        random_state=0)
    f = enc.fit_transform(r[["county"]], r.y)[:, 0]
    coef = LinearRegression().fit(np.column_stack([f, r.floor]), r.y).coef_[0]
    out.append(f"TargetEncoder {name}: coefficient of the cross-fitted encoding in "
               f"the downstream linear model = {coef:.2f}")

# ranking: which counties look most exposed?
top_np = alpha_np.sort_values(ascending=False).head(10).index
top_pp = alpha_pp.sort_values(ascending=False).head(10).index
out.append(f"Top-10 counties by no-pooling intercept: median n_j "
           f"{n_j[top_np].median():.0f}, {int((n_j[top_np] <= 3).sum())} with "
           f"<= 3 houses; by partial pooling: median n_j {n_j[top_pp].median():.0f}, "
           f"{int((n_j[top_pp] <= 3).sum())} with <= 3 houses; overlap "
           f"{len(set(top_np) & set(top_pp))} counties")

# decision: which counties have a typical basement above the EPA action level (4 pCi/L)?
from scipy.stats import norm
act = np.log(4.0)
lam_f = (n_j * tau2 / (n_j * tau2 + s2)).reindex(alpha_pp.index)
post_sd = np.sqrt(tau2 * (1 - lam_f))          # conditional SD of the county effect
p_exceed = pd.Series(1 - norm.cdf((act - alpha_pp) / post_sd), index=alpha_pp.index)
flag_np = alpha_np.index[alpha_np > act]
flag_pp = alpha_pp.index[alpha_pp > act]
flag_90 = p_exceed.index[p_exceed > 0.9]
out.append(f"EPA action level 4 pCi/L (log = {act:.3f}), basement geometric mean: "
           f"no pooling flags {len(flag_np)} counties ({int((n_j[flag_np] <= 3).sum())} "
           f"with <= 3 houses); partial pooling flags {len(flag_pp)} "
           f"({int((n_j[flag_pp] <= 3).sum())} with <= 3 houses); "
           f"{len(set(flag_np) - set(flag_pp))} flagged only by no pooling; "
           f"{len(set(flag_pp) - set(flag_np))} only by partial pooling (their n_j: "
           f"{sorted(n_j[list(set(flag_pp) - set(flag_np))].tolist())}); grand mean "
           f"{pp.beta[0]:.3f} is above the threshold; "
           f"partial pooling with P(exceed) > 0.9: {len(flag_90)} counties "
           f"(median n_j {n_j[flag_90].median():.0f})")

# ------------------------------------------------------ 3. repeated hold-out
def split(rng):
    test = np.zeros(N, bool)
    for c, idx in r.groupby("county").indices.items():
        if len(idx) >= 2:
            k = max(1, int(round(0.3 * len(idx))))
            test[rng.choice(idx, k, replace=False)] = True
    return test


def fit_predict_all(tr, te, seed):
    preds = {}
    Xtr = np.column_stack([np.ones(len(tr)), tr.floor])
    Xte = np.column_stack([np.ones(len(te)), te.floor])
    b = np.linalg.lstsq(Xtr, tr.y, rcond=None)[0]
    preds["Complete pooling"] = Xte @ b

    cats = np.sort(tr.county.unique())
    Dtr = (tr.county.values[:, None] == cats).astype(float)
    Dte = (te.county.values[:, None] == cats).astype(float)
    b = np.linalg.lstsq(np.column_stack([Dtr, tr.floor]), tr.y, rcond=None)[0]
    preds["No pooling (one-hot)"] = np.column_stack([Dte, te.floor]) @ b

    for name, smooth in (("TargetEncoder auto", "auto"), ("TargetEncoder EB", None)):
        if smooth is None:
            nf = LMM().fit(np.ones((len(tr), 1)), np.ones(len(tr)), tr.y, tr.county)
            smooth = nf.sigma2 / float(nf.D[0, 0])
        enc = TargetEncoder(smooth=smooth, target_type="continuous", cv=5,
                            shuffle=True, random_state=seed)
        ftr = enc.fit_transform(tr[["county"]], tr.y)[:, 0]
        fte = enc.transform(te[["county"]])[:, 0]
        lr = LinearRegression().fit(np.column_stack([ftr, tr.floor]), tr.y)
        preds[name] = lr.predict(np.column_stack([fte, te.floor]))

    m = LMM()
    m.fit(Xtr, np.ones(len(tr)), tr.y, tr.county)
    preds["Partial pooling"] = m.predict(Xte, np.ones(len(te)), te.county)
    m = LMM()
    m.fit(np.column_stack([Xtr, tr.u]), np.ones(len(tr)), tr.y, tr.county)
    preds["Partial pooling + uranium"] = m.predict(
        np.column_stack([Xte, te.u]), np.ones(len(te)), te.county)

    hgb = HistGradientBoostingRegressor(categorical_features=[0], random_state=seed)
    hgb.fit(tr[["county", "floor", "u"]].values, tr.y)
    preds["Gradient boosting"] = hgb.predict(te[["county", "floor", "u"]].values)
    return preds


r["county"] = r.county.astype(int)
# county ids for HGB must be 0..n_cat-1
r["county"] = r.county.rank(method="dense").astype(int) - 1
R = 200
rows = []
for rep in range(R):
    test = split(rng)
    tr, te = r[~test].reset_index(drop=True), r[test].reset_index(drop=True)
    ntr = tr.groupby("county").size()
    bucket = pd.cut(te.county.map(ntr), [0, 1, 3, 9, 1000],
                    labels=["1", "2-3", "4-9", "10+"])
    for name, p in fit_predict_all(tr, te, rep).items():
        rows.append(pd.DataFrame({"rep": rep, "model": name, "bucket": bucket,
                                  "se": (te.y.values - p) ** 2}))
ho = pd.concat(rows)
models = ["Complete pooling", "No pooling (one-hot)", "TargetEncoder auto",
          "TargetEncoder EB", "Gradient boosting", "Partial pooling",
          "Partial pooling + uranium"]
overall = ho.groupby("model").se.mean() ** .5
byb = ho.groupby(["model", "bucket"], observed=True).se.mean().unstack() ** .5
nb = ho[ho.model == models[0]].groupby("bucket", observed=True).size() / R
out.append(f"\nHold-out ({R} random splits, ~30% of houses in counties with >=2 "
           f"houses; test houses per split by county training size: "
           + ", ".join(f"{b}: {v:.0f}" for b, v in nb.items()) + ")")
for mname in models:
    out.append(f"  {mname:28s} RMSE all {overall[mname]:.3f} | "
               + " | ".join(f"n_train {b}: {byb.loc[mname, b]:.3f}"
                            for b in byb.columns))
per_rep = ho.groupby(["rep", "model"]).se.mean().unstack() ** .5
for a, b in (("Partial pooling", "No pooling (one-hot)"),
             ("Partial pooling", "Complete pooling"),
             ("Partial pooling", "TargetEncoder auto"),
             ("TargetEncoder EB", "TargetEncoder auto"),
             ("Partial pooling + uranium", "Gradient boosting"),
             ("Partial pooling + uranium", "Partial pooling")):
    d = per_rep[a] - per_rep[b]
    out.append(f"  {a} minus {b}: mean RMSE diff {d.mean():+.4f}, "
               f"{a} better in {(d < 0).mean():.0%} of splits")

# ------------------------------------------------ 4. leave counties out
from sklearn.model_selection import GroupKFold

rows = []
for rep in range(40):
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=rep)
    for trn, tst in gkf.split(r, groups=r.county):
        tr, te = r.iloc[trn].reset_index(drop=True), r.iloc[tst].reset_index(drop=True)
        X1 = lambda d: np.column_stack([np.ones(len(d)), d.floor])
        X2 = lambda d: np.column_stack([np.ones(len(d)), d.floor, d.u])
        res = {}
        b = np.linalg.lstsq(X1(tr), tr.y, rcond=None)[0]; res["Complete pooling"] = X1(te) @ b
        b = np.linalg.lstsq(X2(tr), tr.y, rcond=None)[0]; res["Complete pooling + uranium"] = X2(te) @ b
        m = LMM(); m.fit(X1(tr), np.ones(len(tr)), tr.y, tr.county)
        res["Partial pooling"] = m.predict(X1(te), np.ones(len(te)), te.county)
        m = LMM(); m.fit(X2(tr), np.ones(len(tr)), tr.y, tr.county)
        res["Partial pooling + uranium"] = m.predict(X2(te), np.ones(len(te)), te.county)
        hgb = HistGradientBoostingRegressor(categorical_features=[0], random_state=rep)
        hgb.fit(tr[["county", "floor", "u"]].values, tr.y)
        res["Gradient boosting"] = hgb.predict(te[["county", "floor", "u"]].values)
        for k, p in res.items():
            rows.append(pd.DataFrame({"rep": rep, "model": k, "se": (te.y.values - p) ** 2}))
lco = pd.concat(rows)
lco_rmse = lco.groupby("model").se.mean() ** .5
out.append("\nLeave-counties-out (5-fold by county, 40 repeats): RMSE for houses "
           "in counties never seen in training")
for k, v in lco_rmse.sort_values().items():
    out.append(f"  {k:28s} {v:.3f}")
write("01_radon.txt", "\n".join(out))

# --------------------------------------------------------------- figures
# Figure 1: shrinkage of county intercepts
fig, ax = plt.subplots(figsize=(8.2, 4.6))
nn = n_j.reindex(alpha_np.index)
jit = np.exp(rng.uniform(-0.08, 0.08, len(nn)))
x = nn.values * jit
for xi, a0, a1 in zip(x, alpha_np.values, alpha_pp.reindex(alpha_np.index).values):
    ax.plot([xi, xi], [a0, a1], color="#b8b7b2", lw=0.9, zorder=1)
ax.scatter(x, alpha_np.values, s=30, color=ORANGE, label="No pooling (county OLS)",
           zorder=2, edgecolor="white", linewidth=0.6)
ax.scatter(x, alpha_pp.reindex(alpha_np.index).values, s=30, color=BLUE,
           label="Partial pooling (BLUP)", zorder=3, edgecolor="white", linewidth=0.6)
ax.axhline(pp.beta[0], color=INK2, lw=1, ls="--")
ax.text(118, pp.beta[0] + 0.06, "grand mean", color=INK2, ha="right", fontsize=9)
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100])
ax.set_xlabel("Houses measured in the county (log scale)")
ax.set_ylabel("County intercept, log radon (basement)")
ax.set_title("Small counties are pulled hardest towards the grand mean")
ax.legend(loc="upper right")
fig.savefig(FIG / "fig1_radon_shrinkage.png"); plt.close(fig)

# Figure 3: weight on the county mean, TargetEncoder vs empirical Bayes
fig, ax = plt.subplots(figsize=(8.2, 4.4))
grid = np.linspace(1, 120, 400)
ax.plot(grid, grid * t0 / (grid * t0 + s0), color=BLUE, lw=2,
        label="Empirical Bayes / random intercept")
jt = n_j.values * np.exp(rng.uniform(-0.06, 0.06, len(n_j)))
ax.scatter(jt, lam_te.values, s=26, color=ORANGE, edgecolor="white", linewidth=0.6,
           label="TargetEncoder(smooth='auto')", zorder=3)
ax.set_xscale("log"); ax.set_ylim(-0.03, 1.05)
ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100])
ax.set_xlabel("Houses measured in the county (log scale)")
ax.set_ylabel("Weight on the county's own mean (λ)")
ax.set_title("Same idea, different prior variance: the encoder barely shrinks")
ax.legend(loc="lower right")
fig.savefig(FIG / "fig3_targetencoder_lambda.png"); plt.close(fig)

# Figure 2: hold-out RMSE relative to partial pooling, by county training size
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ref = byb.loc["Partial pooling"]
others = [m_ for m_ in models if m_ != "Partial pooling"]
others = sorted(others, key=lambda m_: overall[m_])
ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ypos = np.arange(len(others))[::-1]
for yi, mname in zip(ypos, others):
    rel = 100 * (byb.loc[mname] / ref - 1)
    ax.plot([rel.min(), rel.max()], [yi, yi], color="#d6d5d0", lw=1.2, zorder=1)
    for k, (b, v) in enumerate(rel.items()):
        ax.scatter(v, yi, s=58, color=ramp[k], edgecolor="white", linewidth=0.8,
                   zorder=3)
ax.axvline(0, color=INK2, lw=1, ls="--")
ax.text(0.3, -0.55, "partial pooling = 0", color=INK2, fontsize=9)
ax.set_yticks(ypos); ax.set_yticklabels(others)
ax.set_xlabel("Hold-out RMSE relative to partial pooling (%)")
ax.set_title("Where the county is thin, pooling decides the error")
ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", color=ramp[k], markersize=8, label=str(b))
           for k, b in enumerate(byb.columns)]
ax.set_ylim(-0.8, len(others) - 0.5)
ax.legend(handles=handles, title="Training houses in the county", fontsize=9,
          title_fontsize=9, loc="upper right")
fig.savefig(FIG / "fig2_radon_holdout.png"); plt.close(fig)
