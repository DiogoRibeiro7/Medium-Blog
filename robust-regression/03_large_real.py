"""Experiment 3 - robust regression as a check on large real data sets.

* ggplot2::diamonds, 53,920 stones with positive recorded dimensions:
  log price ~ log(x * y * z) + cut + colour + clarity. A handful of rows have
  dimensions that cannot be right (implied density far from 0.0061 carat per
  cubic mm). Does OLS move? Does MM find them?
* robustbase::NOxEmissions, 8,088 hourly measurements next to a Swiss motorway:
  log NOx in the air ~ log NOx emitted by cars + sqrt(wind speed).

Run:  python 03_large_real.py      (about 2 minutes)
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.covariance import MinCovDet

from common import BLUE, FIG, INK2, ORANGE, fetch, load_diamonds, style, write
from rob import flag_outliers, mm_estimate, ols

warnings.filterwarnings("ignore")
style()
out = []

# ----------------------------------------------------------------- diamonds
d = load_diamonds()
n_all = len(d)
d = d[d.lv.notna()].reset_index(drop=True)
impossible = ~d.density.between(0.003, 0.012).values
X = np.column_stack([np.ones(len(d)), d.lv, d.cut_n, d.col_n, d.cla_n])
y = d.lp.values
o = ols(X, y)
o_clean = ols(X[~impossible], y[~impossible])
mm = mm_estimate(X, y, n_sub=500, seed=3)
mm_clean = mm_estimate(X[~impossible], y[~impossible], n_sub=500, seed=3)
u = mm.resid / mm.scale
fl3 = np.abs(u) > 3
infl_t = o.resid / (o.scale * np.sqrt(1 - np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)))
out.append(f"Diamonds: {n_all} rows, {n_all - len(d)} with a zero dimension (dropped), {len(d)} used; "
           f"{impossible.sum()} with implied density outside 0.003-0.012 carat/mm^3")
out.append(f"  Impossible rows (carat, x, y, z, price): " + "; ".join(
    f"{r.carat}, {r.x}, {r.y}, {r.z}, {r.price}" for r in d[impossible].itertuples()))
names = ["const", "log volume", "cut", "colour", "clarity"]
for lab, f in (("OLS, all rows", o), ("OLS, impossible rows removed", o_clean),
               ("MM, all rows", mm), ("MM, impossible rows removed", mm_clean)):
    out.append(f"  {lab:30s} " + ", ".join(f"{k} {v:.4f}" for k, v in zip(names, f.coef))
               + f"; SE log volume {f.se[1]:.4f}; scale {f.scale:.4f}")
out.append(f"  OLS log-volume coefficient moves by {o.coef[1] - o_clean.coef[1]:+.4f} "
           f"({(o.coef[1] - o_clean.coef[1]) / o_clean.se[1]:+.1f} SE) when the "
           f"{impossible.sum()} impossible rows are removed; MM by {mm.coef[1] - mm_clean.coef[1]:+.4f}")
out.append(f"  MM vs OLS (all rows), log volume: {mm.coef[1] - o.coef[1]:+.4f} "
           f"({(mm.coef[1] - o.coef[1]) / o.se[1]:+.1f} OLS SE)")
out.append(f"  Rows with |MM residual / scale| > 3: {fl3.sum()} ({fl3.mean():.2%}); of the impossible "
           f"rows, {(fl3 & impossible).sum()} of {impossible.sum()} are among them; median |u| of "
           f"impossible rows {np.median(np.abs(u[impossible])):.1f}")
out.append(f"  Rows with |OLS standardised residual| > 3: {(np.abs(infl_t) > 3).sum()}; of the impossible "
           f"rows {((np.abs(infl_t) > 3) & impossible).sum()}")
mcd = MinCovDet(random_state=0, support_fraction=0.75).fit(X[:, 1:])
rd = np.sqrt(mcd.mahalanobis(X[:, 1:]))
cut = np.sqrt(stats.chi2.ppf(0.975, X.shape[1] - 1))
lev = rd > cut
out.append(f"  MCD robust distance > {cut:.2f}: {lev.sum()} rows ({lev.mean():.2%}); bad leverage "
           f"(distance and |u| > 3): {(lev & fl3).sum()}; impossible rows with large distance: "
           f"{(lev & impossible).sum()}")
srt = np.argsort(-np.abs(u))[:10]
out.append("  Ten largest |MM residual / scale|: " + "; ".join(
    f"u {u[i]:+.1f} (carat {d.carat[i]}, price {d.price[i]}, cut {d.cut[i]}, clarity {d.clarity[i]}, "
    f"dims {d.x[i]}x{d.y[i]}x{d.z[i]})" for i in srt))

# ---------------------------------------------------------------------- NOx
nx = fetch("robustbase", "NOxEmissions")
Xn = np.column_stack([np.ones(len(nx)), nx.LNOxEm, nx.sqrtWS])
yn = nx.LNOx.values
on, mn = ols(Xn, yn), mm_estimate(Xn, yn, n_sub=500, seed=3)
out.append(f"\nNOx: n = {len(nx)}")
for lab, f in (("OLS", on), ("MM", mn)):
    out.append(f"  {lab}: " + ", ".join(f"{k} {v:.4f} (SE {s:.4f})" for k, v, s in
                                        zip(["const", "log emission", "sqrt wind"], f.coef, f.se))
               + f"; scale {f.scale:.4f}")
un = mn.resid / mn.scale
out.append(f"  differences MM - OLS in OLS SE: " + ", ".join(
    f"{(a - b) / s:+.1f}" for a, b, s in zip(mn.coef, on.coef, on.se)) +
    f"; |MM residual / scale| > 3: {(np.abs(un) > 3).sum()} rows ({np.mean(np.abs(un) > 3):.2%})")
write("03_large_real.txt", "\n".join(out))

# --------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.6, 4.8))
ok = ~impossible
ax.scatter(rd[ok], u[ok], s=3, color="#9a9993", alpha=0.35, lw=0, rasterized=True,
           label=f"{ok.sum():,} diamonds")
ax.scatter(rd[impossible], u[impossible], s=46, color=ORANGE, edgecolor="white", linewidth=0.7,
           zorder=3, label="Recorded dimensions physically impossible")
ax.axvline(cut, color=INK2, lw=0.8, ls=":")
for h_ in (-3, 3):
    ax.axhline(h_, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log")
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
ax.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 3, 5]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Robust distance of the covariates (MCD, log scale)")
ax.set_ylabel("MM residual / robust scale")
ax.set_title("Outlier map of 53,920 diamonds: the impossible rows stand apart")
ax.legend(loc="lower left", fontsize=9, markerscale=1.0)
fig.savefig(FIG / "fig6_diamonds_outlier_map.png"); plt.close(fig)
