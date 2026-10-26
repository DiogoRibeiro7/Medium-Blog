"""Experiment 2 - two real data sets whose outliers have a known cause.

* Belgian telephone calls 1950-1973 (MASS::phones; Rousseeuw & Leroy 1987):
  for 1964-1969 the series recorded the total number of minutes of calls
  instead of the number of calls.
* Star cluster CYG OB1 (robustbase::starsCYG): 43 main-sequence stars and four
  red giants (observations 11, 20, 30, 34) that are a different population.

For each: OLS, LAD, Huber M, bisquare M from OLS, LTS, S and MM, the rows each
flags, and an outlier map (robust distance of x from the MCD against the
standardised MM residual; Rousseeuw & van Zomeren 1990).

Run:  python 02_classic_cases.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.covariance import MinCovDet

from common import BLUE, FIG, INK2, ORANGE, AQUA, VIOLET, fetch, style, write
from rob import flag_outliers, lts, m_estimate, mm_estimate, ols

warnings.filterwarnings("ignore")
style()
out = []


def fits(X, y):
    q = sm.QuantReg(y, X).fit(q=0.5)
    return {
        "OLS": ols(X, y, robust_se=False),
        "LAD": q.params,
        "Huber M": m_estimate(X, y, "huber"),
        "Bisquare M from OLS": m_estimate(X, y, "bisquare"),
        "LTS": lts(X, y, n_sub=3000, seed=1),
        "MM": mm_estimate(X, y, n_sub=3000, seed=1),
    }


def coef(f):
    return f if isinstance(f, np.ndarray) else f.coef


ph = fetch("MASS", "phones")
st = fetch("robustbase", "starsCYG")
cases = {
    "phones": (ph.year.values.astype(float), ph.calls.values,
               np.isin(ph.year.values, range(64, 70)), "Year (19xx)", "Calls (millions)"),
    "stars": (st["log.Te"].values, st["log.light"].values,
              np.isin(np.arange(1, len(st) + 1), [11, 20, 30, 34]),
              "log surface temperature", "log light intensity"),
}
res = {}
for name, (x, y, known, *_l) in cases.items():
    X = np.column_stack([np.ones(len(x)), x])
    F = fits(X, y)
    res[name] = F
    out.append(f"== {name}: n = {len(x)}, known anomalies at rows "
               f"{(np.flatnonzero(known) + 1).tolist()}")
    for k, f in F.items():
        c = coef(f)
        extra = ""
        if k == "OLS":
            # externally studentised residuals and Cook's distance
            infl = sm.OLS(y, X).fit().get_influence()
            t = infl.resid_studentized_external
            cook = infl.cooks_distance[0]
            extra = (f"; |t| > 2.5: rows {(np.flatnonzero(np.abs(t) > 2.5) + 1).tolist()}; "
                     f"Cook's D > 4/n: rows {(np.flatnonzero(cook > 4 / len(x)) + 1).tolist()}")
        if k in ("LTS", "MM"):
            fl = flag_outliers(f, 2.5)
            extra = f"; |robust residual / scale| > 2.5: rows {(np.flatnonzero(fl) + 1).tolist()}"
        se = ""
        if hasattr(f, "cov") and f.cov is not None:
            se = f" (SE {f.se[1]:.4f})"
        out.append(f"  {k:20s} intercept {c[0]:9.4f}  slope {c[1]:8.4f}{se}{extra}")
    mm = F["MM"]
    out.append(f"  MM scale {mm.scale:.4f}; S start {np.round(mm.init.coef, 4).tolist()}, "
               f"S scale {mm.init.scale:.4f}")
    if name == "stars":
        mcd = MinCovDet(random_state=0).fit(x[:, None])
        rd = np.sqrt(mcd.mahalanobis(x[:, None]))
        cut = np.sqrt(stats.chi2.ppf(0.975, 1))
        lev = rd > cut
        out.append(f"  MCD robust distance > {cut:.2f}: rows {(np.flatnonzero(lev) + 1).tolist()}; "
                   f"classical leverage h > 2p/n: rows "
                   f"{(np.flatnonzero(np.diag(X @ np.linalg.inv(X.T @ X) @ X.T) > 4 / len(x)) + 1).tolist()}")
        res["stars_rd"] = rd
write("02_classic_cases.txt", "\n".join(out))

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), gridspec_kw={"width_ratios": [1, 1, 0.9]})
lines = {"OLS": (INK2, "--"), "Huber M": (VIOLET, ":"), "LTS": (AQUA, "-"), "MM": (BLUE, "-")}
for ax, name in zip(axes[:2], ("phones", "stars")):
    x, y, known, xl, yl = cases[name]
    ax.scatter(x[~known], y[~known], s=28, color=INK2, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(x[known], y[known], s=40, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=4,
               label="Known anomaly")
    xs = np.linspace(x.min() - 0.02 * np.ptp(x), x.max() + 0.02 * np.ptp(x), 50)
    for k, (col, ls) in lines.items():
        c = coef(res[name][k])
        ax.plot(xs, c[0] + c[1] * xs, color=col, ls=ls, lw=1.8, label=k)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if name == "stars":
        ax.invert_xaxis()
        ax.set_ylim(3.3, 6.6)
    ax.set_title({"phones": "Belgian phone calls: minutes recorded as calls",
                  "stars": "Star cluster CYG OB1: four giants"}[name], fontsize=10.5)
axes[0].legend(fontsize=8.5, loc="upper left")
ax = axes[2]
x, y, known, *_ = cases["stars"]
mm = res["stars"]["MM"]
u = mm.resid / mm.scale
rd = res["stars_rd"]
ax.scatter(rd[~known], u[~known], s=28, color=INK2, edgecolor="white", linewidth=0.5)
ax.scatter(rd[known], u[known], s=40, color=ORANGE, edgecolor="white", linewidth=0.5)
cut = np.sqrt(stats.chi2.ppf(0.975, 1))
ax.axvline(cut, color=INK2, lw=0.8, ls=":")
for h in (-2.5, 2.5):
    ax.axhline(h, color=INK2, lw=0.8, ls=":")
ax.text(cut + 0.2, 4.6, "bad leverage points:\nunusual x, large residual", fontsize=8.5,
        color=INK2, va="center")
ax.text(cut + 0.2, -1.7, "good leverage points", fontsize=8.5, color=INK2, va="center")
ax.set_xlabel("Robust distance of temperature (MCD)")
ax.set_ylabel("MM residual / robust scale")
ax.set_title("Outlier map for the stars", fontsize=10.5)
fig.savefig(FIG / "fig1_classic_cases.png"); plt.close(fig)
