"""Experiment 5 - weights that are right, roughly right and wrong, with classical and HC3 standard errors.

Model: y = 1 + 1 * x + sigma(x) * eps, eps ~ N(0, 1), x drawn with replacement
from the incomes of single-person households in wooldridge::k401ksubs
($1,000s, 10 to 143). True variance: sigma(x) proportional to x^gamma with
gamma in {-0.5, 0, 0.5, 1}, scaled so that E sigma^2 = 1:
  gamma = -0.5  the variance falls with income (every weighting scheme below
                points the wrong way)
  gamma =  0    constant variance (weighting is pure noise)
  gamma =  0.5  variance proportional to income (Wooldridge's 1/inc weights are exact)
  gamma =  1    standard deviation proportional to income (1/inc^2 weights are exact)
n in {50, 200, 1000}; 2,000 samples per cell.

Estimators of the slope:
  OLS
  WLS, weights 1/x        (assumes var proportional to x)
  WLS, weights 1/x^2      (assumes sd proportional to x)
  FGLS, power             (log e^2 regressed on log x; estimates gamma)
  FGLS, exponential       (log e^2 regressed on x; the wrong functional form)
each with classical and HC3 standard errors. Reported: the RMSE of the slope
relative to OLS, the coverage of 95% intervals (t(n - 2) critical values),
and the mean standard error divided by the Monte Carlo SD of the estimator.

Run:  python 05_simulation_wls.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import AQUA, BLUE, FIG, INK, INK2, ORANGE, RES, VIOLET, fetch, style, write
from het import fgls, ols, wls

warnings.filterwarnings("ignore")
style()
k = fetch("wooldridge", "k401ksubs")
INCOME = k[k.fsize == 1].inc.values.astype(float)
GAMMAS = (-0.5, 0.0, 0.5, 1.0)
NS = (50, 200, 1000)
REPS = 2000
EST = ["OLS", "WLS 1/x", "WLS 1/x^2", "FGLS power", "FGLS exponential"]
rows = []
for gi, g in enumerate(GAMMAS):
    scale = np.mean(INCOME ** (2 * g))
    for ni, n in enumerate(NS):
        rng = np.random.default_rng([5, gi, ni])
        tcrit = stats.t.ppf(0.975, n - 2)
        for rep in range(REPS):
            x = rng.choice(INCOME, n, replace=True)
            X = np.column_stack([np.ones(n), x])
            y = 1 + x + x ** g / np.sqrt(scale) * rng.standard_normal(n)
            fits = {"OLS": ols(X, y), "WLS 1/x": wls(X, y, 1 / x), "WLS 1/x^2": wls(X, y, 1 / x ** 2),
                    "FGLS power": fgls(X, y, x, form="power")[0],
                    "FGLS exponential": fgls(X, y, x, form="exponential")[0]}
            for e, f in fits.items():
                b = f.coef[1]
                sc, sh = f.se("const")[1], f.se("HC3")[1]
                rows.append((g, n, rep, e, b, sc, sh, abs(b - 1) <= tcrit * sc, abs(b - 1) <= tcrit * sh))
df = pd.DataFrame(rows, columns=["gamma", "n", "rep", "estimator", "slope", "se_classical", "se_hc3",
                                 "cover_classical", "cover_hc3"])

summ = []
out = [f"Slope estimators under four variance functions ({REPS} samples per cell; Monte Carlo SE of a 95% "
       f"coverage ~ {np.sqrt(.95 * .05 / REPS):.4f})"]
for (g, n), s in df.groupby(["gamma", "n"], sort=False):
    rmse_ols = np.sqrt(np.mean((s[s.estimator == "OLS"].slope - 1) ** 2))
    out.append(f"-- gamma {g:+.1f}, n {n}")
    for e in EST:
        t = s[s.estimator == e]
        sd = t.slope.std(ddof=1)
        rmse = np.sqrt(np.mean((t.slope - 1) ** 2))
        r = {"gamma": g, "n": n, "estimator": e, "bias": t.slope.mean() - 1, "sd": sd,
             "rmse_ratio": rmse / rmse_ols, "cover_classical": t.cover_classical.mean(),
             "cover_hc3": t.cover_hc3.mean(), "se_ratio_classical": t.se_classical.mean() / sd,
             "se_ratio_hc3": t.se_hc3.mean() / sd}
        summ.append(r)
        out.append(f"   {e:17s} RMSE/OLS {r['rmse_ratio']:.3f}; bias {r['bias']:+.5f}; coverage classical "
                   f"{r['cover_classical']:.4f}, HC3 {r['cover_hc3']:.4f}; mean SE / SD classical "
                   f"{r['se_ratio_classical']:.3f}, HC3 {r['se_ratio_hc3']:.3f}")
S = pd.DataFrame(summ)
S.to_csv(RES / "05_wls_summary.csv", index=False)
out.append("Coverage ranges over the four variance functions, by estimator and n (classical; HC3):")
for e in EST:
    for n in NS:
        t = S[(S.estimator == e) & (S.n == n)]
        out.append(f"   {e:17s} n {n:4d}: classical {t.cover_classical.min():.4f}-{t.cover_classical.max():.4f}; "
                   f"HC3 {t.cover_hc3.min():.4f}-{t.cover_hc3.max():.4f}")
w = S[S.estimator != "OLS"]
out.append(f"HC3 coverage of the four weighted estimators over all cells: {w.cover_hc3.min():.4f}-{w.cover_hc3.max():.4f}; "
           f"classical {w.cover_classical.min():.4f}-{w.cover_classical.max():.4f}")
write("05_simulation_wls.txt", "\n".join(out))

# --------------------------------------------------------------- figure
COL = {"OLS": INK2, "WLS 1/x": AQUA, "WLS 1/x^2": BLUE, "FGLS power": ORANGE, "FGLS exponential": VIOLET}
MK = {"OLS": "o", "WLS 1/x": "^", "WLS 1/x^2": "v", "FGLS power": "D", "FGLS exponential": "s"}
fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.4), sharey="row",
                         gridspec_kw={"hspace": 0.55, "wspace": 0.12})
titles = {-0.5: "Variance falls with x\n(γ = −0.5)", 0.0: "Constant variance\n(γ = 0)",
          0.5: "Var ∝ x\n(γ = 0.5)", 1.0: "SD ∝ x\n(γ = 1)"}
for c, g in enumerate(GAMMAS):
    s = S[S.gamma == g]
    ax = axes[0, c]
    for e in EST:
        t = s[s.estimator == e]
        ax.plot(range(len(NS)), t.rmse_ratio, marker=MK[e], color=COL[e], lw=1.5, ms=5, label=e)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_yscale("log")
    ax.set_title(titles[g], fontsize=10)
    ax.set_xticks(range(len(NS)), [str(n) for n in NS])
    if c == 0:
        ax.set_ylabel("RMSE of slope relative to OLS\n(log scale)")
    ax = axes[1, c]
    for e in EST:
        t = s[s.estimator == e]
        ax.plot(range(len(NS)), t.cover_classical, marker=MK[e], color=COL[e], lw=1.2, ms=4, ls="--", alpha=0.85)
        ax.plot(range(len(NS)), t.cover_hc3, marker=MK[e], color=COL[e], lw=1.5, ms=5)
    ax.axhline(0.95, color=INK, lw=0.8)
    ax.set_xticks(range(len(NS)), [str(n) for n in NS])
    ax.set_xlabel("Sample size n")
    if c == 0:
        ax.set_ylabel("Coverage of 95% interval")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
for ax in axes[0]:
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.5, 3.3)
for ax in axes[1]:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
h, l = axes[0, 0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], color=INK2, lw=1.5), Line2D([], [], color=INK2, lw=1.2, ls="--")]
l += ["HC3 standard error (bottom row)", "classical standard error (bottom row)"]
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.suptitle("Weights pay when they point the right way; HC3 keeps the interval honest when they do not",
             fontsize=11.5, fontweight="bold", y=1.0)
fig.savefig(FIG / "fig6_wls_simulation.png"); plt.close(fig)
