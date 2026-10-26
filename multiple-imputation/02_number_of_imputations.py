"""Experiment 2 - how many imputations?

One fixed sample of 1,000 men from the real CPS 1988 population, with 40% of
education deleted under MAR-Y (low earners more likely to be missing). For each
m, the whole multiple imputation (PMM, with the wage interactions of
experiment 1 in the imputation model) is repeated 200 times with different
seeds; the spread of the pooled results across repeats is the Monte Carlo error
that m controls. The observed coefficient of variation of the pooled standard
error is compared with lambda / sqrt(2(m - 1)), which inverts to von Hippel's
(2020) quadratic rule m = 1 + (1/2) (FMI / CV)^2.

Run:  python 02_number_of_imputations.py      (about 2 minutes)
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit

from common import BLUE, FIG, INK2, ORANGE, RES, load_cps, style, write
from mi import mice, ols, pool

warnings.filterwarnings("ignore")
style()
POP = load_cps()
ANALYSIS = ["educ", "exper", "exper2", "afam", "smsa"]
AUX = ["parttime", "northeast", "midwest", "west"]
rng = np.random.default_rng(2020)
df = POP.sample(1000, random_state=7).reset_index(drop=True)
drv = -(df.logwage - df.logwage.mean()) / df.logwage.std()
c = optimize.brentq(lambda c: expit(c + 1.5 * drv).mean() - 0.4, -10, 10)
miss = rng.random(len(df)) < expit(c + 1.5 * drv)
y, ex = df.logwage.values, df.exper.values
P = np.column_stack([df[["logwage"] + ANALYSIS[1:] + AUX].values, y ** 2, y * ex / 10,
                     y * ex ** 2 / 100, y * df.afam, y * df.smsa, y * df.parttime,
                     ex ** 3 / 1000])          # the "PMM with interactions" imputation model
D = np.column_stack([np.where(miss, np.nan, df.educ), P])
Xrest = [df[c].values for c in ANALYSIS[1:]]


def analyse(educ):
    X = np.column_stack([np.ones(len(educ)), educ] + Xrest)
    b, v = ols(X, df.logwage.values, robust=True)
    return b[1], v[1]


M_GRID, REPEATS = (2, 3, 5, 10, 20, 50, 100), 200
rows = []
for m in M_GRID:
    for r in range(REPEATS):
        imps = mice(D, {0: "pmm"}, m=m, seed=[m, r])
        Q, U = zip(*(analyse(Z[:, 0]) for Z in imps))
        p = pool(np.array(Q)[:, None], np.array(U)[:, None], len(df) - 6)
        lo, hi = p.ci()
        rows.append((m, r, p.est[0], p.se[0], p.df[0], p.fmi[0], lo[0], hi[0],
                     2 * stats.t.sf(abs(p.est[0] / p.se[0]) , p.df[0])))
d = pd.DataFrame(rows, columns=["m", "r", "est", "se", "df", "fmi", "lo", "hi", "p"])
d.to_csv(RES / "02_repeats.csv.gz", index=False)

g = d.groupby("m")
fmi_ref = g.fmi.mean().loc[100]
out = [f"One sample, n = {len(df)}, {miss.mean():.1%} of educ missing (MAR-Y); "
       f"{REPEATS} independent MI runs per m (PMM); FMI for educ at m = 100: "
       f"{fmi_ref:.3f} (range over runs {g.fmi.min().loc[100]:.3f}-{g.fmi.max().loc[100]:.3f})",
       f"Complete-data estimate on this sample: {analyse(df.educ.values)[0]:.4f}"]
for m in M_GRID:
    x = g.get_group(m)
    cv = x.se.std() / x.se.mean()
    lam = x.fmi.mean()
    out.append(f"  m = {m:3d}: pooled estimate mean {x.est.mean():.4f}, SD across runs "
               f"{x.est.std():.5f} ({x.est.std() / x.se.mean():.2f} SE); SE mean "
               f"{x.se.mean():.5f}, CV of SE {cv:.3f} (formula FMI/sqrt(2(m-1)) with "
               f"FMI {fmi_ref:.3f}: {fmi_ref / np.sqrt(2 * (m - 1)):.3f}); mean FMI estimate {lam:.3f}, "
               f"SD of FMI estimate {x.fmi.std():.3f}; df median {x.df.median():.0f}; "
               f"upper CI limit range {x.hi.min():.4f}-{x.hi.max():.4f}")
for cvt in (0.10, 0.05, 0.02):
    out.append(f"von Hippel's rule with FMI {fmi_ref:.3f}, target CV(SE) {cvt}: m = "
               f"{1 + 0.5 * (fmi_ref / cvt) ** 2:.1f}")
out.append(f"Rule of thumb m >= 100 x FMI: m = {100 * fmi_ref:.0f}; m >= % incomplete "
           f"cases: m = {100 * miss.mean():.0f}")
write("02_number_of_imputations.txt", "\n".join(out))

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
ms = np.array(M_GRID)
cvs = g.se.std() / g.se.mean()
mm = np.linspace(2, 100, 300)
ax = axes[0]
ax.plot(mm, fmi_ref / np.sqrt(2 * (mm - 1)), color=INK2, lw=1.4, ls="--",
        label="FMI / √(2(m − 1))")
ax.plot(ms, cvs.values, "o", color=BLUE, ms=7, label="Observed over 200 runs")
ax.set_xscale("log"); ax.set_xticks(ms); ax.set_xticklabels(ms)
ax.set_xlabel("Number of imputations m"); ax.set_ylabel("Coefficient of variation of the SE")
ax.set_title("The standard error itself is noisy at small m")
ax.legend()
ax = axes[1]
bins = np.linspace(g.get_group(5).hi.min(), g.get_group(5).hi.max(), 36)
for m, col in ((5, ORANGE), (100, BLUE)):
    x = g.get_group(m)
    ax.hist(x.hi, bins=bins, color=col, alpha=0.75, label=f"m = {m}")
ax.set_xlabel("Upper limit of the 95% interval for the return to schooling")
ax.set_ylabel("MI runs")
ax.set_title("Same data, same method, different seed")
ax.legend()
fig.savefig(FIG / "fig3_number_of_imputations.png"); plt.close(fig)
