"""Figures 1 and 2 from the summaries written by 01_cps_mechanisms.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import BLUE, FIG, INK, INK2, ORANGE, RES, style

style()
MECHS = ["MCAR", "MAR-X", "MAR-Y", "MNAR-X"]
MECH_LABEL = {"MCAR": "MCAR", "MAR-X": "MAR on\nexperience", "MAR-Y": "MAR on\nwage",
              "MNAR-X": "MNAR on\neducation"}


def load(pop, par="educ"):
    t = pd.read_csv(RES / f"01_summary_{pop}_{par}.csv")
    return t.set_index(["mech", "method"])


# ---------------------------------------------------------------- figure 1
cmap = LinearSegmentedColormap.from_list("short", ["#fcfcfb", "#f6c9b5", "#eb6834", "#8f3412"])
tabs = {"model": load("model"), "real": load("real")}
methods = [
    "Complete data (no deletion)", "Complete cases", "Mean imputation",
    "Mean + missing indicator", "Regression imputation", "Stochastic regression (single)",
    "kNN imputation (k = 5)", "Random forest imputation", "MI, Bayesian regression",
    "MI, predictive mean matching", "MI, PMM without the outcome",
    "MI, PMM with interactions", "MI, random forest",
]


def pct(x):
    v = int(round(100 * x))
    return f"{v:+d}%" if v != 0 else "0%"
fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6), sharey=True,
                         gridspec_kw={"wspace": 0.04})
for ax, (pop, t), ttl in zip(axes, tabs.items(),
                             ("Wages generated from the Mincer model",
                              "Real wages (Mincer model only approximate)")):
    grid = np.array([[max(0.0, 0.95 - t.loc[(mc, mt)].cover) for mc in MECHS] for mt in methods])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=0.95, aspect="auto")
    for i, mt in enumerate(methods):
        for j, mc in enumerate(MECHS):
            r = t.loc[(mc, mt)]
            dark = grid[i, j] > 0.45
            ax.text(j, i - 0.15, f"{100 * r.cover:.0f}%", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="white" if dark else INK)
            ax.text(j, i + 0.24, f"bias {pct(r.relbias)}", ha="center", va="center",
                    fontsize=7.8, color="white" if dark else INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels([MECH_LABEL[m] for m in MECHS], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(ttl, pad=40, fontsize=11)
fig.suptitle("Coverage of nominal 95% intervals for the return to schooling, 1,000 samples per cell "
             "(bias in small type)", x=0.02, ha="left", fontweight="bold", fontsize=12, y=1.0)
fig.savefig(FIG / "fig1_coverage_grid.png"); plt.close(fig)
t = tabs["model"]

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
single = {"Mean imputation": ("Mean", (8, -4)), "Regression imputation": ("Regression", (8, 0)),
          "kNN imputation (k = 5)": ("kNN", (8, 0)),
          "Random forest imputation": ("Random forest", (8, 2)),
          "MI, PMM without the outcome": ("MI, PMM\nwithout outcome", (-8, 0))}
for ax, mc in zip(axes, ("MCAR", "MAR-Y")):
    for mt, (lab, (dx, dy)) in single.items():
        r = t.loc[(mc, mt)]
        ax.scatter(r.rmse_imp, 100 * r.relbias, s=70, color=ORANGE, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.annotate(lab, (r.rmse_imp, 100 * r.relbias), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5, color=INK2,
                    ha="right" if dx < 0 else "left", va="center")
    st = t.loc[(mc, "Stochastic regression (single)")]
    ax.scatter(st.rmse_imp, 100 * st.relbias, s=70, facecolor="white", edgecolor=ORANGE,
               linewidth=1.8, zorder=4)
    ax.annotate("Stochastic regression,\nsingle draw (SE too small)",
                (st.rmse_imp, 100 * st.relbias), xytext=(-12, -42), textcoords="offset points",
                fontsize=8.5, color=INK2, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    pts = [t.loc[(mc, k)] for k in ("MI, Bayesian regression", "MI, predictive mean matching")]
    for r in pts:
        ax.scatter(r.rmse_imp, 100 * r.relbias, s=70, color=BLUE, edgecolor="white",
                   linewidth=0.8, zorder=5)
    ax.annotate("MI: Bayesian\nregression, PMM", (pts[0].rmse_imp, 100 * pts[0].relbias),
                xytext=(-14, 30), textcoords="offset points", fontsize=8.5, color=BLUE,
                ha="right", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xlabel("RMSE of the imputed education values (years)")
    ax.set_title({"MCAR": "Missing completely at random", "MAR-Y": "Missing at random, on wage"}[mc])
axes[0].set_ylabel("Bias in the return to schooling (%)")
fig.suptitle("The imputations closest to the truth give the most biased coefficient",
             x=0.06, ha="left", fontweight="bold", fontsize=11.5, y=1.02)
fig.savefig(FIG / "fig2_accuracy_vs_validity.png"); plt.close(fig)
print("figures 1-2 written")
