"""Figures 3, 4 and 5 from results/01_summary.csv (written by 01_contamination.py)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from common import AQUA, BLUE, FIG, INK2, MAGENTA, ORANGE, RES, VIOLET, YELLOW, GREEN, style

style()
t = pd.read_csv(RES / "01_summary.csv").set_index(["kind", "level", "method"])
RATES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]


def series(kind, method, col="relbias"):
    vals = [t.loc[("clean", 0.0, method)][col]]
    vals += [t.loc[(kind, r, method)][col] for r in RATES[1:]]
    return np.array(vals)


def log_axis(ax, ticks):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------ figure 3: precision and bias
show = {"OLS": (INK2, "--"), "OLS, drop Cook's D > 4/n": (YELLOW, "-"),
        "LAD": (MAGENTA, "-"), "Huber M (statsmodels)": (VIOLET, "-"),
        "Bisquare M from OLS": (GREEN, "-"), "Theil-Sen": ("#9a9993", "-"),
        "RANSAC": (AQUA, ":"), "LTS": (ORANGE, ":"), "MM": (BLUE, "-")}
fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
x = 100 * np.array(RATES)
ax = axes[0]
for m, (col, ls) in show.items():
    ax.plot(x, series("price", m, "rmse"), color=col, ls=ls, marker="o", ms=4, lw=1.8, label=m)
log_axis(ax, [0.01, 0.02, 0.05, 0.1, 0.2])
ax.set_ylabel("RMSE of the log-carat coefficient (log scale)")
ax.set_title("Prices typed with an extra zero: precision")
ax = axes[1]
for m, (col, ls) in show.items():
    ax.plot(x, 100 * series("carat", m), color=col, ls=ls, marker="o", ms=4, lw=1.8)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylabel("Bias of the log-carat coefficient (%)")
ax.set_title("Carats with a slipped decimal point: bias")
for ax in axes:
    ax.set_xlabel("Share of corrupted rows (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.suptitle("Two kinds of data-entry error, 300 samples of 500 diamonds per point", x=0.06, ha="left",
             fontweight="bold", fontsize=11.5, y=1.02)
fig.savefig(FIG / "fig3_bias_curves.png"); plt.close(fig)

# ------------------------------------------------------- figure 4: detection
rules = {"OLS, drop |t| > 3": (INK2, "--"), "OLS, drop |t| > 3 repeatedly": (VIOLET, "-"),
         "OLS, drop Cook's D > 4/n": (YELLOW, "-"), "MM": (BLUE, "-")}
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
for ax, kind, title in ((axes[0], "price", "Prices typed with an extra zero"),
                        (axes[1], "carat", "Carats with a slipped decimal point")):
    for m, (col, ls) in rules.items():
        tp = [t.loc[(kind, r, m)].tp for r in RATES[1:]]
        ax.plot(x[1:], 100 * np.array(tp), color=col, ls=ls, marker="o", ms=4, lw=1.8,
                label="MM, |residual / scale| > 3" if m == "MM" else m)
    ax.set_xlabel("Share of corrupted rows (%)")
    ax.set_title(title)
    ax.set_ylim(-3, 103)
axes[0].set_ylabel("Corrupted rows flagged (%)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.suptitle("Deletion rules built on least squares stop seeing the errors as they multiply",
             x=0.06, ha="left", fontweight="bold", fontsize=11.5, y=1.02)
fig.savefig(FIG / "fig4_detection.png"); plt.close(fig)

# ------------------------------------------------------ figure 5: efficiency
meth = ["LAD", "Huber M (statsmodels)", "Bisquare M from OLS", "Theil-Sen", "RANSAC", "LTS", "S", "MM"]
levels = [("clean", 0.0, "normal")] + [("tails", float(v), f"t, {v} df") for v in (5, 3, 2)]
fig, ax = plt.subplots(figsize=(9.6, 4.6))
cols = ["#cde2fb", "#86b6ef", "#3987e5", "#0d366b"]
w = 0.2
for j, (kind, lev, lab) in enumerate(levels):
    base = t.loc[(kind, lev, "OLS")].rmse
    vals = [t.loc[(kind, lev, m)].rmse / base for m in meth]
    ax.bar(np.arange(len(meth)) + (j - 1.5) * w, vals, w * 0.92, color=cols[j], label=lab)
ax.axhline(1, color=INK2, lw=0.9, ls="--")
log_axis(ax, [0.4, 0.5, 0.7, 1, 1.5, 2, 3])
ax.set_ylim(0.33, 3.6)
ax.set_xticks(range(len(meth)))
ax.set_xticklabels([m.replace(" (statsmodels)", "").replace(" from OLS", "\nfrom OLS") for m in meth],
                   fontsize=9)
ax.set_ylabel("RMSE relative to OLS (log scale)")
ax.set_title("The price of robustness on clean data, and its return on heavy tails")
ax.legend(title="Error distribution", fontsize=8.5, title_fontsize=9, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.18))
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig5_efficiency.png"); plt.close(fig)
print("figures 3, 4, 5 written")
