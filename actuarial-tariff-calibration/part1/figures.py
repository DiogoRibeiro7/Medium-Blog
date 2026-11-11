"""Figures for Part 1 of the actuarial-tariff-calibration series.

fig1_bailey_simon_tests.png   The Canadian tariffs of 1960 by Bailey and Simon's chi-square, by their balance test
                              and by an exposure-weighted Gini index
fig2_same_ranking.png         Sweden 1977: one ordered Lorenz curve for every power of the GLM tariff, and what each
                              power charges the seven bonus classes
fig3_deviance_by_power.png    Sweden 1977: Poisson deviance of the powered tariffs against the nested GLM tariffs

Run:  python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

from history_numbers import (LOSS, N, PUBLISHED, R, balance_by, canadian, criteria, distorted_premium, gini,
                             lorenz, marginal_totals, poisson_deviance, swedish)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#7a5cd6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 170,
})

CREDIT = "Figure: Diogo Ribeiro. Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration"


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def figure_bailey_simon(can: dict) -> None:
    methods = [("Customary (Method 1)", can["F1"], 12), ("Min. chi-square, x*y (Method 2)", can["F2"], 12),
               ("Minimum bias (Bailey 1963)", can["Fmb"], 12), ("Min. chi-square, x+y (Method 3)", can["F3"], 12),
               ("Min. chi-square, 3xy-2 (Method 4)", can["F4"], 11)]
    names = [m[0] for m in methods]
    chi = [criteria(F)["chi2"] for _, F, _ in methods]
    worst = []
    for _, F, _ in methods:
        c = criteria(F)
        b = np.r_[c["class"], c["merit"]]
        worst.append(100 * np.max(np.abs(b - 1)))
    gin = [gini(F.ravel(), N.ravel(), LOSS.ravel()) for _, F, _ in methods]
    y = np.arange(len(names))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.15], "wspace": 0.16})
    ax = axes[0]
    ax.hlines(y, 0, chi, color=GRID, lw=2)
    ax.plot(chi, y, "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=3)
    crit = stats.chi2.isf(0.05, 12)
    ax.axvline(crit, color=MUTED, lw=1, ls="--")
    ax.text(crit + 2, 4.55, "5% critical value, 12 df", color=INK2, fontsize=8.5, va="center")
    for xv, yv in zip(chi, y):
        ax.text(xv + 3, yv + 0.22, f"{xv:.0f}", color=INK2, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_yticks(y, names)
    ax.set_title("Chi-square")
    ax.set_xlabel("chi-square, K = 1/200")

    ax = axes[1]
    ax.hlines(y, 0, worst, color=GRID, lw=2)
    ax.plot(worst, y, "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=3)
    for xv, yv in zip(worst, y):
        ax.text(xv + 0.3, yv + 0.22, f"{xv:.1f}%", color=INK2, fontsize=9)
    ax.set_xlim(0, 13)
    ax.set_title("Worst class or merit imbalance")
    ax.set_xlabel("|premium / experience - 1|, %")

    ax = axes[2]
    ceiling = gini(R.ravel(), N.ravel(), LOSS.ravel())
    class_only = gini(np.outer(PUBLISHED["Method 1"][0], np.ones(4)).ravel(), N.ravel(), LOSS.ravel())
    merit_only = gini(np.outer(np.ones(5), PUBLISHED["Method 1"][1]).ravel(), N.ravel(), LOSS.ravel())
    ax.hlines(y, 0, gin, color=GRID, lw=2)
    ax.plot(gin, y, "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=3)
    for xv, yv in zip(gin, y):
        ax.text(0.004, yv + 0.22, f"{xv:.4f}", color=INK2, fontsize=9)
    for val, lab in ((ceiling, "ceiling"), (class_only, "class\nonly"), (merit_only, "merit\nonly")):
        ax.axvline(val, color=MUTED, lw=1, ls=":")
        ax.text(val + 0.002, 4.55, lab, color=INK2, fontsize=8.5, ha="left", va="center")
    ax.set_xlim(0, 0.2)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_title("Gini index")
    ax.set_xlabel("Gini of relative losses, 20 cells")
    for a in axes:
        a.grid(axis="y", visible=False)
        a.set_ylim(-0.6, 4.95)
    fig.suptitle("Canada 1957-58: Bailey and Simon's tests separate the five tariffs; the Gini does not",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=1.0)
    fig.subplots_adjust(left=0.185, right=0.98, top=0.84, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig1_bailey_simon_tests.png")
    plt.close(fig)


def figure_same_ranking(sw: dict) -> None:
    E, C, codes, rate = sw["E"], sw["C"], sw["codes"], sw["rate"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(0.62, 0.66, "flat rate", color=INK2, fontsize=9, rotation=38)
    mb = marginal_totals(C, E, codes, ("Bonus",))[0]
    cw, cl = lorenz(mb / E, E, C)
    ax.plot(cw, cl, color=ORANGE, lw=2)
    ax.annotate(f"bonus class only, Gini {gini(mb / E, E, C):.3f}", (0.30, 0.215), xytext=(0.03, 0.62),
                color=INK2, fontsize=9, arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1})
    for gam in (0.5, 1.0, 2.0):
        cw, cl = lorenz(rate ** gam, E, C)
        ax.plot(cw, cl, color=BLUE, lw=2)
    ax.annotate(f"four-factor GLM raised to\nthe power 0.5, 1 or 2:\none curve, Gini {gini(rate, E, C):.3f}",
                (0.62, 0.43), xytext=(0.55, 0.08), color=INK2, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 1})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("cumulative share of policy-years, lowest premium first")
    ax.set_ylabel("cumulative share of claims")
    ax.set_title("Ordered Lorenz curves")

    ax = axes[1]
    ax.hlines(1, 0.7, 7.0, color=MUTED, lw=1)
    lv = np.arange(1, 8)
    cols = {0.5: AQUA, 1.0: BLUE, 2.0: ORANGE}
    for gam in (0.5, 1.0, 2.0):
        b = balance_by(codes["Bonus"], distorted_premium(sw, gam), C)
        ax.plot(lv, b, "o-", color=cols[gam], lw=2, ms=7, mec="white", mew=1.2)
        lab = {0.5: "power 0.5", 1.0: "power 1 (the GLM)", 2.0: "power 2"}[gam]
        ax.text(7.25, b[-1], lab, color=INK2, fontsize=9, va="center")
    ax.set_xlim(0.7, 9.0)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(lv)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.75, 1, 1.5, 2], ["0.5", "0.75", "1", "1.5", "2"])
    ax.minorticks_off()
    ax.set_xlabel("bonus class (years since last claim + 1)")
    ax.set_ylabel("premium / claims (log scale)")
    ax.set_title("What each tariff charges the bonus classes")
    fig.suptitle("Sweden 1977: tariffs with the same ranking and different prices",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig2_same_ranking.png")
    plt.close(fig)


def figure_deviance(sw: dict) -> None:
    g = np.array([t[0] for t in sw["distortions"]])
    dv = np.array([t[2] for t in sw["distortions"]])
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    refs = [("flat rate, Gini 0", sw["flat_dev"])]
    labels = {("Bonus",): "bonus only", ("Bonus", "Zone"): "bonus + zone",
              ("Bonus", "Zone", "Kilometres"): "bonus + zone + distance"}
    for fs, gi, d in sw["nested"][:-1]:
        refs.append((f"{labels[fs]}, Gini {gi:.3f}", d))
    for lab, val in refs:
        ax.axhline(val, color=MUTED, lw=1, ls=":")
        ax.text(3.02, val, lab, color=INK2, fontsize=8.5, va="center")
    ax.plot(g, dv, color=BLUE, lw=2)
    i1 = int(np.argmin(dv))
    ax.plot([g[i1]], [dv[i1]], "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=3)
    ax.annotate(f"power 1: the GLM, deviance {dv[i1]:,.0f}", (g[i1], dv[i1]), xytext=(1.25, 1900), color=INK2,
                fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.text(0.22, 1.1e5, f"Every point on the curve has Gini {sw['nested'][-1][1]:.4f}", color=INK, fontsize=10)
    cross = optimize.brentq(lambda t: poisson_deviance(sw["C"], distorted_premium(sw, t)) - sw["flat_dev"], 1.0, 3.0)
    ax.annotate(f"above power {cross:.2f}: worse than\ncharging everyone the same rate", (cross, sw["flat_dev"]),
                xytext=(2.05, 1.55e4), color=INK2, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_yscale("log")
    ax.set_xlim(0.2, 3.0)
    ax.set_ylim(1500, 2.0e5)
    ticks = [2000, 5000, 10000, 20000, 50000, 100000]
    ax.set_yticks(ticks, [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("power applied to the four-factor GLM frequency (tariff rebalanced to total claims)")
    ax.set_ylabel("Poisson deviance (log scale)")
    ax.set_title("Sweden 1977: Poisson deviance of the powered tariffs")
    fig.subplots_adjust(left=0.1, right=0.73, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig3_deviance_by_power.png")
    plt.close(fig)


def main() -> None:
    can = canadian()
    figure_bailey_simon(can)
    sw = swedish()
    figure_same_ranking(sw)
    figure_deviance(sw)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
