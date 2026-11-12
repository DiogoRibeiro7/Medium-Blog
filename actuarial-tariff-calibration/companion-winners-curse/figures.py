"""Figures for the winner's-curse companion. Run after curse_numbers.py (reads figure_data.json).

fig1_collected_by_k.png      What a market of K insurers of one class collects, against K, with the lognormal prediction
fig2_book_size.png           One boosting insurer against one GLM insurer as the books grow: twins and head-to-head
fig3_gini_vs_market.png      Single-insurer Gini against what a two-insurer market of that class collects
fig4_breakeven_levels.png    The average level policyholders pay once every insurer has reviewed its rates

Run:  python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DATA = json.loads((HERE / "figure_data.json").read_text())

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#7a5cd6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"
COL = {"GLM": VIOLET, "GBM": BLUE, "BAG": AQUA}
LAB = {"GLM": "GLM", "GBM": "boosting", "BAG": "boosting, subagged"}
PORT = {"freMTPL2": "France (freMTPL2)", "beMTPL97": "Belgium (beMTPL97)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
CREDIT = ("Figure: Diogo Ribeiro. Data: freMTPL2 and beMTPL97 (CASdatasets). "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration")


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def figure_collected() -> None:
    ks = [2, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        for c in ("GLM", "BAG", "GBM"):
            h = DATA[port]["homo"][c]
            mean = [np.mean(h[str(k)]["coll"]) for k in ks]
            lo = [np.min(h[str(k)]["coll"]) for k in ks]
            hi = [np.max(h[str(k)]["coll"]) for k in ks]
            pred = [np.mean(h[str(k)]["lognormal"]) for k in ks]
            ax.vlines(ks, lo, hi, color=COL[c], lw=2, alpha=0.5)
            ax.plot(ks, mean, "o", color=COL[c], ms=7, mec="white", mew=1.2, label=f"{LAB[c]}: market")
            ax.plot(ks, pred, "--", color=COL[c], lw=1.5, label=f"{LAB[c]}: lognormal prediction")
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("number of insurers in the market (each with its own book)")
        ax.set_title(PORT[port])
    axes[0].set_ylabel("premium collected / claims (cheapest quote wins)")
    axes[0].legend(loc="lower left", fontsize=8.5, ncol=1)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig1_collected_by_k.png")
    plt.close(fig)


def figure_book_size() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), gridspec_kw={"wspace": 0.18, "hspace": 0.42})
    for j, port in enumerate(("freMTPL2", "beMTPL97")):
        S = DATA[port]["sizes"]
        ns = sorted(S, key=int)
        x = [int(n) for n in ns]
        ax = axes[0, j]
        for c in ("GLM", "GBM"):
            ax.plot(x, [np.mean(S[n]["twin_coll"][c]) for n in ns], "-o", color=COL[c], lw=2, ms=6, mec="white",
                    mew=1, label=f"two {LAB[c]} insurers")
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v // 1000}k" for v in x])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"{PORT[port]}: what a market of twins collects")
        ax.set_ylabel("premium collected / claims")
        ax.legend(loc="lower right", fontsize=9)
        ax = axes[1, j]
        ax.axhline(0.5, color=MUTED, lw=1)
        for i, n in enumerate(ns):
            v = S[n]["endG"]
            ax.plot([x[i]] * len(v), v, "o", color=BLUE, ms=4, alpha=0.35, mec="none")
        ax.plot(x, [np.mean(S[n]["endG"]) for n in ns], "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1,
                label="mean over runs")
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v // 1000}k" for v in x])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("Boosting against GLM: boosting's share after rate reviews")
        ax.set_ylabel("boosting insurer's share of exposure")
        ax.set_xlabel("policies in each insurer's book")
        ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08)
    credit(fig)
    fig.savefig(OUT / "fig2_book_size.png")
    plt.close(fig)


def figure_gini_vs_market() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.2})
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        S = DATA[port]["sizes"]
        ns = sorted(S, key=int)
        for c in ("GLM", "GBM"):
            g = [np.mean(S[n]["gini"][c]) for n in ns]
            m = [np.mean(S[n]["twin_coll"][c]) for n in ns]
            ax.plot(g, m, "-o", color=COL[c], lw=2, ms=7, mec="white", mew=1, label=LAB[c])
            for gi, mi, n in zip(g, m, ns):
                ax.annotate(f"{int(n) // 1000}k", (gi, mi), xytext=(4, -11 if c == "GBM" else 5),
                            textcoords="offset points", fontsize=8, color=INK2)
        ax.set_xlabel("Gini index of one insurer alone (test sample, claim counts)")
        ax.set_title(PORT[port])
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("what a market of two such insurers collects")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig3_gini_vs_market.png")
    plt.close(fig)


def figure_levels() -> None:
    ks = [2, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        for c in ("GLM", "BAG", "GBM"):
            h = DATA[port]["homo"][c]
            ax.plot(ks, [np.mean(h[str(k)]["level_10"]) for k in ks], "-o", color=COL[c], lw=2, ms=7, mec="white",
                    mew=1, label=LAB[c])
        ax.axhline(1, color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("number of insurers in the market")
        ax.set_title(PORT[port])
    axes[0].set_ylabel("average level paid after rate reviews (beta = 10)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig4_breakeven_levels.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_collected()
    figure_book_size()
    figure_gini_vs_market()
    figure_levels()
    print(f"figures written to {OUT}")
