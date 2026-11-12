"""Figures for the model-choice companion. Run after game_numbers.py (reads figure_data.json).

fig1_noise_and_prices.png        Tariffs that differ only in noise: equilibrium price level and profit against sigma
fig2_two_insurer_payoffs.png     Two insurers, curse-aware pricing: the four GLM/boosting payoffs against book size
fig3_price_levels.png            What policyholders pay in all-GLM, all-subagged and all-boosting markets of two
fig4_four_insurers.png           Four insurers: payoff of a boosting and of a GLM insurer by the number using boosting
fig5_reviews_shares.png          Two insurers under rate reviews to break-even: who keeps the business

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
LAB = {"GLM": "GLM", "GBM": "boosting", "BAG": "subagged boosting"}
PORT = {"freMTPL2": "France (freMTPL2)", "beMTPL97": "Belgium (beMTPL97)"}
IDX = {"GLM": 0, "GBM": 1, "BAG": 2}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.handlelength": 3.4,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
CREDIT = ("Figure: Diogo Ribeiro. Data: freMTPL2 and beMTPL97 (CASdatasets). "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration")


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def sizes(port):
    return sorted(int(n) for n in DATA[port]["aware10"])


def size_axis(ax, ns) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ns, [f"{n // 1000}k" for n in ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("policies in each insurer's book")


def figure_mechanism() -> None:
    sym = DATA["freMTPL2"]["mechanism"]["sym"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), gridspec_kw={"wspace": 0.22})
    styles = {("2", "aware"): (ORANGE, "-"), ("2", "naive"): (ORANGE, "--"),
              ("4", "aware"): (BLUE, "-"), ("4", "naive"): (BLUE, "--")}
    for (k, how), (col, ls) in styles.items():
        row = sym[f"{k}_{how}"]
        sg = [r[0] for r in row]
        lab = f"{k} insurers, {'curse-aware' if how == 'aware' else 'naive'} pricing"
        axes[0].plot(sg, [r[2] for r in row], ls, color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
        axes[1].plot(sg, [r[1] for r in row], ls, color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
    axes[0].axhline(1, color=MUTED, lw=1)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel("premium paid / expected claims")
    axes[1].set_ylabel("profit of each insurer, % of the market's claims")
    axes[0].set_title("What policyholders pay in equilibrium")
    axes[1].set_title("What each insurer earns")
    for ax in axes:
        ax.set_xlabel("noise in each tariff (sd of its log error), equal ranking otherwise")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig1_noise_and_prices.png")
    plt.close(fig)


def figure_two_payoffs() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.18})
    cells = [(("GLM", "GLM"), VIOLET, "-", "GLM against GLM"),
             (("GBM", "GLM"), BLUE, "--", "boosting against GLM"),
             (("GLM", "GBM"), VIOLET, "--", "GLM against boosting"),
             (("GBM", "GBM"), BLUE, "-", "boosting against boosting")]
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        ns = sizes(port)
        for (s, t), col, ls, lab in cells:
            y = [DATA[port]["aware10"][str(n)]["A"][IDX[s]][IDX[t]] for n in ns]
            ax.plot(ns, y, ls, color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
        size_axis(ax, ns)
        ax.set_title(PORT[port])
    axes[0].set_ylabel("profit of the first-named insurer, % of claims")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig2_two_insurer_payoffs.png")
    plt.close(fig)


def figure_levels() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        ns = sizes(port)
        for c in ("GLM", "BAG", "GBM"):
            y = [DATA[port]["aware10"][str(n)]["level"][IDX[c]][IDX[c]] for n in ns]
            yn = [DATA[port]["naive10"][str(n)]["level"][IDX[c]][IDX[c]] for n in ns]
            ax.plot(ns, y, "-", color=COL[c], lw=2, marker="o", ms=5, mec="white", label=f"two {LAB[c]} insurers, curse-aware")
            ax.plot(ns, yn, ":", color=COL[c], lw=1.5)
        size_axis(ax, ns)
        ax.set_title(PORT[port])
    axes[0].plot([], [], ":", color=MUTED, label="dotted: the same with naive pricing")
    axes[0].set_ylabel("premium paid / claims (beta 10)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig3_price_levels.png")
    plt.close(fig)


def figure_four() -> None:
    panels = [("freMTPL2", 20_000), ("freMTPL2", 100_000), ("beMTPL97", 24_000)]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), gridspec_kw={"wspace": 0.22, "hspace": 0.42})
    j = np.arange(5)
    for row, how in enumerate(("aware", "naive")):
        for col, (port, n) in enumerate(panels):
            ax = axes[row, col]
            d = DATA[port]["four"][f"{n}_{how}_10"]
            pg = np.array([np.nan if v is None else v for v in d["pg"]])
            pl = np.array([np.nan if v is None else v for v in d["pl"]])
            ax.plot(j, pg, "-", color=BLUE, lw=2, marker="o", ms=6, mec="white", label="a boosting insurer")
            ax.plot(j, pl, "-", color=VIOLET, lw=2, marker="o", ms=6, mec="white", label="a GLM insurer")
            for e in d["eq"]:
                y = pg[e] if e > 0 else pl[e]
                ax.plot([e], [y], "o", ms=13, mfc="none", mec=ORANGE, mew=2)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_xticks(j)
            ax.set_title(f"{PORT[port].split(' ')[0]}, books of {n // 1000}k, {'curse-aware' if how == 'aware' else 'naive'}",
                         fontsize=10)
            if row == 1:
                ax.set_xlabel("number of the four insurers using boosting")
            if col == 0:
                ax.set_ylabel("profit, % of the market's claims")
    axes[0, 0].plot([], [], "o", ms=10, mfc="none", mec=ORANGE, mew=2, label="equilibrium")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.1)
    credit(fig)
    fig.savefig(OUT / "fig4_four_insurers.png")
    plt.close(fig)


def figure_reviews() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    pairs = [(("GBM", "GLM"), BLUE, "boosting against GLM"), (("BAG", "GLM"), AQUA, "subagged against GLM"),
             (("BAG", "GBM"), ORANGE, "subagged against boosting")]
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        ns = sizes(port)
        for (s, t), col, lab in pairs:
            y = [DATA[port]["review10"][str(n)][IDX[s]][IDX[t]][0] for n in ns]
            ax.plot(ns, y, "-", color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
        ax.axhline(0.5, color=MUTED, lw=1)
        size_axis(ax, ns)
        ax.set_title(PORT[port])
    axes[0].set_ylabel("share of the first-named insurer after 40 rate reviews")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig5_reviews_shares.png")
    plt.close(fig)


def main() -> None:
    figure_mechanism()
    figure_two_payoffs()
    figure_levels()
    figure_four()
    figure_reviews()


if __name__ == "__main__":
    main()
