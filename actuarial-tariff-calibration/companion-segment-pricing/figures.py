"""Figures for the segment-pricing companion. Run after segment_numbers.py (reads figure_data.json).

fig1_noise_by_segmentation.png   Tariffs that differ only in noise: profit per insurer and dispersion of the final
                                 quotes against sigma, for one level, risk bands and market position
fig2_market_position_levels.png  Levels by market-position decile chosen by a GLM and a boosting insurer
fig3_one_level_vs_position.png   Two insurers: payoffs of the uniform markets, one level against market position

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
KIND_COL = {"one level": ORANGE, "risk band": AQUA, "market position": BLUE}
PORT = {"freMTPL2": "France (freMTPL2)", "beMTPL97": "Belgium (beMTPL97)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.handlelength": 3.4, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
CREDIT = ("Figure: Diogo Ribeiro. Data: freMTPL2 and beMTPL97 (CASdatasets). "
          "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration")


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def figure_noise() -> None:
    mech = DATA["freMTPL2"]["mechanism"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), gridspec_kw={"wspace": 0.22, "hspace": 0.38})
    for row, K in enumerate((2, 4)):
        for kind, col in KIND_COL.items():
            for how, ls in (("aware", "-"), ("naive", "--")):
                d = mech[f"{K}|{how}|{kind}"]
                sg = [r[0] for r in d]
                lab = f"{kind}, {'curse-aware' if how == 'aware' else 'naive'}"
                axes[row, 0].plot(sg, [r[1] for r in d], ls, color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
                axes[row, 1].plot(sg, [r[3] for r in d], ls, color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
        axes[row, 0].set_title(f"{K} insurers: profit of each insurer")
        axes[row, 1].set_title(f"{K} insurers: dispersion of the final quotes")
        axes[row, 0].set_ylabel("% of the market's expected claims")
        axes[row, 1].set_ylabel("sd of log quotes for the same policy")
        axes[row, 1].plot([0, 0.4], [0, 0.4], ":", color=MUTED, lw=1)
        for ax in axes[row]:
            ax.set_xlabel("noise in each tariff (sd of its log error)")
    axes[0, 1].text(0.33, 0.36, "quotes as noisy\nas the tariffs", fontsize=8, color=MUTED, ha="right")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08)
    credit(fig)
    fig.savefig(OUT / "fig1_noise_by_segmentation.png")
    plt.close(fig)


def figure_levels() -> None:
    panels = [("freMTPL2", 100_000), ("beMTPL97", 24_000)]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True, gridspec_kw={"wspace": 0.08})
    dec = np.arange(1, 11)
    for ax, (port, n) in zip(axes, panels):
        for c, col in (("GLM", VIOLET), ("GBM", BLUE)):
            for how, ls in (("aware", "-"), ("naive", "--")):
                L = DATA[port]["profile"][f"{how}|{c}|{n}"]
                lab = f"{'boosting' if c == 'GBM' else 'GLM'} insurer, {'curse-aware' if how == 'aware' else 'naive'}"
                ax.plot(dec, L, ls, color=col, lw=2, marker="o", ms=5, mec="white", label=lab)
        ax.axhline(1, color=MUTED, lw=1)
        ax.set_xticks(dec)
        ax.set_xlabel("decile of the insurer's rate relative to its competitor's (1 = relatively cheapest)")
        ax.set_title(f"{PORT[port]}, books of {n // 1000}k")
    axes[0].set_ylabel("level applied to the tariff in that decile")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig2_market_position_levels.png")
    plt.close(fig)


def figure_payoffs() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.18})
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        ns = sorted({int(k.split("|")[2]) for k in DATA[port]["two"]})
        for kind, ls in (("one level", "-"), ("market position", "--")):
            for c, idx, col in (("GLM", 0, VIOLET), ("GBM", 1, BLUE)):
                y = [DATA[port]["two"][f"{kind}|aware|{n}"]["A"][idx][idx] for n in ns]
                ax.plot(ns, y, ls, color=col, lw=2, marker="o", ms=5, mec="white",
                        label=f"two {'boosting' if c == 'GBM' else 'GLM'} insurers, {kind}")
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{n // 1000}k" for n in ns])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("policies in each insurer's book")
        ax.set_title(PORT[port])
    axes[0].set_ylabel("profit of each insurer, % of claims (curse-aware, beta 10)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig3_one_level_vs_position.png")
    plt.close(fig)


def main() -> None:
    figure_noise()
    figure_levels()
    figure_payoffs()


if __name__ == "__main__":
    main()
