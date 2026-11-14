"""Figures for the design-differences companion. Run after design_numbers.py (reads figure_data.json).

fig1_data_and_design_parts.png  The part of the disagreement between two designs that more data would not remove,
                                against book size, beside the twin dispersion of the GLM and the GBM; and the share
                                of the squared disagreement it accounts for
fig2_profit_gap.png             Two curse-aware insurers with different designs: the profit of the first minus the
                                profit of the second, with one level and with market-position pricing

Run:  python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
SIZES = {"freMTPL2": (20_000, 50_000, 100_000, 200_000), "beMTPL97": (5_000, 12_000, 24_000, 48_000)}
PORT = {"freMTPL2": "France (freMTPL2)", "beMTPL97": "Belgium (beMTPL97)"}
# Fixed order: each pair keeps its colour and marker in every panel.
PAIRS = [(("GLM", "linear BM"), BLUE, "o", "GLM vs linear BM"),
         (("GLM", "no territory"), ORANGE, "s", "GLM vs no territory"),
         (("GLM", "GBM"), AQUA, "^", "GLM vs GBM"),
         (("GBM", "other GBM"), VIOLET, "D", "GBM vs other GBM")]

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


def end_labels(ax, x, items, gap) -> None:
    """Direct labels at the right end of lines, nudged apart so that none overlap (gap in data units)."""
    items = sorted(items)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (y0, text), y in zip(items, ys):
        ax.annotate(text, (x, y0), xytext=(x * 1.07, y), textcoords="data", fontsize=8, color=INK2, va="center")


def size_axis(ax, sizes) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n // 1000}k" for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("policies in each insurer's book")


def figure_parts() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), gridspec_kw={"wspace": 0.24, "hspace": 0.42})
    for row, port in enumerate(("freMTPL2", "beMTPL97")):
        d = DATA[port]
        sizes = SIZES[port]
        ax, bx = axes[row]
        for design, ls in (("GLM", "--"), ("GBM", ":")):
            tw = [d["single"][f"{design}|{n}"]["twin"] for n in sizes]
            ax.plot(sizes, tw, ls, color=MUTED, lw=2, label=f"two {design}s on different data (twins)")
            ax.annotate(f"{design} twins", (sizes[-1], tw[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center")
        ends = []
        for (a, b), col, mk, lab in PAIRS:
            des = [d["decomp"][f"{a}|{b}|{n}"]["design"] for n in sizes]
            share = [100 * d["decomp"][f"{a}|{b}|{n}"]["share"] for n in sizes]
            ax.plot(sizes, des, "-", color=col, lw=2, marker=mk, ms=6, mec="white", label=f"{lab}: design part")
            bx.plot(sizes, share, "-", color=col, lw=2, marker=mk, ms=6, mec="white", label=lab)
            ends.append((share[-1], lab))
        end_labels(bx, sizes[-1], ends, 4.0)
        ax.set_title(f"{PORT[port]}: the design part of the disagreement")
        ax.set_ylabel("sd of the log quotes for the same policy")
        ax.set_ylim(0, 0.45)
        bx.set_title(f"{PORT[port]}: design share of the squared dispersion")
        bx.set_ylabel("% of the squared dispersion")
        bx.set_ylim(-5, 80)
        bx.axhline(0, color=MUTED, lw=1)
        for x in (ax, bx):
            size_axis(x, sizes)
            x.set_xlim(sizes[0] * 0.85, sizes[-1] * 2.3)
    axes[0, 0].legend(loc="upper right", fontsize=7.5, ncol=1)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08)
    credit(fig)
    fig.savefig(OUT / "fig1_data_and_design_parts.png")
    plt.close(fig)


def figure_gap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"wspace": 0.24})
    for ax, port in zip(axes, ("freMTPL2", "beMTPL97")):
        d = DATA[port]
        sizes = SIZES[port]
        ends = []
        for (a, b), col, mk, lab in PAIRS:
            for game, ls in (("one level", "-"), ("market position", "--")):
                g = [d["gm"][f"{a}|{b}|{n}|{game}"] for n in sizes]
                gap = [x["pa"] - x["pb"] for x in g]
                ax.plot(sizes, gap, ls, color=col, lw=2, marker=mk, ms=6, mec="white",
                        label=f"{lab}, {game}")
                if game == "one level":
                    ends.append((gap[-1], lab))
        end_labels(ax, sizes[-1], ends, 0.45)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(f"{PORT[port]}: profit gap, first insurer minus second")
        ax.set_ylabel("percentage points of the market's claims")
        size_axis(ax, sizes)
        ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 2.3)
    handles = [plt.Line2D([], [], color=INK2, lw=2, ls="-", label="one level per insurer"),
               plt.Line2D([], [], color=INK2, lw=2, ls="--", label="ten levels by market position")]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.16)
    credit(fig)
    fig.savefig(OUT / "fig2_profit_gap.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_parts()
    figure_gap()
