"""Figures for Part 1. Reads figure_data.json written by premises_numbers.py.

    python figures.py

Writes three PNG files to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {"light": "#86b6ef", "mid": "#2a78d6", "dark": "#104281"}


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def note(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def fig_paths(data: dict) -> None:
    vcm = data["vcm"]
    pun = data["punishment"]
    comp = data["compulsion_lab"]
    rounds = np.arange(1, len(vcm["contribution"]) + 1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    a.plot(rounds, vcm["contribution"], color=BLUE, label="Voluntary")
    a.plot(rounds, pun["contribution"], color=ORANGE, label="Voluntary with peer punishment")
    for r, lo, hi in ((1, 0.40, 0.60), (10, 0.15, 0.25)):
        a.plot([r, r], [lo, hi], color=INK2, lw=1.2)
        a.plot([r - 0.6, r + 0.6], [lo, lo], color=INK2, lw=1.2)
        a.plot([r - 0.6, r + 0.6], [hi, hi], color=INK2, lw=1.2)
    a.annotate("published ranges used\nfor calibration", xy=(10.6, 0.22), xytext=(14, 0.38),
               fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.text(50, vcm["contribution"][-1] + 0.04, "voluntary", color=BLUE, ha="right", fontsize=9)
    a.text(50, pun["contribution"][-1] - 0.08, "with peer punishment", color=ORANGE, ha="right", fontsize=9)
    a.set_ylim(0, 1.05)
    a.set_xlim(0.5, 50.5)
    a.set_xlabel("Round")
    a.set_ylabel("Mean contribution, share of endowment")
    a.set_title("Contributions in groups of four, MPCR 0.4")
    a.legend(loc="center right", fontsize=8.5, bbox_to_anchor=(1.0, 0.62))

    b.axhline(1.6, color=MUTED, lw=1, ls=(0, (4, 3)))
    b.text(1.5, 1.615, "everyone gives everything (1.6)", color=INK2, fontsize=8, va="bottom")
    b.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    b.text(1.5, 0.985, "nobody gives anything (1.0)", color=INK2, fontsize=8, va="top")
    b.plot(rounds, vcm["welfare"], color=BLUE, label="Voluntary")
    b.plot(rounds, pun["welfare"], color=ORANGE, label="Voluntary with peer punishment")
    b.plot(rounds, np.full(rounds.size, comp["welfare"]), color=AQUA,
           label=f"Tax, cost of funds {comp['lam']:.2f}, {comp['w']:.0%} lost")
    b.text(50, comp["welfare"] + 0.02, "tax-financed", color=INK2, ha="right", fontsize=9)
    b.set_ylim(0.55, 1.72)
    b.set_xlim(0.5, 50.5)
    b.set_xlabel("Round")
    b.set_ylabel("Welfare per person and round, units of endowment")
    b.set_title("Welfare per round")
    b.legend(loc="lower right", fontsize=8.5)

    note(fig, "Simulation, 10,000 groups of four. Calibration targets: Ledyard (1995), Isaac and Walker (1988) via "
              "Andreoni (1995), Fehr and Gaechter (2000). Code: taxation-efficiency-drug-prices/part1.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig1_contribution_and_welfare.png", dpi=200)
    plt.close(fig)


def fig_frontier(data: dict) -> None:
    fr = data["frontier"]
    G = np.array(fr["G"])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    styles = {"1.09": RAMP["light"], "1.33": RAMP["mid"], "1.56": RAMP["dark"]}
    for lam, color in styles.items():
        y = np.array(fr["curves"][f"{lam}|0.1"])
        ax.plot(G, np.where(y >= 0, y, np.nan), color=color, label=f"cost of funds {lam}, 10% lost")
    y = np.array(fr["curves"]["1.33|0.3"])
    ax.plot(G, np.where(y >= 0, y, np.nan), color=RAMP["mid"], ls=(0, (5, 3)), lw=1.6,
            label="cost of funds 1.33, 30% lost")

    labels = {
        "voluntary, 10-round mean": ("voluntary, mean of 10 rounds", "right"),
        "voluntary, round 50": ("voluntary, round 50", "right"),
        "peer punishment, 50-round equivalent": ("peer punishment, mean of 50 rounds", "left"),
    }
    for key, (text, side) in labels.items():
        s = fr["shares"][key]
        ax.axhline(s, color=MUTED, lw=0.9, ls=(0, (2, 2)))
        x = 19.5 if side == "right" else 1.07
        ax.text(x, s + 0.012, f"{text} ({s:.2f})", ha=side, va="bottom", fontsize=8, color=INK2)

    ax.axvline(fr["G_lab"], color=AXIS, lw=1)
    ax.text(fr["G_lab"] * 1.02, 0.99, "laboratory, G = 1.6", fontsize=8, color=INK2, va="top")
    ax.text(1.07, 0.81, "above a curve: voluntary\nprovision gives more welfare", fontsize=8.5, color=INK2, va="top",
            bbox=dict(boxstyle="square,pad=0.2", facecolor=SURFACE, edgecolor="none"), zorder=4)
    ax.text(7.0, 0.47, "below a curve:\nthe tax gives more", fontsize=8.5, color=INK2, va="top")

    ax.set_xscale("log")
    ax.set_xlim(1.05, 20)
    ax.set_ylim(0, 1)
    ax.set_xticks([1.2, 1.6, 2, 3, 5, 10, 20])
    ax.set_xticklabels(["1.2", "1.6", "2", "3", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("Marginal group return G: value created by one unit given (log scale)")
    ax.set_ylabel("Break-even voluntary share s*")
    ax.set_title("When a tax beats voluntary provision")
    ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, -0.13), ncol=2)
    note(fig, "s* = (G(1 - w) - cost of funds) / (G - 1). Cost of public funds from Browning (1976) and "
              "Ballard, Shoven and Whalley (1985). Shares from the Part 1 simulation.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig2_break_even_frontier.png", dpi=200)
    plt.close(fig)


def fig_support(data: dict) -> None:
    sup = data["support"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    n_line = np.geomspace(1, 1e8, 50)
    for g in (1, 10):
        ax.plot(n_line, g / n_line, color=GRID, lw=1.2, zorder=1)

    groups = {
        "Isaac, Walker and Williams (1994)": (BLUE, "o"),
        "Fehr and Gaechter (2000)": (ORANGE, "s"),
        "Weimann et al. (2019)": (AQUA, "D"),
    }
    for name, (color, marker) in groups.items():
        pts = [(n, m) for src, n, m in sup["cells"] if src == name]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=46, color=color, marker=marker,
                   edgecolor=SURFACE, linewidth=1.5, label=name, zorder=3)

    n_nat = sup["n_nat"]
    ax.plot([n_nat, n_nat], [1.5 / n_nat, 10 / n_nat], color=INK, lw=3, solid_capstyle="round", zorder=3)
    box = dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none")
    ax.text(n_nat * 0.6, 3.5e-7, "a good shared by 10 million people,\nG between 1.5 and 10",
            ha="right", va="center", fontsize=8.5, color=INK, bbox=box, zorder=4)
    ax.annotate("", xy=(n_nat * 0.8, 2e-2), xytext=(130, 2e-2),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.text(3.5e4, 3.2e-2, "five orders of magnitude in n without observations", ha="center",
            fontsize=8.5, color=INK2)
    for g, n_at in ((1, 3e3), (10, 3e4)):
        ax.text(n_at, g / n_at, f"G = {g}", fontsize=8, color=MUTED, ha="center", va="center",
                bbox=box, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, 4e7)
    ax.set_ylim(5e-8, 2)
    ax.set_xlabel("Group size n (log scale)")
    ax.set_ylabel("Return to the giver per unit given, MPCR (log scale)")
    ax.set_title("Where the experimental evidence on voluntary provision lives")
    ax.legend(loc="lower left", fontsize=8.5)
    note(fig, "Design cells as reported in each study. Diagonals: constant marginal group return G = n x MPCR.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig3_evidence_support.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    with open(HERE / "figure_data.json") as fh:
        data = json.load(fh)
    fig_paths(data)
    fig_frontier(data)
    fig_support(data)


if __name__ == "__main__":
    main()
