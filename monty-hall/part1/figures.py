"""Figures for Part 1 of the Monty Hall series.

fig1_timeline.png   the problem's lineage, 1889-2026, on a broken time axis
fig2_n_doors.png    what the host's work is worth in the n-door game
fig3_evidence.png   how much evidence the dining-room and classroom experiments carry

Run:  python figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from history_numbers import (
    SEED,
    bf_paths,
    outcome_probabilities,
    p_switch_formula,
    power_binomial,
    power_fisher,
)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
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
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 170,
})


# ---------------------------------------------------------------------------
# Figure 1: timeline on a broken axis
# ---------------------------------------------------------------------------

SEGMENTS = [(1887.0, 1891.0, 1.0), (1957.0, 1961.0, 1.0), (1973.5, 1993.0, 1.9), (2007.0, 2028.0, 1.0)]
GAPS = [1.8, 1.8, 7.0]  # horizontal space for each break, in plotting units


def year_to_x(year: float) -> float:
    x = 0.0
    for i, (lo, hi, scale) in enumerate(SEGMENTS):
        if year <= hi:
            return x + (year - lo) * scale
        x += (hi - lo) * scale + GAPS[i]
    raise ValueError(year)


# (year, stem height, label). Within a crowded stretch on one side of the axis,
# later events get shorter stems so that no stem crosses an earlier label.
EVENTS = [
    (1889.0, 2.4, "1889  Bertrand: three boxes,\none gold coin seen"),
    (1959.8, -2.4, "1959  Gardner: three\nprisoners and a warden"),
    (1975.1, 4.3, "Feb 1975  Selvin's letter in\nThe American Statistician"),
    (1975.37, -1.5, "May 1975  Monty Hall\nwrites to Selvin"),
    (1975.6, 1.9, "Aug 1975  Selvin names it\nthe Monty Hall problem"),
    (1987.75, -4.2, "1987  Nalebuff's puzzle column,\nJournal of Economic Perspectives"),
    (1988.5, 5.0, "1988  Pearl uses the prisoners\nin probabilistic AI"),
    (1990.69, 3.3, "Sep 1990  Ask Marilyn:\nswitch"),
    (1991.13, -3.0, "Feb 1991  92% of public\nand 65% of university\nletters say she is wrong"),
    (1991.55, 1.5, "Jul 1991  NYT front\npage; Hall's dining-\nroom games"),
    (1991.85, -1.4, "Nov 1991  Morgan et al.:\nthe conditional\nquestion"),
    (2008.7, 3.6, "2008  Rosenthal: Monty Fall,\nMonty Crawl"),
    (2011.1, 1.7, "2011  Gill: a challenge in\nmathematical modelling"),
    (2017.75, -2.0, "30 Sep 2017\nMonty Hall dies"),
    (2026.75, 2.2, "30 Sep 2026\nSteve Selvin dies"),
]


def figure_timeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    # baseline segments with break marks
    x = 0.0
    for i, (lo, hi, scale) in enumerate(SEGMENTS):
        width = (hi - lo) * scale
        ax.plot([x, x + width], [0, 0], color=INK2, lw=2, solid_capstyle="butt", zorder=1)
        if i < len(SEGMENTS) - 1:
            mid = x + width + GAPS[i] / 2
            for dx in (-0.25, 0.25):
                ax.plot([mid + dx - 0.18, mid + dx + 0.18], [-0.22, 0.22], color=INK2, lw=1.2)
        x += width + GAPS[i] if i < len(GAPS) else width

    # decade ticks
    for year in (1889, 1959, 1980, 1985, 1990, 2010, 2015, 2020, 2025):
        xt = year_to_x(year)
        ax.plot([xt, xt], [-0.12, 0.12], color=INK2, lw=1)
        ax.text(xt, -0.45, str(year), ha="center", va="top", fontsize=8.5, color=INK2)

    for year, h, label in EVENTS:
        xe = year_to_x(year)
        color = ORANGE if year > 2017 else BLUE
        ax.plot([xe, xe], [0, h], color=MUTED, lw=0.9, zorder=1)
        ax.plot(xe, 0, "o", ms=6.5, color=color, mec="white", mew=1.5, zorder=3)
        va = "bottom" if h > 0 else "top"
        ax.text(xe + 0.15, h, label, ha="left", va=va, fontsize=8.6, color=INK, linespacing=1.15)

    ax.set_xlim(-0.8, year_to_x(2028.0) + 6.5)
    ax.set_ylim(-6.5, 6.0)
    ax.set_title("Three doors, two goats and 137 years of the same conditional probability",
                 pad=6)
    ax.text(0, -6.45, "Broken time axis. Blue: publications and public episodes; orange: deaths of "
            "the show's host and of the statistician who named the problem after him.",
            fontsize=8.5, color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_timeline.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: n doors, k opened
# ---------------------------------------------------------------------------

def figure_n_doors() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})

    ns = np.unique(np.round(np.logspace(np.log10(3), 3, 200)).astype(int))
    stay = 1 / ns
    sw_one = np.array([float(p_switch_formula(n, 1)) for n in ns])
    sw_half = 2 * (ns - 1) / ns ** 2  # k = (n - 2) / 2 in the formula, drawn as a smooth curve
    sw_all = np.array([float(p_switch_formula(n, n - 2)) for n in ns])

    ax1.plot(ns, sw_all, color=BLUE, lw=2, label="switch, host opens all but one")
    half = ns >= 4
    ax1.plot(ns[half], sw_half[half], color=AQUA, lw=2, label="switch, host opens half")
    ax1.plot(ns, sw_one, color=ORANGE, lw=2, label="switch, host opens one door")
    ax1.plot(ns, stay, color=INK2, lw=1.5, ls=(0, (4, 3)), label="stay (1/n)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("number of doors n")
    ax1.set_ylabel("probability of winning the car")
    ax1.set_title("Adding doors helps only if the host opens them")
    for n, k, dy in ((3, 1, 1.25), (100, 98, 1.25), (100, 1, 1.9)):
        p = float(p_switch_formula(n, k))
        ax1.plot(n, p, "o", ms=6.5, color=INK, mec="white", mew=1.4, zorder=4)
        ax1.annotate(f"n={n}, k={k}: {p:.3f}", (n, p), xytext=(n * 1.15, p * dy),
                     fontsize=8.5, color=INK)
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.set_ylim(5e-4, 2)

    n = 100
    ks = np.arange(0, n - 1)
    ratio = (n - 1) / (n - ks - 1)
    ax2.plot(ks, ratio, color=BLUE, lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("goat doors opened by the host, k (n = 100)")
    ax2.set_ylabel("switch / stay win ratio")
    ax2.set_title("The last doors opened carry most of the gain")
    for k in (1, 50, 90, 98):
        r = (n - 1) / (n - k - 1)
        ax2.plot(k, r, "o", ms=6.5, color=INK, mec="white", mew=1.4, zorder=4)
        ax2.annotate(f"k={k}: x{r:.2f}" if r < 10 else f"k={k}: x{r:.0f}", (k, r),
                     xytext=(-62 if k > 80 else 6, 6), textcoords="offset points",
                     fontsize=8.5, color=INK)
    ax2.set_xlim(-2, 102)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_n_doors.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: evidence in the experiments
# ---------------------------------------------------------------------------

def figure_evidence() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.7))

    ns = np.arange(1, 201)
    ahead = np.array([outcome_probabilities(n)[0] for n in ns])
    fisher = np.array([power_fisher(n) for n in ns])
    binom = np.array([power_binomial(n) for n in ns])
    ax1.plot(ns, ahead, color=AQUA, lw=2, label="switching comes out ahead")
    ax1.plot(ns, fisher, color=BLUE, lw=2, label="Fisher test, both arms, 5%")
    ax1.plot(ns, binom, color=ORANGE, lw=1.6, label="binomial test, switch arm, 5%")
    for n_mark, text in ((10, "Monty Hall's\ndining room\n(10 + 10)"), (200, "vos Savant's\nprotocol\n(200 + 200)")):
        ax1.axvline(n_mark, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax1.text(n_mark * 0.93 if n_mark < 100 else n_mark - 8, 0.34 if n_mark < 100 else 0.42, text,
                 fontsize=8.5, color=INK2, ha="right")
    ax1.set_xscale("log")
    ax1.set_xlim(1, 230)
    ax1.set_ylim(0, 1.03)
    ax1.set_xlabel("games played with each strategy (log scale)")
    ax1.set_ylabel("probability, if switching truly wins 2/3")
    ax1.set_title("A classroom needs dozens of games, not ten")
    ax1.legend(loc="lower right", fontsize=8.5, bbox_to_anchor=(1.0, 0.0))

    rng = np.random.default_rng(SEED + 1)
    games = 300
    paths = bf_paths(30, games, rng)
    x = np.arange(1, games + 1)
    for i, path in enumerate(paths):
        ax2.plot(x, path, color=BLUE, lw=0.8, alpha=0.35,
                 label="30 simulated runs, switch truly wins 2/3" if i == 0 else None)
    drift = (2 / 3 * math.log10(4 / 3) + 1 / 3 * math.log10(2 / 3)) * x
    ax2.plot(x, drift, color=INK, lw=2, label="expected log10 Bayes factor")
    for level in (2, -2):
        ax2.axhline(level, color=ORANGE, lw=1.4, ls=(0, (4, 3)))
    box = {"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9}
    ax2.text(games - 3, 2.12, "Bayes factor 100 for 'switch wins 2/3'", ha="right", fontsize=8.5,
             color=INK2, bbox=box)
    ax2.text(games - 3, -2.12, "Bayes factor 100 for 'switch wins 1/2'", ha="right", va="top",
             fontsize=8.5, color=INK2, bbox=box)
    bf_hall = math.log10((2 / 3) ** 8 * (1 / 3) ** 2 / 0.5 ** 10)
    ax2.plot(10, bf_hall, "o", ms=8, color=ORANGE, mec="white", mew=1.5, zorder=5)
    ax2.annotate("Monty Hall's 8 cars in 10 switches:\nBayes factor 4.4", (10, bf_hall),
                 xytext=(40, -1.15), fontsize=8.5, color=INK,
                 bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9},
                 arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax2.set_xlim(0, games)
    ax2.set_ylim(-3.2, 7.2)
    ax2.set_xlabel("switching games played")
    ax2.set_ylabel("log10 Bayes factor, 2/3 against 1/2")
    ax2.set_title("How long a sceptic's simulation has to run")
    ax2.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_evidence.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_timeline()
    figure_n_doors()
    figure_evidence()
    print("figures written to", OUT)
