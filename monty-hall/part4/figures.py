"""Figures for Part 4 of the Monty Hall series.

fig1_decision_and_belief.png  what outcomes teach: the decision of a counting learner and the conviction of a believer in fifty-fifty
fig2_humans_and_pigeons.png   published switch rates against the counting learner and probability matching
fig3_matching_fits.png        a matching learner fitted to the published human curves, and where it leads
fig4_reinforcement.png        delta-rule learners: own-outcome against counterfactual feedback, and recency
fig5_asymmetry.png            how much extra weight on losses after switching it takes to hold switching down

Run:  python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import math

import matplotlib.pyplot as plt
import numpy as np

from learning_numbers import (
    HS_HUMANS,
    HS_PIGEONS,
    P_SWITCH,
    PETROCELLI_HARRIS,
    SEED,
    asymptote,
    conviction_after,
    fit_matching_two_blocks,
    matching_curve,
    memory_distortion_rate,
    p_counts_favour_switch,
    simulate_learners,
)

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
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 170,
})


def ref_line(ax, y, label, x_text, color=INK2, va="bottom"):
    ax.axhline(y, color=color, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(x_text, y + (0.012 if va == "bottom" else -0.012), label, fontsize=8.5, color=color, va=va)


def figure_decision_and_belief() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    n = np.arange(0, 101)
    dec = np.array([0.5] + [p_counts_favour_switch(k) for k in n[1:]])
    match = matching_curve(0.5, 2.0, P_SWITCH, n + 1)
    ax1.step(n + 1, dec, where="post", color=BLUE, lw=2.2, label="counting learner: switch if switching has won more often")
    ax1.plot(n + 1, match, color=ORANGE, lw=2.2, label="matching learner: switch as often as switching has won")
    ref_line(ax1, P_SWITCH, "2/3", 93, va="bottom")
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xlabel("game")
    ax1.set_ylabel("P(switch)")
    ax1.set_title("Outcomes alone settle the decision quickly")
    ax1.legend(loc="lower right", fontsize=8.5)

    ns = np.arange(1, 301, 2)  # odd numbers of games: no ties, so the lattice does not zigzag
    for pi0, color in ((0.5, BLUE), (0.9, ORANGE), (0.99, VIOLET)):
        share = np.array([conviction_after(k, pi0) for k in ns])
        ax2.plot(ns, share, color=color, lw=2.2, label=f"prior P(fifty-fifty) = {pi0}")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_xlim(0, 300)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("games played (odd numbers)")
    ax2.set_ylabel("share with P(not fifty-fifty) > 0.95")
    ax2.set_title("Conviction against a fifty-fifty belief is slow")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_decision_and_belief.png")
    plt.close(fig)


def figure_humans_and_pigeons() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw={"width_ratios": [1.45, 1]})
    games = np.arange(1, 201)
    dec = np.array([0.5] + [p_counts_favour_switch(k) for k in games[:-1]])
    ax1.plot(games, dec, color=MUTED, lw=1.8, zorder=2)
    ax1.text(130, 0.975, "counting learner", fontsize=8.5, color=INK2, va="top")
    ref_line(ax1, P_SWITCH, "probability matching, 2/3", 6.5, va="top")
    for (name, d), color in zip(PETROCELLI_HARRIS.items(), (BLUE, AQUA)):
        n = d["n"]
        xs = [5.5, 55.5]
        ys = [d["first"][0], d["last"][0]]
        es = [1.96 * d["first"][1] / math.sqrt(n), 1.96 * d["last"][1] / math.sqrt(n)]
        ax1.errorbar(xs, ys, yerr=es, fmt="o-", color=color, ms=6.5, lw=1.4, capsize=0, zorder=4,
                     label=f"Petrocelli & Harris (2011), {name}, N = {n}")
    ax1.plot([1, 45.5], [0.12, 0.50], "s--", color=ORANGE, ms=6.5, mfc="white", mew=1.6, lw=1.2, zorder=4,
             label="Granberg & Brown (1995), approximate")
    d = HS_HUMANS["switching pays 2/3"]
    ax1.errorbar([25.5, 175.5], [d["first"][0], d["last"][0]],
                 yerr=[[d["first"][0] - d["first"][1], d["last"][0] - d["last"][1]],
                       [d["first"][2] - d["first"][0], d["last"][2] - d["last"][0]]],
                 fmt="D-", color=VIOLET, ms=6, lw=1.4, capsize=0, zorder=4,
                 label="Herbranson & Schroeder (2010), no story, N = 6")
    ax1.set_xscale("log")
    ax1.set_xlim(0.8, 220)
    ax1.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax1.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("game (block midpoints; log scale)")
    ax1.set_ylabel("share of games in which people switched")
    ax1.set_title("People learn to switch, but stop near matching")
    ax1.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.93))

    rows = [("pigeons, day 1 → day 30", HS_PIGEONS["day 1"], HS_PIGEONS["day 30"], ORANGE),
            ("humans, block 1 → block 4", HS_HUMANS["switching pays 2/3"]["first"],
             HS_HUMANS["switching pays 2/3"]["last"], VIOLET)]
    for j, (label, a, b, color) in enumerate(rows):
        y = 1 - j
        ax2.plot([a[0], b[0]], [y, y], color=GRID, lw=4, zorder=1)
        ax2.errorbar([a[0]], [y], xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="o", color=color, mfc="white",
                     mew=1.6, ms=8, capsize=0, zorder=3)
        ax2.errorbar([b[0]], [y], xerr=[[b[0] - b[1]], [b[2] - b[0]]], fmt="o", color=color, ms=8,
                     capsize=0, zorder=3)
        ax2.text(0.02, y + 0.22, label, fontsize=9, color=INK)
    ax2.axvline(P_SWITCH, color=INK2, lw=1, ls=(0, (4, 3)))
    ax2.axvline(1.0, color=INK2, lw=1, ls=(0, (1, 2)))
    ax2.text(P_SWITCH - 0.01, -0.62, "matching", fontsize=8.5, color=INK2, ha="right")
    ax2.text(0.99, -0.62, "maximising", fontsize=8.5, color=INK2, ha="right")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(-0.75, 1.55)
    ax2.set_yticks([])
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("switch rate (95% interval; hollow = start, filled = end)")
    ax2.set_title("Same task: pigeons maximise, people match")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_humans_and_pigeons.png")
    plt.close(fig)


def figure_matching_fits() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    g = np.unique(np.round(np.logspace(0, 3, 300)).astype(int))
    ref_line(ax, P_SWITCH, "2/3: where every matching learner ends", 1.1, va="bottom")
    series = []
    for (name, d), color in zip(PETROCELLI_HARRIS.items(), (BLUE, AQUA)):
        p0, m = fit_matching_two_blocks(d["first"][0], d["last"][0], (1, 10), (51, 60))
        series.append((f"Petrocelli & Harris, {name}: prior worth {m:.0f} games", p0, m, color,
                       [(5.5, d["first"][0]), (55.5, d["last"][0])], 60))
    p0, m = fit_matching_two_blocks(0.12, 0.50, (1, 1), (41, 50))
    series.append((f"Granberg & Brown, approximate: prior worth {m:.0f} games", p0, m, ORANGE,
                   [(1, 0.12), (45.5, 0.50)], 50))
    for label, p0, m, color, pts, end in series:
        curve = matching_curve(p0, m, P_SWITCH, g)
        ax.plot(g[g <= end], curve[g <= end], color=color, lw=2.2, label=label)
        ax.plot(g[g >= end], curve[g >= end], color=color, lw=1.6, ls=(0, (3, 2)))
        for x, y in pts:
            ax.plot(x, y, "o", color=color, ms=7, mec="white", mew=1.4, zorder=5)
    ax.axvspan(60, 1000, color="#f4f3f0", lw=0, zorder=0)
    ax.text(70, 0.04, "extrapolation beyond the experiments", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 60, 100, 1000])
    ax.set_xticklabels(["1", "10", "60", "100", "1000"])
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("game (log scale)")
    ax.set_ylabel("P(switch)")
    ax.set_title("Matching learners calibrated to the human curves, and where they lead")
    ax.legend(loc="lower right", fontsize=8.5, bbox_to_anchor=(1.0, 0.13))
    fig.tight_layout()
    fig.savefig(OUT / "fig3_matching_fits.png")
    plt.close(fig)


def figure_reinforcement() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    b = math.log(9)
    games = np.arange(1, 201)
    for fb, color, label in (("full", BLUE, "learns from the door not chosen too"),
                             ("own", ORANGE, "learns only from its own outcome")):
        curve, _ = simulate_learners(20_000, 200, 0.1, 10.0, b, feedback=fb, rng=np.random.default_rng(SEED + 1))
        ax1.plot(games, curve, color=color, lw=2.2, label=label)
    ax1.axhline(asymptote(10.0, b), color=INK2, lw=1, ls=(0, (4, 3)))
    ax1.text(198, asymptote(10.0, b) + 0.015, "settled values, no noise", fontsize=8.5, color=INK2, ha="right")
    ax1.set_xlim(1, 200)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("game")
    ax1.set_ylabel("mean P(switch) over 20,000 learners")
    ax1.set_title("The other door speeds learning and adds noise")
    ax1.legend(loc="lower right", fontsize=8.5)

    for alpha, color in ((0.03, VIOLET), (0.1, BLUE), (0.3, AQUA)):
        curve, _ = simulate_learners(20_000, 200, alpha, 10.0, b, feedback="full", rng=np.random.default_rng(SEED + 1))
        ax2.plot(games, curve, color=color, lw=2.2, label=f"learning rate {alpha}")
    ax2.axhline(asymptote(10.0, b), color=INK2, lw=1, ls=(0, (4, 3)))
    ax2.set_xlim(1, 200)
    ax2.set_xlabel("game")
    ax2.set_title("Chasing recent outcomes lowers the plateau")
    ax2.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_reinforcement.png")
    plt.close(fig)


def figure_asymmetry() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    w = np.linspace(1, 6, 201)
    s = np.array([memory_distortion_rate(x) for x in w])
    ax1.plot(w, s, color=BLUE, lw=2.2)
    ax1.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax1.text(5.95, 0.515, "indifference", fontsize=8.5, color=INK2, ha="right")
    for x in (1, 2, 3):
        y = memory_distortion_rate(x)
        ax1.plot(x, y, "o", color=INK, ms=7, mec="white", mew=1.4, zorder=5)
        ax1.annotate(f"{y:.2f}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8.5)
    ax1.set_xlim(1, 6)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("each loss after switching remembered as w losses")
    ax1.set_ylabel("long-run switch rate")
    ax1.set_title("A matching learner with a biased memory")

    k = np.linspace(0, 2, 201)
    for beta, color in ((3.0, ORANGE), (10.0, BLUE), (30.0, VIOLET)):
        ax2.plot(k, [asymptote(beta, 0.0, x) for x in k], color=color, lw=2.2, label=f"choice sharpness β = {beta:g}")
    ax2.axvline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    ax2.text(1.03, 0.9, "κ = 1: a loss after switching costs\nas much as a car is worth", fontsize=8.5, color=INK2)
    ax2.set_xlim(0, 2)
    ax2.set_xlabel("κ: regret for a loss after switching, in cars")
    ax2.set_title("A softmax learner with regret for switching")
    ax2.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_asymmetry.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_decision_and_belief()
    figure_humans_and_pigeons()
    figure_matching_fits()
    figure_reinforcement()
    figure_asymmetry()
    print("figures written to", OUT)
