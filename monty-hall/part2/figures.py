"""Figures for Part 2 of the Monty Hall series.

fig1_components.png  P(switch wins | what was seen) as three components of the host's rules vary
fig2_offer_policy.png P(switch wins | offer) over the offer policy (a, b), with the decision boundary
fig3_offer_game.png  the value of the game when the host may decline to offer, by minimum offer rate
fig4_learning.png    a contestant who learns the offer policy from past episodes

Run:  python figures.py
"""

from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from protocol_numbers import HOSTS, SEED, game_value, learning_curves

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e6e5e1"
STAY_ZONE = "#fbe3d9"

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


def stay_zone(ax, xmax=1.0):
    ax.axhspan(0, 0.5, color=STAY_ZONE, zorder=0, lw=0)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.02 * xmax, 0.03, "staying is better", fontsize=8.5, color=INK2)


def figure_components() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    x = np.linspace(0, 1, 401)

    ax = axes[0]
    stay_zone(ax)
    ax.plot(x, 1 / (1 + x), color=BLUE, lw=2, label="host opened door 3")
    ax.plot(x, 1 / (2 - x), color=ORANGE, lw=2, label="host opened door 2")
    ax.set_xlabel("q: chance the host opens door 3 when free to choose")
    ax.set_ylabel("P(switching wins | what was seen)")
    ax.set_title("Tie-break")
    ax.legend(loc="upper center", fontsize=8.5)

    ax = axes[1]
    stay_zone(ax)
    for q, color, label in ((0.5, BLUE, "fair tie-break"), (1.0, ORANGE, "always door 3 (q = 1)")):
        ax.plot(x, (1 + x) / (2 * (1 + x * q)), color=color, lw=2, label=label)
    ax.set_xlabel("φ: chance the host knows where the car is")
    ax.set_title("Host's knowledge (door 3 opened)")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[2]
    stay_zone(ax)
    ax.plot(x, 1 - x, color=BLUE, lw=2, label="fixed pick: door 1")
    ax.axhline(2 / 3, color=AQUA, lw=2, label="random pick, always switch")
    ax.set_xlabel("weight of the car on the contestant's door")
    ax.set_title("Placement (others equally likely)")
    ax.legend(loc="upper right", fontsize=8.5)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_components.png")
    plt.close(fig)


def figure_offer_policy() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.grid(False)
    n = 400
    a = np.linspace(0, 1, n)
    b = np.linspace(0, 1, n)
    A, B = np.meshgrid(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where((A == 0) & (B == 0), np.nan, 2 * B / (2 * B + A))
    # diverging: orange below 1/2 (stay), blue above (switch), grey at 1/2
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#c24a1c", "#f2b59a", "#efeeea", "#9cc3ef", "#1f5fae"])
    im = ax.imshow(P, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="equal")
    cs = ax.contour(A, B, P, levels=[1 / 3, 0.5, 2 / 3], colors=[INK2, INK, INK2],
                    linewidths=[0.8, 1.8, 0.8], linestyles=["--", "-", "--"])
    ax.clabel(cs, fmt={1 / 3: "1/3", 0.5: "1/2", 2 / 3: "2/3"}, fontsize=8.5)
    marks = {
        "standard host": (1.0, 1.0, (-88, -16)),
        "trap": (1.0, 0.0, (-34, 8)),
        "gift": (0.0, 1.0, (6, -16)),
        "sparing, leans right": (0.3, 0.1, (6, -14)),
        "sparing, leans wrong": (0.3, 0.2, (6, 6)),
    }
    for label, (aa, bb, off) in marks.items():
        ax.plot(aa, bb, "o", ms=7, color=INK, mec="white", mew=1.5, zorder=5, clip_on=False)
        ax.annotate(label, (aa, bb), xytext=off, textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("a: chance of an offer when the first pick is right")
    ax.set_ylabel("b: chance of an offer when the first pick is wrong")
    ax.set_title("P(switching wins | the host offers) = 2b / (2b + a)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("probability that switching wins", color=INK2)
    cb.outline.set_edgecolor(MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_offer_policy.png")
    plt.close(fig)


def figure_offer_game() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ms = np.linspace(0, 1, 201)
    vals = np.array([game_value(m, grid=201)[0] for m in ms])
    ax.plot(ms, vals, color=BLUE, lw=2.2, label="value with the best rule")
    ax.plot(ms, np.full_like(ms, 1 / 3), color=ORANGE, lw=1.5, ls=(0, (4, 3)), label="never switch")
    ax.plot(ms, np.maximum(0, ms - 1 / 3), color=AQUA, lw=1.5, ls=(0, (1, 2)),
            label="always switch, worst case")
    ax.axvline(2 / 3, color=MUTED, lw=1)
    ax.text(2 / 3 + 0.01, 0.06, "offers must reach\ntwo games in three", fontsize=8.5, color=INK2)
    ax.plot(1, 2 / 3, "o", ms=7, color=INK, mec="white", mew=1.5, zorder=5, clip_on=False)
    ax.annotate("the textbook game:\nthe offer is compulsory", (1, 2 / 3), xytext=(0.55, 0.68),
                fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("minimum share of games in which the host must offer the switch")
    ax.set_ylabel("win probability the contestant can guarantee")
    ax.set_title("Against a host who chooses when to offer, the offer is worth nothing")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_offer_game.png")
    plt.close(fig)


def figure_learning() -> None:
    names = ["trap: offers only to right picks (a=1, b=0)",
             "sparing, leans right (a=0.3, b=0.1)",
             "sparing, leans wrong (a=0.3, b=0.2)"]
    titles = ["Trap host (a = 1, b = 0)", "Sparing host, leans right (0.3, 0.1)",
              "Sparing host, leans wrong (0.3, 0.2)"]
    ns = np.unique(np.round(np.logspace(0, np.log10(2000), 45)).astype(int))
    ns = np.concatenate([[0], ns])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    rng = np.random.default_rng(SEED + 7)
    for ax, name, title in zip(axes, names, titles):
        a, b = HOSTS[name]
        full, broad, rf, rb = learning_curves(a, b, ns, 20_000, rng)
        x = np.maximum(ns, 0.7)
        ax.plot(x, rf, color=BLUE, lw=2, label="sees every past episode")
        ax.plot(x, rb, color=ORANGE, lw=1.6, label="sees only episodes with an offer")
        ax.axhline(0.95, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_xlim(0.7, 2000)
        ax.set_ylim(-0.02, 1.03)
        ax.set_title(title)
        ax.set_xlabel("past episodes watched (log scale; 0 at left edge)")
        verdict = "switch" if 2 * b > a else "stay"
        ax.text(0.98, 0.04, f"right call: {verdict}", transform=ax.transAxes, ha="right",
                fontsize=8.5, color=INK2)
    axes[0].set_ylabel("probability of the right call")
    axes[0].legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_learning.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_components()
    figure_offer_policy()
    figure_offer_game()
    figure_learning()
    print("figures written to", OUT)
