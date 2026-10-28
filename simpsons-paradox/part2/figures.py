"""Figures for Part 2 of the Simpson's paradox series.

fig1_three_structures.png  one table, three causal structures, two different answers
fig3_berkeley_direct.png   Berkeley: direct effects by department, and what an unmeasured qualification does
fig2_colliders.png         bias from adjusting or not adjusting in M and butterfly structures
fig4_lord.png              Lord's paradox: change scores and covariance adjustment on the same data

Run:  python figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causal_numbers import (
    COUNTS,
    SEED,
    factorised_joint,
    effect_under,
    load_berkeley,
    mediation,
    ols_coef,
    sem_covariance,
    stylised_admissions,
    unit_variance_sem,
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

POS = {0: (-0.05, 0.0), 1: (0.5, 0.8), 2: (1.05, 0.0)}
NAMES = {0: "treatment", 1: "stone\ngroup", 2: "success"}


def draw_dag(ax, dag, title, answer, color):
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.75, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    for (a, b) in dag:
        xa, ya = POS[a]
        xb, yb = POS[b]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK2, shrinkA=30, shrinkB=30,
                                    mutation_scale=16))
    for v, (x, y) in POS.items():
        ax.add_patch(plt.Circle((x, y), 0.23, color=color if v == 1 else "white", ec=INK2, lw=1.4,
                                alpha=0.25 if v == 1 else 1, zorder=3))
        ax.text(x, y, NAMES[v], ha="center", va="center", fontsize=8, color=INK, zorder=4)
    ax.set_title(title, fontsize=10.5)
    ax.text(0.5, -0.5, answer, ha="center", va="center", fontsize=9.5, color=INK)


def figure_three_structures() -> None:
    p = COUNTS / COUNTS.sum()
    structures = [
        ("Stone group is a confounder", frozenset({(1, 0), (1, 2), (0, 2)}), BLUE),
        ("Stone group is a mediator", frozenset({(0, 1), (1, 2), (0, 2)}), ORANGE),
        ("Stone group is a collider", frozenset({(0, 1), (2, 1), (0, 2)}), VIOLET),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, (title, dag, color) in zip(axes, structures):
        e, _ = effect_under(dag, factorised_joint(dag, p))
        rule = "compare within stone groups" if (1, 0) in dag else "compare the totals"
        answer = (f"{rule}\nP(success | do(open)) = {e[0]:.3f}\nP(success | do(PCNL)) = {e[1]:.3f}\n"
                  f"open − PCNL = {e[0] - e[1]:+.3f}")
        draw_dag(ax, dag, title, answer, color)
    fig.suptitle("The same 700 patients, three causal structures that fit them exactly", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_three_structures.png")
    plt.close(fig)


def figure_berkeley() -> None:
    depts, adm, app = load_berkeley()
    r = adm / app
    mix = app / app.sum(axis=1, keepdims=True)
    te, cde, nde, nie = mediation(r, mix, x1=0, x0=1)
    _, _, nde_m, _ = mediation(r, mix, x1=1, x0=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    y = np.arange(len(depts))[::-1]
    size = app.sum(axis=0)
    ax1.scatter(cde * 100, y, s=size / 6, color=BLUE, alpha=0.6, zorder=3)
    for yi, d in zip(y, depts):
        ax1.text(-34, yi, f"department {d}", va="center", fontsize=9, color=INK2)
    for val, lab, col in ((te, "total effect", ORANGE), (nde, "natural direct, women's choices", AQUA),
                          (-nde_m, "natural direct, men's choices", VIOLET)):
        ax1.axvline(val * 100, color=col, lw=1.6, ls=(0, (4, 2)))
        ax1.text(val * 100 + 0.6, 5.55 if col == ORANGE else (5.25 if col == AQUA else 4.95), lab,
                 fontsize=8.3, color=col)
    ax1.axvline(0, color=INK2, lw=0.8)
    ax1.set_yticks([])
    ax1.set_xlim(-36, 26)
    ax1.set_ylim(-0.6, 5.9)
    ax1.set_xlabel("men's admission rate minus women's (percentage points)")
    ax1.set_title("Direct effects by department, and the totals")
    ax1.grid(axis="y", visible=False)

    g = np.linspace(-1.5, 1.5, 61)
    hard, easy = [], []
    for gi in g:
        rates, _ = stylised_admissions(max(gi, 0.0), max(-gi, 0.0))
        hard.append(rates[0, 0] - rates[1, 0])
        easy.append(rates[0, 1] - rates[1, 1])
    ax2.plot(g, np.array(hard) * 100, color=BLUE, lw=2.2, label="hard department")
    ax2.plot(g, np.array(easy) * 100, color=ORANGE, lw=2.2, label="easy department")
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.axvline(0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax2.set_xlabel("right of 0: men's slope (women's 0); left of 0: minus women's slope (men's 0)\n"
                   "logistic slope of choosing the hard department on qualification")
    ax2.set_ylabel("apparent advantage of men within department (points)")
    ax2.set_title("No direct effect, yet stratified differences of either sign")
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_berkeley_direct.png")
    plt.close(fig)


def figure_colliders() -> None:
    beta = 0.30
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    a_grid = np.linspace(0, 0.70, 71)
    adj = []
    for a in a_grid:
        B, nv = unit_variance_sem({(0, 3): a, (0, 2): a, (1, 2): a, (1, 4): a, (3, 4): beta}, 5)
        S = sem_covariance(B, nv)
        adj.append(ols_coef(S, 4, [3, 2])[0] - beta)
    ax1.plot(a_grid, np.zeros_like(a_grid), color=BLUE, lw=2.2, label="not adjusted for Z")
    ax1.plot(a_grid, adj, color=ORANGE, lw=2.2, label="adjusted for Z")
    ax1.set_xlabel("strength of each path through the unobserved U1 and U2")
    ax1.set_ylabel("bias in the effect of X on Y (true value 0.30)")
    ax1.set_title("M-structure: adjusting opens a path")
    ax1.legend(loc="lower left", fontsize=9)

    c_grid = np.linspace(0, 0.42, 43)
    un, ad = [], []
    for c in c_grid:
        B, nv = unit_variance_sem({(0, 3): 0.5, (0, 2): 0.5, (1, 2): 0.5, (1, 4): 0.5, (2, 3): c, (2, 4): c,
                                   (3, 4): beta}, 5)
        S = sem_covariance(B, nv)
        un.append(ols_coef(S, 4, [3])[0] - beta)
        ad.append(ols_coef(S, 4, [3, 2])[0] - beta)
    ax2.plot(c_grid, np.abs(un), color=BLUE, lw=2.2, label="not adjusted for Z (bias positive)")
    ax2.plot(c_grid, np.abs(ad), color=ORANGE, lw=2.2, label="adjusted for Z (bias negative)")
    ax2.set_xlabel("strength of Z's own paths to X and to Y (U-paths fixed at 0.5)")
    ax2.set_ylabel("size of the bias")
    ax2.set_title("Butterfly: both choices are biased")
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_colliders.png")
    plt.close(fig)


def figure_lord() -> None:
    rng = np.random.default_rng(SEED)
    mu = np.array([70.0, 56.0])
    sd, rho, n = 8.0, 0.7, 1000
    w1 = [rng.normal(mu[g], sd, n) for g in (0, 1)]
    e = rng.normal(0, sd * math.sqrt(1 - rho ** 2), 2 * n)
    w2 = [mu[0] + rho * (w1[0] - mu[0]) + e[:n], mu[1] + rho * (w1[1] - mu[1]) + e[n:]]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True, sharey=True)
    xs = np.linspace(35, 95, 2)
    for ax, title in zip(axes, ("First statistician: compare the gains", "Second statistician: compare at equal start")):
        for g, (col, lab) in enumerate(((BLUE, "boys"), (ORANGE, "girls"))):
            ax.scatter(w1[g][:300], w2[g][:300], s=8, color=col, alpha=0.35, lw=0, label=lab)
        ax.set_xlim(35, 95)
        ax.set_ylim(35, 95)
        ax.set_aspect("equal")
        ax.set_xlabel("weight in September (kg)")
        ax.set_title(title)
    axes[0].set_ylabel("weight in June (kg)")
    axes[0].plot(xs, xs, color=INK2, lw=1.4, ls=(0, (4, 2)))
    axes[0].text(88, 92.5, "no gain", fontsize=8.5, color=INK2, ha="right")
    for g, col in enumerate((BLUE, ORANGE)):
        axes[0].plot(mu[g], mu[g], "o", color=col, ms=9, mec="white", mew=1.5, zorder=5)
        axes[1].plot(xs, mu[g] + rho * (xs - mu[g]), color=col, lw=2.2)
    axes[1].axvline(63, color=INK2, lw=1, ls=(0, (2, 2)))
    axes[1].annotate("", xy=(63, mu[0] + rho * (63 - mu[0])), xytext=(63, mu[1] + rho * (63 - mu[1])),
                     arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    axes[1].text(64.8, 62.5, "4.2 kg", fontsize=9, color=INK)
    axes[0].legend(loc="upper left", fontsize=9, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_lord.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_three_structures()
    figure_berkeley()
    figure_colliders()
    figure_lord()
    print("figures written to", OUT)
