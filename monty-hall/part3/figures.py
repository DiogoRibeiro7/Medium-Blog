"""Figures for Part 3 of the Monty Hall series.

fig1_identification.png  the same identification problem in the game and in the Coat ratings
fig2_coat_selection.png  what self-selection does to the Coat ratings
fig3_evaluation.png      offline evaluation of six rating models, naive and reweighted, against the truth

Run:  python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from missing_data_numbers import (
    SEED,
    repeated_evaluation,
    load_coat,
    nb_selection,
    rating_shares,
    tilt_mean,
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
BAND = "#e8f0fb"

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


def figure_identification(train, test) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    lam = np.linspace(0, 1, 301)
    ax1.axhspan(0.5, 1.0, color=BAND, lw=0, zorder=0)
    ax1.plot(lam, 1 / (1 + lam), color=BLUE, lw=2.2)
    for x, label, off in ((1.0, "ignorable: the door says\nnothing about the car", (-150, -6)),
                          (0.5, "knowing host,\nfair tie-break", (8, 6))):
        y = 1 / (1 + x)
        ax1.plot(x, y, "o", ms=7.5, color=INK, mec="white", mew=1.5, zorder=5, clip_on=False)
        ax1.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=INK)
    ax1.text(0.98, 0.985, "shaded: every value a knowing host allows", fontsize=8.5, color=INK2, va="top", ha="right")
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xlabel("P(host opens door 3 | car at 1) / P(host opens door 3 | car at 2)")
    ax1.set_ylabel("P(switching wins | door 3 opened)")
    ax1.set_title("Monty Hall: the answer is the selection ratio")

    obs = train > 0
    sel = train[obs].astype(float)
    mar = test[test > 0]
    p_obs = obs.mean()
    lower = p_obs * sel.mean() + (1 - p_obs) * 1
    upper = p_obs * sel.mean() + (1 - p_obs) * 5
    g = np.linspace(0, 0.6, 301)
    m = np.array([tilt_mean(sel, x) for x in g])
    ax2.axhspan(lower, upper, color=BAND, lw=0, zorder=0)
    ax2.plot(g, m, color=BLUE, lw=2.2)
    ax2.axhline(mar.mean(), color=ORANGE, lw=1.6, ls=(0, (4, 3)))
    ax2.text(0.6, mar.mean() + 0.05, f"random-sample mean {mar.mean():.2f}", ha="right", fontsize=8.5, color=INK2)
    ax2.plot(0, sel.mean(), "o", ms=7.5, color=INK, mec="white", mew=1.5, zorder=5, clip_on=False)
    ax2.annotate(f"ignorable: {sel.mean():.2f}", (0, sel.mean()), xytext=(8, 6), textcoords="offset points",
                 fontsize=8.5, color=INK)
    from scipy.optimize import brentq
    gs = brentq(lambda x: tilt_mean(sel, x) - mar.mean(), 0, 5)
    ax2.plot(gs, mar.mean(), "o", ms=7.5, color=ORANGE, mec="white", mew=1.5, zorder=5)
    ax2.annotate(f"γ = {gs:.2f}", (gs, mar.mean()), xytext=(8, -16), textcoords="offset points",
                 fontsize=8.5, color=INK)
    ax2.text(0.01, upper - 0.08, f"no-assumption bounds [{lower:.2f}, {upper:.2f}]", fontsize=8.5,
             color=INK2, va="top")
    ax2.set_xlim(0, 0.6)
    ax2.set_ylim(1.0, 5.0)
    ax2.set_xlabel("γ: log of the selection odds per rating point")
    ax2.set_ylabel("implied mean rating of all coats")
    ax2.set_title("Coat: the answer is the selection slope")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_identification.png")
    plt.close(fig)


def figure_selection(train, test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    obs = train > 0
    sel = train[obs]
    mar = test[test > 0]
    k = np.arange(1, 6)

    ax = axes[0]
    w = 0.38
    ax.bar(k - w / 2 - 0.01, rating_shares(sel), width=w, color=BLUE, label="coats people chose to rate")
    ax.bar(k + w / 2 + 0.01, rating_shares(mar), width=w, color=ORANGE, label="coats assigned at random")
    ax.set_xticks(k)
    ax.set_xlabel("rating")
    ax.set_ylabel("share of ratings")
    ax.set_title("Chosen coats are rated higher")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[1]
    p_obs = obs.mean()
    pr = nb_selection(sel, mar, p_obs)
    rng = np.random.default_rng(SEED + 3)
    n_u = train.shape[0]
    boot = []
    for _ in range(2000):
        us = rng.integers(0, n_u, n_u)
        boot.append(nb_selection(train[us][train[us] > 0], test[us][test[us] > 0], p_obs))
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    ax.vlines(k, lo, hi, color=BLUE, lw=2)
    ax.plot(k, pr, "o", ms=7.5, color=BLUE, mec="white", mew=1.5, zorder=5)
    ax.axhline(p_obs, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(5.6, p_obs - 0.004, "overall share rated: 8%", fontsize=8.5, color=INK2, va="top", ha="right")
    ax.set_xticks(k)
    ax.set_xlim(0.6, 5.7)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("rating the person would give")
    ax.set_ylabel("P(the coat is among those rated)")
    ax.set_title("Selection depends on the hidden rating")

    ax = axes[2]
    n_i = train.shape[1]
    sm = np.array([train[:, i][obs[:, i]].mean() for i in range(n_i)])
    rm = np.array([test[:, i][test[:, i] > 0].mean() for i in range(n_i)])
    ax.plot([1, 5], [1, 5], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.scatter(rm, sm, s=14, color=BLUE, alpha=0.55, lw=0)
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("mean rating from random assignments")
    ax.set_ylabel("mean rating from people who chose it")
    ax.set_title("Same coat, rated by its choosers")
    above = np.mean(sm > rm)
    ax.text(4.9, 1.15, f"{above:.0%} of the 300 coats\nlie above the diagonal", ha="right",
            fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_coat_selection.png")
    plt.close(fig)


def figure_evaluation(train, test) -> None:
    short = {
        "global mean": "global mean",
        "coat mean (shrunk)": "coat mean",
        "user + coat biases": "user + coat biases",
        "matrix factorisation (k = 4)": "matrix factorisation",
        "user + coat biases, IPS-weighted": "biases, IPS-weighted",
        "matrix factorisation, IPS-weighted": "factorisation, IPS-weighted",
    }
    names, res = repeated_evaluation(train, test, 100)
    mean = res.mean(axis=0)
    sd = res.std(axis=0, ddof=1)
    order = np.argsort(mean[:, 2])[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(len(names))
    styles = ((0, ORANGE, "o", "naive: held-out chosen coats"),
              (1, AQUA, "D", "reweighted held-out (SNIPS)"),
              (2, BLUE, "s", "truth: randomly assigned coats"))
    for j, i in enumerate(order):
        ax.plot([mean[i].min(), mean[i].max()], [j, j], color=GRID, lw=3, zorder=1)
    for col, color, marker, label in styles:
        ax.errorbar(mean[order, col], y + (col - 1) * 0.12, xerr=sd[order, col], fmt=marker, ms=7,
                    color=color, ecolor=color, elinewidth=1.2, capsize=0, label=label, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([short[names[i]] for i in order])
    ax.set_xlabel("mean squared error of the predicted rating (mean and s.d. over 100 splits)")
    ax.set_title("Evaluated on chosen coats, the debiased models look worse")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_evaluation.png")
    plt.close(fig)


if __name__ == "__main__":
    train, test, _, _ = load_coat()
    figure_identification(train, test)
    figure_selection(train, test)
    figure_evaluation(train, test)
    print("figures written to", OUT)
