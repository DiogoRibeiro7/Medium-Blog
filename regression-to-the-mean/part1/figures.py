"""Figures for Part 1 of the regression-to-the-mean series.

fig1_galton.png           Galton's mid-parents and children: both regressions and the line of no regression
fig2_pearson_lee.png      Pearson and Lee's fathers and sons: regression both ways, and two equal spreads
fig3_secrist.png          Secrist's design on baseball teams, and the spread of winning percentage by season

Run:  python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from history_numbers import ols_slope, read

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
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 170,
})


def figure_galton() -> None:
    g = read("Galton.csv")
    parent = np.array([float(r["parent"]) for r in g])
    child = np.array([float(r["child"]) for r in g])
    rng = np.random.default_rng(1886)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.scatter(parent + rng.uniform(-0.25, 0.25, len(parent)), child + rng.uniform(-0.25, 0.25, len(child)),
               s=7, color=MUTED, alpha=0.45, lw=0)
    pv = sorted(set(parent))
    ax.plot(pv, [child[parent == v].mean() for v in pv], "o", color=BLUE, ms=6,
            label="mean child height for each mid-parent height")
    cv = sorted(set(child))
    ax.plot([parent[child == v].mean() for v in cv], cv, "s", color=ORANGE, ms=5,
            label="mean mid-parent height for each child height")
    xs = np.array([63.5, 73.5])
    b, a, _ = ols_slope(parent, child)
    ax.plot(xs, a + b * xs, color=BLUE, lw=2, label=f"child on mid-parent: slope {b:.2f}")
    b2, a2, _ = ols_slope(child, parent)
    ys = np.array([61, 74.5])
    ax.plot(a2 + b2 * ys, ys, color=ORANGE, lw=2, label=f"mid-parent on child: slope {b2:.2f}, read against the vertical axis")
    mx, my = parent.mean(), child.mean()
    ax.plot([63.5, 73.5], [my + (63.5 - mx), my + (73.5 - mx)], color=INK, lw=1.2, ls=(0, (4, 2)),
            label="no regression: children deviate as much as parents")
    ax.set_xlim(63.5, 73.5)
    ax.set_ylim(61, 74.5)
    ax.set_xlabel("mid-parent height (inches)")
    ax.set_ylabel("adult child's height (inches; daughters x 1.08)")
    ax.set_title("Galton, 1886: regression in both directions")
    ax.legend(loc="upper left", fontsize=8.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_galton.png")
    plt.close(fig)


def figure_pearson_lee() -> None:
    rows = [r for r in read("PearsonLee.csv") if r["gp"] == "fs"]
    x = np.array([float(r["parent"]) for r in rows])
    y = np.array([float(r["child"]) for r in rows])
    w = np.array([float(r["frequency"]) for r in rows])
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(x, y, s=8 + 6 * w, color=MUTED, alpha=0.35, lw=0)
    xv = sorted(set(x))
    keep = [v for v in xv if w[x == v].sum() >= 10]
    ax1.plot(keep, [np.average(y[x == v], weights=w[x == v]) for v in keep], "o-", color=BLUE, ms=5, lw=1.6,
             label="mean height of sons, by father's height")
    yv = sorted(set(y))
    keep = [v for v in yv if w[y == v].sum() >= 10]
    ax1.plot([np.average(x[y == v], weights=w[y == v]) for v in keep], keep, "s-", color=ORANGE, ms=4.5, lw=1.6,
             label="mean height of fathers, by son's height")
    lo, hi = 60, 76
    ax1.plot([lo, hi], [my + lo - mx, my + hi - mx], color=INK, lw=1.2, ls=(0, (4, 2)), label="no regression")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo + 1, hi + 1)
    ax1.set_xlabel("father's height (inches)")
    ax1.set_ylabel("son's height (inches)")
    ax1.set_title("1,078 fathers and sons: each regresses on the other")
    ax1.legend(loc="upper left", fontsize=8.5)
    bins = np.arange(58.5, 80.5, 1.0)
    ax2.hist(x, bins=bins, weights=w, color=BLUE, alpha=0.55, label="fathers")
    ax2.hist(y, bins=bins, weights=w, color=ORANGE, alpha=0.55, label="sons")
    sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = np.sqrt(np.average((y - my) ** 2, weights=w))
    ax2.text(0.03, 0.95, f"SD fathers {sx:.2f} in\nSD sons {sy:.2f} in", transform=ax2.transAxes, va="top",
             fontsize=9, color=INK2)
    ax2.set_xlabel("height (inches)")
    ax2.set_ylabel("number of men")
    ax2.set_title("...and the spread does not shrink")
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_pearson_lee.png")
    plt.close(fig)


def figure_secrist() -> None:
    teams = read("Teams.csv")
    wp = {}
    for r in teams:
        y_ = int(r["yearID"])
        if r["lgID"] in ("AL", "NL") and y_ >= 1901:
            wp[(r["franchID"], y_)] = int(r["W"]) / (int(r["W"]) + int(r["L"]))
    years = sorted({y_ for _, y_ in wp})
    league = {y_: np.mean([v for (f, yy), v in wp.items() if yy == y_]) for y_ in years}
    k = 5
    offsets = list(range(-k, k + 1))
    groups = {q: {o: [] for o in offsets} for q in range(4)}
    for y0 in range(1961, 2026 - k):
        fr = sorted(f for (f, yy) in wp if yy == y0)
        vals = np.array([wp[(f, y0)] for f in fr])
        quart = np.digitize(vals, np.quantile(vals, [0.25, 0.5, 0.75]))
        for f, q in zip(fr, quart):
            if all((f, y0 + o) in wp for o in offsets):
                for o in offsets:
                    groups[q][o].append(wp[(f, y0 + o)] - league[y0 + o])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = (ORANGE, VIOLET, AQUA, BLUE)
    names = ("worst quarter", "second", "third", "best quarter")
    for q in range(4):
        ax1.plot(offsets, [1000 * np.mean(groups[q][o]) for o in offsets], "o-", color=cols[q], lw=2, ms=4,
                 label=names[q])
    ax1.axhline(0, color=MUTED, lw=1)
    ax1.axvline(0, color=MUTED, lw=1, ls=(0, (3, 2)))
    ax1.set_xticks(offsets)
    ax1.set_xlabel("seasons before (-) and after (+) the season used to form the groups")
    ax1.set_ylabel("winning percentage above league mean (x 1000)")
    ax1.set_title("Grouped in one season: converging forwards and backwards")
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)
    sd = [np.std([v for (f, yy), v in wp.items() if yy == y_]) for y_ in years]
    ax2.plot(years, [1000 * s for s in sd], color=INK2, lw=1.4)
    ax2.set_xlabel("season")
    ax2.set_ylabel("SD of winning percentage across teams (x 1000)")
    ax2.set_title("The spread across teams, season by season")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_secrist.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_galton()
    figure_pearson_lee()
    figure_secrist()
    print("figures written to", OUT)
