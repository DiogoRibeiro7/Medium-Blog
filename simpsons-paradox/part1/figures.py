"""Figures for Part 1 of the Simpson's paradox series.

fig1_berkeley.png       the six largest Berkeley departments: rates by sex, and where each sex applied
fig2_kidney_stones.png  open surgery and PCNL: the rates within stone sizes and the mixes behind the totals
fig3_evidence.png       stratum and aggregate differences with 95% intervals for Berkeley, the kidney stones and Jeter-Justice

Run:  python figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from history_numbers import load_batting, load_berkeley, rates, standardised_interval

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

KIDNEY = np.array([[[81, 87], [234, 270]], [[192, 263], [55, 80]]])


def figure_berkeley() -> None:
    depts, t = load_berkeley()
    r, agg = rates(t)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    y = np.arange(len(depts) + 1)[::-1]
    labels = [f"department {d}" for d in depts] + ["all six"]
    men = list(r[:, 0]) + [agg[0]]
    women = list(r[:, 1]) + [agg[1]]
    for yi, m, w in zip(y, men, women):
        ax1.plot([m, w], [yi, yi], color=GRID, lw=3, zorder=1)
    ax1.plot(men, y, "o", color=BLUE, ms=8, label="men", zorder=3)
    ax1.plot(women, y, "o", color=ORANGE, ms=8, label="women", zorder=3)
    ax1.axhline(0.5, color=MUTED, lw=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(0, 0.9)
    ax1.set_xlabel("share of applicants admitted")
    ax1.set_title("Women's rate is higher in four of six departments")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="y", visible=False)

    share_w = t[:, 1, 1] / t[:, :, 1].sum(axis=1)
    adm = t[:, :, 0].sum(axis=1) / t[:, :, 1].sum(axis=1)
    size = t[:, :, 1].sum(axis=1)
    ax2.scatter(share_w, adm, s=size / 2.2, color=BLUE, alpha=0.35, lw=1.2, edgecolor=BLUE)
    for d, x, yv in zip(depts, share_w, adm):
        ax2.annotate(d, (x, yv), xytext=(0, 0), textcoords="offset points", ha="center", va="center",
                     fontsize=9, color=INK, fontweight="bold")
    ax2.set_xlim(0, 0.8)
    ax2.set_ylim(0, 0.8)
    ax2.set_xlabel("share of the department's applicants who are women")
    ax2.set_ylabel("share of the department's applicants admitted")
    ax2.set_title("Women applied where admission was hard")
    ax2.text(0.79, 0.76, "circle area: applicants", fontsize=8.5, color=INK2, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_berkeley.png")
    plt.close(fig)


def figure_kidney() -> None:
    t = KIDNEY
    r, agg = rates(t)
    share_large = t[1, :, 1] / t[:, :, 1].sum(axis=0)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.linspace(0, 1, 2)
    for g, (name, color) in enumerate((("open surgery, 1972-80", BLUE), ("PCNL, 1980-85", ORANGE))):
        line = r[0, g] + (r[1, g] - r[0, g]) * x
        ax.plot(x, line, color=color, lw=2.2, label=name)
        ax.plot(share_large[g], agg[g], "o", color=color, ms=10, mec="white", mew=1.6, zorder=5)
        ax.axvline(share_large[g], color=color, lw=0.9, ls=(0, (3, 3)), ymax=0.92)
        ax.annotate(f"{agg[g]:.1%} overall\n{share_large[g]:.0%} large or multiple",
                    (share_large[g], agg[g]), xytext=(10, 10 if g == 1 else -34), textcoords="offset points",
                    fontsize=8.5, color=color)
    for g, color in enumerate((BLUE, ORANGE)):
        ax.plot([0, 1], [r[0, g], r[1, g]], "s", color=color, ms=6, zorder=4)
    ax.text(0.01, 0.955, "stones < 2 cm", fontsize=8.5, color=INK2)
    ax.text(0.99, 0.755, "stones ≥ 2 cm\nor multiple", fontsize=8.5, color=INK2, ha="right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("share of a treatment's patients with large or multiple stones")
    ax.set_ylabel("share of patients treated successfully")
    ax.set_title("Each overall rate sits on its own line, at its own mix of patients")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_kidney_stones.png")
    plt.close(fig)


def diff_ci(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2, p1 - p2 - 1.96 * se, p1 - p2 + 1.96 * se


def figure_evidence() -> None:
    depts, tb = load_berkeley()
    tj = load_batting({"jeterde01": "Jeter", "justida01": "Justice"}, [1995, 1996])
    panels = [
        ("Berkeley: men minus women", tb, [f"dept {d}" for d in depts], 100),
        ("Kidney stones: open minus PCNL", KIDNEY, ["< 2 cm", "≥ 2 cm or multiple"], 100),
        ("Batting: Jeter minus Justice", tj, ["1995", "1996"], 1000),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for ax, (title, t, labels, scale) in zip(axes, panels):
        rows = []
        for k, lab in enumerate(labels):
            rows.append((lab, *diff_ci(t[k, 0, 0], t[k, 0, 1], t[k, 1, 0], t[k, 1, 1]), BLUE))
        tot = t.sum(axis=0)
        rows.append(("standardised", *standardised_interval(t), AQUA))
        rows.append(("aggregate", *diff_ci(tot[0, 0], tot[0, 1], tot[1, 0], tot[1, 1]), ORANGE))
        y = np.arange(len(rows))[::-1]
        for yi, (lab, est, lo, hi, color) in zip(y, rows):
            ax.plot([lo * scale, hi * scale], [yi, yi], color=color, lw=2)
            ax.plot(est * scale, yi, "o", color=color, ms=7, zorder=3)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("difference in percentage points" if scale == 100 else "difference in batting average (×1000)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_evidence.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_berkeley()
    figure_kidney()
    figure_evidence()
    print("figures written to", OUT)
