"""Figures for Part 1 of the statistical-process-control series.

fig1_three_sigma.png      In-control ARL of k-sigma limits: the normal, two non-normal distributions and the bounds
fig2_runs_rules.png       ARL against the size of a shift: test 1, the four Western Electric tests, the British scheme
fig3_estimated_limits.png Conditional in-control ARL of charts with estimated limits: 25 subgroups, and the band by Phase I size

Run:  python figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy import stats

from history_numbers import (SEED, conditional_p, phase1_individuals, phase1_xbar, warning_scheme_arl,
                             we_arl)

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

CREDIT = "Figure: Diogo Ribeiro. Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/statistical-process-control"


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def label_end(ax, x, y, text, color, dy=0.0, dx=0.03):
    ax.annotate(text, (x, y), xytext=(x + dx, y * (10 ** dy)), color=INK2, fontsize=9, va="center",
                ha="left", annotation_clip=False)
    ax.plot([x], [y], "o", color=color, ms=4.5, mec="white", mew=1.2, zorder=5)


def figure_three_sigma() -> None:
    k = np.linspace(1.5, 4.0, 400)

    def two_sided(dist):
        m, v = dist.stats(moments="mv")
        sd = math.sqrt(float(v))
        return dist.sf(float(m) + k * sd) + dist.cdf(float(m) - k * sd)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_yscale("log")
    cheb = k ** 2
    uni = np.where(k >= math.sqrt(8 / 3), 9 * k ** 2 / 4, np.nan)
    ax.plot(k, cheb, color=MUTED, lw=1.6, ls=(0, (4, 2)))
    ax.plot(k, uni, color=INK2, lw=1.6, ls=(0, (1.5, 1.5)))
    curves = [("normal", stats.norm(), BLUE), ("Laplace", stats.laplace(), ORANGE),
              ("exponential", stats.expon(), AQUA)]
    for name, d, col in curves:
        ax.plot(k, 1 / two_sided(d), color=col, lw=2.2)
    ax.axvline(3.0, color=INK, lw=0.9, alpha=0.6)
    ax.axhline(1 / (2 * stats.norm.sf(3)), color=BLUE, lw=0.8, alpha=0.35)
    xe = 4.0
    for name, d, col in curves:
        y = 1 / two_sided(d)[-1]
        label_end(ax, xe, y, f"{name}", col)
    label_end(ax, xe, cheb[-1], "Chebyshev: any distribution", MUTED, dy=0.04)
    label_end(ax, xe, uni[-1], "unimodal distributions", INK2, dy=-0.04)
    for name, d, col in curves:
        y3 = 1 / two_sided_at(d, 3.0)
        ax.plot([3.0], [y3], "o", color=col, ms=7, mec="white", mew=1.5, zorder=6)
        ax.annotate(f"{y3:,.0f}", (3.0, y3), xytext=(-8, 6), textcoords="offset points", ha="right", fontsize=9,
                    color=INK2)
    ax.plot([3.0, 3.0], [9, 20.25], "o", color=INK2, ms=5, mec="white", mew=1.2, zorder=6)
    ax.annotate("at least 9", (3.0, 9), xytext=(-8, 0), textcoords="offset points", fontsize=8.5, color=INK2, ha="right", va="center")
    ax.annotate("at least 20.25", (3.0, 20.25), xytext=(-8, 0), textcoords="offset points", fontsize=8.5, color=INK2, ha="right", va="center")
    ax.set_xlim(1.5, 4.0)
    ax.set_ylim(2, 3e4)
    ax.set_xlabel("limits at k standard deviations from the mean")
    ax.set_ylabel("in-control ARL of individual values, known parameters (log scale)")
    ax.set_title("Three sigma means an ARL of 370 only for the normal")
    fig.subplots_adjust(right=0.74, bottom=0.13)
    credit(fig)
    fig.savefig(OUT / "fig1_three_sigma.png")
    plt.close(fig)


def two_sided_at(dist, kk: float) -> float:
    m, v = dist.stats(moments="mv")
    sd = math.sqrt(float(v))
    return float(dist.sf(float(m) + kk * sd) + dist.cdf(float(m) - kk * sd))


def figure_runs_rules() -> None:
    deltas = np.round(np.arange(0.0, 3.01, 0.1), 2)
    t1 = [we_arl(frozenset({1}), d)[0] for d in deltas]
    t12 = [we_arl(frozenset({1, 2}), d)[0] for d in deltas]
    t1234 = [we_arl(frozenset({1, 2, 3, 4}), d)[0] for d in deltas]
    brit = [warning_scheme_arl(3.09, 1.96, d) for d in deltas]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.set_yscale("log")
    series = [("test 1 alone (three sigma)", t1, BLUE), ("tests 1 and 2", t12, VIOLET),
              ("all four Western Electric tests", t1234, ORANGE),
              ("British: 3.09 action, 1.96 warning lines", brit, AQUA)]
    for name, ys, col in series:
        ax.plot(deltas, ys, color=col, lw=2.2, label=f"{name}: in control {ys[0]:.1f}")
        ax.plot([0.0], [ys[0]], "o", color=col, ms=7, mec="white", mew=1.5, zorder=6)
    ax.legend(loc="upper right")
    ax.set_xlim(-0.05, 3.0)
    ax.set_ylim(1, 600)
    ax.set_xlabel("shift of the process mean, in standard deviations of the plotted statistic")
    ax.set_ylabel("average run length (log scale)")
    ax.set_title("Every supplementary test buys speed with false alarms paid for in advance")
    fig.subplots_adjust(bottom=0.13)
    credit(fig)
    fig.savefig(OUT / "fig2_runs_rules.png")
    plt.close(fig)


def ecdf(x: np.ndarray):
    xs = np.sort(x)
    return xs, np.arange(1, xs.size + 1) / xs.size


def figure_estimated_limits() -> None:
    children = np.random.SeedSequence(SEED).spawn(6)
    rng_x = np.random.default_rng(children[2])
    rng_i = np.random.default_rng(children[3])
    mu, s = phase1_xbar(rng_x, 200_000, 25, 5)
    carl_x = 1 / conditional_p(mu, s, 5)
    mu_i, s_i = phase1_individuals(rng_i, 200_000, 25)
    carl_i = 1 / conditional_p(mu_i, s_i, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for carl, col, name in ((carl_x, BLUE, "X-bar chart, 25 subgroups of 5 (A2 x Rbar limits)"),
                            (carl_i, ORANGE, "individuals chart, 25 values (MRbar limits)")):
        xs, ys = ecdf(carl)
        step = max(1, xs.size // 4000)
        ax1.plot(xs[::step], ys[::step], color=col, lw=2.2, label=name)
        share = np.mean(carl < 370.4)
        ax1.plot([370.4], [share], "o", color=col, ms=7, mec="white", mew=1.5, zorder=6)
        if col == BLUE:
            ax1.annotate(f"{share:.0%} of X-bar charts below 370", (370.4, share), xytext=(10, 6),
                         textcoords="offset points", fontsize=9, color=INK2, ha="left")
        else:
            ax1.annotate(f"{share:.0%} of individuals charts below 370", (370.4, share), xytext=(10, -14),
                         textcoords="offset points", fontsize=9, color=INK2, ha="left")
    ax1.axvline(370.4, color=INK, lw=0.9, alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_xlim(10, 1e5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("conditional in-control ARL of one chart (log scale)")
    ax1.set_ylabel("share of charts with a smaller ARL")
    ax1.set_title("Two charts built the textbook way")
    ax1.legend(loc="lower right", fontsize=8.5)

    ms = np.array([20, 25, 30, 50, 75, 100, 150, 200, 300, 500, 1000])
    children_grid = np.random.SeedSequence(SEED + 1).spawn(2)
    rg_x = np.random.default_rng(children_grid[0])
    rg_i = np.random.default_rng(children_grid[1])
    for kind, col, rng in (("X-bar, n = 5", BLUE, rg_x), ("individuals", ORANGE, rg_i)):
        lo, med, hi = [], [], []
        for m in ms:
            if kind == "individuals":
                a, b = phase1_individuals(rng, 40_000, int(m))
                c = 1 / conditional_p(a, b, 1)
            else:
                a, b = phase1_xbar(rng, 40_000, int(m), 5)
                c = 1 / conditional_p(a, b, 5)
            q = np.percentile(c, [10, 50, 90])
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        ax2.fill_between(ms, lo, hi, color=col, alpha=0.12, lw=0)
        ax2.plot(ms, med, color=col, lw=2.2, label=f"{kind}: median and 10th-90th percentile")
        ax2.plot(ms, lo, color=col, lw=0.9, alpha=0.7)
        ax2.plot(ms, hi, color=col, lw=0.9, alpha=0.7)
    ax2.axhline(370.4, color=INK, lw=0.9, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(20, 1000)
    ax2.set_ylim(30, 6000)
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xticks([20, 50, 100, 200, 500, 1000])
    ax2.set_xticklabels(["20", "50", "100", "200", "500", "1000"])
    ax2.set_xlabel("Phase I size m: subgroups (X-bar) or values (individuals), log scale")
    ax2.set_ylabel("conditional in-control ARL (log scale)")
    ax2.set_title("How slowly the band closes on 370")
    ax2.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("With estimated limits, each chart's in-control ARL depends on the Phase I sample it was built from",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.28)
    credit(fig)
    fig.savefig(OUT / "fig3_estimated_limits.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_three_sigma()
    figure_runs_rules()
    figure_estimated_limits()
