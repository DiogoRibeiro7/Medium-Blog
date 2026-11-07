"""Figures for Part 2 of the statistical-process-control series.

fig1_one_chart.png        Exact distribution of the conditional in-control ARL, and the mean that does not exist
fig2_parallel.png         Line-level conditional in-control ARL for 1, 10, 50 and 500 machines
fig3_short_runs.png       Fixed limits against a self-starting chart after a changeover, and the shift it absorbs
fig4_cause_selecting.png  The downstream residual chart when stage 1 drifts, with the regression estimated
fig5_t2.png               Hotelling's T^2 with an estimated covariance matrix, by number of sensors

Run after mechanism_numbers.py (which writes figure_data.json):  python figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from mechanism_numbers import ARL0, aarl_truncated, carl_cdf_exact

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


def log_axis(ax, which="x"):
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def figure_one_chart() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ts = np.exp(np.linspace(math.log(10), math.log(1e5), 70))
    cases = [("X-bar, 25 subgroups of 5, pooled S", 25, 5, 100, BLUE),
             ("individuals, 10 values, S", 10, 1, 9, ORANGE),
             ("individuals, 50 values, S", 50, 1, 49, AQUA)]
    for name, m, n, nu, col in cases:
        cdf = [carl_cdf_exact(t, m, n, nu) for t in ts]
        ax1.plot(ts, cdf, color=col, lw=2.2, label=name)
    ax1.axvline(ARL0, color=INK, lw=0.9, alpha=0.6)
    ax1.set_xscale("log")
    log_axis(ax1, "x")
    ax1.set_xlim(10, 1e5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("conditional in-control ARL (log scale)")
    ax1.set_ylabel("share of charts with a smaller ARL (exact)")
    ax1.set_title("Exact distribution across Phase I samples")
    ax1.legend(loc="lower right", fontsize=8.5)

    smax = np.linspace(1.5, 6.0, 19)
    for m, col in ((10, ORANGE), (11, VIOLET), (25, AQUA)):
        vals = [aarl_truncated(m, m - 1, s) for s in smax]
        ax2.plot(smax, vals, color=col, lw=2.2, label=f"{m} values ({m - 1} degrees of freedom)")
    ax2.set_yscale("log")
    ax2.set_xlabel("upper limit s_max of sigma_hat / sigma in the integral")
    ax2.set_ylabel("partial mean of the conditional ARL (log scale)")
    ax2.set_title("The mean exists only above 9 degrees of freedom")
    ax2.legend(loc="upper left", fontsize=8.5, title="individuals chart, sigma from S", title_fontsize=8.5,
               alignment="left")
    fig.suptitle("One chart: a median below 370, and a mean that can be infinite",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.28)
    credit(fig)
    fig.savefig(OUT / "fig1_one_chart.png")
    plt.close(fig)


def figure_parallel() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    q = np.linspace(0.5, 99.5, 199) / 100
    for K, col in (("1", MUTED), ("10", BLUE), ("50", VIOLET), ("500", ORANGE)):
        xs = np.array(DATA["parallel"][K])
        ax.plot(xs, q, color=col, lw=2.4 if K != "1" else 1.8,
                label=f"{K} machine{'s' if K != '1' else ''}: median {np.interp(0.5, q, xs):.0f}")
    ax.axvline(ARL0, color=INK, lw=0.9, alpha=0.6)
    ax.annotate("370.4: the line's target", (ARL0, 0.03), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=INK2)
    ax.set_xscale("log")
    log_axis(ax, "x")
    ax.set_xlim(30, 3000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("conditional in-control ARL of the whole line (log scale)")
    ax.set_ylabel("share of simulated lines with a smaller ARL")
    ax.set_title("Each machine's limits from its own 25 subgroups, widened for K")
    ax.legend(loc="lower right", title="line of K machines, X-bar charts with Sidak limits", title_fontsize=9,
              alignment="left", fontsize=9)
    fig.subplots_adjust(bottom=0.13)
    credit(fig)
    fig.savefig(OUT / "fig2_parallel.png")
    plt.close(fig)


def figure_short_runs() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    ms = ["5", "10", "20"]
    for i, m in enumerate(ms):
        d = DATA["selfstart"][m]
        for j, (key, col, lab) in enumerate((("fixed", ORANGE, "fixed limits from the first m values"),
                                             ("q", BLUE, "self-starting (Q) chart"))):
            v = np.array(d[key])
            lo, med, hi = np.percentile(v, [10, 50, 90])
            y = i + (0.13 if j == 0 else -0.13)
            ax1.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round", label=lab if i == 0 else None)
            ax1.plot([med], [y], "o", color=col, ms=7, mec="white", mew=1.5, zorder=5)
    ax1.axvline(ARL0, color=INK, lw=0.9, alpha=0.6)
    ax1.set_xscale("log")
    log_axis(ax1, "x")
    ax1.set_xlim(3, 3e5)
    ax1.set_yticks(range(len(ms)))
    ax1.set_yticklabels([f"m = {m}" for m in ms])
    ax1.set_ylim(-0.6, len(ms) - 0.4)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("conditional in-control ARL (log scale): median, 10th-90th")
    ax1.set_title("In control, from the same starting units")
    ax1.legend(loc="upper right", fontsize=8.5)

    for delta, ls in (("1", "-"), ("2", (0, (5, 2)))):
        d = DATA["shift"][delta]
        u = np.array(d["u"])
        ax2.plot(u, d["fixed"], color=ORANGE, lw=2.2, ls=ls, label=f"fixed limits, shift of {delta} sigma")
        ax2.plot(u, d["q"], color=BLUE, lw=2.2, ls=ls, label=f"self-starting, shift of {delta} sigma")
    ax2.set_xlim(0, 380)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("units since a sustained shift that starts at unit 21")
    ax2.set_ylabel("share of runs that have signalled")
    ax2.set_title("After a shift it does not catch at once")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8), fontsize=8.5)
    fig.suptitle("Short runs: the self-starting chart keeps its false-alarm rate and loses the shift",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.22, left=0.07)
    credit(fig)
    fig.savefig(OUT / "fig3_short_runs.png")
    plt.close(fig)


def figure_cause_selecting() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    d1 = np.array([0, 1, 2, 3])
    beta, sig_e = 0.8, 0.6
    from scipy import stats
    naive = [1 / (stats.norm.sf(3 - beta * x) + stats.norm.cdf(-3 - beta * x)) for x in d1]
    ax.plot(d1, naive, color=MUTED, lw=2, marker="o", label="chart of the stage-2 measurement itself (known parameters)")
    offs = {"25": -0.06, "50": 0.0, "100": 0.06}
    for m, col in (("25", ORANGE), ("50", VIOLET), ("100", BLUE)):
        vals = DATA["cause"][m]
        lo = [vals[str(x)][0] for x in d1]
        med = [vals[str(x)][1] for x in d1]
        hi = [vals[str(x)][2] for x in d1]
        xs = d1 + offs[m]
        ax.vlines(xs, lo, hi, color=col, lw=2.6, alpha=0.75)
        ax.plot(xs, med, color=col, marker="o", lw=1.6, mec="white", mew=1.2, ms=7,
                label=f"cause-selecting chart, regression from {m} pairs: median and 10th-90th")
    ax.axhline(ARL0, color=INK, lw=0.9, alpha=0.6, label="cause-selecting chart, known regression: 370.4 at every shift")
    ax.set_yscale("log")
    log_axis(ax, "y")
    ax.set_xticks(d1)
    ax.set_xlabel("size of a fault at stage 1 (shift in the stage-1 mean, in its standard deviations); stage 2 unchanged")
    ax.set_ylabel("ARL of the stage-2 chart: time until it blames stage 2 (log scale)")
    ax.set_title("A fault upstream, blamed downstream")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.subplots_adjust(bottom=0.13)
    credit(fig)
    fig.savefig(OUT / "fig4_cause_selecting.png")
    plt.close(fig)


def figure_t2() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True)
    cols = {2: MUTED, 5: AQUA, 10: BLUE, 20: VIOLET, 50: ORANGE}
    rows = DATA["t2"]
    for ax, key, title in ((ax1, "chi", "Known-parameter chi-square limit"),
                           (ax2, "f", "F-based limit, right on average")):
        for p, col in cols.items():
            sel = [r for r in rows if r["p"] == p]
            ms = [r["m"] for r in sel]
            med = [r[key][1] for r in sel]
            lo = [r[key][0] for r in sel]
            hi = [r[key][2] for r in sel]
            ax.fill_between(ms, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(ms, med, color=col, lw=2.2, marker="o", ms=5, label=f"p = {p} sensors")
        ax.axhline(ARL0, color=INK, lw=0.9, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        log_axis(ax, "x")
        log_axis(ax, "y")
        ax.set_xticks([50, 100, 250, 1000])
        ax.set_xlabel("Phase I observations m (log scale)")
        ax.set_title(title)
    ax1.set_ylabel("conditional in-control ARL (log scale)")
    ax1.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Hotelling's T^2: every sensor added needs more Phase I data (median and 10th-90th percentile)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.08)
    credit(fig)
    fig.savefig(OUT / "fig5_t2.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_one_chart()
    figure_parallel()
    figure_short_runs()
    figure_cause_selecting()
    figure_t2()
