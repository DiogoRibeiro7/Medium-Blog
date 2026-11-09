"""Figures for Part 4 of the statistical-process-control series.

fig1_guaranteed_k.png        one chart: the guaranteed limit width against Phase I size, and its price
fig2_nonnormal.png           how often each guarantee holds under four distributions
fig3_line.png                a line of charts: per-chart, joint and Sidak limits, and their price
fig4_calibration_size.png    in-control data needed for a distribution-free guarantee
fig5_liveline.png            the continuous-flow line under a budget of one false alarm per hour
fig6_secom.png               the SECOM line under a budget of one false alarm per 100 runs

Run after budget_numbers.py (which writes figure_data.json):  python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

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
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})

CODE = "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/statistical-process-control"
SIM = "Simulation."
LIVELINE = "Data: Liveline Technologies, multi-stage continuous-flow manufacturing process (Kaggle, 2020)."
SECOM = "Data: SECOM, UCI Machine Learning Repository (McCann and Johnston 2008)."


def credit(fig, data: str) -> None:
    fig.text(0.01, 0.005, f"Figure: Diogo Ribeiro. {data}\n{CODE}", fontsize=7.2, color=MUTED, ha="left",
             va="bottom", linespacing=1.4)


def title(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def log_m_axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def figure_guaranteed_k() -> None:
    d = DATA["one_chart"]
    m = np.array(d["m"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    ax1.plot(m, d["k10"], color=ORANGE, lw=2.4, label="guaranteed with probability 0.90")
    ax1.plot(m, d["k05"], color=VIOLET, lw=2.0, ls="--", label="guaranteed with probability 0.95")
    ax1.axhline(3.0, color=INK, lw=0.9, alpha=0.7)
    ax1.annotate("known parameters: 3", (m[0], 3.0), xytext=(2, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    log_m_axis(ax1, [20, 50, 100, 250, 500, 1000, 2000])
    ax1.set_ylim(2.9, 4.3)
    ax1.set_xlabel("Phase I size m (individual values, log scale)")
    ax1.set_ylabel("limit width k (multiples of the Phase I SD)")
    ax1.set_title("Width that keeps the in-control ARL at 370 or more")
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.plot(m, d["ic_k3_p10"], color=BLUE, lw=2.0, ls="--", label="k = 3: in-control ARL, 10th percentile")
    ax2.plot(m, d["ic_g10_p10"], color=ORANGE, lw=2.0, ls="--", label="guaranteed: in-control ARL, 10th percentile")
    ax2.plot(m, d["med1_k3"], color=BLUE, lw=2.4, label="k = 3: ARL after a 1-sigma shift, median")
    ax2.plot(m, d["med1_g10"], color=ORANGE, lw=2.4, label="guaranteed: ARL after a 1-sigma shift, median")
    ax2.axhline(370.4, color=INK, lw=0.9, alpha=0.6)
    ax2.axhline(d["known1"], color=INK, lw=0.9, alpha=0.6)
    ax2.annotate("370.4", (2000, 370.4), xytext=(-4, 4), textcoords="offset points", fontsize=8.5, color=INK2, ha="right")
    ax2.annotate(f"{d['known1']:.1f}, known parameters", (2000, d["known1"]), xytext=(-4, 4), textcoords="offset points",
                 fontsize=8.5, color=INK2, ha="right")
    log_m_axis(ax2, [20, 50, 100, 250, 500, 1000, 2000])
    ax2.set_yscale("log")
    ax2.set_ylim(30, 4000)
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([50, 100, 200, 500, 1000, 2000]))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("Phase I size m (log scale)")
    ax2.set_ylabel("conditional ARL (log scale)")
    ax2.set_title("What the guarantee costs in detection")
    ax2.legend(loc="upper right", fontsize=8.0)
    title(fig, "One chart: guaranteeing the in-control ARL instead of averaging it")
    fig.subplots_adjust(top=0.86, bottom=0.17, wspace=0.25)
    credit(fig, SIM + " Individuals chart, normal data, limits from the Phase I mean and SD.")
    fig.savefig(OUT / "fig1_guaranteed_k.png")
    plt.close(fig)


def figure_nonnormal() -> None:
    d = DATA["nonnormal"]
    cover = d["cover"]
    dists = list(cover)
    methods = [("normal_theory_100", "normal-theory guarantee, m = 100", ORANGE, "o"),
               ("normal_theory_1000", "normal-theory guarantee, m = 1,000", VIOLET, "s"),
               ("np_bootstrap_100", "nonparametric bootstrap, m = 100", BLUE, "D"),
               (f"order_stat_{d['n2']}", f"smallest and largest of n = {d['n2']:,}", AQUA, "^")]
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    y = np.arange(len(dists))[::-1]
    for i, (key, label, col, mk) in enumerate(methods):
        off = (i - 1.5) * 0.14
        ax.scatter([cover[n][key] for n in dists], y + off, s=60, color=col, marker=mk, edgecolor="white",
                   linewidth=1.0, zorder=3, label=label)
    ax.axvline(0.9, color=INK, lw=0.9, alpha=0.7)
    ax.annotate("target 0.90", (0.9, y[0] + 0.45), xytext=(-4, 0), textcoords="offset points", fontsize=8.5,
                color=INK2, ha="right", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(dists)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.6, len(dists) - 0.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of Phase I samples whose limits keep the in-control ARL at 370 or more")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8.5, borderaxespad=0)
    title(fig, "Each guarantee holds only under the assumptions it was built on")
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.13, right=0.97)
    credit(fig, SIM + " Each distribution has mean 0 and variance 1; 20,000 samples (300 for the bootstrap).")
    fig.savefig(OUT / "fig2_nonnormal.png")
    plt.close(fig)


def figure_line() -> None:
    d = DATA["line"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    styles = {"Sidak": (BLUE, "Šidák, known-parameter width (budget not met)", "--"),
              "every chart": (VIOLET, "a guarantee for every chart", "-"),
              "joint line": (ORANGE, "one joint guarantee for the line", "-")}
    for key, (col, label, ls) in styles.items():
        K = d[key]["K"]
        ax1.plot(K, d[key]["k"], color=col, lw=2.4, ls=ls, marker="o", ms=6, mec="white", label=label)
        ax2.plot(K, d[key]["arl2"], color=col, lw=2.4, ls=ls, marker="o", ms=6, mec="white", label=label)
    for ax in (ax1, ax2):
        log_m_axis(ax, [1, 10, 50, 100, 500])
        ax.set_xlabel("charts on the line, K (log scale)")
    ax1.set_ylabel("limit width k for every chart")
    ax1.set_title("Width needed (m = 100 per chart)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.set_ylabel("median ARL of one chart after a 2-sigma shift (log)")
    ax2.set_title("What it costs in detection")
    title(fig, "A line of charts: guaranteeing the line is far cheaper than guaranteeing each chart")
    fig.subplots_adjust(top=0.86, bottom=0.17, wspace=0.25)
    credit(fig, SIM + " Line in-control ARL of 370 or more with probability 0.90; normal data.")
    fig.savefig(OUT / "fig3_line.png")
    plt.close(fig)


def figure_sizes() -> None:
    d = DATA["sizes"]
    a = np.array(d["alpha"])
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.plot(a, d["n10"], color=ORANGE, lw=2.4, label="guarantee with probability 0.90")
    ax.plot(a, d["n05"], color=VIOLET, lw=2.0, ls="--", label="guarantee with probability 0.95")
    offsets = [(-10, -18), (-10, -16), (14, 26), (-10, -34), (-10, -16)]
    names = ["one chart at the 3-sigma rate", "a line, 1 alarm per 100 runs", "one of 428 sensors, same line budget",
             "a line, 1 alarm per hour, scored each minute", "one of 21 series, same line budget"]
    for (label, alpha, n), name, off in zip(d["marks"], names, offsets):
        ax.scatter([alpha], [n], s=50, color=INK, zorder=4)
        ax.annotate(f"{name}: {n:,}", (alpha, n), xytext=off, textcoords="offset points", fontsize=8.5, color=INK2,
                    ha="right" if off[0] < 0 else "left", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("false-alarm probability to guarantee per score (log scale)")
    ax.set_ylabel("in-control scores needed (log scale)")
    ax.legend(loc="upper right", fontsize=8.5)
    title(fig, "Distribution-free guarantees: the budget decides how much in-control data you need")
    fig.subplots_adjust(top=0.90, bottom=0.17, left=0.10, right=0.97)
    credit(fig, "Calculation (Wilks 1941): threshold = the largest of n in-control scores.")
    fig.savefig(OUT / "fig4_calibration_size.png")
    plt.close(fig)


def figure_liveline() -> None:
    r = DATA["live_rates"]
    det = DATA["live_detect"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [0.8, 1.2]})
    labels = ["A: Šidák limits", "B: joint line\nguarantee (normal)", "C: adaptive\nthreshold"]
    vals = [r["A"], r["B"], r["C"]]
    cols = [BLUE, VIOLET, ORANGE]
    ax1.bar(range(3), vals, color=cols, width=0.6, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax1.annotate(f"{v:.1f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax1.axhline(1.0, color=INK, lw=0.9, alpha=0.7)
    ax1.annotate("budget:\n1 per hour", (2.36, 1.0), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax1.set_xlim(-0.5, 3.0)
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 120)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.grid(axis="x", visible=False)
    ax1.set_ylabel("minutes with a line alarm, per hour (log scale)")
    ax1.set_title("Alarms, minutes 90-234")

    sizes = det["sizes"]
    for key, col in (("A", BLUE), ("B", VIOLET), ("C", ORANGE)):
        ax2.plot(sizes, det["step"][key], color=col, lw=2.4, marker="o", ms=6, mec="white", label=f"{key}, step")
        ax2.plot(sizes, det["ramp"][key], color=col, lw=1.6, ls="--", marker="o", ms=5, mfc="white", label=f"{key}, 60-minute ramp")
    ax2.axhline(0, color=INK, lw=0.8, alpha=0.6)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f"{s:g}" for s in sizes])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_xlabel("size of the injected shift (SDs of that series' minute means, log scale)")
    ax2.set_ylabel("extra detections over the unaltered data")
    ax2.set_title("Injected faults: what each design still catches")
    ax2.legend(loc="upper left", fontsize=8.0, ncol=2)
    title(fig, "Continuous-flow line, 21 series: holding a budget of one false alarm per hour")
    fig.subplots_adjust(top=0.86, bottom=0.19, wspace=0.28)
    credit(fig, LIVELINE)
    fig.savefig(OUT / "fig5_liveline.png")
    plt.close(fig)


def figure_secom() -> None:
    d = DATA["secom"]
    des = d["designs"]
    names = list(des)
    nice = {names[0]: "A: Šidák limits, known-parameter width", names[1]: "B: joint line guarantee, normal theory",
            names[2]: "C: max |z|, threshold calibrated once", names[3]: "D: max |z|, recalibrated before each run",
            names[4]: "E: max |z|, adaptive threshold", names[5]: "F: BH on per-sensor conformal p-values"}
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    y = np.arange(len(names))[::-1]
    floor = 1e-3
    for off, key, col, label in ((0.13, "pass", BLUE, "later passing runs"), (-0.13, "fail", ORANGE, "later failing runs")):
        v = np.maximum([des[n][key] for n in names], floor)
        ax.scatter(v, y + off, s=60, color=col, edgecolor="white", linewidth=1.0, zorder=3, label=label)
    ax.axvline(d["budget"], color=INK, lw=0.9, alpha=0.7)
    ax.annotate("budget 1%", (d["budget"], y[0] + 0.5), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.7, 1.3)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0", "1%", "10%", "100%"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([nice[n] for n in names], fontsize=8.8)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.set_xlabel("share of runs with an alarm (log scale; 0 drawn at the left edge)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8.5, borderaxespad=0)
    title(fig, "SECOM, 428 sensors: a budget of one false alarm per 100 runs")
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.34, right=0.97)
    credit(fig, SECOM)
    fig.savefig(OUT / "fig6_secom.png")
    plt.close(fig)


def main() -> None:
    figure_guaranteed_k()
    figure_nonnormal()
    figure_line()
    figure_sizes()
    figure_liveline()
    figure_secom()


if __name__ == "__main__":
    main()
