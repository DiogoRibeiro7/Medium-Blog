"""Figures for Part 3 of the statistical-process-control series.

fig1_time_scales.png    Continuous-flow line: the semivariogram, and the realized alarm rate of four designs
fig2_phase1_windows.png Continuous-flow line: realized alarm rates across Phase I windows, against references
fig3_secom_sensors.png  SECOM: per-sensor realized alarm rates, and line-level alarms against limit width
fig4_secom_pca.png      SECOM: PCA-based T^2 and SPE on later passing runs, against a normal reference

Run after audit_numbers.py (which writes figure_data.json):  python figures.py
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

CODE = "Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/statistical-process-control"
LIVELINE = "Data: Liveline Technologies (2019), multi-stage continuous-flow manufacturing process."
SECOM = "Data: SECOM, UCI Machine Learning Repository (McCann and Johnston 2008)."
FLOOR = 1e-4   # realized rates of exactly 0 are drawn here, on log axes


def credit(fig, data: str) -> None:
    fig.text(0.01, 0.005, f"Figure: Diogo Ribeiro. {data}\n{CODE}", fontsize=7.2, color=MUTED, ha="left", va="bottom",
             linespacing=1.4)


def rate_axis(ax, which="y", lo=FLOOR, hi=1.0):
    axis = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_yscale if which == "y" else ax.set_xscale)("log")
    (ax.set_ylim if which == "y" else ax.set_xlim)(lo * 0.7, hi * 1.25)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    labels = ["0", "0.1%", "1%", "10%", "100%"] if lo <= FLOOR else ["0.01%", "0.1%", "1%", "10%", "100%"]
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def floor(v):
    return np.maximum(np.asarray(v, dtype=float), FLOOR)


def nominal_line(ax, p0, which="y", text=True):
    if which == "y":
        ax.axhline(p0, color=INK, lw=0.9, alpha=0.7)
        if text:
            ax.annotate("nominal 0.27%", (ax.get_xlim()[0], p0), xytext=(4, 3), textcoords="offset points",
                        fontsize=8.5, color=INK2)
    else:
        ax.axvline(p0, color=INK, lw=0.9, alpha=0.7)


def figure_time_scales() -> None:
    d = DATA["sigma"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.6), gridspec_kw={"width_ratios": [1.0, 1.05]})
    lags = np.array(d["lags"])
    vg = np.array(d["variogram"])
    stage = np.array(d["stage"])
    for s, col in ((1, BLUE), (2, ORANGE)):
        for row in vg[stage == s]:
            ax1.plot(lags, row, color=col, lw=0.9, alpha=0.28)
        ax1.plot(lags, np.median(vg[stage == s], axis=0), color=col, lw=2.6,
                 label=f"stage {s}: median of {int(np.sum(stage == s))} locations")
    ax1.axhline(1.0, color=INK, lw=0.9, alpha=0.6)
    ax1.annotate("variance over the whole record", (1, 1.0), xytext=(2, 4), textcoords="offset points",
                 fontsize=8.5, color=INK2)
    ax1.axvspan(1, 60, color=GRID, alpha=0.45, lw=0)
    ax1.annotate("what a moving range\nor a one-minute\nsubgroup sees", (1.25, 1.33), fontsize=8.5, color=INK2, va="top")
    ax1.set_xscale("log")
    ax1.set_xticks([1, 10, 60, 600, 3600])
    ax1.set_xticklabels(["1 s", "10 s", "1 min", "10 min", "1 h"])
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlim(0.9, 4000)
    ax1.set_ylim(0, 1.5)
    ax1.set_xlabel("lag between two readings (log scale)")
    ax1.set_ylabel("semivariance / variance")
    ax1.set_title("Variation grows with the time scale")
    ax1.legend(loc="lower right", fontsize=8.5)

    names = [("D1", "1 Hz,\nmoving\nrange"), ("D2", "minute X̄,\nwithin-\nminute σ"),
             ("D3", "minute\nmeans,\nmoving range"), ("D4", "minute\nmeans,\nPhase I SD")]
    rng = np.random.default_rng(3)
    for i, (key, _) in enumerate(names):
        v = np.array(d["designs"][key])
        for s, col, off in ((1, BLUE, -0.13), (2, ORANGE, 0.13)):
            vv = v[stage == s]
            x = i + off + rng.uniform(-0.07, 0.07, len(vv))
            ax2.scatter(x, floor(vv), s=26, color=col, edgecolor="white", linewidth=0.8, zorder=3,
                        label=f"stage {s}" if i == 0 else None)
        ax2.plot([i - 0.3, i + 0.3], [max(np.median(v), FLOOR)] * 2, color=INK, lw=1.6, zorder=4)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n for _, n in names], fontsize=8.5)
    ax2.set_xlim(-0.6, len(names) - 0.4)
    ax2.grid(axis="x", visible=False)
    rate_axis(ax2, "y")
    nominal_line(ax2, d["p0"])
    ax2.set_ylabel("share of later points outside the limits (log scale)")
    ax2.set_title("Phase I = first 30 minutes; black bar = median")
    ax2.legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.0, 0.08))
    fig.suptitle("A four-hour continuous-flow line: which sigma the chart is built on decides how often it sounds",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.24)
    credit(fig, LIVELINE)
    fig.savefig(OUT / "fig1_time_scales.png")
    plt.close(fig)


def figure_windows() -> None:
    d = DATA["lottery"]
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    designs = [("SD", "SD of minute means"), ("MR", "moving range of minute means"), ("AR1", "AR(1) residuals")]
    rows = []
    for m in ("30", "60", "120"):
        for key, label in designs:
            rows.append((m, key, label))
    y = np.arange(len(rows))[::-1].astype(float)
    for yy, (m, key, label) in zip(y, rows):
        for kind, col, dy in (("real", ORANGE, 0.14), ("ref", BLUE, -0.14)):
            q = floor(d[m][key][kind])
            ax.plot([q[0], q[4]], [yy + dy] * 2, color=col, lw=1.4, solid_capstyle="round")
            ax.plot([q[1], q[3]], [yy + dy] * 2, color=col, lw=5.0, solid_capstyle="round")
            ax.scatter([q[2]], [yy + dy], s=34, color="white", edgecolor=col, linewidth=1.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label}, m = {m} min" for m, _, label in rows], fontsize=8.8)
    for i in (3, 6):
        ax.axhline(y[i] + 0.5, color=GRID, lw=1.2)
    ax.grid(axis="y", visible=False)
    rate_axis(ax, "x")
    nominal_line(ax, 0.0027, which="x")
    ax.annotate("nominal 0.27%", (0.0027, y[0] + 0.55), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=INK2, va="center")
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.8)
    ax.plot([], [], color=ORANGE, lw=5, label="the line: every window, every usable measurement")
    ax.plot([], [], color=BLUE, lw=5, label="reference: independent normal (AR(1), φ = 0.7, for residuals)")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", fontsize=8.5, bbox_to_anchor=(0.01, 0.93), ncol=2, borderaxespad=0)
    ax.set_xlabel("share of the other minutes outside the limits (log scale)\nthin bar 10–90%, thick bar 25–75%, dot median")
    fig.suptitle("Which Phase I window: limits from m consecutive minutes, used on the rest of the record",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.32, right=0.97)
    credit(fig, LIVELINE)
    fig.savefig(OUT / "fig2_phase1_windows.png")
    plt.close(fig)


def ecdf(v):
    v = np.sort(floor(v))
    return v, np.arange(1, len(v) + 1) / len(v)


def figure_secom_sensors() -> None:
    d = DATA["secom_uni"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.3), gridspec_kw={"width_ratios": [1.1, 1]})
    for key, col, label in (("ref", BLUE, "100 runs of simulated normal data"),
                            ("oracle", VIOLET, "all passing runs (shape only)"),
                            ("random", AQUA, "100 random passing runs"),
                            ("first", ORANGE, "first 100 passing runs")):
        x, q = ecdf(d[key])
        ax1.step(x, q, where="post", color=col, lw=2.2, label=label)
    rate_axis(ax1, "x")
    nominal_line(ax1, d["p0"], which="x")
    ax1.annotate("nominal\n0.27%", (d["p0"], 0.9), xytext=(-4, 0), textcoords="offset points", fontsize=8.5,
                 color=INK2, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("sensor's share of later passing runs outside its limits (log scale)")
    ax1.set_ylabel("share of charted sensors with a smaller rate")
    ax1.set_title(f"{d['K']} individuals charts, 3-sigma limits")
    ax1.legend(loc="lower right", fontsize=8.3, title="limits estimated from", title_fontsize=8.3, alignment="left")

    k = np.array(d["kgrid"])
    ax2.plot(k, d["any_real"], color=ORANGE, lw=2.4, label="realized, limits from the first 100 passing runs")
    ax2.plot(k, d["any_nom"], color=BLUE, lw=2.0, ls="--", label="known parameters, normal, independent")
    ax2.axvline(d["sidak"], color=INK, lw=0.9, alpha=0.6)
    ax2.annotate(f"Šidák limit k = {d['sidak']:.2f}:\nline nominal 0.27%", (d["sidak"], 0.5), xytext=(5, 0),
                 textcoords="offset points", fontsize=8.5, color=INK2)
    ax2.set_xlim(3, 8)
    ax2.set_ylim(0, 1.02)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_xlabel("limit width k (multiples of the Phase I SD)")
    ax2.set_ylabel("later passing runs with at least one alarm")
    ax2.set_title("The line as a whole")
    ax2.legend(loc="lower left", fontsize=8.3)
    fig.suptitle("SECOM: hundreds of sensors, each charted on its own",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.17, wspace=0.26)
    credit(fig, SECOM)
    fig.savefig(OUT / "fig3_secom_sensors.png")
    plt.close(fig)


def figure_secom_pca() -> None:
    d = DATA["pca"]
    ks = d["ks"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3), sharey=True)
    for j, (ax, stat) in enumerate(zip(axes, ("T²", "SPE"))):
        for m, col in (("100", ORANGE), ("200", VIOLET), ("400", AQUA)):
            real = [d["real"][m][str(k)]["pass"][j] for k in ks]
            ref = [row[j] for row in d["ref"][m]]
            ax.plot(ks, floor(real), color=col, lw=2.4, marker="o", ms=7, mec="white", mew=1.2,
                    label=f"m = {m}, SECOM passing runs")
            ax.plot(ks, floor(ref), color=col, lw=1.4, ls="--", marker="o", ms=6, mfc="white", mew=1.4,
                    label=f"m = {m}, normal reference")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        rate_axis(ax, "y")
        nominal_line(ax, 0.0027, text=(j == 0))
        ax.set_xlabel("principal components retained, k (log scale)")
        ax.set_title(f"{stat} on the first k components" if j == 0 else "SPE on the discarded components")
    axes[0].set_ylabel("later passing runs above the limit (log scale)")
    axes[0].annotate("normal references (hollow):\nat or near 0; m = 100 and\nm = 200 coincide at 0",
                     (11, 3.2e-4), fontsize=8.3, color=INK2)
    axes[1].legend(loc="lower right", fontsize=8.0, ncol=1)
    fig.suptitle("SECOM: principal-component monitoring with limits estimated from m passing runs",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, bottom=0.17, wspace=0.08)
    credit(fig, SECOM)
    fig.savefig(OUT / "fig4_secom_pca.png")
    plt.close(fig)


def main() -> None:
    figure_time_scales()
    figure_windows()
    figure_secom_sensors()
    figure_secom_pca()


if __name__ == "__main__":
    main()
