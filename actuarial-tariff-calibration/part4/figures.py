"""Figures for Part 4 of the actuarial-tariff-calibration series. Run after market_numbers.py (reads figure_data.json).

fig1_double_lift.png          Double lift charts: GLM-A against GLM-B, and the squared boosted rate against GLM-B
fig2_relativity_lorenz.png    Ordered Lorenz curves with relativities, with the market cut-off marked
fig3_bonus_malus_market.png   GLM-A against GLM-B: the share GLM-A wins and the loss ratio of each book, by band
fig4_rate_reviews.png         Rate reviews round by round: market share and level of the first insurer
fig5_same_gini.png            Powers of the boosted rate (one Gini) against GLM-B: loss ratios and share after review
fig6_twins.png                Twins: how much two fits of one model disagree, and the market loss ratio by beta

Run:  python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
COL = {"GLM-A": ORANGE, "GLM-B": VIOLET, "GBM": BLUE, "GBM^2": AQUA, "GBM^1.25": AQUA}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
})
CREDIT = "Figure: Diogo Ribeiro. Data: freMTPL2 (CASdatasets). Code: github.com/DiogoRibeiro7/Medium-Blog/tree/main/actuarial-tariff-calibration"


def credit(fig) -> None:
    fig.text(0.01, 0.005, CREDIT, fontsize=7.5, color=MUTED, ha="left", va="bottom")


def figure_double_lift() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.22})
    for ax, (key, a, b, title) in zip(axes, [
            ("GLM-A|GLM-B", "GLM-A", "GLM-B", "GLM-A against GLM-B"),
            ("GBM^2|GLM-B", "GBM^2", "GLM-B", "Squared boosted rate (Gini 0.3218) against GLM-B")]):
        rows = DATA["double_lift"][key]
        x = np.arange(1, 11)
        ax.plot(x, [r["actual"] for r in rows], "-o", color=INK, lw=2, ms=5, label="actual claim frequency",
                mec="white", mew=1)
        ax.plot(x, [r["A"] for r in rows], "-o", color=COL[a], lw=2, ms=5, label=f"{a} prediction", mec="white", mew=1)
        ax.plot(x, [r["B"] for r in rows], "-o", color=COL[b], lw=2, ms=5, label=f"{b} prediction", mec="white", mew=1)
        ax.set_yscale("log")
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5]) if a == "GBM^2" else ax.set_yticks([0.6, 0.8, 1, 1.2, 1.5, 1.8])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(x)
        ax.set_xlabel(f"decile of {a} / {b} (equal exposure): {b} relatively dearer on the left")
        ax.set_title(title)
        ax.legend(loc="upper center", fontsize=9)
    axes[0].set_ylabel("relative to the overall average (log scale)")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)
    credit(fig)
    fig.savefig(OUT / "fig1_double_lift.png")
    plt.close(fig)


def figure_relativity_lorenz() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), gridspec_kw={"wspace": 0.18})
    panels = [(axes[0], [("GLM-A|GLM-B", "base GLM-A, competitor GLM-B", ORANGE),
                         ("GLM-B|GLM-A", "base GLM-B, competitor GLM-A", VIOLET)], "Two GLMs against each other"),
              (axes[1], [("GLM-B|GBM", "base GLM-B, competitor boosting", VIOLET),
                         ("GBM|GLM-B", "base boosting, competitor GLM-B", BLUE)], "GLM-B and the boosting")]
    for ax, curves, title in panels:
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
        for key, lab, col in curves:
            c = DATA["lorenz"][key]
            ax.plot(c["x"], c["y"], color=col, lw=2, label=lab)
            xs, ys = c["cut"]["1"]
            ax.plot(xs, ys, "o", color=col, ms=8, mec="white", mew=1.5, zorder=5)
            for lv in ("0.97", "1.03"):
                ax.plot(*c["cut"][lv], "o", color=col, ms=4, mec="white", mew=0.8, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("share of the base insurer's premium, lowest relativity first")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("share of claims")
    axes[0].annotate("large dot: the competitor at its fitted level\ntakes everything to the left;\nsmall dots: competitor 3% cheaper / dearer",
                     (DATA["lorenz"]["GLM-A|GLM-B"]["cut"]["1"][0], DATA["lorenz"]["GLM-A|GLM-B"]["cut"]["1"][1]),
                     xytext=(0.36, 0.12), fontsize=8.5, color=INK2,
                     arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.11)
    credit(fig)
    fig.savefig(OUT / "fig2_relativity_lorenz.png")
    plt.close(fig)


def figure_bonus_malus() -> None:
    rows = DATA["bands"]
    labels = [r["band"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"wspace": 0.22})
    ax = axes[0]
    ax.bar(x, [r["shareA"] for r in rows], color=ORANGE, width=0.62, label="won by GLM-A")
    ax.bar(x, [1 - r["shareA"] for r in rows], bottom=[r["shareA"] for r in rows], color=VIOLET, width=0.62,
           label="won by GLM-B", edgecolor="white", linewidth=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the band's exposure")
    ax.set_xlabel("bonus-malus band")
    ax.set_title("Who wins each band (cheaper quote wins)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.axhline(1, color=MUTED, lw=1)
    la = [r["lrA"] if r["shareA"] > 0.02 else np.nan for r in rows]
    lb = [r["lrB"] if r["shareA"] < 0.98 else np.nan for r in rows]
    ax.plot(x - 0.12, la, "o", color=ORANGE, ms=8, mec="white", mew=1.2, label="GLM-A's book")
    ax.plot(x + 0.12, lb, "o", color=VIOLET, ms=8, mec="white", mew=1.2, label="GLM-B's book")
    ax.set_xticks(x, labels)
    ax.set_ylabel("loss ratio (claims / premium)")
    ax.set_xlabel("bonus-malus band (bands where a side wins under 2% are not shown for it)")
    ax.set_title("What each insurer's book in the band costs it")
    ax.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.25)
    credit(fig)
    fig.savefig(OUT / "fig3_bonus_malus_market.png")
    plt.close(fig)


def figure_rate_reviews() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.22})
    series = [("GLM-A|GLM-B", "GLM-A against GLM-B", ORANGE), ("GLM-B|GBM", "GLM-B against the boosting", VIOLET),
              ("GBM^1.25|GBM", "boosting^1.25 against the boosting (same Gini)", AQUA)]
    for key, lab, col in series:
        for beta, ls in (("inf", "-"), ("10", "--")):
            path = DATA["reviews"][f"{key}|{beta}"]
            r = [p["round"] for p in path]
            axes[0].plot(r, [p["share"] for p in path], ls, color=col, lw=2, marker="o" if beta == "inf" else None,
                         ms=4, label=f"{lab}, {'cheapest wins' if beta == 'inf' else 'beta = 10'}")
            axes[1].plot(r, [p["la"] for p in path], ls, color=col, lw=2, marker="o" if beta == "inf" else None, ms=4)
    axes[0].set_ylabel("first insurer's share of exposure")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Market share after each rate review")
    axes[1].set_ylabel("first insurer's level (1 = as fitted)")
    axes[1].set_title("The level each rate review imposes")
    for ax in axes:
        ax.set_xlabel("round (each round: quote, lose or win business, rescale by own loss ratio)")
        ax.set_xlim(0.5, 25.5)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig4_rate_reviews.png")
    plt.close(fig)


def figure_same_gini() -> None:
    P = DATA["powers"]
    pw = sorted(P, key=float)
    x = [float(k) for k in pw]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.22})
    ax = axes[0]
    ax.axhline(1, color=MUTED, lw=1)
    ax.plot(x, [P[k]["inf"]["lrA0"] for k in pw], "-o", color=BLUE, lw=2, ms=5, mec="white", mew=1,
            label="boosting^power's book")
    ax.plot(x, [P[k]["inf"]["lrB0"] for k in pw], "-o", color=VIOLET, lw=2, ms=5, mec="white", mew=1,
            label="GLM-B's book")
    ax.set_ylabel("loss ratio in the first period (cheapest wins)")
    ax.set_title("One period: every power has the boosting's Gini, 0.3218")
    ax.legend(loc="upper center", fontsize=9)
    ax = axes[1]
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.plot(x, [P[k]["inf"]["share_end"] for k in pw], "-o", color=BLUE, lw=2, ms=5, mec="white", mew=1,
            label="cheapest wins")
    ax.plot(x, [P[k]["10"]["share_end"] for k in pw], "--o", color=BLUE, lw=1.5, ms=4, mec="white", mew=1,
            alpha=0.7, label="beta = 10")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("boosting^power's share after the rate reviews")
    ax.set_title("After repeated rate reviews against GLM-B (Gini 0.2924)")
    ax.legend(loc="lower left", fontsize=9)
    for a_ in axes:
        a_.set_xlabel("power applied to the boosted rate (1 = as fitted; above 1 more spread out)")
        a_.set_xticks(x, [k for k in pw], fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.14)
    credit(fig)
    fig.savefig(OUT / "fig5_same_gini.png")
    plt.close(fig)


def figure_twins() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"wspace": 0.22})
    ax = axes[0]
    edges = np.linspace(-1.2, 1.2, 97)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for k, col, lab in (("GLM-B", VIOLET, "GLM-B fitted on each half"), ("GBM", BLUE, "boosting fitted on each half")):
        h = np.array(DATA["twin_ratio"][k]["hist"], dtype=float)
        ax.plot(np.exp(mid), h / h.sum() / np.diff(edges)[0], color=col, lw=2, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([0.4, 0.6, 0.8, 1, 1.25, 1.6, 2.5])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.35, 2.9)
    ax.set_xlabel("rate of the first twin / rate of the second (log scale)")
    ax.set_ylabel("density of policies")
    ax.set_title("Same model, two halves of the data")
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    betas = ["0", "2", "5", "10", "20", "inf"]
    x = np.arange(len(betas))
    ax.axhline(1, color=MUTED, lw=1)
    for key, col, lab, ls in (("twins GBM", BLUE, "twin boosting models", "-"),
                              ("twins GLM-B", VIOLET, "twin GLM-B models", "-"),
                              ("GLM-B|GBM", INK2, "GLM-B against the boosting", "--"),
                              ("GLM-A|GLM-B", ORANGE, "GLM-A against GLM-B", "--")):
        ax.plot(x, [DATA["sens"][f"{key}|{b}"]["mkt"] for b in betas], ls, marker="o", color=col, lw=2, ms=5,
                mec="white", mew=1, label=lab)
    ax.set_xticks(x, ["0", "2", "5", "10", "20", "cheapest\nwins"])
    ax.set_xlabel("price sensitivity beta")
    ax.set_ylabel("loss ratio of the market as a whole")
    ax.set_title("Both insurers balanced, the market is not")
    ax.legend(loc="upper left", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.16)
    credit(fig)
    fig.savefig(OUT / "fig6_twins.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_double_lift()
    figure_relativity_lorenz()
    figure_bonus_malus()
    figure_rate_reviews()
    figure_same_gini()
    figure_twins()
    print(f"figures written to {OUT}")
