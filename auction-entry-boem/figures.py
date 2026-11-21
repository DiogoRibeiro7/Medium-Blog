"""Figures for the auction-entry-boem article; reads results/ written by entry.py.

    python figures.py     # figures/fig1_*.png ... fig3_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SOURCE = "Data: BOEM lease-sale files, Gulf sales 231 to BBG3 (2013-2026). Analysis: Diogo Ribeiro."

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
})


def finish(fig, name: str, note: str = "") -> None:
    fig.text(0.01, 0.01, (note + "  " if note else "") + SOURCE, fontsize=7.5, color=INK2, ha="left", va="bottom")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ecdf(ax, v, color, label, ls="-"):
    v = np.sort(np.asarray(v))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, label=label, linestyle=ls)


def fig1(t: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ecdf(ax, t[t.n == 1].ratio, BLUE, f"high bid, one bid ({(t.n == 1).sum():,} tracts)")
    ecdf(ax, t[t.n == 2].ratio, ORANGE, f"high bid, two bids ({(t.n == 2).sum():,})")
    ecdf(ax, t[t.n == 2].ratio2, ORANGE, "losing bid, two bids", ls=(0, (4, 2)))
    ecdf(ax, t[t.n >= 3].ratio, AQUA, f"high bid, three or more ({(t.n >= 3).sum():,})")
    ax.set_xscale("log")
    ax.set_xlim(0.95, 300)
    ax.set_xticks([1, 2, 5, 10, 30, 100, 300], ["1", "2", "5", "10", "30", "100", "300"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("bid as a multiple of the minimum bid (log scale)")
    ax.set_ylabel("share of tracts at or below")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("The losing bid on a contested tract is usually above the winning bid on an uncontested one",
                 loc="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    finish(fig, "fig1_bids_by_competition")


def fig2() -> None:
    m = pd.read_csv(RES / "multipliers.csv")
    labels = {
        "two or more vs one, same sale and depth band": "contested vs uncontested tracts, same sale and depth",
        "same block, different sales": "same block, contested in one sale and not in another",
        "same company, same sale, own bids": "the same company's own bids, same sale",
        "second draw at contested-tract dispersion": "one more bid, as dispersed as on two-bid tracts",
        "second draw from the cell": "one more bid, drawn from single-bid tracts of the cell",
    }
    m = m[m.comparison.isin(labels)].set_index("comparison").loc[list(labels)]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    y = np.arange(len(m))[::-1]
    for yi, (name, r) in zip(y, m.iterrows()):
        colr = ORANGE if name.startswith("second draw") else BLUE
        ax.plot([r.lo, r.hi], [yi, yi], color=colr, linewidth=3, solid_capstyle="round", alpha=0.45)
        ax.scatter(r.multiplier, yi, s=80, color=colr, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(r.hi * 1.04, yi, f"x{r.multiplier:.2f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [labels[n] for n in m.index])
    ax.set_xscale("log")
    ax.set_xlim(0.95, 4.2)
    ax.set_xticks([1, 1.5, 2, 3, 4], ["1", "1.5", "2", "3", "4"])
    ax.axvline(1, color=INK2, linewidth=0.8)
    ax.set_xlabel("multiple of the high bid on a tract with one bid (95% intervals)")
    ax.set_title("What the data attribute to a second bidder, by comparison", loc="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    finish(fig, "fig2_what_a_second_bidder_is_worth",
           "Blue: observed associations (fixed effects, block-clustered intervals). Orange: mechanical value of one more sealed bid (bootstrap).")


def fig3() -> None:
    rr = pd.read_csv(RES / "bidder_or_reserve.csv")
    rr = rr[rr.depth == "deep"]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    q = np.linspace(0, 1, 101)
    for col, colr, lab in (("uplift_cell", ORANGE, "one more bid, drawn from the cell"),
                           ("uplift_pair", AQUA, "one more bid, contested-tract dispersion")):
        u = rr[col].iloc[0]
        ax.plot(100 * q, 100 * q * u, color=colr, label=lab)
    for r in rr.itertuples():
        ax.plot([0, 100], [100 * r.best, 100 * r.best], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(101, 100 * r.best, f"${r.new_per_acre:.0f}/acre if every\nbidder below it pays it", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.set_xlabel("share of single-bid tracts that attract one more bidder (%)")
    ax.set_ylabel("change in high bids on deep-water\nsingle-bid tracts (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("One more bidder against a higher deep-water minimum bid", loc="left")
    fig.tight_layout(rect=(0, 0.05, 0.86, 1))
    finish(fig, "fig3_bidder_or_reserve",
           "Dashed lines: best case for each minimum (no bidder below it walks away); its worst case is a loss of 26% to 58%.")


if __name__ == "__main__":
    t = pd.read_csv(RES / "tracts.csv")
    fig1(t); fig2(); fig3()
    print("figures written:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
