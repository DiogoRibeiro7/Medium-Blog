"""Figures for the case study. Run after build_dataset.py; writes PNG files to ./figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from offshore_bids import (block_history, first_and_second, high_bids, load_bids,
                           minimum_bid_breakeven, rejected_then_rebid)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#dddcd6"
NEUTRAL = "#a9a8a2"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": RULE,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": RULE,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def title(fig, main, sub, y=0.965):
    fig.text(0.04, y, main, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.04, y - 0.36 / fig.get_figheight(), sub, fontsize=11,
             color=INK_2, ha="left", va="top")


def credit(fig, source="Data: BOEM lease sale files, sales 249 to BBG3. Figure by the author"):
    fig.text(0.96, 0.012, source, fontsize=8.5, color=INK_2, ha="right", va="bottom")


def times(x, _pos=None):
    return f"{x:g}×"


def millions(x, _pos=None):
    return f"${x:g}m"


# ---------------------------------------------------------------------------
# Figure 1: high bid relative to the minimum bid
# ---------------------------------------------------------------------------


def figure_minimum():
    tracts = high_bids(load_bids())
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.subplots_adjust(left=0.09, right=0.955, top=0.80, bottom=0.13)
    title(fig, "Most high bids sit at, or just above, the minimum bid",
          "Share of blocks whose high bid is at most a given multiple of the minimum bid, "
          "13 Gulf lease sales, 2017 to 2026.")
    groups = [("one bid", tracts[tracts.n_bids == 1], BLUE),
              ("two or more bids", tracts[tracts.n_bids > 1], ORANGE)]
    for label, sub, colour in groups:
        x = np.sort(sub.ratio.to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(np.r_[1.0, x], np.r_[0.0, y], where="post", color=colour, linewidth=2.2,
                zorder=3)
    one = tracts[tracts.n_bids == 1]
    multi = tracts[tracts.n_bids > 1]
    ax.text(2.3, 0.76, f"one bid ({len(one):,} blocks)\nmedian {one.ratio.median():.2f}× the minimum",
            color=INK, fontsize=10.5, ha="left", va="center")
    ax.text(5.6, 0.40, f"two or more bids ({len(multi)} blocks)\nmedian {multi.ratio.median():.2f}×",
            color=INK, fontsize=10.5, ha="left", va="center")
    def on_curve(sub, x):
        return float((sub.ratio <= x).mean())
    ax.plot([2.27, 2.05], [0.79, on_curve(one, 2.05) - 0.01], color=BLUE, linewidth=1.0)
    ax.plot([5.55, 4.2], [0.41, on_curve(multi, 4.2) - 0.01], color=ORANGE, linewidth=1.0)
    share = (tracts.ratio <= 1.10 + 1e-12).mean()
    ax.axvline(1.10, color=INK_2, linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
    ax.text(1.115, 0.985, f"10% above the minimum:\n{100 * share:.1f}% of all blocks",
            fontsize=10, color=INK_2, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 50)
    ax.set_xticks([1, 1.5, 2, 3, 5, 10, 20, 50])
    ax.xaxis.set_major_formatter(FuncFormatter(times))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("High bid as a multiple of the minimum bid (log scale)")
    ax.set_ylabel("Share of blocks")
    credit(fig)
    fig.savefig(OUT / "fig1_high_bid_vs_minimum.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: first and second bids on the same block
# ---------------------------------------------------------------------------


def figure_disagreement():
    bids = load_bids()
    fs = first_and_second(bids)
    first, second = fs["first"] / 1e6, fs["second"] / 1e6
    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    fig.subplots_adjust(left=0.11, right=0.955, top=0.835, bottom=0.10)
    title(fig, "Bidders on the same block often disagree by a factor of two or more",
          f"Highest and second-highest bid on each of the {len(fs)} blocks that drew "
          "more than one bid.\nDashed lines: high bid equal to, twice and ten times the second.")
    lo, hi = 0.08, 60
    grid = np.array([lo, hi])
    for k in (1, 2, 10):
        ax.plot(grid, grid / k, color=NEUTRAL, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(52, 52 / k * 1.1, f"{k}:1", fontsize=10, color=INK_2, ha="right",
                va="bottom", rotation=45, rotation_mode="anchor")
    big = fs.n_bids >= 3
    ax.scatter(first[~big], second[~big], s=34, color=BLUE, alpha=0.75, edgecolor=SURFACE,
               linewidth=0.8, zorder=3, label="two bids")
    ax.scatter(first[big], second[big], s=40, color=ORANGE, alpha=0.9, edgecolor=SURFACE,
               linewidth=0.8, zorder=4, label="three or more bids")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_aspect("equal")
    ax.set_xlabel("Highest bid (log scale)")
    ax.set_ylabel("Second-highest bid (log scale)")
    ax.legend(loc="upper left", fontsize=10.5, markerscale=1.2)
    credit(fig)
    fig.savefig(OUT / "fig2_first_and_second_bids.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: rejected high bids and the next bid on the same block
# ---------------------------------------------------------------------------


def figure_rejections():
    bids = load_bids()
    tracts = high_bids(bids)
    rb = rejected_then_rebid(tracts)
    again = rb[rb.rebid].copy()
    x, y = again.amount / 1e6, again.next_amount / 1e6
    reviewed = again.next_reviewed.astype(bool)
    rejected_again = reviewed & again.next_rejected.astype(bool)
    accepted = reviewed & ~again.next_rejected.astype(bool)

    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    fig.subplots_adjust(left=0.11, right=0.955, top=0.835, bottom=0.10)
    title(fig, "Most rejected blocks that came back drew a higher bid",
          f"Each of the {len(again)} rejected high bids whose block drew bids again, "
          "against the next high bid on that block.")
    lo, hi = 0.1, 30
    ax.plot([lo, hi], [lo, hi], color=NEUTRAL, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(20, 20 * 1.12, "next bid equal to the rejected one", fontsize=9.5, color=INK_2,
            ha="right", va="bottom", rotation=45, rotation_mode="anchor")
    ax.scatter(x[accepted], y[accepted], s=40, color=BLUE, edgecolor=SURFACE, linewidth=0.8,
               zorder=3, label=f"next bid accepted ({int(accepted.sum())})")
    ax.scatter(x[rejected_again], y[rejected_again], s=40, color=ORANGE, edgecolor=SURFACE,
               linewidth=0.8, zorder=4, label=f"rejected again ({int(rejected_again.sum())})")
    ax.scatter(x[~reviewed], y[~reviewed], s=40, facecolor="none", edgecolor=INK_2,
               linewidth=1.0, zorder=4, label=f"review not yet published ({int((~reviewed).sum())})")

    hist = block_history(tracts, "NH16-10-40")
    amounts = hist.amount.to_numpy() / 1e6
    ax.plot(amounts[:-1], amounts[1:], color=INK, linewidth=1.0, zorder=5)
    ax.scatter(amounts[:-1], amounts[1:], s=12, color=INK, zorder=6)
    ax.annotate(f"Mississippi Canyon 40: rejected in {len(hist) - 1} sales,\n"
                f"accepted in {hist.sale_date.iloc[-1]:%Y} at "
                f"${hist.amount.iloc[-1] / 1e6:.1f} million",
                xy=(amounts[-2], amounts[-1]), xytext=(0.115, 15.5), fontsize=10,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK,
                                                        linewidth=0.9, shrinkB=5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_aspect("equal")
    ax.set_xlabel("Rejected high bid (log scale)")
    ax.set_ylabel("Next high bid on the same block (log scale)")
    ax.legend(loc="lower right", fontsize=10.5)
    credit(fig)
    fig.savefig(OUT / "fig3_rejected_and_next_bid.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: what a higher minimum bid puts at stake
# ---------------------------------------------------------------------------


def figure_breakeven():
    tracts = high_bids(load_bids())
    multiples = np.linspace(1.0, 3.0, 201)[1:]
    groups = [("water 400 m or deeper (now $100 per acre)", tracts[~tracts.shallow], 100.0, BLUE),
              ("water shallower than 400 m (now $25 per acre)", tracts[tracts.shallow], 25.0, ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.79, bottom=0.14, wspace=0.22)
    title(fig, "How many bidders a higher minimum bid would have to keep",
          "Left: share of blocks whose high bid is below the new minimum. Right: share of those "
          "bidders who must still bid for bonus revenue not to fall.")
    for label, sub, base, colour in groups:
        rows = [minimum_bid_breakeven(sub, m * base) for m in multiples]
        below = [r["share_tracts_below"] for r in rows]
        breakeven = [r["breakeven_share"] for r in rows]
        axes[0].plot(multiples, below, color=colour, linewidth=2.2, label=label)
        axes[1].plot(multiples, breakeven, color=colour, linewidth=2.2, label=label)
    for ax in axes:
        ax.set_xlim(1.0, 3.0)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([1, 1.5, 2, 2.5, 3])
        ax.xaxis.set_major_formatter(FuncFormatter(times))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.set_xlabel("New minimum bid as a multiple of the current one")
    axes[0].set_ylabel("Blocks with a high bid below the new minimum")
    axes[1].set_ylabel("Break-even share of those bidders")
    axes[0].legend(loc="lower right", fontsize=10)
    deep = tracts[~tracts.shallow]
    b = minimum_bid_breakeven(deep, 150.0)
    axes[1].scatter([1.5], [b["breakeven_share"]], s=50, color=BLUE, zorder=5,
                    edgecolor=SURFACE, linewidth=1.5)
    axes[1].annotate(f"$150 per acre in deep water:\n{100 * b['breakeven_share']:.0f}% must stay",
                     xy=(1.5, b["breakeven_share"]), xytext=(1.85, 0.93), fontsize=10,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.9, shrinkB=6))
    credit(fig)
    fig.savefig(OUT / "fig4_minimum_bid_breakeven.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    figure_minimum()
    figure_disagreement()
    figure_rejections()
    figure_breakeven()
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")
