"""Figures for the uk-sales-under-water article.

    python figures.py     # reads results/*.csv; writes figures/fig*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"

# First three slots of a validated colour-blind-safe categorical palette (all pairs pass).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SOURCE = ("Contains HM Land Registry data © Crown copyright and database right 2026, licensed "
          "under the Open Government Licence v3.0. Chart: Diogo Ribeiro.")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})
PCT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, title, subtitle=None):
    fig.text(0.01, 0.975, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 0.93, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")


def _finish(fig, name, top=0.86):
    fig.text(0.01, 0.01, SOURCE, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, top))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


def fig1_timing():
    s = pd.read_csv(RES / "monthly_series.csv", parse_dates=["date"])
    s = s[s.date >= "2005-01-01"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    a1.plot(s.date, 100 * s.rate, color=BLUE, linewidth=1.6)
    a1.set_ylabel("Resold in the month, %")
    a1.set_ylim(0, None)
    a1.set_title("Monthly resale rate of owners who bought since 1995", loc="left")
    a2.fill_between(s.date, 0, 100 * s.share_under, color=ORANGE, alpha=0.25, linewidth=0)
    a2.plot(s.date, 100 * s.share_under, color=ORANGE, linewidth=1.6, label="Below the purchase price")
    a2.plot(s.date, 100 * s.share_under10, color=INK2, linewidth=1.4, label="More than 10% below")
    a2.set_ylabel("Share of those owners, %")
    a2.yaxis.set_major_formatter(PCT)
    a2.set_title("Owners whose area index puts them below the price they paid", loc="left")
    a2.legend(loc="upper right", fontsize=9)
    for ax in (a1, a2):
        _grid(ax)
        ax.axvspan(pd.Timestamp("2007-10-01"), pd.Timestamp("2008-06-30"), color=GRID, alpha=0.6, linewidth=0)
    a1.text(pd.Timestamp("2008-08-01"), a1.get_ylim()[1] * 0.9, "Oct 2007 – Jun 2008", color=INK2, fontsize=9)
    _titles(fig, "Sales collapsed before owners were under water",
            "Monthly resales as a share of owners at the start of the month, and the share of those owners "
            "below their purchase price,\nmeasured with the repeat-sales index of their postcode area.")
    _finish(fig, "fig1_timing", top=0.88)


def fig2_lock_in_curve():
    c = pd.read_csv(RES / "lock_in_curve.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for period, col in [("2008-2012", ORANGE), ("2013-2021", BLUE)]:
        d = c[(c.period == period) & c.m_bin.between(-14, 24)].sort_values("m_bin")
        x = 100 * (d.m_lo + 0.01)
        rel = 100 * d.effect / d.ref_rate
        se = 100 * d.se / d.ref_rate
        ax.fill_between(x, rel - 1.96 * se, rel + 1.96 * se, color=col, alpha=0.18, linewidth=0)
        ax.plot(x, rel, color=col, label=f"Resales in {period} (owners at 0 to +2% sold "
                                         f"{100 * d.ref_rate.iloc[0]:.1f}% a year)")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel("Index-implied gain since purchase at the start of the year (m)")
    ax.set_ylabel("Resale probability relative to owners at 0 to +2%")
    ax.legend(loc="lower right", fontsize=9)
    _grid(ax)
    _titles(fig, "Owners below their purchase price sell less often, but there is no step at zero",
            "Difference in the probability of reselling within the year, by 2% band of m, within the same "
            "postcode area and year,\nfor owners with the same years since purchase, type of home and new-build "
            "status. Shaded: 95% intervals clustered by area.")
    _finish(fig, "fig2_lock_in_curve", top=0.86)


def fig3_counterfactual():
    c = pd.read_csv(RES / "counterfactual_by_period.csv")
    c = c[c.year >= 2005]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(c.year, 100 * c.rate, color=INK, marker="o", ms=4, label="Actual")
    ax.plot(c.year, 100 * c.rate_without_lock_in, color=ORANGE, marker="o", ms=4,
            label="If owners below the purchase price had sold like owners at 0 to +2%")
    ax.plot(c.year, 100 * c.rate_without_position, color=BLUE, marker="o", ms=4,
            label="If owners below +10% had sold like owners at +10% to +20%")
    ax.set_ylabel("Resold within the year, %")
    ax.set_ylim(0, None)
    ax.set_xticks(range(2005, 2026, 2))
    ax.legend(loc="upper right", fontsize=9)
    _grid(ax)
    _titles(fig, "How many sales the owners' position held back",
            "Annual resale rate of owners who bought since 1995, and the rate with the effect of their position "
            "removed, using the curve\nestimated for each period (2001-2007, 2008-2012, 2013-2021, 2022-2025).")
    _finish(fig, "fig3_counterfactual", top=0.86)


def fig4_districts():
    b = pd.read_csv(RES / "district_binned.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=False)
    for ax, ep in zip(axes, ["2007-2009", "2007-2010"]):
        for spec, col, lab in [("area only", MUTED, "Same postcode area only"),
                               ("controls", BLUE, "Plus age, type and new-build mix of the stock, and prior trend")]:
            d = b[(b.episode == ep) & (b.spec == spec)]
            ax.plot(100 * d.x, d.y, color=col, marker="o", ms=5, label=lab)
        ax.set_title(f"Change in resale rate, {ep.replace('-', ' to ')}", loc="left")
        ax.set_xlabel(f"Share of the district's owners under water, start of {ep[-4:]}")
        ax.xaxis.set_major_formatter(PCT)
        _grid(ax)
    axes[0].set_ylabel("Change in log resale rate")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.875), ncol=2, fontsize=9)
    _titles(fig, "Districts: the relation depends on what else is held fixed",
            "Postcode districts grouped into tenths of their share of owners under water, compared within "
            "the same postcode area.")
    _finish(fig, "fig4_districts", top=0.78)


def main() -> int:
    fig1_timing()
    fig2_lock_in_curve()
    fig3_counterfactual()
    fig4_districts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
