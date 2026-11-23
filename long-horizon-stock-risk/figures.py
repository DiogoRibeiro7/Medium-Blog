"""Draw the article's figures from the tables horizon_numbers.py writes to results/.

python figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import horizon_core as C  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT_BLUE, LIGHT_ORANGE = "#cde2fb", "#fbd9c9"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)
OECD_CREDIT = (
    "Data: OECD share price indices, ten-year bond yields and consumer prices (CC BY 4.0); Shiller's S&P composite."
    " Code: Medium-Blog/long-horizon-stock-risk. Figure: Diogo Ribeiro."
)
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 25, 30]


def frame(fig, title, subtitle, top_in=0.85, bottom_in=0.7, credit=OECD_CREDIT):
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h)
    fig.text(0.01, 1 - 0.08 / h, title, fontsize=11, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 1 - 0.36 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.01, 0.06 / h, credit, fontsize=7.2, color=MUTED, ha="left", va="bottom")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def percent(ax, axis="y", digits=0):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.{digits}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def horizon_axis(ax):
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0, 31)
    ax.set_xlabel("Holding period, years")


# ----------------------------------------------------------------------------------------------- figure 1


def fig_how_often_how_much():
    panel = pd.read_csv(RES / "panel_horizons.csv")
    sens = pd.read_csv(RES / "dividend_sensitivity.csv")
    us = pd.read_csv(RES / "us_horizons.csv")
    band = sens[sens["yield"].between(0.019, 0.041)].groupby("horizon")
    lo, hi = band.min(), band.max()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.8))
    frame(
        fig,
        "A longer holding period made a real loss rarer, but not smaller",
        "Real stock wealth at the end of every holding period, every start year. 23 OECD markets from the 1950s"
        " to 2025 (share prices plus a 3% dividend\nyield; band: lowest to highest with 2%, 3% or 4%) and the"
        " United States from 1871 to 2022 (Shiller, actual dividends).",
        top_in=1.1,
    )
    for ax in (a, b):
        grid(ax)
        horizon_axis(ax)
        percent(ax)
    a.fill_between(HORIZONS, lo.p_loss.loc[HORIZONS], hi.p_loss.loc[HORIZONS], color=LIGHT_BLUE, lw=0)
    a.plot(panel.horizon, panel.p_loss, color=BLUE, lw=2, marker="o", ms=4)
    a.plot(us.horizon, us.p_loss, color=ORANGE, lw=2, ls="--", marker="o", ms=4)
    a.set_title("How often: share of start years ending in a real loss", fontsize=9.5, color=INK, loc="left")
    a.set_ylim(0, 0.45)
    a.annotate(
        "23 OECD markets",
        (10, panel.set_index("horizon").p_loss[10]),
        (12.5, 0.27),
        color=INK2,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    a.annotate(
        "US, 1871-2022",
        (3, us.set_index("horizon").p_loss[3]),
        (1.2, 0.115),
        color=INK2,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    b.fill_between(HORIZONS, lo.median_loss.loc[HORIZONS], hi.median_loss.loc[HORIZONS], color=LIGHT_BLUE, lw=0)
    b.plot(panel.horizon, panel.median_loss, color=BLUE, lw=2, marker="o", ms=4)
    u = us.dropna(subset=["median_loss"])
    b.plot(u.horizon, u.median_loss, color=ORANGE, lw=2, ls="--", marker="o", ms=4)
    b.set_title("How much: median real loss, among the start years that lost", fontsize=9.5, color=INK, loc="left")
    b.set_ylim(0, 0.45)
    b.annotate(
        "only 6 start years\nin 2 markets",
        (30, panel.set_index("horizon").median_loss[30]),
        (24.5, 0.41),
        color=INK2,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    b.annotate(
        "no US losses\nbeyond 15 years",
        (15, u.set_index("horizon").median_loss[15]),
        (16.5, 0.03),
        color=INK2,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(FIG / "fig1_how_often_how_much.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 2


def fig_funnel():
    f = pd.read_csv(RES / "funnel.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))
    frame(
        fig,
        "The yearly rate narrows with the horizon; the wealth it compounds into spreads out",
        "5th to 95th percentile across every start year in 23 OECD markets, real, share prices plus a 3% dividend"
        " yield. Line: the median.",
    )
    for ax in (a, b):
        grid(ax)
        horizon_axis(ax)
    a.fill_between(f.horizon, f.r05, f.r95, color=LIGHT_BLUE, lw=0)
    a.plot(f.horizon, f.r50, color=BLUE, lw=2)
    a.plot([0, 30], [0, 0], color=INK2, lw=0.8)
    percent(a)
    a.set_title("Real return, per year", fontsize=9.5, color=INK, loc="left")
    a.set_xlim(0, 35)
    fr = f.set_index("horizon")
    a.text(1.6, fr.r95[1], f"{100 * fr.r95[1]:+.0f}%", fontsize=8, color=INK2, ha="left", va="center")
    a.text(1.6, fr.r05[1], f"{100 * fr.r05[1]:+.0f}%", fontsize=8, color=INK2, ha="left", va="center")
    a.text(30.6, fr.r95[30], f"{100 * fr.r95[30]:+.0f}%", fontsize=8, color=INK2, ha="left", va="bottom")
    a.text(30.6, fr.r05[30], f"{100 * fr.r05[30]:+.0f}%", fontsize=8, color=INK2, ha="left", va="top")
    b.fill_between(f.horizon, f.w05, f.w95, color=LIGHT_BLUE, lw=0)
    b.plot(f.horizon, f.w50, color=BLUE, lw=2)
    b.set_yscale("log")
    b.plot([0, 30], [1, 1], color=INK2, lw=0.8)
    b.set_yticks([0.5, 1, 2, 5, 10, 20])
    b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    b.set_title("Real wealth per unit invested (log scale)", fontsize=9.5, color=INK, loc="left")
    b.set_xlim(0, 35)
    b.set_ylim(0.4, 35)
    for v in (fr.w95[30], fr.w05[30]):
        b.text(30.6, v, f"{v:.1f}x", fontsize=8, color=INK2, ha="left", va="center")
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(FIG / "fig2_rate_and_wealth.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 3


def fig_shuffles():
    s = pd.read_csv(RES / "shuffles.csv")
    p = s[s.series == "panel"].set_index("h").loc[HORIZONS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6))
    frame(
        fig,
        "Did the order of the years matter? A little for how often, not for how much",
        "History against 2,000 shuffles of each market's own yearly real returns (same years, random order)."
        " 23 OECD markets, share prices\nplus a 3% dividend yield. Band: 5th to 95th percentile of the shuffles;"
        " dotted line: their median.",
        top_in=1.1,
    )
    for ax in (a, b):
        grid(ax)
        horizon_axis(ax)
        percent(ax)
    a.fill_between(HORIZONS, p.perm_loss_p05, p.perm_loss_p95, color=LIGHT_ORANGE, lw=0)
    a.plot(HORIZONS, p.perm_loss_median, color=ORANGE, lw=1.4, ls=":")
    a.plot(HORIZONS, p.hist_loss, color=BLUE, lw=2, marker="o", ms=4)
    a.set_ylim(0, 0.42)
    a.set_title("Share of start years ending in a real loss", fontsize=9.5, color=INK, loc="left")
    a.text(23, 0.135, "shuffled", color=INK2, fontsize=8.5)
    a.text(16.5, 0.035, "history", color=INK2, fontsize=8.5)
    b.fill_between(HORIZONS, p.perm_size_p05, p.perm_size_p95, color=LIGHT_ORANGE, lw=0)
    b.plot(HORIZONS, p.perm_size_median, color=ORANGE, lw=1.4, ls=":")
    b.plot(HORIZONS, p.hist_size, color=BLUE, lw=2, marker="o", ms=4)
    b.set_ylim(0, 0.5)
    b.set_title("Median real loss, among the start years that lost", fontsize=9.5, color=INK, loc="left")
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(FIG / "fig3_shuffle_test.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 4


def fig_markets():
    w = pd.read_csv(RES / "windows20.csv")
    rows = []
    for area, g in w.groupby("area"):
        m = g.multiple.to_numpy()
        rows.append(
            {
                "area": area,
                "n": len(m),
                "min": m.min(),
                "p25": np.quantile(m, 0.25),
                "median": np.median(m),
                "p75": np.quantile(m, 0.75),
                "max": m.max(),
            }
        )
    r = pd.DataFrame(rows).sort_values("min").reset_index(drop=True)
    labels = {"MIX": "Equal-weight mix of the markets", "USA_SHILLER": "United States 1871-2022 (Shiller)"}
    fig, ax = plt.subplots(figsize=(10, 8.2))
    frame(
        fig,
        "Twenty years in one stock market: 13 of 23 markets had a start year that ended in a real loss",
        "Real stock wealth after 20 years per unit invested, every start year. Thin line: worst to best start;"
        " thick bar: middle half; dot: median.\nShare prices plus a 3% dividend yield (OECD, start years 1955-2005);"
        " the United States from 1871 also with actual dividends. Log scale.",
        top_in=0.95,
    )
    grid(ax, "x")
    for i, row in r.iterrows():
        special = row.area in labels
        col = ORANGE if special else BLUE
        ax.plot([row["min"], row["max"]], [i, i], color=col, lw=1.2, solid_capstyle="round")
        ax.plot([row.p25, row.p75], [i, i], color=col, lw=5.5, solid_capstyle="round")
        ax.plot(row["median"], i, "o", ms=6, color="white", mec=col, mew=1.6)
    names = [labels.get(a, C.NAMES.get(a, a)) + f"  ({n})" for a, n in zip(r.area, r.n, strict=True)]
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(1, color=INK2, lw=0.9)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_xlabel("Real wealth after 20 years, per unit invested (start years in brackets)")
    ax.text(1.03, -0.9, "break-even", fontsize=8, color=INK2, ha="left", va="center")
    fig.savefig(FIG / "fig4_markets_20y.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- figure 5


def fig_where_losses_were():
    w = pd.read_csv(RES / "windows20.csv")
    w = w[~w.area.isin(["MIX", "USA_SHILLER"])]
    order = sorted(w.area.unique(), key=lambda a: (w[w.area == a].start.min(), C.NAMES[a]))
    years = np.arange(w.start.min(), w.start.max() + 1)
    grid_ = np.full((len(order), len(years)), np.nan)
    for i, a in enumerate(order):
        g = w[w.area == a]
        for s, m in zip(g.start, g.multiple, strict=True):
            grid_[i, s - years[0]] = np.log(m)
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f6f5f2", BLUE])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(10, 7))
    frame(
        fig,
        "The 20-year real losses came in two waves: starts around the 1960s and around 2000",
        "Each cell: one market and start year, coloured by real stock wealth 20 years later (share prices plus a"
        " 3% dividend yield).\nOrange: a real loss; grey: about break-even; blue: a gain. Blank: no data.",
    )
    norm = TwoSlopeNorm(vmin=np.log(0.2), vcenter=0.0, vmax=np.log(5))
    im = ax.imshow(
        grid_,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(years[0] - 0.5, years[-1] + 0.5, len(order) - 0.5, -0.5),
    )
    for i in range(len(order)):
        for j, y in enumerate(years):
            v = grid_[i, j]
            if not np.isnan(v) and v < 0:
                ax.add_patch(plt.Rectangle((y - 0.5, i - 0.5), 1, 1, fill=False, ec=INK2, lw=0.4))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([C.NAMES[a] for a in order], fontsize=8.5)
    ax.set_xlabel("Start year of the 20-year holding period")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ticks = [0.2, 0.5, 1, 2, 5]
    cb.set_ticks(np.log(ticks))
    cb.set_ticklabels([f"{t:g}x" for t in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    fig.savefig(FIG / "fig5_where_losses_were.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_how_often_how_much()
    fig_funnel()
    fig_shuffles()
    fig_markets()
    fig_where_losses_were()


if __name__ == "__main__":
    main()
