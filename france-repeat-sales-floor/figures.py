"""Figures for the france-repeat-sales-floor article.

    python figures.py     # reads results/*.csv; writes figures/fig*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"

# First slots of a validated colour-blind-safe categorical palette: France blue, England and
# Wales orange, throughout.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
LIGHT_BLUE, LIGHT_ORANGE = "#a9c8ef", "#f5b89f"
SOURCE = ("DVF: DGFiP, Licence Ouverte 2.0 (Etalab). Contains HM Land Registry data © Crown copyright "
          "and database right 2026 (OGL v3.0). OECD house prices. Chart: Diogo Ribeiro.")
COL = {"France": BLUE, "England and Wales": ORANGE}
LIGHT = {"France": LIGHT_BLUE, "England and Wales": LIGHT_ORANGE}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})
PCT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
PCT1 = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}%")


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, title, subtitle=None):
    fig.text(0.01, 0.975, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 0.925, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")


def _finish(fig, name, top=0.85):
    fig.text(0.01, 0.01, SOURCE, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, top))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


def fig1_market():
    hp = pd.read_csv(RES / "oecd_hpi.csv").set_index("quarter")
    vol = pd.read_csv(RES / "volume_12m.csv", parse_dates=["month"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    q = hp.index.tolist()
    x = np.arange(len(q))
    for iso, lab in [("FRA", "France"), ("GBR", "England and Wales")]:
        s = 100 * hp[iso] / hp.loc["2021-Q1", iso]
        a1.plot(x, s, color=COL[lab], linewidth=2)
        a1.text(x[-1] + 0.4, s.iloc[-1], "France" if iso == "FRA" else "United Kingdom", color=COL[lab],
                fontsize=9.5, va="center")
    a1.axvspan(q.index("2021-Q1") - 0.5, q.index("2025-Q4") + 0.5, color=GRID, alpha=0.45, linewidth=0)
    a1.text(q.index("2021-Q1"), a1.get_ylim()[0] + 1, "  resales in the DVF files", color=INK2, fontsize=8.5)
    ticks = [i for i, s in enumerate(q) if s.endswith("Q1")]
    a1.set_xticks(ticks, [q[i][:4] for i in ticks])
    a1.set_xlim(x[0] - 0.5, x[-1] + 6)
    a1.set_ylabel("Nominal house prices, 2021 Q1 = 100")
    a1.set_title("House prices (OECD index)", loc="left")
    _grid(a1)
    a2.plot(vol.month, vol.sales_12m / 1000, color=BLUE, linewidth=2)
    a2.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    a2.set_ylabel("Thousands, 12 months to the date")
    a2.set_title("French sales of a single dwelling", loc="left")
    a2.set_ylim(0, None)
    _grid(a2)
    _titles(fig, "French prices fell after 2022; British prices dipped and recovered",
            "Left: OECD nominal house price indices. Right: ordinary sales of one house or one flat in "
            "the DVF files, 12-month totals.")
    _finish(fig, "fig1_market", top=0.86)


def fig2_near_zero():
    nz = pd.read_csv(RES / "near_zero.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, lab in zip(axes, ["France", "England and Wales"]):
        d = nz[(nz["sample"] == lab) & (nz.x > -0.12) & (nz.x < 0.12)]
        left = d.x - 0.0025
        other = 100 * (d.share - d.exact_share)
        ax.bar(left, other, width=0.0045, align="edge", color=LIGHT[lab], linewidth=0)
        ax.bar(left, 100 * d.exact_share, bottom=other, width=0.0045, align="edge", color=COL[lab],
               linewidth=0)
        ax.plot(d.x, 100 * d.cf_share, color=INK, linewidth=1.4, linestyle="--")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_xlim(-0.12, 0.12)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}%"))
        ax.set_xlabel("Log change from the purchase price")
        ax.set_title(lab, loc="left", color=COL[lab])
        ax.yaxis.set_major_formatter(PCT1)
        _grid(ax)
        top = d[d.bin == 0]
        ax.annotate("exactly the\nold price", xy=(0.0, 100 * top.share.iloc[0]), xytext=(0.035, 100 * top.share.iloc[0]),
                    fontsize=8.5, color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[0].set_ylabel("Share of repeat sales in the 0.5% bin")
    axes[1].text(-0.115, axes[1].get_ylim()[1] * 0.93, "dashed: smooth counterfactual fitted outside\n"
                 "-10% to +2% (degree-7 polynomial)", fontsize=8.5, color=INK2, va="top")
    _titles(fig, "The hole below the purchase price is shallow in France",
            "Resales 2021-2025 held under five years. Dark segment: resales at exactly the old price. "
            "France 166,780 pairs; England and Wales 575,335.")
    _finish(fig, "fig2_near_zero", top=0.86)


def fig3_scan():
    sc = pd.read_csv(RES / "jump_scan.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, lab in zip(axes, ["France", "England and Wales"]):
        d = sc[(sc["sample"] == lab) & ((sc.cut.abs() >= 0.02 - 1e-9) | (sc.cut == 0))]
        off, zero = d[d.cut != 0], d[d.cut == 0]
        ax.errorbar(off.cut, off.theta, yerr=2 * off.se, fmt="o", ms=3.5, color=MUTED, ecolor=GRID,
                    elinewidth=1.2, capsize=0)
        ax.errorbar(zero.cut, zero.theta, yerr=2 * zero.se, fmt="o", ms=7, color=COL[lab],
                    ecolor=COL[lab], elinewidth=1.6, capsize=0)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvspan(0.07, 0.08, color=GRID, alpha=0.6, linewidth=0)
        ax.text(0.075, -0.17, "price plus\nnotary fees", fontsize=8, color=INK2, ha="center", va="bottom")
        ax.annotate("the old\nprice", xy=(0, zero.theta.iloc[0]), xytext=(0.03, zero.theta.iloc[0] + 0.07),
                    fontsize=8.5, color=COL[lab], arrowprops=dict(arrowstyle="-", color=COL[lab], lw=0.8))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}%"))
        ax.set_xlabel("Cutoff (log change from the purchase price)")
        ax.set_title(lab, loc="left", color=COL[lab])
        _grid(ax)
    axes[0].set_ylabel("Jump in log density at the cutoff")
    axes[0].set_ylim(-0.2, 0.4)
    _titles(fig, "In France the old price looks like any round number",
            "Jump in the density of price changes at each cutoff, with two standard errors. Grey: cutoffs "
            "at least 2% from zero, where the jumps come from round price ratios.")
    _finish(fig, "fig3_scan", top=0.86)


def _wls_line(d):
    w = 1 / d.se_ex ** 2
    X = np.column_stack([np.ones(len(d)), d.median_lr_2_5y])
    XtW = X.T * w.to_numpy()
    return np.linalg.solve(XtW @ X, XtW @ d.theta_ex.to_numpy())


def fig4_market_state():
    ry = pd.read_csv(RES / "region_years.csv").dropna(subset=["theta_ex", "se_ex", "median_lr_2_5y"])
    ry = ry[np.isfinite(ry.se_ex) & (ry.se_ex > 0)]
    uk, fr = ry[ry.country == "England and Wales"], ry[ry.country == "France"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    size = lambda d: np.clip(6 / d.se_ex, 8, 160)  # noqa: E731
    ax.scatter(100 * uk.median_lr_2_5y, uk.theta_ex, s=size(uk), color=LIGHT_ORANGE, edgecolor=SURFACE,
               linewidth=0.8, label="England and Wales region-years, 1996-2025")
    ax.scatter(100 * fr.median_lr_2_5y, fr.theta_ex, s=size(fr), color=BLUE, edgecolor=SURFACE,
               linewidth=0.8, label="French region-years, 2023-2025")
    b = _wls_line(uk)
    xs = np.linspace(min(ry.median_lr_2_5y.min(), -0.1), ry.median_lr_2_5y.max(), 50)
    ax.plot(100 * xs, b[0] + b[1] * xs, color=ORANGE, linewidth=1.8)
    X = np.column_stack([np.ones(len(ry)), ry.median_lr_2_5y, (ry.country == "France").astype(float)])
    w = 1 / ry.se_ex.to_numpy() ** 2
    XtW = X.T * w
    bp = np.linalg.solve(XtW @ X, XtW @ ry.theta_ex.to_numpy())
    xf = np.linspace(fr.median_lr_2_5y.min() - 0.02, fr.median_lr_2_5y.max() + 0.02, 20)
    ax.plot(100 * xf, bp[0] + bp[2] + bp[1] * xf, color=BLUE, linewidth=1.8, linestyle="--")
    ax.text(100 * xf[-1] + 0.5, bp[0] + bp[2] + bp[1] * xf[-1], f"France: {bp[2]:+.2f}\nat the same market",
            color=BLUE, fontsize=9, va="center")
    ax.axhline(0, color=INK2, linewidth=0.8)
    lo, hi = -0.45, 0.95
    out = int(((ry.theta_ex < lo) | (ry.theta_ex > hi)).sum())
    ax.set_ylim(lo, hi)
    if out:
        ax.text(0.99, 0.01, f"{out} imprecise region-year{'s' if out > 1 else ''} outside the range not shown",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Median nominal change of resales after 2-5 years (local market strength)")
    ax.set_ylabel("Jump at the old price, exact resales removed (log)")
    ax.legend(loc="upper left", fontsize=9)
    _grid(ax)
    _titles(fig, "A weaker market does not explain the French result",
            "Each dot is a region in a year (resales held under five years); size grows with precision. "
            "Lines: weighted fits, with a common slope for France.")
    _finish(fig, "fig4_market_state", top=0.87)


def fig5_worlds():
    sim = pd.read_csv(RES / "simulations.csv")
    obs = pd.read_csv(RES / "bunching_observed.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    lim = 0.5
    ax.plot([0, lim], [0, lim], color=GRID, linewidth=1.5)
    ax.text(0.40, 0.42, "every missing seller sells\nat the old price", color=MUTED, fontsize=8.5, rotation=0)
    ax.axhline(0, color=GRID, linewidth=1.5)
    ax.text(0.14, 0.008, "missing sellers do not sell at all", color=MUTED, fontsize=8.5)
    short = {"no floor, no rounding": "no floor",
             "30% sell at exactly the old price": "30% sell at the old price",
             "30% withdraw": "30% withdraw",
             "rounding only: half within 1% go to zero": "rounding only",
             "30% at the old price + rounding": "30% at the old price\n+ rounding",
             "10% at the old price + 20% withdraw + rounding": "10% at the old price,\n20% withdraw, + rounding"}
    for r in sim.itertuples():
        ax.scatter(r.bunch_missing, r.bunch_excess, s=30, color=INK2, marker="s", zorder=3)
        ax.annotate(short.get(r.scenario, r.scenario), (r.bunch_missing, r.bunch_excess), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
    for r in obs.itertuples():
        c = COL[r.sample]
        ax.errorbar(r.missing, r.excess, xerr=2 * r.se_missing, yerr=2 * r.se_excess, fmt="o", ms=9,
                    color=c, ecolor=c, elinewidth=1.4, zorder=4)
        ax.annotate(r.sample, (r.missing, r.excess), xytext=(8, -14), textcoords="offset points", fontsize=10,
                    color=c, fontweight="bold")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Missing below the old price (share of the counterfactual from -10% to 0)")
    ax.set_ylabel("Excess at the old price and up to +2%")
    _grid(ax, "both")
    _titles(fig, "Where the missing sellers go",
            "Squares: simulated markets. Dots: resales 2021-2025 held under five years, with two standard "
            "errors (Poisson bootstrap).")
    _finish(fig, "fig5_worlds", top=0.87)


def main():
    fig1_market()
    fig2_near_zero()
    fig3_scan()
    fig4_market_state()
    fig5_worlds()


if __name__ == "__main__":
    main()
