"""Draw the six figures for the article from ``results/``, numbered in the order the article shows them.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import hubs as H  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
IE_C, LU_C, EU_C, NONEU_C, OTHER_C = SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[3], BASE
SOURCE = ("Sources: Eurostat balance of payments and international investment position by partner (bop_eu6_q, "
          "bop_c6_q, bop_iip6_q) and financial balance sheets (nasa_10_f_bs), snapshot 13 Nov 2026; "
          "author's calculations.")
NAMES = {"IE": "Ireland", "LU": "Luxembourg", "DE": "Germany", "FR": "France", "NL": "Netherlands",
         "SE": "Sweden", "DK": "Denmark", "IT": "Italy", "FI": "Finland", "BE": "Belgium", "ES": "Spain",
         "AT": "Austria"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK2,
    "hatch.linewidth": 0.8,
})


def tidy(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, SOURCE, ha="left", va="bottom", fontsize=7.2, color=MUTED, wrap=True)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def stacked(ax, x, parts: list[tuple[np.ndarray, dict]], width: float = 0.72) -> None:
    """Stacked bars that stack positives up and negatives down, with a 2px surface gap between fills."""
    pos = np.zeros(len(x))
    neg = np.zeros(len(x))
    for vals, style in parts:
        vals = np.asarray(vals, float)
        bottom = np.where(vals >= 0, pos, neg)
        ax.bar(x, vals, width, bottom=bottom, edgecolor=SURFACE, linewidth=1.0, **style)
        pos = pos + np.where(vals >= 0, vals, 0)
        neg = neg + np.where(vals < 0, vals, 0)


def fig1() -> None:
    a = pd.read_csv(RES / "us_purchases_annual.csv", index_col=0) / 1000
    h = pd.read_csv(RES / "us_holdings_2024.csv", index_col=0).iloc[:, 0] / 1000
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.17, wspace=0.42)
    x = np.arange(len(a))
    stacked(ax, x, [(a.IE_flow.values, {"color": IE_C}), (a.LU_flow.values, {"color": LU_C}),
                    (a.other_flow.values, {"color": OTHER_C})])
    ax.scatter(x, a.eu27_flow, s=22, color=INK, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, a.index.astype(str), fontsize=8.5)
    ax.set_ylabel("EUR billion")
    ax.set_title("A. Net purchases of US securities by EU residents, by member state")
    tidy(ax)
    share = (a.IE_flow + a.LU_flow).sum() / a.eu27_flow.sum()
    ax.text(0.02, 0.96, f"Ireland and Luxembourg: {share:.0%} of the 2015–2025 total",
            transform=ax.transAxes, fontsize=9, color=INK, va="top")
    ax.legend(handles=[Patch(color=IE_C, label="Ireland"), Patch(color=LU_C, label="Luxembourg"),
                       Patch(color=OTHER_C, label="Other 25 members"),
                       plt.Line2D([], [], marker="o", linestyle="", color=INK, label="EU27 total")],
              loc="upper left", bbox_to_anchor=(0.0, 0.90), fontsize=8.5)
    top = h.head(8)
    rest = h.iloc[8:].sum()
    names = [NAMES.get(g, g) for g in top.index] + [f"Other {len(h) - 8} members"]
    vals = list(top.values) + [rest]
    cols = [IE_C if g == "IE" else LU_C if g == "LU" else OTHER_C for g in top.index] + [OTHER_C]
    y = np.arange(len(vals))[::-1]
    bx.barh(y, vals, color=cols, height=0.7, edgecolor=SURFACE)
    for yi, v in zip(y, vals):
        bx.text(v + 25, yi, f"{v:,.0f}", va="center", fontsize=8.5, color=INK2)
    bx.set_yticks(y, names, fontsize=8.5)
    bx.set_xlabel("EUR billion")
    bx.set_xlim(0, max(vals) * 1.18)
    bx.set_title("B. Holdings of US securities, end-2024")
    tidy(bx, "x")
    finish(fig, "fig1_us_purchases_by_member",
           "Most of the EU's buying of US securities is done by residents of Ireland and Luxembourg",
           "Portfolio investment in US equity, investment fund shares and debt securities, net of sales; dots are "
           "the EU27 aggregate, which differs slightly from the sum of members.")


def fig2() -> None:
    p = pd.read_csv(RES / "hub_positions.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.08)
    for ax, h in zip(axes, H.HUBS):
        x = np.arange(len(p))
        rep, imp = p[f"{h}_eu_reported"] / 1e6, p[f"{h}_eu_imputed"] / 1e6
        non = p[f"{h}_non_eu"] / 1e6
        ax.bar(x, rep, 0.72, color=EU_C, edgecolor=SURFACE)
        ax.bar(x, imp, 0.72, bottom=rep, color=SURFACE, edgecolor=EU_C, hatch="////", linewidth=0.8)
        ax.bar(x, non, 0.72, bottom=rep + imp, color=NONEU_C, edgecolor=SURFACE)
        lo = (p[f"{h}_nonresident"] - p[f"{h}_non_eu_high"]) / 1e6
        hi = (p[f"{h}_nonresident"] - p[f"{h}_non_eu_low"]) / 1e6
        ax.vlines(x + 0.43, lo, hi, color=INK, linewidth=1.2)
        ax.set_xticks(x, p.index.astype(str), fontsize=8, rotation=0)
        ax.tick_params(axis="x", labelsize=7.5)
        share = (p[f"{h}_non_eu"] / p[f"{h}_nonresident"]).iloc[-1]
        ax.set_title(f"{'A' if h == 'IE' else 'B'}. {NAMES[h]}: {share:.0%} held outside the EU at end-{p.index[-1]}")
        tidy(ax)
    axes[0].set_ylabel("EUR trillion")
    fig.legend(handles=[Patch(color=EU_C, label="Other EU members, published"),
                        Patch(facecolor=SURFACE, edgecolor=EU_C, hatch="////", label="Other EU members, imputed"),
                        Patch(color=NONEU_C, label="Outside the EU (residual)"),
                        plt.Line2D([], [], color=INK, label="Range of the EU / non-EU split")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.05), fontsize=8.5)
    finish(fig, "fig2_who_holds_hub_securities",
           "Who holds the equity and fund shares issued in Ireland and Luxembourg",
           "Holdings by non-residents at the end of each year; EU holders from other members' reported assets "
           "(mirror data), the rest by residual.")


def fig3() -> None:
    f = pd.read_csv(RES / "hub_flows_annual.csv", index_col=0) / 1000
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20)
    x = np.arange(len(f))
    rest = f.eu_inflow - f.hub_non_eu
    stacked(ax, x, [(f.hub_non_eu.values, {"color": NONEU_C}), (rest.values, {"color": OTHER_C})])
    ax.vlines(x - 0.2, f.hub_non_eu_low, f.hub_non_eu_high, color=INK, linewidth=1.2)
    ax.scatter(x, f.eu_inflow, s=24, color=INK, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, f.index.astype(str))
    ax.set_ylabel("EUR billion")
    tidy(ax)
    share = f.hub_non_eu.sum() / f.eu_inflow.sum()
    ax.text(0.01, 0.97, f"2015–2025: hub equity and fund shares = {share:.0%} of the total "
            f"({f.hub_non_eu.sum():,.0f} of {f.eu_inflow.sum():,.0f} billion)", transform=ax.transAxes,
            fontsize=9, va="top")
    fig.legend(handles=[Patch(color=NONEU_C, label="Irish and Luxembourg equity and fund shares"),
                        Patch(color=OTHER_C, label="All other EU securities (residual)"),
                        plt.Line2D([], [], marker="o", linestyle="", color=INK, label="Total non-EU purchases"),
                        plt.Line2D([], [], color=INK, label="Range for the hub part")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.05), fontsize=8.5)
    finish(fig, "fig3_gross_inflow_decomposition",
           "Non-EU purchases of EU securities are mostly purchases of Irish and Luxembourg equity and fund shares",
           "Net purchases of EU securities by non-EU residents (EU27 portfolio investment liabilities), "
           "split by the mirror estimate for the two hubs.")


def fig4() -> None:
    a = pd.read_csv(RES / "attribution_annual.csv", index_col=0) / 1000
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"width_ratios": [4.2, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.28)
    x = np.arange(len(a))
    parts = [(a.other_members.values, {"color": OTHER_C}), (a.hubs_eu_owned.values, {"color": EU_C}),
             (a.hubs_non_eu.values, {"color": NONEU_C})]
    stacked(ax, x, parts)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, a.index.astype(str), fontsize=8.5)
    ax.set_ylabel("EUR billion")
    ax.set_title("A. By year")
    tidy(ax)
    tot = a.sum()
    cum = 0.0
    for key, col in (("other_members", OTHER_C), ("hubs_eu_owned", EU_C), ("hubs_non_eu", NONEU_C)):
        bx.bar(0, tot[key], 0.6, bottom=cum, color=col, edgecolor=SURFACE)
        bx.text(0.36, cum + tot[key] / 2, f"{tot[key] / tot.eu27:.0%}", va="center", fontsize=9, color=INK)
        cum += tot[key]
    lo, hi = tot.other_members + tot.hubs_eu_owned + tot.hubs_non_eu - tot.hubs_non_eu_high, \
        tot.other_members + tot.hubs_eu_owned + tot.hubs_non_eu - tot.hubs_non_eu_low
    bx.vlines(-0.36, lo, hi, color=INK, linewidth=1.2)
    bx.set_xlim(-0.6, 0.8)
    bx.set_xticks([0], ["2015–2025"])
    bx.set_title("B. Total")
    tidy(bx)
    fig.legend(handles=[Patch(color=OTHER_C, label="Bought directly by the other 25 members"),
                        Patch(color=EU_C, label="Bought by Irish and Luxembourg residents for EU owners"),
                        Patch(color=NONEU_C, label="Bought by Irish and Luxembourg residents for non-EU owners"),
                        plt.Line2D([], [], color=INK, label="Range of the non-EU part (panel B)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.03), fontsize=8.5)
    finish(fig, "fig4_attribution_of_us_purchases",
           "About a third of the EU's purchases of US securities are made for owners outside the EU",
           "EU27 net purchases of US securities; the hubs' purchases are split in proportion to the estimated "
           "ownership of their fund shares.")


def fig5() -> None:
    q = pd.read_csv(RES / "quarterly_flows.csv", index_col=0)
    panels = [("non_eu_in", "hub_us", "A. Hubs' US purchases against\nnon-EU purchases of hub shares", NONEU_C),
              ("eu_in", "hub_us", "B. Hubs' US purchases against\nEU purchases of hub shares", EU_C),
              ("non_eu_in", "other_us", "C. Other members' US purchases against\nnon-EU purchases of hub shares",
               OTHER_C)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.20, wspace=0.10)
    for ax, (xc, yc, title, col) in zip(axes, panels):
        x, y = q[xc].values, q[yc].values
        ax.scatter(x, y, s=34, color=col, edgecolor=INK2 if col == OTHER_C else SURFACE, linewidth=0.6, zorder=3)
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=INK, linewidth=1.5)
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.04, 0.95, f"r = {r:.2f}, slope = {b[0]:.2f}", transform=ax.transAxes, fontsize=9, va="top")
        ax.axhline(0, color=BASE, linewidth=0.8)
        ax.axvline(0, color=BASE, linewidth=0.8)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Purchases of hub equity and fund shares,\nEUR billion per quarter", fontsize=8.5)
        tidy(ax, "both")
    axes[0].set_ylabel("Purchases of US securities, EUR billion per quarter")
    finish(fig, "fig5_quarterly_comovement",
           "The hubs buy US securities when money flows into their funds, from inside and outside the EU",
           "44 quarters, 2015Q1–2025Q4; each point is one quarter. Panel C is a placebo: the other members' "
           "purchases should not track flows into the hubs.")


def fig6() -> None:
    s = pd.read_csv(RES / "sensitivity.csv")
    flows = s[~s.variant.str.startswith("holdings")].copy()
    order = ["lower bound", "central", "ownership at start of year", "no correction for company shares",
             "owners outside the hub only", "upper bound", "ownership of net purchases 2015-2025"]
    flows = flows.set_index("variant").loc[order]
    label = {
        "lower bound": "Lower bound: missing EU holders' intra-EU holdings all in the hubs",
        "central": "Central: imputed EU holders, mid-year ownership",
        "ownership at start of year": "Ownership at the start of each year",
        "no correction for company shares": "No correction for shares of Irish companies",
        "owners outside the hub only": "Ownership among holders outside the hub only",
        "upper bound": "Upper bound: missing EU holders own nothing in the hubs",
        "ownership of net purchases 2015-2025": "Ownership of net new money, 2015–2025",
    }
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.42, right=0.96, top=0.80, bottom=0.17)
    y = np.arange(len(flows))[::-1]
    lo, hi = flows.loc["lower bound", "share"], flows.loc["upper bound", "share"]
    ax.axvspan(lo * 100, hi * 100, color=GRID, alpha=0.6, zorder=0)
    for yi, (k, row) in zip(y, flows.iterrows()):
        col = NONEU_C if k == "central" else INK2
        ax.scatter(row.share * 100, yi, s=70 if k == "central" else 42, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.text(row.share * 100 + 0.8, yi, f"{row.share:.1%}", va="center", fontsize=9, color=INK2)
    hold = s[s.variant.str.startswith("holdings")].iloc[0]
    yh = -1.3
    ax.hlines(yh, hold.low * 100, hold.high * 100, color=INK2, linewidth=2)
    ax.scatter(hold.share * 100, yh, s=60, color=NONEU_C, marker="D", zorder=3, edgecolor=SURFACE)
    ax.text(hold.high * 100 + 0.8, yh, f"{hold.share:.1%} ({hold.low:.0%}–{hold.high:.0%})", va="center",
            fontsize=9, color=INK2)
    ax.set_yticks(list(y) + [yh], [label[k] for k in flows.index] + ["Holdings of US securities, end-2024"],
                  fontsize=8.8)
    ax.set_xlim(0, 50)
    ax.set_ylim(yh - 0.8, len(flows) - 0.4)
    ax.set_xlabel("Share of EU27 purchases (or holdings) of US securities attributed to non-EU owners, %")
    tidy(ax, "x")
    finish(fig, "fig6_sensitivity",
           "The non-EU share runs from about a fifth to about two-fifths under every assumption tried",
           "Purchases 2015–2025 unless stated; the shaded band spans the lower and upper bounds of the central method.")


def main() -> None:
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()


if __name__ == "__main__":
    main()
