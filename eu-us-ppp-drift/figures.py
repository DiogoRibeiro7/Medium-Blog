"""Draw the five figures for the article from ``results/``, numbered in the order the article shows them.

    python figures.py

Static PNGs for Medium (light surface), one fixed categorical order, direct labels where series are few.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOURCE = ("Sources: OECD national accounts (Tables 1, 4, 5, 6 and GFCF by asset) and Eurostat prc_ppp_ind, "
          "snapshot 12 Nov 2026; author's calculations.")
NAMES = {
    "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia", "CZE": "Czechia", "DNK": "Denmark",
    "EST": "Estonia", "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece", "HUN": "Hungary",
    "IRL": "Ireland", "ITA": "Italy", "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg", "NLD": "Netherlands",
    "POL": "Poland", "PRT": "Portugal", "ROU": "Romania", "SVK": "Slovakia", "SVN": "Slovenia", "ESP": "Spain",
    "SWE": "Sweden", "GBR": "United Kingdom", "CHE": "Switzerland", "NOR": "Norway", "ISL": "Iceland",
    "TUR": "Türkiye", "SRB": "Serbia", "MKD": "North Macedonia", "ALB": "Albania", "JPN": "Japan", "KOR": "Korea",
    "CAN": "Canada", "AUS": "Australia", "NZL": "New Zealand", "ISR": "Israel", "MEX": "Mexico", "CHL": "Chile",
    "COL": "Colombia", "CRI": "Costa Rica",
}
REGIONS = [("EU", SLOTS[0]), ("Other Europe", SLOTS[1]), ("Non-European OECD", SLOTS[2])]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def tidy(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str, source: str = SOURCE) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, source, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig1() -> None:
    p = pd.read_csv(RES / "drift_paths.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.07, right=0.79, top=0.84, bottom=0.12)
    lines = [("GDP", "GDP", SLOTS[0], "-"), ("AIC", "Actual individual consumption", SLOTS[1], "-"),
             ("HFCE", "Household consumption", SLOTS[2], "-"),
             ("HFCE_Japan_vs_US", "Japan against the US,\nhousehold consumption", MUTED, (0, (4, 2)))]
    for col, label, colour, ls in lines:
        s = p[col]
        ax.plot(s.index, s.values, color=colour, linewidth=2 if colour != MUTED else 1.6, linestyle=ls, label=label)
    # direct labels at the right edge: keep a minimum gap, then shift the stack down if it overflows the top
    short = {"GDP": "GDP", "AIC": "Actual individual cons.", "HFCE": "Household consumption",
             "HFCE_Japan_vs_US": "Japan vs US, household"}
    ends = sorted(((p[c].iloc[-1], short[c], colour) for c, _, colour, _ in lines), key=lambda x: x[0])
    pos = [ends[0][0]]
    for y, _, _ in ends[1:]:
        pos.append(max(y, pos[-1] + 3.4))
    over = max(0.0, pos[-1] - 13.0)
    pos = [q - over for q in pos]
    for (y, lab, colour), yy in zip(ends, pos):
        ax.annotate(lab, xy=(p.index[-1], y), xytext=(p.index[-1] + 1.0, yy), color=INK2, fontsize=8.6,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=1, shrinkA=0, shrinkB=2))
    for x0, x1 in ((2005, 2019),):
        ax.axvspan(x0, x1, color=GRID, alpha=0.35, linewidth=0)
    ax.text(2012, -24.5, "2005–2019", ha="center", color=MUTED, fontsize=8.5)
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xlim(1995, 2024)
    ax.set_ylim(-26, 14)
    ax.set_ylabel("log points (×100), 2020 = 0")
    ax.legend(loc="upper left", fontsize=8.5)
    tidy(ax)
    finish(fig, "fig1_drift_by_aggregate",
           "The drift between price bases, EU group against the United States, 1995–2024",
           "Change in relative prices recorded by national deflators minus the change recorded by the PPPs. "
           "24 EU members (EU27 without Cyprus,\nMalta and Estonia), sums at current and at 2020 PPPs. "
           "Japan against the US shown for comparison.")


def fig2() -> None:
    c = pd.read_csv(RES / "country_drift.csv")
    h = c[c.measure == "HFCE"].dropna(subset=["drift_2005_2019"]).sort_values("drift_2005_2019")
    fig, ax = plt.subplots(figsize=(9, 10.5))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.91, bottom=0.07)
    ypos = np.arange(len(h))
    colour = {r: col for r, col in REGIONS}
    for i, (_, r) in enumerate(h.iterrows()):
        ax.plot([0, r.drift_2005_2019], [i, i], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(r.drift_2005_2019, i, s=46, color=colour[r.region], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(ypos, [NAMES.get(e, e) for e in h.economy], fontsize=8.6)
    ax.axvline(0, color=BASE, linewidth=1)
    for reg, col in REGIONS:
        ax.scatter([], [], s=46, color=col, label=reg)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("drift 2005–2019 against the United States, household consumption (log points ×100)")
    ax.set_ylim(-0.8, len(h) - 0.2)
    tidy(ax, "x")
    finish(fig, "fig2_drift_by_economy",
           "Household-consumption drift against the US, 2005–2019, by economy",
           "Positive: the deflators recorded relative inflation that the PPPs did not. "
           "OECD Tables 1 and 4; economies with data for both years.")


def fig3() -> None:
    g = pd.read_csv(RES / "ict_price_gaps.csv")
    labels = {"CP08": "Communication", "CP091": "Audio-visual and IT equipment",
              "C26": "Computers, electronic and optical products", "J": "Information and communication",
              "J58": "  of which publishing, incl. software", "J59_60": "  of which film, video, broadcasting",
              "J61": "  of which telecommunications", "J62_63": "  of which IT and information services",
              "N1132G": "ICT equipment", "N1173G": "Software and databases"}
    blocks = [("Household consumption, price relative to all consumption", ["CP08", "CP091"]),
              ("Value added, price relative to all value added", ["C26", "J", "J58", "J59_60", "J61", "J62_63"]),
              ("Investment, price relative to the GDP deflator", ["N1132G", "N1173G"])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), gridspec_kw={"height_ratios": [2, 6, 2]}, sharex=True)
    fig.subplots_adjust(left=0.33, right=0.86, top=0.84, bottom=0.09, hspace=0.42)
    for ax, (title, codes) in zip(axes, blocks):
        sub = g.set_index("code").loc[codes]
        y = np.arange(len(codes))[::-1]
        for yi, (code, r) in zip(y, sub.iterrows()):
            nudge = 0.13 if abs(r.us - r.eu) < 0.03 else 0.0   # keep coincident markers both visible
            ax.plot([100 * r.us, 100 * r.eu], [yi, yi], color=GRID, linewidth=3, zorder=1, solid_capstyle="butt")
            ax.scatter(100 * r.us, yi + nudge, s=55, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter(100 * r.eu, yi - nudge, s=55, color=SLOTS[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(1.005, yi, f"{100 * r.share_us:.1f}% / {100 * r.share_eu:.1f}%", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8.3, color=INK2)
        ax.set_yticks(y, [labels[c] for c in codes], fontsize=8.8)
        ax.set_title(title, fontsize=9.6)
        ax.axvline(0, color=BASE, linewidth=1)
        ax.set_ylim(-0.6, len(codes) - 0.4)
        tidy(ax, "x")
    axes[0].text(1.005, 1.25, "share US / Europe", transform=axes[0].get_yaxis_transform(), fontsize=8.3,
                 color=MUTED, va="bottom")
    h1 = axes[0].scatter([], [], s=55, color=SLOTS[0])
    h2 = axes[0].scatter([], [], s=55, color=SLOTS[1])
    fig.legend([h1, h2], ["United States", "Europe (EU members reporting, weighted)"], loc="upper left",
               bbox_to_anchor=(0.33, 0.925), ncol=2, fontsize=8.6)
    axes[-1].set_xlabel("cumulative change 2005–2019 (log points ×100)")
    finish(fig, "fig3_ict_relative_prices",
           "How fast ICT prices fell, United States and Europe, 2005–2019",
           "Each item's price relative to its aggregate. Shares are of household consumption, of value added "
           "and of GDP, averaged over 2006–2019.")


def fig4() -> None:
    h = pd.read_csv(RES / "hfce_counterfactuals.csv", index_col=0)
    gdp = pd.read_csv(RES / "gdp_counterfactuals.csv", index_col=0)
    rows = [("GDP", gdp, [("actual", "As published"), ("excluded", "ICT industries left out"),
                          ("us_like_eu", "US ICT priced like Europe"),
                          ("eu_like_us", "Europe's ICT priced like US")]),
            ("Household consumption", h, [("actual", "As published"), ("excluded", "ICT items left out"),
                                          ("us_like_eu", "US ICT priced like Europe"),
                                          ("eu_like_us", "Europe's ICT priced like US")])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.80, bottom=0.14, wspace=0.62)
    for ax, (title, frame, variants) in zip(axes, rows):
        vals = [frame.loc[2019, k] - frame.loc[2005, k] for k, _ in variants]
        y = np.arange(len(variants))[::-1]
        cols = [SLOTS[0]] + [SLOTS[3]] * (len(variants) - 1)
        ax.barh(y, vals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(y, [lab for _, lab in variants], fontsize=8.8)
        ax.set_title(title)
        ax.set_xlim(0, 20)
        tidy(ax, "x")
    axes[0].set_xlabel("drift 2005–2019 (log points ×100)")
    axes[1].set_xlabel("drift 2005–2019 (log points ×100)")
    finish(fig, "fig4_ict_counterfactuals",
           "The drift with ICT prices removed or harmonised, EU group against the US, 2005–2019",
           "Blue: as published. Amber: counterfactuals that change only the deflators of ICT industries "
           "(left) or ICT consumer items (right); PPPs unchanged.")


def fig5() -> None:
    x = pd.read_csv(RES / "eu_cross_section.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.18, wspace=0.22)
    panels = [(a1, "ict_consumer_effect", "hfce_drift", "effect of own ICT consumer prices relative to the US path\n"
               "(log points ×100)", "Household-consumption drift"),
              (a2, "ict_share", "gdp_drift", "ICT industries' share of value added (%)", "GDP drift")]
    for ax, xc, yc, xl, title in panels:
        s = x[[xc, yc]].dropna()
        ax.scatter(s[xc], s[yc], s=40, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if xc == "ict_share":
            offsets = {"FIN": (5, 4), "SWE": (5, 4), "IRL": (-4, 7), "HUN": (5, 4), "DEU": (5, -10), "FRA": (5, 4),
                       "ITA": (-5, 4), "ESP": (5, -10), "ROU": (5, 4), "GRC": (5, 4)}
        else:
            offsets = {"FIN": (5, 4), "SWE": (5, 4), "HUN": (5, -10), "DEU": (5, 5), "FRA": (5, 4),
                       "ITA": (-5, 4), "ESP": (5, -10), "GRC": (-5, 5)}
        for e, r in s.iterrows():
            if e in offsets:
                ax.annotate(NAMES[e], (r[xc], r[yc]), xytext=offsets[e], textcoords="offset points", fontsize=8,
                            color=INK2, ha="right" if offsets[e][0] < 0 else "left")
        b = np.polyfit(s[xc], s[yc], 1)
        xs = np.linspace(s[xc].min(), s[xc].max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=MUTED, linewidth=1.2, linestyle=(0, (4, 2)))
        r = s.corr().iloc[0, 1]
        ax.text(0.98, 0.04, f"r = {r:.2f}, n = {len(s)}", transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
        ax.set_xlabel(xl)
        ax.set_ylabel("drift 2005–2019 against the US (log points ×100)")
        ax.set_title(title)
        tidy(ax, "both")
    finish(fig, "fig5_eu_cross_section",
           "Across EU members, the drift does not line up with ICT",
           "Left: how much each member's own ICT consumer prices, compared with the US path, move its consumption "
           "deflator.\nRight: the size of the ICT-producing industries. One dot per EU member with data, 2005–2019.")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
