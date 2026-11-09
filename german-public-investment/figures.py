"""Figures for the article, from ``data/`` and ``results/``.

    python figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from common import FIGURES, NAMES, PEERS, RESULTS, SUBSECTORS, fixed_assets, gdp, gov, peer_mean, share

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#184f95"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
PEER_COL = {"NL": ORANGE, "AT": AQUA, "FI": YELLOW, "DK": MAGENTA, "SE": VIOLET}
BRAKE = (2011, 2019)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
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
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    }
)


def note(fig, text: str, y: float = -0.02) -> None:
    fig.text(0.01, y, f"{text}   |   {CREDIT}", ha="left", va="top", fontsize=8.2, color=MUTED)


def suptitle(fig, text: str, y: float = 1.02) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=y)


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def shade(ax) -> None:
    ax.axvspan(BRAKE[0] - 0.5, BRAKE[1] + 0.5, color=SHADE, lw=0, zorder=0)


def countries(ax, frame: pd.DataFrame, years: slice, label_last: bool = True) -> None:
    f = frame.loc[years]
    for p in PEERS:
        ax.plot(f.index, f[p], color=PEER_COL[p], lw=1.3, alpha=0.9, label=NAMES[p])
    ax.plot(f.index, peer_mean(f), color=INK2, lw=1.5, ls="--", label="Peer mean")
    ax.plot(f.index, f["DE"], color=BLUE, lw=2.6, label="Germany", zorder=5)


def fig1_net() -> None:
    Y = gdp()
    net = share(gov("P51G") - gov("P51C"), Y)
    broad = net + share(gov("D92PAY"), Y)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, f, title in ((a, net, "Net public investment (own fixed capital)"), (b, broad, "Net public investment plus investment grants paid")):
        shade(ax)
        countries(ax, f, slice(1995, 2025))
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_title(title)
    a.set_ylabel("% of GDP")
    a.text(BRAKE[0] - 0.3, 2.85, "debt brake in force,\nbefore the pandemic", fontsize=8.5, color=INK2, va="top")
    a.legend(loc="upper left", ncol=2, fontsize=8.6, bbox_to_anchor=(0.0, 0.86))
    a.set_ylim(-0.9, 3.0)
    suptitle(fig, "Germany's net public investment was near zero before the debt brake, under it and after it")
    note(fig, "Net = gross fixed capital formation (P.51G) minus consumption of fixed capital (P.51C), general government. Right: plus investment grants paid (D.92), which include grants for rail and hospitals\noutside government and subsidies to private investment, so the two panels bracket public capital spending. Peers: unweighted mean. Source: Eurostat gov_10a_main, nama_10_gdp", y=-0.01)
    fig.tight_layout()
    save(fig, "fig1_net")


def fig2_stock() -> None:
    Y = gdp()
    k = share(fixed_assets("N11N"), Y)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    countries(a, k, slice(1995, 2024))
    a.set_title("Net public capital stock, all fixed assets")
    a.set_ylabel("% of GDP")
    a.legend(loc="lower left", ncol=3, fontsize=8.6)
    a.set_ylim(30, 90)
    for code, col, lab in (("N1121N", BLUE_DARK, "Non-residential buildings (schools, offices)"), ("N1122N", BLUE, "Other structures (roads, bridges, networks)")):
        kk = share(fixed_assets(code), Y).loc[1995:2024]
        b.plot(kk.index, kk["DE"], color=col, lw=2.4, label=f"Germany: {lab.lower()}")
        b.plot(kk.index, peer_mean(kk), color=col, lw=1.4, ls="--", label=f"Peer mean: {lab.split(' (')[0].lower()}")
    b.set_title("Germany against the peer mean, by asset")
    b.legend(loc="lower left", fontsize=8.4)
    b.set_ylim(0, 30)
    suptitle(fig, "Germany's public capital stock fell from 54% to 44% of GDP by 2019, twice the peers' average fall")
    note(fig, "Net fixed assets of general government at current replacement cost (balance sheets for non-financial assets). Ratios rose in 2021-2024 partly because construction prices rose faster than GDP.\nAustria's fall in 1995-2000 reflects public companies leaving the government sector. Source: Eurostat nama_10_nfa_bs, nama_10_gdp", y=-0.01)
    fig.tight_layout()
    save(fig, "fig2_stock")


def fig3_brake() -> None:
    B = pd.read_csv(RESULTS / "space_balances.csv", index_col=0)
    C = pd.read_csv(RESULTS / "space_counterfactual.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    yrs = B.index
    a.axvspan(2013.5, 2019.5, color=SHADE, lw=0, zorder=0)
    w = 0.27
    a.bar(yrs - w, B["S1311"], width=w, color=BLUE, label="Federal", edgecolor=SURFACE, linewidth=1)
    a.bar(yrs, B["S1312"], width=w, color=AQUA, label="Länder", edgecolor=SURFACE, linewidth=1)
    a.bar(yrs + w, B["S1313"], width=w, color=YELLOW, label="Municipalities", edgecolor=SURFACE, linewidth=1)
    a.plot(yrs, B["S13"], color=INK, lw=1.8, marker="o", ms=4, label="General government")
    a.plot([2016, 2025.4], [-0.35, -0.35], color=RED, lw=1.4, ls="--")
    a.text(2015.2, -0.5, "federal limit −0.35%\n(structural, from 2016)", fontsize=8.4, color=INK2, va="top")
    a.axhline(0, color=MUTED, lw=0.9)
    a.text(2013.7, 2.25, "surpluses well inside the brake", fontsize=8.6, color=INK2)
    a.set_ylim(-5, 2.6)
    a.set_title("Net lending, % of GDP (headline)")
    a.legend(loc="lower left", fontsize=8.6, ncol=2)
    for var, col, ls in (("peer mean, 2011-2019", ORANGE, "-"), ("Sweden, 2011-2019", MAGENTA, "--")):
        f = C[C.variant == var].set_index("year").loc[2005:2025]
        b.plot(f.index, f["debt_cf"], color=col, lw=1.8, ls=ls, label=f"Investing at the {var.split(',')[0]} level, 2011-2019")
    act = C[C.variant == "peer mean, 2011-2019"].set_index("year").loc[2005:2025]
    full = B["debt"].loc[2005:2025]
    b.plot(full.index, full, color=BLUE, lw=2.6, label="Actual")
    b.axhline(60, color=GREY, lw=1, ls=":")
    b.text(2005.3, 60.8, "60% reference", fontsize=8.3, color=INK2)
    b.set_title("Debt ratio, actual and counterfactual, % of GDP")
    b.legend(loc="lower left", fontsize=8.6)
    b.set_ylim(40, 90)
    b.set_xticks(range(2005, 2026, 5))
    _ = act
    suptitle(fig, "In 2014-2019 the brake did not bind: Germany ran surpluses, and could have matched its peers' investment")
    note(fig, "Left: national-accounts net lending by subsector; the brake applies to structural balances in budget accounting. Right: extra net investment equal to the gap with the peers, financed by debt\nat Germany's implicit interest rate, no growth dividend assumed. Source: Eurostat gov_10a_main, gov_10dd_edpt1; author's calculations", y=-0.01)
    fig.tight_layout()
    save(fig, "fig3_brake")


def fig4_subsectors() -> None:
    S = pd.read_csv(RESULTS / "investment_subsectors.csv", index_col=0).loc[1995:2025]
    Y = gdp()
    mun = (gov("P51G", "S1313") - gov("P51C", "S1313"))["DE"].loc[1995:2025] / 1000
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    shade(a)
    cols = {"S1311": BLUE, "S1312": AQUA, "S1313": YELLOW, "S1314": GREY}
    for s, lab in SUBSECTORS.items():
        a.plot(S.index, S[s], color=cols[s], lw=2.4 if s == "S1313" else 1.7, label=lab)
    a.axhline(0, color=MUTED, lw=0.9)
    a.set_title("Germany, net investment by level of government, % of GDP")
    a.legend(loc="upper left", fontsize=8.8)
    cum = mun.cumsum()
    b.bar(mun.index, mun, color=[YELLOW if v >= 0 else RED for v in mun], width=0.75, edgecolor=SURFACE, linewidth=1)
    b2 = cum
    b.plot(b2.index, b2, color=INK, lw=1.6, label="Cumulative since 1995")
    b.axhline(0, color=MUTED, lw=0.9)
    b.set_title("Municipal net investment, EUR bn")
    b.legend(loc="lower left", fontsize=8.8)
    _ = Y
    suptitle(fig, "German municipalities ran down their capital for twenty years")
    note(fig, "Gross fixed capital formation minus consumption of fixed capital, by subsector of general government; shaded: 2011-2019. Red bars: years of negative municipal net investment.\nSource: Eurostat gov_10a_main", y=-0.01)
    fig.tight_layout()
    save(fig, "fig4_subsectors")


def fig5_functions() -> None:
    F = pd.read_csv(RESULTS / "investment_functions.csv")
    F = F[F.code != "GF04"].sort_values("gap")
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    y = np.arange(len(F))
    cols = [RED if c in ("GF07",) else (ORANGE if c == "GF0405" else BLUE) for c in F.code]
    ax.barh(y, F["gap"], color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
    for yy, r in zip(y, F.itertuples()):
        ax.text(r.gap + (0.012 if r.gap >= 0 else -0.012), yy, f"{r.gap:+.2f}  ({r.germany:.2f} vs {r.peers:.2f})", va="center", ha="left" if r.gap >= 0 else "right", fontsize=8.8, color=INK2)
    ax.set_yticks(y, [f.capitalize() if not f.startswith("of which") else "Economic affairs: transport" for f in F.function])
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(-0.62, 0.2)
    ax.set_xlabel("Germany minus peer mean, gross investment, % of GDP, average 2011-2019")
    handles = [Patch(color=BLUE, label="Mostly own investment"), Patch(color=ORANGE, label="Partly financed by grants (rail)"), Patch(color=RED, label="Mostly financed by grants (hospitals)")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.8)
    suptitle(fig, "Germany's gross investment gap is widest in health, general services, transport and education")
    note(fig, "Gross fixed capital formation of general government by function (COFOG); in brackets: Germany and the peer mean. Health and rail investment in Germany is largely made outside the government\nsector and financed by grants, so those gaps overstate the shortfall. Source: Eurostat gov_10a_exp, nama_10_gdp; author's calculations", y=-0.01)
    fig.tight_layout()
    save(fig, "fig5_functions")


def fig6_reform() -> None:
    R = pd.read_csv(RESULTS / "reform_quarterly.csv")
    W = R.pivot_table(index="quarter", columns="geo", values="gfcf_4q").loc["2015-Q4":]
    t = pd.PeriodIndex(W.index, freq="Q").to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for p in PEERS:
        ax.plot(t, W[p], color=PEER_COL[p], lw=1.2, alpha=0.9, label=NAMES[p])
    ax.plot(t, W[list(PEERS)].mean(axis=1), color=INK2, lw=1.5, ls="--", label="Peer mean")
    ax.plot(t, W["DE"], color=BLUE, lw=2.6, label="Germany")
    for date, lab in ((pd.Timestamp("2023-11-15"), "Court ruling,\nNov 2023"), (pd.Timestamp("2025-03-18"), "Reform,\nMar 2025")):
        ax.axvline(date, color=MUTED, lw=1, ls=":")
        ax.text(date + pd.Timedelta(days=25), 6.15, lab, fontsize=8.3, color=INK2, va="top")
    ax.set_ylabel("% of GDP, rolling four quarters")
    ax.set_ylim(2, 6.3)
    ax.legend(loc="upper left", ncol=4, fontsize=8.6)
    ax.set_title("Gross public investment, general government")
    suptitle(fig, "German public investment has risen since 2023, with no visible acceleration after the 2025 reform")
    note(fig, "Gross fixed capital formation of general government over GDP, sums of the latest four quarters, not seasonally adjusted, to 2026-Q2. Investment financed by the special fund through grants\n(for example to Deutsche Bahn) is not included. Source: Eurostat gov_10q_ggnfa, namq_10_gdp", y=-0.01)
    fig.tight_layout()
    save(fig, "fig6_reform")


def main() -> None:
    fig1_net()
    fig2_stock()
    fig3_brake()
    fig4_subsectors()
    fig5_functions()
    fig6_reform()


if __name__ == "__main__":
    main()
