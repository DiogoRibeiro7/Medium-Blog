"""Figures for the article, from ``results/``.

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#184f95"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
NAMES = {"EL": "Greece", "IE": "Ireland", "PT": "Portugal", "CY": "Cyprus", "ES": "Spain", "DE": "Germany", "FR": "France", "NL": "Netherlands", "FI": "Finland", "AT": "Austria", "IT": "Italy", "BE": "Belgium"}
PROGRAMME = ("EL", "IE", "PT", "CY", "ES")
CORE = ("DE", "FR", "NL", "FI", "AT")
COLORS = dict(zip(PROGRAMME, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA))) | dict(zip(CORE, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)))
CF_COLOR = {"no_fragmentation": ORANGE, "safe_asset": BLUE}
CF_NAME = {"no_fragmentation": "No-fragmentation counterfactual", "safe_asset": "Safe-asset counterfactual"}
WINDOW = ("2010-01", "2014-12")

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
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RES / name, keep_default_na=False, na_values=[""], **kw)


def months(s: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(s + "-01")


def shade(ax) -> None:
    ax.axvspan(pd.Timestamp(WINDOW[0] + "-01"), pd.Timestamp(WINDOW[1] + "-28"), color=SHADE, lw=0, zorder=0)


# 1. ten-year yields, programme and core countries
def fig1_yields() -> None:
    cf = read("yields_monthly_cf.csv")
    cf = cf[cf["time"] >= "2000-01"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.18})
    for ax, group, title in ((a, PROGRAMME, "Programme countries"), (b, CORE, "Core countries")):
        shade(ax)
        for c in group:
            s = cf[cf["geo"] == c]
            ax.plot(months(s["time"]), s["actual"], color=COLORS[c], lw=1.6, label=NAMES[c])
        ax.set_title(title)
        ax.set_ylabel("10-year yield, percent")
        ax.legend(loc="upper right", fontsize=8.5, ncol=1)
    a.set_ylim(-1, 30)
    b.set_ylim(-1, 6.5)
    suptitle(fig, "Ten-year government bond yields, 2000-2026: divergence in 2010-2014, then compression")
    note(fig, "Monthly EMU convergence-criterion yields. Shaded: the crisis window used for the ledger. Different vertical scales.\nSource: Eurostat irt_lt_mcby_m", y=-0.03)
    save(fig, "fig1_yields")


# 2. Germany against two counterfactuals, and the placebo test
def fig2_germany() -> None:
    cf = read("yields_monthly_cf.csv")
    de = cf[(cf["geo"] == "DE") & (cf["time"] >= "2003-01") & (cf["time"] <= "2019-12")]
    pl = read("yields_placebo.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"wspace": 0.25, "width_ratios": [1.6, 1]})
    shade(a)
    t = months(de["time"])
    a.plot(t, de["no_fragmentation"], color=ORANGE, lw=1.5, label="No-fragmentation: euro-area average + pre-crisis spread")
    a.plot(t, de["safe_asset"], color=BLUE, lw=1.5, label="Safe-asset: synthetic from non-euro yields")
    a.plot(t, de["actual"], color=INK, lw=2.0, label="Actual Bund yield")
    a.set_ylabel("percent")
    a.set_title("Germany: actual 10-year yield and two counterfactuals")
    a.legend(loc="lower left", fontsize=8.3)
    a.set_ylim(-0.9, 4.9)
    pl = pl.sort_values("ratio")
    names = {"DE": "Germany", "US": "United States", "GB": "United Kingdom", "CA": "Canada", "SE": "Sweden", "NO": "Norway", "CH": "Switzerland", "JP": "Japan", "AU": "Australia", "NZ": "New Zealand"}
    b.barh(range(len(pl)), pl["ratio"], color=[BLUE if u == "DE" else GREY for u in pl["unit"]], height=0.6)
    b.set_yticks(range(len(pl)), [names[u] for u in pl["unit"]])
    b.grid(axis="x", color=GRID)
    b.grid(axis="y", visible=False)
    b.set_title("Placebo: window error / pre-fit error")
    for i, v in enumerate(pl["ratio"]):
        b.text(v + 0.08, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK2)
    b.set_xlim(0, pl["ratio"].max() * 1.18)
    suptitle(fig, "How far below its counterfactual Germany borrowed depends on the counterfactual")
    note(fig, "Left: monthly yields, 2003-2019; shaded: crisis window. Right: synthetic control fitted 1999-2009 for Germany and for each donor from the others;\nratio of the root-mean-square gap in 2010-2014 to the pre-fit error. Sources: Eurostat irt_lt_mcby_m; OECD via FRED (IRLTLT01)", y=-0.03)
    save(fig, "fig2_germany")


# 3. average yield gap in the window, by country and counterfactual
def fig3_gaps() -> None:
    a = read("yields_annual.csv")
    a = a[(a["year"] >= 2010) & (a["year"] <= 2014)]
    g = a.groupby("geo")[["gap_no_fragmentation", "gap_safe_asset"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"wspace": 0.35})
    for ax, group, title, lim in ((axes[0], CORE, "Core countries", (-1.7, 0.75)), (axes[1], PROGRAMME, "Programme countries", (0, 12.5))):
        gg = g.reindex(group)
        y = np.arange(len(group))
        ax.hlines(y, gg.min(axis=1), gg.max(axis=1), color=GREY, lw=2, zorder=1)
        ax.scatter(gg["gap_no_fragmentation"], y, color=ORANGE, s=60, zorder=3, label=CF_NAME["no_fragmentation"], edgecolor=SURFACE, lw=1)
        ax.scatter(gg["gap_safe_asset"], y, color=BLUE, s=60, zorder=3, label=CF_NAME["safe_asset"], edgecolor=SURFACE, lw=1)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, [NAMES[c] for c in group])
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(*lim)
        ax.set_title(title)
        ax.set_xlabel("percentage points, 2010-2014 average")
        for i, (c, row) in enumerate(gg.iterrows()):
            ax.text(row.max() + (lim[1] - lim[0]) * 0.03, i, f"{row['gap_no_fragmentation']:+.2f} / {row['gap_safe_asset']:+.2f}", va="center", fontsize=8.3, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 1.0), fontsize=8.8, ncol=2)
    suptitle(fig, "Both counterfactuals agree on the programme countries and disagree on the core", y=1.08)
    note(fig, "Actual minus counterfactual ten-year yield; labels give no-fragmentation / safe-asset. Different horizontal scales.\nSources: Eurostat irt_lt_mcby_m; OECD via FRED", y=-0.06)
    save(fig, "fig3_gaps")


# 4. implicit interest rate against market yields
def fig4_passthrough() -> None:
    r = read("interest_by_year.csv")
    r = r[r["cf"] == "no_fragmentation"]
    a = read("yields_annual.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"wspace": 0.22}, sharey=False)
    for ax, c in zip(axes, ("DE", "EL", "PT")):
        y = a[(a["geo"] == c) & (a["year"].between(2005, 2025))]
        s = r[r["geo"] == c]
        ax.axvspan(2009.5, 2014.5, color=SHADE, lw=0, zorder=0)
        ax.plot(y["year"], y["actual"], color=GREY, lw=1.6, label="Market 10-year yield")
        ax.plot(s["year"], s["actual_rate"], color=INK, lw=2.0, label="Implicit rate on all debt")
        ax.plot(s["year"], s["model_rate"], color=MUTED, lw=1.2, ls=":", label="Model with actual yields")
        ax.plot(s["year"], s["cf_rate"], color=ORANGE, lw=1.6, ls="--", label="Model with counterfactual yields")
        ax.set_title(NAMES[c])
        ax.set_xlim(2005, 2025.5)
        if c == "EL":
            ax.set_ylim(0, 24)
    axes[0].set_ylabel("percent")
    axes[0].legend(loc="upper right", fontsize=8.2)
    suptitle(fig, "The implicit rate follows market yields slowly, and not at all where official lenders took over")
    note(fig, "Implicit rate: interest payable (D.41) over average general government debt. Model: simulated from the 2009 rate with pass-through rho = 0.17 a year,\nwith actual yields and with the no-fragmentation yields in 2010-2014. Sources: Eurostat gov_10a_main, gov_10dd_edpt1, irt_lt_mcby_m", y=-0.04)
    save(fig, "fig4_passthrough")


# 5. cumulative interest effect over time
def fig5_cumulative() -> None:
    r = read("interest_by_year.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.22})
    for c in CORE:
        for cf, ls in (("no_fragmentation", "-"), ("safe_asset", "--")):
            s = r[(r["geo"] == c) & (r["cf"] == cf)].sort_values("year")
            a.plot(s["year"], s["market_effect"].cumsum() / 1e3, color=COLORS[c], ls=ls, lw=1.7, label=NAMES[c] if cf == "no_fragmentation" else None)
    for c in PROGRAMME:
        for cf, ls in (("no_fragmentation", "-"), ("safe_asset", "--")):
            s = r[(r["geo"] == c) & (r["cf"] == cf)].sort_values("year")
            b.plot(s["year"], s["market_effect"].cumsum() / 1e3, color=COLORS[c], ls=ls, lw=1.7, label=NAMES[c] if cf == "no_fragmentation" else None)
    for ax, t in ((a, "Core countries"), (b, "Programme countries, at market yields")):
        ax.axhline(0, color=INK2, lw=1)
        ax.axvspan(2009.5, 2014.5, color=SHADE, lw=0, zorder=0)
        ax.set_title(t)
        ax.set_ylabel("EUR bn, cumulative since 2010")
        ax.set_xlim(2010, 2025.4)
    a.legend(loc="lower left", fontsize=8.5)
    b.legend(loc="upper left", fontsize=8.5)
    a.plot([], [], color=INK2, ls="-", label="no-fragmentation")
    a.plot([], [], color=INK2, ls="--", label="safe-asset")
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles, labels, loc="lower left", fontsize=8.2, ncol=2)
    suptitle(fig, "Crisis-era yields kept acting for a decade through debt issued in 2010-2014")
    note(fig, "Market repricing: the 2010-2014 yield gap passes into the implicit rate at rho = 0.17 a year and is applied to actual debt; negative = interest saved.\nSolid: no-fragmentation counterfactual; dashed: safe-asset counterfactual. Sources: Eurostat; OECD via FRED; author's calculations", y=-0.03)
    save(fig, "fig5_cumulative")


# 6. the core ledger
def fig6_ledger() -> None:
    led = read("ledger_interest.csv")
    ex = read("ledger_exposures.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"wspace": 0.3, "width_ratios": [1.3, 1]})
    core = led[led["country"].isin(CORE)].set_index("country").reindex(CORE)
    y = np.arange(len(CORE))
    a.hlines(y, core["market_effect_min"], core["market_effect_max"], color=GREY, lw=6, alpha=0.6, zorder=1, label="Range across counterfactuals and pass-through")
    a.scatter(core["market_effect_no_fragmentation"], y, color=ORANGE, s=55, zorder=3, edgecolor=SURFACE, lw=1, label=CF_NAME["no_fragmentation"])
    a.scatter(core["market_effect_safe_asset"], y, color=BLUE, s=55, zorder=3, edgecolor=SURFACE, lw=1, label=CF_NAME["safe_asset"])
    a.axvline(0, color=INK2, lw=1)
    a.set_yticks(y, [NAMES[c] for c in CORE])
    a.invert_yaxis()
    a.grid(axis="x", color=GRID)
    a.grid(axis="y", visible=False)
    a.set_xlabel("EUR bn, 2010-2025; negative = interest saved")
    a.set_title("Interest account")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.2, ncol=2)
    ex = ex.set_index("country").reindex(CORE)
    mid = (ex["efsf_esm_low"] + ex["efsf_esm_high"]) / 2
    b.barh(y, mid, color=BLUE_LIGHT, height=0.55, label="EFSF and ESM loans at the contribution key")
    b.barh(y, ex["glf"], left=mid, color=BLUE, height=0.55, label="Bilateral loans to Greece (GLF)")
    for i, (m, g) in enumerate(zip(mid, ex["glf"])):
        b.text(m + g + 1, i, f"{m + g:.0f}", va="center", fontsize=8.5, color=INK2)
    b.set_yticks(y, [NAMES[c] for c in CORE])
    b.invert_yaxis()
    b.grid(axis="x", color=GRID)
    b.grid(axis="y", visible=False)
    b.set_xlabel("EUR bn outstanding, 2025-2026")
    b.set_title("Risk position: loans still outstanding")
    b.legend(loc="lower right", fontsize=8.2)
    b.set_xlim(0, (mid + ex["glf"]).max() * 1.2)
    suptitle(fig, "Germany saved under every counterfactual; its loans remain a position, not yet a cost")
    note(fig, "Left: market repricing, cumulative 2010-2025, two counterfactuals and five pass-through assumptions. Right: approximate shares of official loans outstanding;\nno losses have been realised on them. Sources: Eurostat; OECD via FRED; ESM Annual Report 2025; ESM and EFSF contribution keys", y=-0.13)
    save(fig, "fig6_ledger")


# 7. banks leaving the programme countries' public debt
def fig7_banks() -> None:
    c = read("ledger_bank_claims.csv")
    o = c[(c["sector"] == "O") & (c["reporter"] == "5A")]
    o = o[o["quarter"] >= "2005-Q1"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"wspace": 0.22})
    q2d = lambda q: pd.PeriodIndex(q.str.replace("-", ""), freq="Q").to_timestamp()
    for d, col in (("EL", BLUE), ("PT", AQUA), ("IE", ORANGE), ("CY", YELLOW)):
        s = o[o["debtor"] == d].sort_values("quarter")
        a.plot(q2d(s["quarter"]), s["eur_bn"], color=col, label=NAMES[d])
    gr = c[(c["sector"] == "O") & (c["debtor"] == "EL") & c["reporter"].isin(["DE", "FR"])]
    for rep, col in (("FR", ORANGE), ("DE", BLUE)):
        s = gr[gr["reporter"] == rep].sort_values("quarter")
        s = s[s["quarter"] >= "2005-Q1"]
        b.plot(q2d(s["quarter"]), s["eur_bn"], color=col, label={"FR": "French banks", "DE": "German banks"}[rep])
    for ax in (a, b):
        ax.axvline(pd.Timestamp("2012-03-09"), color=INK2, lw=1, ls="--")
        ax.set_ylabel("EUR bn")
        ax.set_xlim(pd.Timestamp("2005-01-01"), pd.Timestamp("2025-06-30"))
    a.text(pd.Timestamp("2012-06-01"), a.get_ylim()[1] * 0.92, "Greek debt\nexchange, March 2012", fontsize=8.3, color=INK2, va="top")
    a.set_title("Claims of all foreign banks on the public sector")
    b.set_title("Claims of German and French banks on the Greek public sector")
    a.legend(loc="upper right", fontsize=8.5)
    b.legend(loc="upper right", fontsize=8.5)
    suptitle(fig, "Most foreign bank exposure to the Greek state was gone before the 2012 haircut")
    note(fig, "BIS consolidated banking statistics, foreign claims on the official sector, ultimate risk basis, converted from US dollars at the quarterly average rate.\nDeclines include repayments at maturity, sales and impairments. Sources: BIS WS_CBS_PUB; ECB rate via FRED (EXUSEU)", y=-0.03)
    save(fig, "fig7_banks")


# 8. who holds the debt now
def fig8_holders() -> None:
    p = RES / "ledger_debt_instruments.csv"
    if not p.exists():
        return
    q = read("ledger_debt_instruments.csv")
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    q2d = lambda s: pd.PeriodIndex(s.str.replace("-", ""), freq="Q").to_timestamp()
    for c in PROGRAMME:
        s = q[(q["geo"] == c)].sort_values("time")
        s = s[s["time"] >= "2005-Q1"]
        ax.plot(q2d(s["time"]), s["loan_share"], color=COLORS[c], label=NAMES[c])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 1)
    ax.set_ylabel("loans as a share of government debt")
    ax.legend(loc="upper left", fontsize=8.5, ncol=5)
    ax.axvline(pd.Timestamp("2012-03-09"), color=INK2, lw=1, ls="--")
    suptitle(fig, "Where official lenders replaced bondholders, debt turned into loans", y=1.0)
    note(fig, "Loans (F.4) as a share of general government debt; in the programme countries most loans are from the EFSF, ESM, EFSM, the GLF and the IMF.\nDashed: Greek debt exchange, March 2012. Source: Eurostat gov_10q_ggdebt", y=-0.02)
    save(fig, "fig8_holders")


def main() -> int:
    for f in (fig1_yields, fig2_germany, fig3_gaps, fig4_passthrough, fig5_cumulative, fig6_ledger, fig7_banks, fig8_holders):
        f()
        print(f.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
