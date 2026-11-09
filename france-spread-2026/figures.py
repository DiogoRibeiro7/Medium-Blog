"""Figures for the article, from ``data/`` and ``results/``.

    python figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, NAMES, RESULTS, annual, monthly_yields

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
BLUE_LIGHT, BLUE_DARK, ORANGE_LIGHT = "#86b6ef", "#184f95", "#f5b090"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
COL = {"FR": BLUE, "IT": ORANGE, "ES": AQUA}
CRISIS = ("2010-01", "2011-12")

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


def ts(index) -> pd.DatetimeIndex:
    return pd.PeriodIndex(index, freq="M").to_timestamp()


def spreads() -> pd.DataFrame:
    m = monthly_yields()
    return m.drop(columns="DE").sub(m["DE"], axis=0)


def fig1_spreads() -> None:
    s = spreads()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
    w = s.loc["2008-01":]
    a.axvspan(pd.Timestamp(CRISIS[0] + "-01"), pd.Timestamp(CRISIS[1] + "-28"), color=SHADE, lw=0, zorder=0)
    for c in ("IT", "ES", "FR"):
        a.plot(ts(w.index), w[c], color=COL[c], lw=1.7 if c != "FR" else 2.2, label=NAMES[c], zorder=3 if c == "FR" else 2)
    a.set_title("Spread to Germany, 2008 to 2026")
    a.set_ylabel("percentage points")
    a.set_ylim(bottom=0)
    a.legend(loc="upper right")
    a.text(pd.Timestamp("2010-02-01"), 3.9, "24 months to\nDec 2011", fontsize=8.5, color=INK2, va="top")
    r = s.loc["2024-01":]
    others = [c for c in s.columns if c not in ("FR", "EL")]
    for c in others:
        b.plot(ts(r.index), r[c], color=GREY, lw=1.1, zorder=1)
    for c in ("IT", "ES"):
        b.plot(ts(r.index), r[c], color=COL[c], lw=1.4, zorder=2)
    b.plot(ts(r.index), r["FR"], color=BLUE, lw=2.4, zorder=3)
    last = r.index[-1]
    lab = {"FR": "France", "IT": "Italy", "ES": "Spain", "BE": "Belgium", "PT": "Portugal", "AT": "Austria", "NL": "Netherlands", "IE": "Ireland", "FI": "Finland"}
    ypos = r.loc[last, others + ["FR"]].sort_values()
    placed = []
    for c, v in ypos.items():
        y = v
        while any(abs(y - p) < 0.065 for p in placed):
            y += 0.065
        placed.append(y)
        b.text(ts([last])[0] + pd.Timedelta(days=12), y, lab[c], fontsize=8.3, va="center", color=INK if c in ("FR", "IT", "ES") else MUTED)
    b.set_xlim(right=ts([last])[0] + pd.Timedelta(days=110))
    b.set_title(f"2024 to {last.strftime('%b %Y')}: France now has the widest spread")
    import matplotlib.dates as mdates
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    b.set_ylim(bottom=0)
    suptitle(fig, "In 2011 France's spread rose with everyone's; in 2026 it rose with Italy's and Belgium's only")
    note(fig, "Monthly averages of 10-year government bond yields minus Germany's (EMU convergence series); Greece omitted from the right panel. Source: Eurostat irt_lt_mcby_m, ECB IRS")
    fig.tight_layout()
    save(fig, "fig1_spreads")


def fig2_shared() -> None:
    R = pd.read_csv(RESULTS / "contagion_rolling.csv")
    ep = pd.read_csv(RESULTS / "contagion_episodes.csv")
    t = ts(R["end"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    a.plot(t, R["r2_fr"], color=BLUE, lw=2.0, label="France")
    a.plot(t, R["r2_it"], color=ORANGE, lw=1.3, alpha=0.9, label="Italy (its own leave-one-out factor)")
    for w, col in (("2011", INK), ("latest", INK)):
        r = ep[(ep.window == w) & (ep.geo == "FR")].iloc[0]
        x = ts([r.end])[0]
        a.errorbar([x], [r.r2], yerr=[[r.r2 - r.r2_lo], [r.r2_hi - r.r2]], fmt="o", ms=7, color=BLUE_DARK, ecolor=BLUE_DARK, capsize=3, zorder=5)
        a.annotate(f"{r.r2:.2f}", (x, r.r2), xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK)
    a.set_ylim(0, 1)
    a.set_ylabel("R²")
    a.set_title("Share of France's monthly spread changes explained by the other countries' common factor (24-month windows)")
    a.legend(loc="lower left", ncol=2)
    b.plot(t, 100 * R["sd_fr"], color=BLUE, lw=1.8, label="All monthly changes")
    b.plot(t, 100 * R["sd_specific_fr"], color=BLUE_LIGHT, lw=1.8, label="Part not shared with the others")
    b.set_ylabel("basis points")
    b.set_title("Size of France's monthly spread changes (standard deviation, 24-month windows)")
    b.legend(loc="upper left", ncol=2)
    b.set_ylim(bottom=0)
    b.set_xlim(pd.Timestamp("2000-01-01"), t.max() + pd.Timedelta(days=150))
    suptitle(fig, "In 2011 France moved with the others; in 2024-2026 most of its movement was its own")
    note(fig, "Leave-one-out first principal component of standardised monthly spread changes of IT, ES, BE, AT, NL, FI, PT, IE (France excluded from its own factor); dots: windows ending Dec 2011\nand at the latest month, with 90% moving-block bootstrap intervals. Low R² before 2008 reflects spreads close to zero. Source: Eurostat, ECB; author's calculations", y=-0.01)
    fig.tight_layout()
    save(fig, "fig2_shared")


def fig3_split() -> None:
    ep = pd.read_csv(RESULTS / "contagion_episodes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    ep_latest = ep[ep.window == "latest"].iloc[0]
    latest_title = f"{pd.Period(ep_latest.start, 'M').strftime('%b %Y')} to {pd.Period(ep_latest.end, 'M').strftime('%b %Y')}"
    for ax, (w, title) in zip(axes, (("2011", "January 2010 to December 2011"), ("latest", latest_title))):
        f = ep[ep.window == w].set_index("geo").loc[["FR", "IT", "ES"]]
        y = np.arange(len(f))[::-1]
        ax.barh(y + 0.18, f["common"], height=0.34, color=GREY, label="Shared with the others", edgecolor=SURFACE, linewidth=2)
        ax.barh(y - 0.18, f["specific"], height=0.34, color=[COL[c] for c in f.index], label="Country-specific", edgecolor=SURFACE, linewidth=2)
        for yy, (c, r) in zip(y, f.iterrows()):
            ax.text(r.common + (0.06 if r.common >= 0 else -0.06), yy + 0.18, f"{r.common:+.2f}", va="center", ha="left" if r.common >= 0 else "right", fontsize=8.8, color=INK2)
            ax.text(r.specific + (0.06 if r.specific >= 0 else -0.06), yy - 0.18, f"{r.specific:+.2f}", va="center", ha="left" if r.specific >= 0 else "right", fontsize=8.8, color=INK2)
            ax.text(1.0, yy, f"total {r.cum:+.2f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=INK)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(y, [NAMES[c] for c in f.index])
        lo, hi = min(f[["common", "specific"]].min().min(), 0), f[["common", "specific"]].max().max()
        ax.set_xlim(lo - 0.25 * (hi - lo) - 0.1, hi + 0.45 * (hi - lo) + 0.1)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title(title)
        ax.set_xlabel("change in spread to Germany, percentage points")
    from matplotlib.patches import Patch

    handles = [Patch(color=GREY, label="Shared with the other countries' factor")] + [Patch(color=COL[c], label=f"{NAMES[c]}-specific") for c in ("FR", "IT", "ES")]
    suptitle(fig, "France's 2011 widening was shared; its 2024-2026 widening is its own")
    note(fig, "Cumulative change in monthly spreads over each window, split into the part loaded on the other countries' common factor and the rest. Source: Eurostat, ECB; author's calculations", y=-0.09)
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9)
    save(fig, "fig3_split")


def fig4_gap() -> None:
    a = annual()
    m = monthly_yields()
    latest = m.dropna(subset=["FR"]).index.max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    spans = {"FR": (1999, 2025), "IT": (1999, 2019), "ES": (1999, 2019)}
    for ax, c in zip(axes, ("FR", "IT", "ES")):
        s = a[a.geo == c].set_index("year").loc[spans[c][0] : spans[c][1]]
        ax.fill_between(s.index, s["implicit"], s["yield"], where=s["yield"] > s["implicit"], color=ORANGE_LIGHT, alpha=0.55, lw=0, interpolate=True, label="Market above implicit: bill still to come")
        ax.plot(s.index, s["yield"], color=MUTED, lw=1.6, label="10-year market yield")
        ax.plot(s.index, s["implicit"], color=COL[c], lw=2.2, label="Implicit rate on debt")
        if c == "FR":
            x = 2025 + (latest.month - 0.5) / 12 + 1
            ax.plot([2025, x], [s.loc[2025, "yield"], m.loc[latest, "FR"]], color=MUTED, lw=1.2, ls=":")
            ax.plot([x], [m.loc[latest, "FR"]], "o", color=MUTED, ms=6)
            ax.annotate(f"{latest.strftime('%b %Y')}: {m.loc[latest, 'FR']:.2f}%", (x, m.loc[latest, "FR"]), xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=8.8, color=INK2)
        else:
            ax.axvspan(2010.5, 2012.5, color=SHADE, lw=0, zorder=0)
        ax.set_title(NAMES[c])
        ax.set_ylim(-0.5, 7.2)
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("percent")
    axes[0].legend(loc="upper right", fontsize=8.5)
    suptitle(fig, "The bill in the pipe: France's market yield now sits far above what it pays on its debt")
    note(fig, "Implicit rate = interest payable (D.41) / average Maastricht debt; market yield = annual average 10-year convergence yield; Italy and Spain shown to 2019; shaded 2011-2012. Source: Eurostat")
    fig.tight_layout()
    save(fig, "fig4_gap")


def fig5_backtest() -> None:
    bt = pd.read_csv(RESULTS / "fiscal_backtest.csv")
    a = annual()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, c in zip(axes, ("IT", "ES", "FR")):
        f = bt[bt.geo == c]
        s = a[a.geo == c].set_index("year")
        y0 = f.year.min() - 1
        pre = s.loc[y0 - 4 : y0]
        ax.plot(list(pre.index) + list(f.year), list(pre["implicit"]) + list(f.actual), color=COL[c], lw=2.2, label="Actual implicit rate")
        ax.plot([y0] + list(f.year), [s.loc[y0, "implicit"]] + list(f.predicted), color=INK, lw=1.5, ls="--", label=f"Model fitted to {f.year.min() - 12 if c != 'FR' else 1999}-{y0}, actual yields")
        if "held" in f and f["held"].notna().any():
            ax.plot([y0] + list(f.year), [s.loc[y0, "implicit"]] + list(f.held), color=RED, lw=1.5, ls=":", label="If 2012 yields had persisted")
        ax.plot(list(pre.index) + list(f.year), list(pre["yield"]) + list(f["yield"]), color=MUTED, lw=1.1, alpha=0.8, label="10-year market yield")
        ax.axvline(y0 + 0.5, color=MUTED, lw=0.8)
        err = f.predicted - f.actual
        ax.set_title(f"{NAMES[c]}, {f.year.min()}-{f.year.max()}: RMSE {np.sqrt((err ** 2).mean()):.2f} pp")
        ax.set_ylim(-0.5, 7.2)
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("percent")
    axes[0].legend(loc="upper right", fontsize=8.3)
    suptitle(fig, "Out of sample, the bill arrived slowly; in 2011-2013 even more slowly than a pre-crisis model predicted")
    note(fig, "Pass-through i(t) = i(t-1) + rho [x(t) - i(t-1)] with rho fitted only on years before the test period (left of the vertical line); 'persisted' holds yields at their 2012 level from 2013 with the full-sample rho. Source: Eurostat; author's calculations")
    fig.tight_layout()
    save(fig, "fig5_backtest")


def fig6_paths() -> None:
    P = pd.read_csv(RESULTS / "stress_paths.csv")
    mc = pd.read_csv(RESULTS / "stress_mc.csv")
    a = annual()
    fr = a[a.geo == "FR"].set_index("year").loc[2015:2025]
    it12 = float(a[(a.geo == "IT") & (a.year == 2012)].int_gdp.iloc[0])
    names = {"hold": "Yields stay at the latest level", "spread_back": "German yield stays, French spread back to 2025", "back_to_2025": "Yields back to 2025 average", "italy_2011": "Italy's 2011-2014 spread rise added"}
    cols = {"hold": BLUE, "spread_back": AQUA, "back_to_2025": GREEN, "italy_2011": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, pk, title in zip(axes, ("plan", "held"), ("Primary balance follows the plan", "No further adjustment after 2026")):
        m = mc[mc.primary_path == pk]
        x = [2025] + list(m.year)
        start = float(fr.loc[2025, "int_gdp"])
        ax.fill_between(x, [start] + list(m.p5), [start] + list(m.p95), color=BLUE_LIGHT, alpha=0.25, lw=0, label="Monte Carlo 90% band")
        ax.fill_between(x, [start] + list(m.p25), [start] + list(m.p75), color=BLUE_LIGHT, alpha=0.45, lw=0, label="Monte Carlo 50% band")
        ax.plot(fr.index, fr["int_gdp"], color=INK, lw=2.0, label="Actual")
        for yk in ("italy_2011", "hold", "spread_back", "back_to_2025"):
            f = P[(P.yields == yk) & (P.primary_path == pk) & (P.growth_path == "base") & (P.rho_case == "france")]
            ax.plot([2025] + list(f.year), [start] + list(f.int_gdp), color=cols[yk], lw=2.0 if yk == "hold" else 1.6, label=names[yk])
        ax.axhline(it12, color=RED, lw=1, ls="--")
        ax.text(2015.2, it12 + 0.08, f"Italy 2012: {it12:.2f}%", fontsize=8.6, color=INK2)
        ax.set_title(title)
        ax.set_xlim(2015, 2036.5)
        ax.set_xticks(range(2015, 2037, 3))
    axes[0].set_ylabel("interest payable, % of GDP")
    axes[0].set_ylim(0, 10)
    axes[0].legend(loc="upper left", fontsize=8.2)
    suptitle(fig, "France's interest bill rises in every scenario, including yields falling back to their 2025 level")
    note(fig, "France rho 0.165, nominal growth 2.5% from 2027; 'plan' = deficit 5.0% of GDP in 2027, then primary balance +0.5 pp a year up to +1%; bands: yields as a random walk (sd 0.63 pp a year), growth sd 1.8 pp.\nSource: Eurostat (to 2025); author's projections", y=-0.01)
    fig.tight_layout()
    save(fig, "fig6_paths")


def fig7_decomp() -> None:
    txt = (RESULTS / "stress_summary.txt").read_text()
    rows = {}
    year = None
    import re

    for ln in txt.splitlines():
        mm = re.match(r"Decomposition of the change in interest % of GDP, \d+ to (\d+)", ln)
        if mm:
            year = int(mm.group(1))
            rows[year] = {}
            continue
        if year is None:
            continue
        for key, pat in (("locked", r"already locked in.*?: ([+-][\d.]+)"), ("german", r"rise in German yields.*?: ([+-][\d.]+)"), ("spread", r"wider French spread.*?: ([+-][\d.]+)")):
            k = re.search(pat, ln)
            if k:
                rows[year][key] = float(k.group(1))
        k = re.search(r"no adjustment instead of plan ([+-][\d.]+); growth 1.5% instead of 2.5% ([+-][\d.]+); growth 3.5% ([+-][\d.]+); an Italian 2011 spread shock in 2027-2030 ([+-][\d.]+)", ln)
        if k:
            rows[year].update(noadj=float(k.group(1)), lowg=float(k.group(2)), highg=float(k.group(3)), italy=float(k.group(4)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    rowspec = [(None, "Built into the path: yields held, plan"), ("locked", "Already locked in by 2025 yields"), ("german", "Rise in German yields since 2025"), ("spread", "Wider French spread since 2025"),
               (None, "Levers, one at a time against that path"), ("noadj", "No adjustment instead of plan"), ("lowg", "Growth 1.5% instead of 2.5%"), ("highg", "Growth 3.5% instead of 2.5%"), ("italy", "An Italian 2011 spread shock, 2027-2030")]
    colors = {"locked": BLUE_DARK, "german": BLUE, "spread": BLUE_LIGHT, "noadj": ORANGE, "lowg": YELLOW, "highg": AQUA, "italy": RED}
    pos = np.arange(len(rowspec))[::-1]
    for ax, y in zip(axes, (2030, 2036)):
        r = rows[y]
        for p, (k, _) in zip(pos, rowspec):
            if k is None:
                continue
            v = r[k]
            ax.barh(p, v, color=colors[k], height=0.62, edgecolor=SURFACE, linewidth=2)
            ax.text(v + (0.04 if v >= 0 else -0.04), p, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(pos, [lab for _, lab in rowspec])
        for tick, (k, _) in zip(ax.get_yticklabels(), rowspec):
            if k is None:
                tick.set_color(MUTED)
                tick.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title(f"Change in interest, 2025 to {y}, points of GDP")
        ax.set_xlim(-0.9, 2.2)
    axes[1].set_yticklabels([])
    suptitle(fig, "Most of the coming bill comes from yields France already pays in the market, not from the 2026 spread")
    note(fig, "Top three bars add up to the rise in interest as % of GDP with yields held at the latest level and the primary balance on plan; the lower bars change one assumption at a time. Source: author's projections")
    fig.tight_layout()
    save(fig, "fig7_decomp")


def main() -> None:
    fig1_spreads()
    fig2_shared()
    fig3_split()
    fig4_gap()
    fig5_backtest()
    fig6_paths()
    fig7_decomp()


if __name__ == "__main__":
    main()
