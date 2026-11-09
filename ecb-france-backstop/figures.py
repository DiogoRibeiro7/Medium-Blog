"""Figures for the article, from ``data/`` and ``results/``.

    python figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from common import FIGURES, NAMES, RESULTS, monthly_spreads

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#184f95"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
COL = {"FR": BLUE, "IT": ORANGE, "ES": AQUA, "PT": YELLOW, "IE": MAGENTA, "BE": VIOLET, "AT": GREY, "NL": GREY, "FI": GREY}
OCT_PEAK = 1.78  # 22 October 2026, daily close reported by FXStreet (documented.csv)
TOOLS = {"SMP": "2010-05", "OMT": "2012-08", "PSPP": "2015-01", "PEPP": "2020-03", "TPI": "2022-07"}

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


def years(ax, step: int = 2) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))


def headline() -> pd.DataFrame:
    p = pd.read_csv(RESULTS / "warranted_panel.csv.gz")
    p["q"] = pd.PeriodIndex(p["quarter"], freq="Q")
    p["x"] = p["q"].dt.year + (p["q"].dt.quarter - 1) / 4 + 0.125
    return p


def fig1_spreads() -> None:
    s = monthly_spreads().loc["2007-01":]
    x = s.index.year + (s.index.month - 0.5) / 12
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for c in ("IT", "ES", "BE", "FR"):
        ax.plot(x, s[c], color=COL[c], lw=2.6 if c == "FR" else 1.6, label=NAMES[c], zorder=5 if c == "FR" else 3)
    for name, m in TOOLS.items():
        t = pd.Period(m, "M")
        xv = t.year + (t.month - 0.5) / 12
        ax.axvline(xv, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.text(xv + 0.08, 6.05, name, fontsize=8.5, color=INK2, va="top")
    ax.scatter([2026 + 9.7 / 12], [OCT_PEAK], s=46, facecolor=SURFACE, edgecolor=BLUE, lw=1.8, zorder=6)
    fr_last = s["FR"].dropna()
    ax.annotate(f"France: Sep 2026 monthly average {fr_last.iloc[-1]:.2f};\n22 Oct 2026 daily close 1.78 (circle)", (2026 + 9.7 / 12, OCT_PEAK), xytext=(2022.9, 2.75), fontsize=8.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylim(0, 6.2)
    ax.set_xlim(2007, 2027.2)
    years(ax, 2)
    ax.set_ylabel("Spread to German 10-year yield, percentage points")
    ax.legend(loc="upper right", ncol=4, bbox_to_anchor=(1.0, 0.93))
    ax.set_title("Monthly averages, January 2007 to September 2026; dotted lines mark ECB announcements")
    suptitle(fig, "France's spread is at its highest since July 2012, but far from the levels the ECB acted against")
    note(fig, "Source: Eurostat irt_lt_mcby_m and ECB IRS (EMU convergence-criterion yields); October 2026 point from FXStreet (27 Oct 2026)")
    save(fig, "fig1_spreads")


def fig2_france_warranted() -> None:
    p = headline()
    p = p[(p["window"] != "pre_crisis") & (p["geo"] == "FR") & (p["q"] >= pd.Period("2005Q1", "Q"))]
    g = p.groupby("x")
    w = g["warranted"].median()
    lo, hi = g["warranted"].quantile(0.1), g["warranted"].quantile(0.9)
    trig = w + g["threshold"].median()
    act = g["spread"].first()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(w.index, lo, hi, color=BLUE_LIGHT, alpha=0.35, lw=0, label="Warranted, 10-90% of 96 models")
    ax.plot(w.index, w, color=INK2, lw=1.6, ls="--", label="Warranted, median model")
    ax.plot(trig.index, trig, color=MUTED, lw=1.2, ls=":", label="Level at which half the models flag")
    ax.plot(act.index, act, color=BLUE, lw=2.6, label="Actual spread (quarterly average)", zorder=5)
    ax.scatter([2026.8], [OCT_PEAK], s=46, facecolor=SURFACE, edgecolor=BLUE, lw=1.8, zorder=6)
    ax.annotate("22 Oct 2026, daily: 1.78", (2026.8, OCT_PEAK), xytext=(2020.4, 2.55), fontsize=8.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(2005, 2027.3)
    years(ax, 2)
    ax.set_ylim(-0.4, 3.0)
    ax.set_ylabel("Percentage points over Germany")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("France, 2005Q1-2026Q3: actual spread against the spread its fundamentals justify")
    suptitle(fig, "By most models France now trades at, or below, the spread its debt and deficit warrant")
    note(fig, "Models: 6 feature sets x German yield yes/no x country effects yes/no x 4 windows (2000-07 window shown separately); fundamentals lagged one quarter. Source: Eurostat; author's estimates")
    save(fig, "fig2_france_warranted")


def fig3_omt() -> None:
    p = headline()
    p = p[(p["window"] != "pre_crisis") & (p["q"] >= pd.Period("2008Q1", "Q")) & (p["q"] <= pd.Period("2016Q4", "Q"))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, title in ((axes[0], "residual", "Residual: actual minus warranted (median model)"), (axes[1], "warranted", "Warranted spread (median model)")):
        for c in ("PT", "IE", "ES", "IT"):
            m = p[p["geo"] == c].groupby("x")[col].median()
            ax.plot(m.index, m, color=COL[c], lw=2.0, label=NAMES[c])
        ax.axvline(2012 + 7.5 / 12, color=INK2, lw=1.0, ls=":")
        ax.text(2012 + 8 / 12, 10.3, "OMT\nannounced", fontsize=8.5, color=INK2, va="top")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(2008, 2017)
        years(ax, 2)
    axes[0].set_ylabel("Percentage points")
    axes[0].set_ylim(-2, 10.5)
    axes[1].legend(loc="upper right")
    suptitle(fig, "After the OMT announcement the unexplained part of periphery spreads collapsed; the warranted part did not")
    note(fig, "Median across the 96 headline models, quarterly. Source: Eurostat; author's estimates")
    save(fig, "fig3_omt")


def fig4_flags() -> None:
    p = headline()
    p = p[(p["window"] != "pre_crisis") & (p["q"] >= pd.Period("2008Q1", "Q"))]
    order = ["PT", "IE", "IT", "ES", "BE", "FR", "AT", "FI", "NL"]
    share = p.groupby(["geo", "x"])["flag"].mean().unstack("x").reindex(order)
    cmap = LinearSegmentedColormap.from_list("blues", ["#f3f6fb", BLUE_LIGHT, BLUE, BLUE_DARK])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = share.columns.to_numpy()
    im = ax.pcolormesh(np.r_[xs - 0.125, xs[-1] + 0.125], np.arange(len(order) + 1), share.to_numpy(), cmap=cmap, vmin=0, vmax=1, shading="flat", edgecolors=SURFACE, linewidth=0.4)
    ax.set_yticks(np.arange(len(order)) + 0.5, [NAMES[c] for c in order])
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xlim(2008, xs[-1] + 0.125)
    years(ax, 2)
    for name, m in TOOLS.items():
        t = pd.Period(m, "M")
        if t.year >= 2008:
            xv = t.year + (t.month - 0.5) / 12
            ax.axvline(xv, color=INK, lw=0.8, ls=":")
            ax.text(xv + 0.08, -0.15, name, fontsize=8.2, color=INK2, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Share of models flagging", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Share of the 96 headline models whose residual exceeds their own 95th percentile, by quarter", pad=16)
    suptitle(fig, "The models agreed in 2011-2012; for France in 2024-2026 almost none of them flag the spread", y=1.06)
    note(fig, "Source: Eurostat; author's estimates", y=-0.04)
    save(fig, "fig4_flags")


def fig5_grid() -> None:
    p = headline()
    last = p[p["geo"] == "FR"].dropna(subset=["spread"])["q"].max()
    f = p[(p["geo"] == "FR") & (p["q"] == last)].copy()
    f["fe"] = f["spec"].str.split("|").str[2]
    order = ["pre_crisis", "normal_times", "post_omt", "long", "real_time"]
    labels = {"pre_crisis": "2000-2007", "normal_times": "2000-2019 without\n2010Q2-2012Q4", "post_omt": "2013-2019", "long": "2000Q1-2024Q2", "real_time": "Real time\n(data to t-4)"}
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k, w in enumerate(order):
        g = f[f["window"] == w]
        for fe, col, lab in (("pooled", BLUE, "Pooled (no country effects)"), ("fe", ORANGE, "Country fixed effects")):
            h = g[g["fe"] == fe]
            y = k + (0.17 if fe == "fe" else -0.17) + rng.uniform(-0.07, 0.07, len(h))
            ax.scatter(h["residual"][~h["flag"]], y[~h["flag"].to_numpy()], s=34, facecolor=SURFACE, edgecolor=col, lw=1.4, zorder=4,
                       label=f"{lab}, not flagged" if k == 1 else None)
            ax.scatter(h["residual"][h["flag"]], y[h["flag"].to_numpy()], s=34, color=col, edgecolor=SURFACE, lw=0.8, zorder=5,
                       label=f"{lab}, flagged" if k == 0 else None)
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_yticks(range(len(order)), [labels[w] for w in order])
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel(f"France, {last}: actual minus warranted spread, percentage points")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.6, ncol=2)
    ax.set_title("Each dot is one model: estimation window (rows), country effects (colour), flagged or not (filled or hollow)")
    suptitle(fig, "Whether France's spread counts as 'unwarranted' depends on the window and on country effects")
    note(fig, "Flag: residual above the 95th percentile of the model's own residuals in its window. Source: Eurostat; author's estimates", y=-0.13)
    save(fig, "fig5_grid")


def fig6_bill() -> None:
    b = pd.read_csv(RESULTS / "bill_paths.csv")
    b = b[b["rho_case"] == "france"]
    show = [
        ("panic", "Panic like Italy 2011-14, no cap", RED, "-", 2.0),
        ("panic_capped_at_trigger", "Panic capped where models flag", ORANGE, "-", 1.8),
        ("held", "Yields held at Sep 2026", BLUE, "-", 2.6),
        ("panic_capped_at_warranted", "Panic capped at warranted spread", INK2, "--", 1.4),
        ("rates_back_to_2025", "Euro-area rates back to 2025", AQUA, "-", 1.8),
        ("spread_zero", "No spread over Germany at all", VIOLET, "-", 1.8),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    with open(RESULTS / "bill_summary.txt", encoding="utf-8") as fh:
        int0 = float(fh.readline().split("interest ")[1].split("%")[0])
    ends = []
    for k, lab, col, ls, lw in show:
        f = b[b["scenario"] == k].set_index("year")["int_gdp"]
        xs = np.r_[2025, f.index]
        ys = np.r_[int0, f.to_numpy()]
        ax.plot(xs, ys, color=col, ls=ls, lw=lw, label=lab, zorder=5 if k == "held" else 3)
        ends.append((ys[-1], lab, col))
    ax.set_xlim(2025, 2039.6)
    ax.set_xticks(range(2025, 2037, 2))
    ax.set_ylim(2.0, 6.2)
    ends.sort()
    last_y = -9
    for y, lab, col in ends:
        yy = max(y, last_y + 0.2)
        ax.text(2036.2, yy, f"{y:.2f}  {lab}", fontsize=8.6, color=INK2, va="center")
        ax.plot([2036, 2036.15], [y, yy], color=col, lw=1.0)
        last_y = yy
    ax.set_ylabel("Interest payable, % of GDP")
    ax.set_title("France, 2025-2036: deficit plan (5.0% in 2027, then +0.5 point a year to a 1% primary surplus), growth 2.5%")
    suptitle(fig, "A cap can stop a panic from adding to the bill; it cannot take back what is already locked in")
    note(fig, "Pass-through of the 10-year yield to the implicit rate estimated for France, 2000-2025. Source: Eurostat; author's projections")
    save(fig, "fig6_bill")


def fig7_spillover() -> None:
    d = pd.read_csv(RESULTS / "channels_spillover.csv")
    order = ["BE", "IT", "ES", "PT", "IE", "AT", "NL", "FI"]
    periods = [("2013-2019", GREY, -0.16), ("2020-2026", BLUE_LIGHT, 0.0), ("June 2024 on", BLUE, 0.16)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for per, col, off in periods:
        f = d[d["period"] == per].set_index("country").reindex(order)
        y = np.arange(len(order)) + off
        ax.errorbar(f["beta"], y, xerr=[f["beta"] - f["lo"], f["hi"] - f["beta"]], fmt="o", color=col, ecolor=col, elinewidth=1.4, capsize=0, ms=6, label=per)
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_yticks(range(len(order)), [NAMES[c] for c in order])
    ax.invert_yaxis()
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlim(-2.8, 1.2)
    ax.set_xlabel("Basis-point move in each country's spread per 1 bp France-specific move (90% bootstrap interval)")
    ax.legend(loc="lower left", title="Months used", title_fontsize=9)
    ax.set_title("France's own shock: residual of its spread change on the other countries' common factor")
    suptitle(fig, "A French-specific shock has spread to Belgium; elsewhere the response is small or not distinguishable from zero")
    note(fig, "Monthly changes in 10-year spreads to Germany; moving-block bootstrap, blocks of 3 months, 2,000 draws. Source: Eurostat, ECB; author's estimates")
    save(fig, "fig7_spillover")


def fig8_banks() -> None:
    b = pd.read_csv(RESULTS / "channels_banks.csv", index_col=0)
    b = b[b["exposure"] > 0.5].sort_values("exposure", ascending=False).head(10)
    names = {"OT": "Other banks (EBA aggregate)", "NO": "Norway", "SE": "Sweden", "DK": "Denmark", "PL": "Poland", "HU": "Hungary", "LU": "Luxembourg", "GR": "Greece", "LI": "Liechtenstein", "SI": "Slovenia"}
    lab = [NAMES.get(c, names.get(c, c)) for c in b.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    y = np.arange(len(b))
    cols = [BLUE if c == "FR" else GREY for c in b.index]
    axes[0].barh(y, b["exp_pct_cet1"], color=cols, height=0.68, edgecolor=SURFACE, lw=2)
    for yi, v, e in zip(y, b["exp_pct_cet1"], b["exposure"]):
        axes[0].text(v + 2, yi, f"{v:.0f}%  (EUR {e:.0f} bn)", va="center", fontsize=8.6, color=INK2)
    axes[0].set_xlim(0, 175)
    axes[0].set_title("Exposure to French general government, % of CET1", fontsize=10.5)
    axes[1].barh(y, b["loss_pct_cet1"], color=cols, height=0.68, edgecolor=SURFACE, lw=2)
    for yi, v, e in zip(y, b["loss_pct_cet1"], b["loss_100bp"]):
        amount = f"EUR {e:.1f} bn" if abs(e) >= 0.05 else "under EUR 0.1 bn"
        axes[1].text(max(v, 0) + 0.02, yi, f"{v:.2f}%  ({amount})", va="center", fontsize=8.6, color=INK2)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_title("Loss from a 100 bp rise in French yields, % of CET1", fontsize=10.5)
    axes[0].set_yticks(y, lab)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    suptitle(fig, "French banks carry most of the exposure; for banks elsewhere a 100 bp repricing costs under 1% of capital")
    note(fig, "Banks' home country; ten largest holders; loss on fair-valued holdings net of short positions only. Source: EBA 2025 transparency exercise (June 2025); author's calculations")
    save(fig, "fig8_banks")


def fig9_eurosystem() -> None:
    h = pd.read_csv(RESULTS / "channels_eurosystem.csv", index_col=0)
    h.index = pd.PeriodIndex(h.index, freq="M")
    x = h.index.year + (h.index.month - 0.5) / 12
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    fig.subplots_adjust(wspace=0.62)
    ax = axes[0]
    ax.stackplot(x, h["pspp"], h["pepp"], colors=[BLUE, BLUE_LIGHT], labels=["PSPP (from 2015)", "PEPP (from 2020)"], edgecolor=SURFACE, lw=0.6)
    ax.set_xlim(2015, 2029.2)
    ax.set_xticks(range(2016, 2028, 2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_ylabel("EUR bn, book value")
    ax.legend(loc="upper left")
    pk = h["total"].idxmax()
    ax.annotate(f"Peak {h['total'].max():.0f} bn\n({pk.strftime('%b %Y')})", (pk.year + (pk.month - 0.5) / 12, h["total"].max()), xytext=(2024.0, 900), fontsize=8.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(x[-1] + 0.15, h["total"].iloc[-1], f"{h['total'].iloc[-1]:.0f} bn\n({h.index[-1].strftime('%b %Y')})", fontsize=8.8, color=INK2, va="center")
    ax.set_ylim(0, 1000)
    ax.set_title("French public-sector bonds held by the Eurosystem", fontsize=10.5)
    ax = axes[1]
    fr_key = 16.3575 / (100 - 18.2319 + 0.9783)
    shared = 10 / 90
    rows = [
        ("Holdings today\n(PSPP/PEPP rules)", 1 - shared * (1 - fr_key)),
        ("TPI, if run on\nPSPP rules", 1 - shared * (1 - fr_key)),
        ("TPI, if shared\nby capital key", fr_key),
    ]
    y = np.arange(len(rows))
    inside = np.array([r[1] for r in rows]) * 100
    ax.barh(y, inside, color=BLUE, height=0.6, edgecolor=SURFACE, lw=2, label="Borne in France")
    ax.barh(y, 100 - inside, left=inside, color=ORANGE, height=0.6, edgecolor=SURFACE, lw=2, label="Borne by other central banks")
    for yi, v in zip(y, inside):
        ax.text(2, yi, f"{v:.0f}% in France", ha="left", va="center", fontsize=9, color=SURFACE, fontweight="bold")
        if 100 - v > 25:
            ax.text(98, yi, f"{100 - v:.0f}% elsewhere", ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
        else:
            ax.text(101.5, yi, f"{100 - v:.0f}%\nelsewhere", ha="left", va="center", fontsize=9, color=INK2, clip_on=False)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of a loss on French bonds, %")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2, fontsize=9)
    ax.set_title("Who would bear a loss", fontsize=10.5)
    suptitle(fig, "Today a loss on French bonds stays mostly in France; a TPI with full risk sharing would change that")
    note(fig, "Cumulative net purchases at book value. Loss sharing: ECB-held 10/90 of government bonds shared by capital key (Banque de France 19.8% of the euro area); the TPI has no published rule. Source: ECB; author's calculations", y=-0.16)
    save(fig, "fig9_eurosystem")


def run() -> None:
    fig1_spreads()
    fig2_france_warranted()
    fig3_omt()
    fig4_flags()
    fig5_grid()
    fig6_bill()
    fig7_spillover()
    fig8_banks()
    fig9_eurosystem()


if __name__ == "__main__":
    run()
