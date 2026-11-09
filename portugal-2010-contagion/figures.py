"""Figures for the article, from ``data/`` and ``results/``.

    python figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from common import DATA, FIGURES, NAMES, RESULTS, annual_panel, quarterly_panel
from warranted import REALTIME_2009

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
BLUE_LIGHT = "#86b6ef"
INK, INK2, MUTED, GRID, SURFACE, SHADE, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#efeeea", "#b5b4ae"
CREDIT = "Diogo Ribeiro"
COL = {"PT": BLUE, "EL": ORANGE, "IE": AQUA, "ES": VIOLET, "IT": YELLOW, "DE": GREY, "BE": MAGENTA, "FR": INK2}
PART = {"common": GREY, "greek": ORANGE, "irish": AQUA, "own": BLUE}
PART_LABEL = {"common": "rest of the euro area", "greek": "Greece-specific", "irish": "Ireland-specific", "own": "Portugal's own"}

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


def years(ax, step: int = 1) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))


def month_x(months: pd.Series) -> np.ndarray:
    p = pd.PeriodIndex(months.astype(str), freq="M")
    return (p.year + (p.month - 1) / 12 + 1 / 24).to_numpy()


def fig1_scorecard() -> None:
    a = annual_panel().set_index(["geo", "year"])
    q = quarterly_panel()
    order = ["PT", "EL", "IE", "ES", "IT", "DE"]
    growth = annual_panel().query("2000 <= year <= 2009").groupby("geo")["growth"].mean()
    ca = q[q["quarter"].astype(str) == "2009Q4"].set_index("geo")["ca"]
    panels = [
        ("Public debt, 2009", lambda g: a.loc[(g, 2009), "debt"], {k: v[0] for k, v in REALTIME_2009.items()}, "% of GDP"),
        ("Government balance, 2009", lambda g: a.loc[(g, 2009), "balance"], {k: v[1] for k, v in REALTIME_2009.items()}, "% of GDP"),
        ("Net international investment position, end-2009", lambda g: a.loc[(g, 2009), "niip"], {}, "% of GDP"),
        ("Private debt, 2009", lambda g: a.loc[(g, 2009), "private_debt"], {}, "% of GDP, firms and households"),
        ("Real growth, 2000-2009 average", lambda g: growth[g], {}, "% a year"),
        ("Current account, 2009", lambda g: ca[g], {}, "% of GDP"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    y = np.arange(len(order))[::-1]
    for ax, (title, f, rt, unit) in zip(axes.flat, panels):
        vals = [f(g) for g in order]
        ax.hlines(y, 0, vals, color=GRID, lw=1.2, zorder=1)
        for yi, g, v in zip(y, order, vals):
            ax.scatter(v, yi, s=70, color=COL[g], zorder=3, edgecolor="white", lw=0.8)
            if g in rt:
                ax.scatter(rt[g], yi, s=70, facecolor="none", edgecolor=COL[g], lw=1.6, zorder=3)
                ax.annotate("", xy=(v, yi), xytext=(rt[g], yi), arrowprops=dict(arrowstyle="->", color=COL[g], lw=1.0), zorder=2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [NAMES[g] for g in order])
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(unit, fontsize=9)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        lo, hi = min(vals + list(rt.values()) + [0]), max(vals + list(rt.values()) + [0])
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
    h1 = axes[0, 0].scatter([], [], s=60, facecolor="none", edgecolor=INK2)
    h2 = axes[0, 0].scatter([], [], s=60, color=INK2)
    fig.legend([h1, h2], ["published April 2010", "today's figures"], loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=2, fontsize=9, handletextpad=0.3)
    suptitle(fig, "Portugal had the smallest deficit of the three and far less debt than Greece, but Irish-sized external debt and the slowest growth")
    note(fig, "Hollow dots: Eurostat's first notification of 2009 data (22 April 2010). Filled dots: Eurostat today (gov_10dd_edpt1, tipsii10, tipspd10, nama_10_gdp, bop_gdp6_q).", y=-0.005)
    fig.tight_layout()
    save(fig, "fig1_scorecard")


def headline() -> pd.DataFrame:
    p = pd.read_csv(RESULTS / "warranted_panel.csv.gz")
    p = p[~p["spec"].str.endswith("pre_crisis")]
    p["q"] = pd.PeriodIndex(p["quarter"], freq="Q")
    p["x"] = p["q"].dt.year + (p["q"].dt.quarter - 1) / 4 + 0.125
    return p


def fig2_warranted() -> None:
    p = headline()
    p = p[(p["x"] >= 2007) & (p["x"] < 2015)]
    requests = {"EL": 2010.29, "IE": 2010.88, "PT": 2011.26}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=False)
    for ax, g in zip(axes, ("PT", "EL", "IE")):
        f = p[p["geo"] == g]
        s = f.groupby("x").agg(actual=("spread", "first"), med=("warranted", "median"),
                               lo=("warranted", lambda v: v.quantile(.1)), hi=("warranted", lambda v: v.quantile(.9)), flag=("flag", "mean"))
        ax.fill_between(s.index, s.lo, s.hi, color=SHADE, lw=0, label="warranted, 10-90% of models")
        ax.plot(s.index, s.med, color=INK2, lw=1.6, ls="--", label="warranted, median model")
        ax.plot(s.index, s.actual, color=COL[g], lw=2.4, label="actual")
        ax.axvline(requests[g], color=COL[g], lw=0.9, ls=":")
        ax.axvline(2012.59, color=MUTED, lw=0.9, ls=":")
        top = s[["actual", "hi"]].max().max()
        ax.text(requests[g] - 0.08, top * 0.80, "request", ha="right", va="top", fontsize=8.5, color=COL[g])
        ax.text(2012.67, top * 0.80, "OMT", ha="left", va="top", fontsize=8.5, color=MUTED)
        ax.set_title(NAMES[g])
        ax.set_ylim(0, top * 1.05)
        years(ax, 2)
        ax.set_ylabel("10-year spread to Germany, pp" if g == "PT" else "")
        flagged = s[(s.index >= 2010) & (s.index < 2013)]["flag"]
        ax.text(0.02, 0.62, f"2010-2012: flagged by half\nthe models in {int((flagged >= 0.5).sum())} of 12 quarters", transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=3, fontsize=9)
    suptitle(fig, "From 2010 all three traded far above what their fundamentals warranted; Portugal's warranted spread was the lowest", y=1.06)
    note(fig, "Warranted spread: 128 regressions of the spread on lagged fundamentals (9 sovereigns, 2000-2024); Greece predicted out of sample. Each panel has its own scale.")
    fig.tight_layout()
    save(fig, "fig2_warranted")


def fig3_contagion() -> None:
    r = pd.read_csv(RESULTS / "contagion_responses.csv")
    m = pd.read_csv(RESULTS / "contagion_monthly_pt.csv", dtype={"month": str})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.35]))
    wins = [("before_request", "Nov 2009 - Mar 2011"), ("crisis_no_exchange", "Nov 2009 - Jul 2012*"), ("after_programme", "Jun 2014 - 2019")]
    shades = [ORANGE, RED, GREY]
    ctry = ["PT", "ES", "IT", "BE", "FR"]
    for k, ((w, lab), c) in enumerate(zip(wins, shades)):
        f = r[r["window"] == w].set_index("country")
        yy = np.arange(len(ctry))[::-1] + (1 - k) * 0.24
        a1.hlines(yy, f.loc[ctry, "b_greek_lo"], f.loc[ctry, "b_greek_hi"], color=c, lw=2.2, alpha=0.55)
        a1.scatter(f.loc[ctry, "b_greek"], yy, color=c, s=40, zorder=3, label=lab)
    a1.axvline(0, color=MUTED, lw=0.8)
    a1.set_yticks(np.arange(len(ctry))[::-1], [NAMES[g] for g in ctry])
    a1.grid(axis="x", color=GRID)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("pp of spread per 1 pp Greece-specific widening")
    a1.set_title("Response to a Greece-specific move (5-95% interval)")
    a1.legend(loc="lower right", fontsize=8.5)
    x = month_x(m["month"])
    for part in ("common", "greek", "irish", "own"):
        a2.plot(x, m[part].cumsum(), color=PART[part], lw=2.0, label=PART_LABEL[part])
    a2.plot(x, m["pt"].cumsum(), color=INK, lw=2.6, label="Portugal, total")
    a2.axvspan(2012 + 1 / 12, 2012 + 4 / 12, color=SHADE, lw=0)
    a2.set_ylim(-1.0, 12.8)
    a2.text(2012 + 2.5 / 12, 12.7, "Greek\nexchange", ha="center", va="top", fontsize=8, color=MUTED)
    for xv, lab in ((2011 + 3.2 / 12, "request"), (2012 + 7.05 / 12, "OMT")):
        a2.axvline(xv, color=MUTED, lw=0.9, ls=":")
        a2.text(xv - 0.04, -0.6, lab, ha="right", fontsize=8.5, color=MUTED)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_title("Portugal's spread since October 2009, cumulative, by source (pp)")
    a2.legend(loc="upper left", fontsize=8.5)
    years(a2, 1)
    suptitle(fig, "Before its request Portugal moved with Greece and Ireland; from April 2011 its own component took over")
    note(fig, "Monthly changes in 10-year spreads to Germany. *Without February-April 2012 (the Greek series switches to new bonds). Right: slopes from that window applied to every month.")
    fig.tight_layout()
    save(fig, "fig3_contagion")


def fig4_events() -> None:
    m = pd.read_csv(RESULTS / "contagion_monthly_pt.csv", dtype={"month": str})
    ev = pd.read_csv(RESULTS / "events.csv", dtype={"month": str})
    show = {"2010-04": "Greece asks", "2010-05": "EFSF, SMP", "2010-10": "Deauville", "2010-11": "Ireland asks", "2011-03": "PEC IV",
            "2011-04": "Portugal asks, hike", "2011-07": "junk, hike, PSI", "2011-12": "LTRO", "2012-01": "S&P junk", "2012-03": "Greek exchange", "2012-09": "OMT"}
    x = month_x(m["month"])
    fig, ax = plt.subplots(figsize=(13, 5.0))
    w = 1 / 12 * 0.8
    pos = np.zeros(len(m))
    neg = np.zeros(len(m))
    for part in ("common", "greek", "irish", "own"):
        v = m[part].to_numpy()
        b = np.where(v >= 0, pos, neg)
        ax.bar(x, v, w, bottom=b, color=PART[part], label=PART_LABEL[part], lw=0)
        pos = pos + np.clip(v, 0, None)
        neg = neg + np.clip(v, None, 0)
    ax.scatter(x, m["pt"], s=14, color=INK, zorder=4, label="Portugal, total change")
    for mo, lab in show.items():
        i = int(np.where(m["month"] == mo)[0][0])
        top = max(pos[i], m["pt"].iloc[i]) + 0.12
        bot = min(neg[i], m["pt"].iloc[i]) - 0.12
        up = mo not in ("2010-10", "2012-09")
        ax.text(x[i], top if up else bot, lab, rotation=90, ha="center", va="bottom" if up else "top", fontsize=8.2, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylim(-2.4, 3.4)
    ax.set_ylabel("monthly change in Portugal's spread, pp")
    years(ax, 1)
    ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    big = ev.set_index("month")
    suptitle(fig, "Portugal's biggest jumps came after its request and were mostly its own; the Deauville month barely registers")
    note(fig, f"Monthly averages, so events late in a month spill into the next; labels mark each event's month. Largest rises: July 2011 {big.loc['2011-07', 'pt']:+.2f} pp, April 2011 {big.loc['2011-04', 'pt']:+.2f} pp.")
    fig.tight_layout()
    save(fig, "fig4_events")


def fig5_programme() -> None:
    f = pd.read_csv(RESULTS / "programme_forecasts.csv")
    r = pd.read_csv(RESULTS / "programme_rates.csv", dtype={"month": str})
    mro = pd.read_csv(DATA / "mro_rate.csv", dtype={"TIME_PERIOD": str})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.2]))
    xt, labs = [], []
    k = 0
    for g in ("EL", "PT"):
        h = f[f["geo"] == g]
        for row in h.itertuples():
            a1.bar(k - 0.19, row.forecast, 0.36, color="none", edgecolor=COL[g], lw=1.6, label="programme projection" if k == 0 else None)
            a1.bar(k + 0.19, row.outturn, 0.36, color=COL[g], label="outturn" if k == 0 else None)
            xt.append(k)
            labs.append(str(row.year))
            k += 1
        a1.text(k - 2.5, 3.6, f"{NAMES[g]}, {h['source'].iloc[0].split(',')[1].strip()}", ha="center", fontsize=9, color=COL[g], fontweight="bold")
        k += 0.8
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xticks(xt, labs)
    a1.set_ylim(-11, 4.5)
    a1.set_ylabel("real GDP growth, %")
    a1.set_title("Programme projections against outturns")
    a1.legend(loc="lower left", fontsize=8.5)
    x = month_x(r["month"])
    sel = (x >= 2010) & (x < 2013.0)
    a2.axvspan(2011 + 3 / 12, 2011 + 10 / 12, color=SHADE, lw=0)
    a2.text(2011 + 6.5 / 12, 0.25, "rate above 1%\n(Apr-Nov 2011)", ha="center", fontsize=8.2, color=MUTED)
    mm = mro[(mro["TIME_PERIOD"] >= "2009-05-01")][["TIME_PERIOD", "OBS_VALUE"]]
    d = pd.to_datetime(mm["TIME_PERIOD"])
    xs = (d.dt.year + (d.dt.dayofyear - 1) / 365).to_numpy()
    xs = np.r_[2010.0, xs[(xs > 2010) & (xs < 2013)], 2013.0]
    ys = np.r_[1.0, mm["OBS_VALUE"].to_numpy()[(d.dt.year + (d.dt.dayofyear - 1) / 365 > 2010).to_numpy() & (d.dt.year < 2013).to_numpy()], 0.75]
    a2.step(xs, ys, where="post", color=INK, lw=1.8, label="ECB main rate")
    a2.plot(x[sel], r["euribor_3m"][sel], color=GREY, lw=1.8, label="3-month Euribor")
    for g, ls in (("PT", "-"), ("ES", "-"), ("DE", "--")):
        a2.plot(x[sel], r[g][sel], color=COL[g], lw=2.2, ls=ls, label=f"new mortgages, {NAMES[g]}")
    a2.set_ylim(0, 5.6)
    a2.set_ylabel("%")
    a2.set_title("One policy rate, different mortgage rates")
    a2.legend(loc="upper left", fontsize=8.2, ncol=2)
    years(a2, 1)
    suptitle(fig, "The programmes assumed far milder recessions; in 2011 Portuguese mortgage rates rose with Euribor while German ones fell")
    note(fig, "Left: European Commission Occasional Papers 61 (May 2010) and 79 (June 2011); outturns Eurostat. Right: ECB MIR (new business, house purchase), FM.")
    fig.tight_layout()
    save(fig, "fig5_programme")


def run() -> None:
    fig1_scorecard()
    fig2_warranted()
    fig3_contagion()
    fig4_events()
    fig5_programme()


if __name__ == "__main__":
    run()
