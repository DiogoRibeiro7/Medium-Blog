"""Figures for Part 3, drawn from the same functions as realdata_numbers.py.

Run: python figures.py  (writes figures/*.png)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import realdata as r  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT = "#e1e0d9"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})
FIRST, LAST = "2021-04-12", "2023-08-31"


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_arrival() -> None:
    melde, onset = r.load_vintages("melde"), r.load_vintages("onset")
    final = r.final_melde()
    ages = list(range(0, 81))
    fig, ax = plt.subplots(figsize=(10, 4.6))

    def band(table, color, label):
        med = [100 * table[a].median() for a in table.columns]
        lo = [100 * table[a].quantile(0.1) for a in table.columns]
        hi = [100 * table[a].quantile(0.9) for a in table.columns]
        ax.fill_between(table.columns, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(table.columns, med, color=color, label=label)

    ct = r.completeness_table(melde, final, ages, FIRST, LAST)
    band(ct, BLUE, "cases by reporting date (RKI files)")
    oc = r.completeness_table(onset, onset.at_age(r.MAX_AGE), ages, FIRST, LAST)
    band(oc, ORANGE, "cases by onset date (RKI files)")
    tri = r.load_hub_triangle()
    i0, i1 = tri.index("2021-10-25"), tri.index(r.HUB_LAST)
    hosp = pd.DataFrame({k: tri.cum[i0:i1 + 1, k] / tri.final[i0:i1 + 1] for k in ages})
    band(hosp, AQUA, "hospitalised cases by reporting date (nowcast hub)")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 102)
    ax.set_xlabel("days after the reference date (reporting date or onset)")
    ax.set_ylabel("share of the final count published (%)")
    ax.set_title("Three clocks on the same epidemic: days for reported cases, weeks for onsets, months for hospitalisations")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig1_arrival.png")


def fig2_weekday_heatmap() -> None:
    melde = r.load_vintages("melde")
    ct = r.completeness_table(melde, r.final_melde(), [1], FIRST, LAST)[1].dropna()
    df = pd.DataFrame({"share": ct, "wd": ct.index.dayofweek, "month": ct.index.to_period("M")})
    grid = df.pivot_table(index="wd", columns="month", values="share", aggfunc="median")
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP[1:])
    fig, ax = plt.subplots(figsize=(12, 3.6))
    im = ax.imshow(100 * grid.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    months = grid.columns
    ticks = [i for i, m in enumerate(months) if m.month in (1, 4, 7, 10)]
    ax.set_xticks(ticks, [f"{months[i].year}-{months[i].month:02d}" for i in ticks])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% published next day", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("month of the reporting date")
    ax.set_ylabel("weekday of the\nreporting date")
    ax.set_title("Share of each day's cases in the next morning's file: from mid-2022, weekend cases waited until Tuesday")
    fig.tight_layout()
    save(fig, "fig2_next_day_by_weekday.png")


def fig3_seven_day() -> None:
    melde = r.load_vintages("melde")
    final = r.final_melde()
    sd = r.seven_day_sums(melde, final).loc[FIRST:LAST]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax.plot(sd.index, sd.final / 1e3, color=LIGHT, linewidth=5, label="final (file of 10 Nov 2026)")
    ax.plot(sd.index, sd.published / 1e3, color=BLUE, linewidth=1.4, label="as published that day")
    ax.set_yscale("log")
    ax.set_yticks([3, 10, 30, 100, 300, 1000], ["3", "10", "30", "100", "300", "1,000"])
    ax.minorticks_off()
    ax.set_ylabel("cases in the 7 reporting\ndates before (thousands)")
    ax.set_title("The 7-day count as published each day, and the same seven days in the final file")
    ax.legend(loc="lower left")

    wk_pub = sd.published / sd.published.shift(7) - 1
    wk_fin = sd.final / sd.final.shift(7) - 1
    live = {}
    for p in sd.index:
        this = pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))
        prev = this - pd.Timedelta(days=7)
        if prev[0] in melde.V.columns:
            live[p] = melde.V.loc[p, this].fillna(0).sum() / melde.V.loc[p, prev].fillna(0).sum() - 1
    live = pd.Series(live)
    bx.axhline(0, color=AXIS, linewidth=1)
    bx.plot(live.index, 100 * (live - wk_fin.reindex(live.index)), color=ORANGE, linewidth=1.2,
            label="both weeks read from one day's file")
    bx.plot(wk_pub.index, 100 * (wk_pub - wk_fin), color=BLUE, linewidth=1.2,
            label="each week as published on its own day")
    bx.set_ylabel("error of the week-over-week\nchange (points)")
    bx.set_ylim(-60, 30)
    bx.set_title("Week-over-week change: comparing at equal maturity removes most of the error")
    bx.legend(loc="lower left", ncols=2)
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    save(fig, "fig3_seven_day_count.png")


def fig4_hospital_series() -> None:
    tri = r.load_hub_triangle()
    q = r.load_hub_nowcasts()
    ens = q[(q.model == "NowcastHub-MeanEnsemble") & (q.h == 0)].set_index("target_end_date")
    days = pd.date_range(r.HUB_FIRST, r.HUB_LAST)
    idx = [tri.index(d) for d in days]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(days, [tri.final7(t) / 1e3 for t in idx], color=LIGHT, linewidth=5, label="latest value (December 2024)")
    ax.plot(days, [tri.known7(t, t + 7) / 1e3 for t in idx], color=AQUA, linewidth=1.6, label="known a week later")
    ax.plot(days, [tri.known7(t, t) / 1e3 for t in idx], color=BLUE, linewidth=1.6, label="as first published")
    e = ens.reindex(days)
    ax.fill_between(days, e["0.025"] / 1e3, e["0.975"] / 1e3, color=ORANGE, alpha=0.15, linewidth=0)
    ax.plot(days, e["point"] / 1e3, color=ORANGE, linewidth=1.6, label="mean-ensemble nowcast that day (95% interval)")
    ax.set_ylabel("hospitalised cases, 7 reporting dates (thousands)")
    ax.set_title("Germany's 7-day hospitalisation count: first published at half its final size")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylim(0, None)
    fig.tight_layout()
    save(fig, "fig4_hospitalisation_count.png")


def fig5_nowcast_scores() -> None:
    tri = r.load_hub_triangle()
    q = r.evaluate_hub(tri, r.load_hub_nowcasts())
    cl = r.evaluate_hub(tri, r.chain_ladder_hub(tri, 56))
    allm = pd.concat([q, cl], ignore_index=True)
    show = {"NowcastHub-MeanEnsemble": (ORANGE, "mean ensemble"), "ILM-prop": (AQUA, "ILM-prop"),
            "chain ladder (56-day window)": (BLUE, "chain ladder (Part 2), 56-day window")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, truth, title in ((axes[0], "truth", "Against the latest value"),
                             (axes[1], "truth80", "Against the value 80 days after")):
        for m, g in allm.groupby("model"):
            sc = g.groupby("h").apply(lambda x: 100 * ((x.point - x[truth]) / x[truth]).abs().mean())
            sc = sc.sort_index()
            if m in show:
                continue
            ax.plot(-sc.index, sc.values, color=AXIS, linewidth=1)
        for m, (color, label) in show.items():
            g = allm[allm.model == m]
            sc = g.groupby("h").apply(lambda x: 100 * ((x.point - x[truth]) / x[truth]).abs().mean()).sort_index()
            ax.plot(-sc.index, sc.values, color=color, label=label)
        ax.set_xlabel("days before the forecast date")
        ax.set_title(title)
        ax.set_xlim(0, 28)
    axes[0].plot([], [], color=AXIS, linewidth=1, label="the other eight nowcasts")
    axes[0].set_ylabel("mean absolute error of the 7-day count (%)")
    axes[0].set_ylim(0, 14)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Ten published nowcasts (eight models, two ensembles) and the Part 2 chain ladder, 22 Nov 2021 to 29 Apr 2022",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig5_nowcast_errors.png")


def fig6_gdp_revisions() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharex=True, sharey=True)
    for ax, cid in zip(axes, r.GDP_IDS):
        s = r.revision_stats(cid)
        f, l = s["first"], s["latest"]
        ok = (f.abs() < 3) & (l.abs() < 3)
        ax.plot([-3, 3], [-3, 3], color=AXIS, linewidth=1)
        changed = np.sign(f) != np.sign(l)
        ax.scatter(f[ok & ~changed], l[ok & ~changed], s=16, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.scatter(f[ok & changed], l[ok & changed], s=22, color=ORANGE, edgecolor=SURFACE, linewidth=0.6,
                   label="sign changed")
        ax.set_title(f"{r.GDP_NAMES[cid]}: {s['mean_abs_revision']:.2f} points")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.set_xlabel("first release (%)")
    axes[0].set_ylabel("vintage of 2024 Q4 (%)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Quarter-on-quarter real GDP growth, 2002 Q3 to 2021 Q4: first release against the latest vintage "
                 "(title: mean absolute revision)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig6_gdp_first_latest.png")


def fig7_gdp_levels() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = {"US": BLUE, "EA": ORANGE, "JP": AQUA, "CHE": INK2}
    for cid in r.GDP_IDS:
        lv = r.gdp_levels(cid).loc["2000-01-01"]
        ax.step(lv.index, 100 * lv / lv.iloc[0], where="post", color=colors[cid], label=r.GDP_NAMES[cid])
        ax.text(lv.index[-1] + pd.Timedelta(days=60), 100 * lv.iloc[-1] / lv.iloc[0], r.GDP_NAMES[cid],
                color=INK2, va="center", fontsize=9)
    ax.set_ylabel("level of 2000 Q1 in each vintage\n(first vintage = 100)")
    ax.set_xlabel("vintage")
    ax.set_xlim(pd.Timestamp("2002-07-01"), pd.Timestamp("2026-12-31"))
    ax.set_title("The same quarter, restated: real GDP for 2000 Q1 in each vintage, steps at benchmark revisions")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig7_gdp_level_restatements.png")


if __name__ == "__main__":
    fig1_arrival()
    fig2_weekday_heatmap()
    fig3_seven_day()
    fig4_hospital_series()
    fig5_nowcast_scores()
    fig6_gdp_revisions()
    fig7_gdp_levels()
