"""Draw the figures for the article from ``results/`` and ``data/ifr_published.csv``.

    python figures.py

Static PNGs for Medium (light surface), one fixed categorical order, direct labels on
every line because several palette slots sit below 3:1 contrast on the light surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
RES, DATA, FIG = HERE / "results", HERE / "data", HERE / "figures"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOURCE_OECD = "Source: OECD Trade in Employment and TiVA 2025 editions, national accounts (snapshot 12 Nov 2026); author's calculations."
NAMES = {"CHN": "China", "VNM": "Vietnam", "THA": "Thailand", "MEX": "Mexico", "ROU": "Romania", "POL": "Poland",
         "PRT": "Portugal", "DEU": "Germany", "USA": "United States", "KOR": "Korea", "JPN": "Japan"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def tidy(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name: str, title: str, subtitle: str, source: str = SOURCE_OECD) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, source, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def end_labels(ax, x, items, min_gap):
    """Place end-of-line labels, spreading them apart only as much as needed (in data units)."""
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for label, y in items:
        yy = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(yy)
        ax.text(x, yy, label, va="center", fontsize=8.5, color=INK2)


def fig1() -> None:
    d = pd.read_csv(RES / "ladder_2022.csv").sort_values("compensation").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.88, bottom=0.08)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        ax.hlines(i, r.compensation, r.value_added, color=BASE, linewidth=2)
    ax.scatter(d.compensation, y, s=55, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Compensation of employees per person engaged")
    ax.scatter(d.value_added, y, s=60, facecolor="none", edgecolor=SLOTS[1], linewidth=2, zorder=4,
               label="Value added per person engaged")
    labels = [f"{n}  ({ls:.0%})" for n, ls in zip(d.name, d.labour_share)]
    ax.set_yticks(y, labels, fontsize=8.5)
    for t, c in zip(ax.get_yticklabels(), d.country):
        if c == "CHN":
            t.set_fontweight("bold")
            t.set_color(INK)
    ci = int(d.index[d.country == "CHN"][0])
    r = d.loc[ci]
    ax.text(r.compensation / 1.08, ci, f"{r.compensation:,.0f}", va="center", ha="right", fontsize=8, color=INK2)
    ax.text(r.value_added * 1.08, ci, f"{r.value_added:,.0f}", va="center", ha="left", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.set_xticks([2000, 5000, 10000, 20000, 50000, 100000, 200000])
    ax.set_xlim(2000, 300000)
    ax.set_xlabel("US dollars per person engaged per year, market exchange rates (log scale); labour share in brackets")
    tidy(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=8.5, handletextpad=0.3)
    finish(fig, "fig1_manufacturing_ladder_2022", "Manufacturing pay and value added per worker, 2022",
           "Twenty-five economies; China pays more per worker than Mexico, Vietnam, Thailand, Türkiye, Romania and Bulgaria")


def fig2() -> None:
    d = pd.read_csv(RES / "china_relative_compensation.csv").set_index("year")
    show = ["VNM", "THA", "MEX", "ROU", "POL", "PRT", "DEU"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(left=0.08, right=0.83, top=0.86, bottom=0.11)
    ax.axhline(1.0, color=INK, linewidth=1.2)
    ax.text(2003, 1.06, "China pays the same", fontsize=8.5, color=INK2, va="bottom")
    items = []
    for k, c in enumerate(show):
        ax.plot(d.index, d[c], color=SLOTS[k], linewidth=2)
        ax.scatter(d.index[-1], d[c].iloc[-1], s=40, color=SLOTS[k], edgecolor=SURFACE, linewidth=2, zorder=3)
        items.append((f"{NAMES[c]} {d[c].iloc[-1]:.2f}", np.log10(d[c].iloc[-1])))
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.015, 3.2)
    ax.set_xlim(1995, 2022)
    # labels in log space
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for label, ly in items:
        yy = ly if not placed else max(ly, placed[-1] + 0.075)
        placed.append(yy)
        ax.text(2022.6, 10 ** yy, label, va="center", fontsize=8.5, color=INK2)
    ax.set_ylabel("China's compensation per person ÷ the country's (log scale)")
    tidy(ax, "y")
    finish(fig, "fig2_china_relative_pay", "China's manufacturing pay relative to seven economies, 1995–2022",
           "Compensation of employees per person engaged, US dollars at market exchange rates; above 1, China pays more")


def fig3() -> None:
    m = pd.read_csv(RES / "china_manufacturing.csv").set_index("TIME_PERIOD")
    w = pd.read_csv(RES / "world_shares.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
    fig.subplots_adjust(left=0.07, right=0.84, top=0.82, bottom=0.12, wspace=0.42)
    ax = axes[0]
    ax.plot(m.index, m.employment_thousands / 1e3, color=SLOTS[0], linewidth=2)
    pk = m.employment_thousands.idxmax()
    ax.scatter(pk, m.employment_thousands.max() / 1e3, s=40, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(pk, m.employment_thousands.max() / 1e3 + 4, f"peak {pk}: {m.employment_thousands.max() / 1e3:.1f}m",
            ha="center", fontsize=8.5, color=INK2)
    ax.scatter(m.index[-1], m.employment_thousands.iloc[-1] / 1e3, s=40, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(m.index[-1], m.employment_thousands.iloc[-1] / 1e3 - 9, f"{m.employment_thousands.iloc[-1] / 1e3:.1f}m",
            ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 175)
    ax.set_xlim(1995, 2022)
    ax.set_title("Persons engaged in manufacturing, millions")
    tidy(ax, "y")
    ax = axes[1]
    series = [("manufacturing", "All manufacturing"), ("textiles, apparel and leather", "Textiles and apparel"),
              ("computers and electronics", "Electronics"), ("motor vehicles", "Motor vehicles")]
    items = []
    for k, (ind, lab) in enumerate(series):
        s = w[w.industry == ind].set_index("year").china * 100
        ax.plot(s.index, s, color=SLOTS[k], linewidth=2)
        ax.scatter(s.index[-1], s.iloc[-1], s=40, color=SLOTS[k], edgecolor=SURFACE, linewidth=2, zorder=3)
        items.append((f"{lab} {s.iloc[-1]:.0f}%", s.iloc[-1]))
    end_labels(ax, 2022.8, items, 2.6)
    ax.set_xlim(1995, 2022)
    ax.set_ylim(0, 55)
    ax.set_title("China's share of world value added, %")
    tidy(ax, "y")
    finish(fig, "fig3_china_jobs_and_world_share", "Fewer workers, a larger share of world output",
           "China's manufacturing employment peaked in 2013; its share of world manufacturing value added kept rising")


def fig4() -> None:
    tim = pd.read_csv(DATA / "oecd_tim.csv", low_memory=False)
    ls = tim[(tim.MEASURE == "LABR") & (tim.UNIT_MEASURE == "PT_VA") & (tim.ACTIVITY == "C")].pivot_table(
        index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
    show = ["CHN", "MEX", "POL", "PRT", "DEU", "USA"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.08, right=0.83, top=0.86, bottom=0.11)
    items = []
    for k, c in enumerate(show):
        ax.plot(ls.index, ls[c], color=SLOTS[k], linewidth=2.6 if c == "CHN" else 2)
        ax.scatter(ls.index[-1], ls[c].iloc[-1], s=40, color=SLOTS[k], edgecolor=SURFACE, linewidth=2, zorder=3)
        items.append((f"{NAMES[c]} {ls[c].iloc[-1]:.0f}%", ls[c].iloc[-1]))
    end_labels(ax, 2022.6, items, 2.2)
    ax.set_xlim(1995, 2022)
    ax.set_ylim(10, 72)
    ax.set_yticks([20, 30, 40, 50, 60, 70])
    ax.set_ylabel("Compensation of employees, % of manufacturing value added")
    tidy(ax, "y")
    finish(fig, "fig4_labour_share", "Labour's share of manufacturing value added, 1995–2022",
           "China's share rose from 36.1% to 40.5% while it fell in Mexico, Poland, Portugal, Germany and the United States")


def fig5() -> None:
    ifr = pd.read_csv(DATA / "ifr_published.csv")
    d = ifr[ifr.metric == "robot_density"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.08, right=0.82, top=0.86, bottom=0.11)
    order = ["CHN", "KOR", "JPN", "DEU", "USA"]
    items = []
    for k, c in enumerate(order):
        s = d[d.country == c].sort_values("year")
        ax.plot(s.year, s.value, color=SLOTS[k], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        items.append((f"{NAMES[c]} {int(s.value.iloc[-1])}", np.log10(s.value.iloc[-1])))
    rev = ifr[(ifr.country == "CHN") & (ifr.metric == "robot_density_revised")]
    ax.scatter(rev.year, rev.value, s=70, facecolor="none", edgecolor=SLOTS[0], linewidth=2, zorder=4)
    ax.annotate("", xy=(2024, rev.value.iloc[0] * 1.08), xytext=(2024, 567 / 1.08),
                arrowprops=dict(arrowstyle="->", color=INK2, linewidth=1))
    ax.text(2023.75, rev.value.iloc[0], "same robots, new count\nof employees: 166", ha="right", va="center", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_yticks([25, 50, 100, 200, 500, 1000])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(18, 1600)
    ax.set_xlim(2012.5, 2024.5)
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for label, ly in items:
        yy = ly if not placed else max(ly, placed[-1] + 0.07)
        placed.append(yy)
        ax.text(2024.8, 10 ** yy, label, va="center", fontsize=8.5, color=INK2)
    ax.set_ylabel("Industrial robots per 10,000 manufacturing employees (log scale)")
    tidy(ax, "y")
    finish(fig, "fig5_robot_density", "Robot density as published by the IFR, 2013–2024",
           "China's 2024 figure was published as 567 and revised to 166 when the employment denominator was updated",
           source="Source: International Federation of Robotics press releases and World Robotics 2025 summary (see data/ifr_sources.csv); published figures, not reproduced.")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
    print("figures written to", FIG)
