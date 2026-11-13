"""Figures for Part 7.

Run: python figures.py  (writes figures/*.png; a few seconds)
"""
from __future__ import annotations

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import jhu as J  # noqa: E402

OUT = J.HERE / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_state_calendars() -> None:
    hol = J.holidays()
    dr = J.us_daily_regime(J.us_day_types(J.us_states(), hol))
    order = ["ordinary", "day after a holiday", "Christmas Eve or New Year's Eve", "federal holiday, state open",
             "state holiday only", "federal and state holiday"]
    labels = {"ordinary": "ordinary weekday", "day after a holiday": "day after a holiday",
              "Christmas Eve or New Year's Eve": "24 or 31 December, not a legal holiday",
              "federal holiday, state open": "federal holiday the state does not observe",
              "state holiday only": "state holiday that is not federal",
              "federal and state holiday": "federal holiday the state observes"}
    colors = {"ordinary": MUTED, "day after a holiday": MUTED, "Christmas Eve or New Year's Eve": ORANGE,
              "federal holiday, state open": ORANGE, "state holiday only": BLUE, "federal and state holiday": BLUE}
    r = J.rates(dr, "day").reindex(order)
    w = J.within_date(dr)
    a = w.groupby("holiday")[["obs_flat", "obs_n", "non_flat", "non_n"]].sum()
    a = a[(a.obs_n >= 10) & (a.non_n >= 10)].sort_values("obs_n")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.75})
    y = np.arange(len(order))
    ax.barh(y, 100 * r.rate, color=[colors[k] for k in order], height=0.62)
    for i, k in enumerate(order):
        ax.text(100 * r.rate[k] + 0.8, i, f"{100 * r.rate[k]:.1f}%  ({int(r['sum'][k])} of {int(r['size'][k]):,})",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [labels[k] for k in order], fontsize=9)
    ax.set_xlim(0, 52)
    ax.set_xlabel("weekday files in which the state's count had not changed (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("US states, weekdays when the state updated daily")

    yb = np.arange(len(a))
    for i, (k, row) in enumerate(a.iterrows()):
        o, n = 100 * row.obs_flat / row.obs_n, 100 * row.non_flat / row.non_n
        bx.plot([n, o], [i, i], color=AXIS, linewidth=2, zorder=1)
        bx.scatter([o], [i], s=48, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        bx.scatter([n], [i], s=44, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.6, zorder=2)
    bx.set_yticks(yb, [f"{k}\n{int(row.obs_n)} observing, {int(row.non_n)} not" for k, row in a.iterrows()],
                  fontsize=8.5)
    bx.scatter([], [], s=48, color=BLUE, label="states observing the holiday")
    bx.scatter([], [], s=44, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.6, label="states working that day")
    bx.legend(loc="lower right", fontsize=8.5)
    bx.set_xlim(-2, 45)
    bx.set_xlabel("files unchanged on that date (%)")
    bx.grid(axis="y", visible=False)
    bx.set_title("The same dates, by the state's own calendar")
    save(fig, "fig2_state_calendars.png")


def fig_womens_day() -> None:
    inc = J.rki_increments()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw={"wspace": 0.3})
    for ax, year in zip(axes, (2022, 2023, 2024)):
        after = pd.Timestamp(f"{year}-03-09")
        days = pd.date_range(after - pd.Timedelta(days=17), after + pd.Timedelta(days=17))
        s = inc.reindex(days)
        col = [ORANGE if d == after else (AXIS if d.dayofweek in (6, 0) else MUTED) for d in days]
        ax.bar(days, s.fillna(0), color=col, width=0.8)
        ax.set_ylim(0, 1.15 * s.max())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xticks([after - pd.Timedelta(days=14), after, after + pd.Timedelta(days=14)],
                      [(after - pd.Timedelta(days=14)).strftime("%d %b"), after.strftime("%d %b"),
                       (after + pd.Timedelta(days=14)).strftime("%d %b")])
        ax.grid(axis="x", visible=False)
        v = s.get(after)
        ax.plot([after], [0], marker="^", markersize=9, color=ORANGE, clip_on=False, zorder=4)
        ax.set_title(f"{year}: {after.strftime('%a')} 9 March, {v:,.0f}", fontsize=10)
    axes[0].set_ylabel("new cases in each daily file")
    fig.suptitle("New cases in the RKI file published the day after International Women's Day, a holiday in Berlin only",
                 x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
    save(fig, "fig1_womens_day.png")


def fig_errors() -> None:
    e = J.us_errors(J.us_day_types(J.us_states(), J.holidays()))
    groups = [("files that changed", e[e.flat == 0], MUTED),
              ("unchanged, still unchanged in the final version", e[(e.flat == 1) & e.still_flat], AQUA),
              ("unchanged, filled in later", e[(e.flat == 1) & ~e.still_flat], ORANGE)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    grid = np.logspace(-2, 3, 400)
    for label, g, col in groups:
        err = 100 * g.rel.abs().to_numpy()
        share = [(err <= x).mean() * 100 for x in grid]
        ax.plot(grid, share, color=col, label=f"{label} ({len(g):,})")
    ax.set_xscale("log")
    ticks = [0.01, 0.1, 1, 10, 100, 1000]
    ax.set_xticks(ticks, ["0.01", "0.1", "1", "10", "100", "1,000"])
    ax.set_ylim(0, 101)
    ax.set_xlabel("absolute error of the 7-day count as first published, against the final version (%)")
    ax.set_ylabel("share of state-days with at most that error (%)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("US states: how far each morning's 7-day count was from the final version")
    save(fig, "fig3_errors.png")


if __name__ == "__main__":
    fig_womens_day()
    fig_state_calendars()
    fig_errors()
