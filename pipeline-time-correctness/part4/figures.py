"""Figures for Part 4. Replays the pipeline (about a minute and a half) and draws four figures.

Run: python figures.py  (writes figures/*.png)
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT = "#e1e0d9"
PALE_BLUE, PALE_ORANGE, PALE_AQUA = "#cde2fb", "#fde0d3", "#d3f0e4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1_layers() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    def box(x, y, w, h, title, lines, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + 0.15, y + h - 0.22, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
        for i, line in enumerate(lines):
            ax.text(x + 0.15, y + h - 0.62 - 0.36 * i, line, fontsize=8.8, color=INK2, va="top")

    def arrow(x0, y0, x1, y1, label=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=INK2, linewidth=1.3))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, fontsize=8.5, color=INK2, ha="center")

    box(0.1, 2.5, 2.2, 1.9, "source", ["one file per day", "(publication date)", "may arrive late,", "may be republished"],
        "#f3f2ee", AXIS)
    box(3.0, 2.3, 2.6, 2.3, "landing (append-only)", ["files: name, sha256,", "  publication date, loaded_at",
                                                      "cases: rows copied as is", "never updated or deleted"],
        PALE_BLUE, BLUE)
    box(6.3, 3.1, 2.5, 1.5, "core.known_at(t)", ["latest version of each file", "loaded by t"], PALE_AQUA, AQUA)
    box(6.3, 1.2, 2.5, 1.5, "core.delay_profile", ["share known by weekday", "and age, as of each run"], PALE_AQUA, AQUA)
    box(9.5, 2.0, 2.4, 2.4, "mart.daily_report", ["one row per run date:", "published, nowcast,", "interval, completeness,",
                                                  "week-over-week,", "computed_at, code version"], PALE_ORANGE, ORANGE)
    arrow(2.3, 3.45, 3.0, 3.45)
    ax.text(2.65, 3.62, "load on\narrival", fontsize=8, color=INK2, ha="center", va="bottom")
    arrow(5.6, 3.85, 6.3, 3.85)
    arrow(7.55, 3.1, 7.55, 2.7)
    arrow(8.8, 3.85, 9.5, 3.5)
    arrow(8.8, 1.95, 9.5, 2.6)
    ax.text(0.1, 0.55, "Each run at 08:00: load the files that have arrived; compute the profile and the report as of the run time; "
            "replace that run's partition.", fontsize=9, color=INK2)
    ax.text(0.1, 0.2, "Tests: every past run recomputes exactly from landing; a later file never changes an earlier run; "
            "landing a file twice changes nothing.", fontsize=9, color=INK2)
    ax.set_title("Three layers, two clocks: the source's publication date and the pipeline's load time",
                 loc="left")
    save(fig, "fig1_layers.png")


def fig2_report(rep: pd.DataFrame) -> None:
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1.15, 1]})
    z = rep[(rep.run_date >= "2022-09-01") & (rep.run_date <= "2022-12-31")]
    ax.fill_between(z.run_date, z.lower_7d / 1e3, z.upper_7d / 1e3, color=ORANGE, alpha=0.18, linewidth=0,
                    label="90% interval from past errors")
    ax.plot(z.run_date, z.final_7d / 1e3, color=LIGHT, linewidth=5, label="final")
    ax.plot(z.run_date, z.published_7d / 1e3, color=BLUE, linewidth=1.5, label="published")
    ax.plot(z.run_date, z.nowcast_7d / 1e3, color=ORANGE, linewidth=1.5, label="nowcast")
    ax.set_ylabel("cases, 7 reporting dates\n(thousands)")
    ax.set_title("The daily report, September to December 2022")
    ax.legend(loc="upper right", ncols=2, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    bx.axhline(0, color=AXIS, linewidth=1)
    bx.plot(rep.run_date, 100 * (rep.published_7d / rep.final_7d - 1), color=BLUE, linewidth=1, label="published")
    bx.plot(rep.run_date, 100 * (rep.nowcast_7d / rep.final_7d - 1), color=ORANGE, linewidth=1, label="nowcast")
    bx.set_ylim(-60, 25)
    bx.set_ylabel("error against final (%)")
    bx.set_title("Every run, June 2021 to August 2023")
    bx.legend(loc="lower left", ncols=2)
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    save(fig, "fig2_daily_report.png")


def fig3_weekday(rep: pd.DataFrame) -> None:
    e_pub = (rep.published_7d / rep.final_7d - 1).abs()
    e_now = (rep.nowcast_7d / rep.final_7d - 1).abs()
    wd = rep.run_date.dt.dayofweek
    a = [100 * e_pub[wd == k].mean() for k in range(7)]
    b = [100 * e_now[wd == k].mean() for k in range(7)]
    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.18, a, width=0.34, color=BLUE, label="published")
    ax.bar(x + 0.18, b, width=0.34, color=ORANGE, label="nowcast")
    for i in x:
        ax.text(i - 0.18, a[i] + 0.15, f"{a[i]:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.text(i + 0.18, b[i] + 0.15, f"{b[i]:.1f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, WEEKDAYS)
    ax.set_xlabel("weekday of the run")
    ax.set_ylabel("mean absolute error of the 7-day count (%)")
    ax.set_title("The nowcast removes most of the shortfall on every weekday; Monday and Sunday stay hardest")
    ax.set_ylim(0, 12.5)
    ax.legend(loc="upper center", ncols=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig3_error_by_weekday.png")


def fig4_coverage(rep: pd.DataFrame) -> None:
    iv = rep[(rep.run_date >= "2021-08-01") & rep.lower_7d.notna()].copy()
    iv["inside"] = (iv.lower_7d <= iv.final_7d) & (iv.final_7d <= iv.upper_7d)
    iv["above"] = iv.final_7d > iv.upper_7d
    iv["below"] = iv.final_7d < iv.lower_7d
    m = iv.groupby(iv.run_date.dt.to_period("M"))[["inside", "above", "below"]].mean()
    idx = m.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(idx, 100 * m.inside, width=24, color=AQUA, label="final inside the interval")
    ax.bar(idx, 100 * m.above, width=24, bottom=100 * m.inside, color=ORANGE, label="final above",
           edgecolor=SURFACE, linewidth=1)
    ax.bar(idx, 100 * m.below, width=24, bottom=100 * (m.inside + m.above), color=BLUE, label="final below",
           edgecolor=SURFACE, linewidth=1)
    ax.axhline(90, color=INK, linewidth=1, linestyle=":")
    ax.text(idx[-1] + pd.Timedelta(days=17), 90, "nominal\n90%", fontsize=8.5, color=INK2, va="center")
    ax.set_xlim(idx[0] - pd.Timedelta(days=20), idx[-1] + pd.Timedelta(days=75))
    ax.set_ylim(0, 114)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("share of runs in the month (%)")
    ax.set_title("The interval from eight weeks of past errors covered 74% of runs, not 90%")
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig4_interval_coverage.png")


def run_replay() -> pd.DataFrame:
    tmp = Path(tempfile.mkdtemp())
    incoming = P.write_incoming(tmp / "incoming", dt.date(2021, 4, 2), dt.date(2023, 12, 31), {dt.date(2022, 3, 15): 2})
    pipe = P.replay(dt.date(2021, 6, 1), dt.date(2023, 8, 31), incoming)
    rep = pipe.report()
    rep["run_date"] = pd.to_datetime(rep.run_date)
    rep["as_of_publication"] = pd.to_datetime(rep.as_of_publication)
    final = P.final_national()
    rep["final_7d"] = [final.reindex(pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))).sum()
                       for p in rep.as_of_publication]
    return rep


if __name__ == "__main__":
    fig1_layers()
    rep = run_replay()
    fig2_report(rep)
    fig3_weekday(rep)
    fig4_coverage(rep)
