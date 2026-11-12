"""Figures for Part 5. Replays four variants of the pipeline (a few minutes on two cores) and draws three figures.

Run: python figures.py  (writes figures/*.png)
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part4"))
import pipeline as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})

FIRST_FILE, LAST_FILE = dt.date(2021, 4, 2), dt.date(2023, 12, 31)
FIRST_RUN, LAST_RUN = dt.date(2021, 6, 1), dt.date(2023, 8, 31)
LATE = {dt.date(2022, 3, 15): 2}
TEN = ["2022-10-04", "2023-04-08", "2023-04-11", "2023-05-02", "2023-05-19", "2023-05-30",
       "2023-08-25", "2023-08-26", "2023-08-27", "2023-08-28"]
SHOWN = [((False, False, "level"), "Part 4", MUTED), ((True, False, "level"), "content check", BLUE),
         ((True, True, "level"), "content check + holidays", ORANGE),
         ((True, True, "last_week"), "all three (+ last-week fallback)", AQUA)]


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def replay(v):
    tmp = Path(tempfile.mkdtemp())
    incoming = P.write_incoming(tmp / "incoming", FIRST_FILE, LAST_FILE, LATE)
    rep = P.replay(FIRST_RUN, LAST_RUN, incoming, content_check=v[0], holidays=v[1], fallback=v[2]).report()
    return v, rep


def fig1_what_each_file_added() -> None:
    tmp = Path(tempfile.mkdtemp())
    inc = P.write_incoming(tmp / "in", dt.date(2023, 8, 17), dt.date(2023, 8, 29))
    pipe = P.Pipeline()
    for i in inc:
        pipe.land(i, i.arrives_at)
    k = pipe.con.execute("select publication_date, reporting_date, sum(cases) as cases from landing.cases group by all").df()
    k["publication_date"] = pd.to_datetime(k.publication_date)
    k["reporting_date"] = pd.to_datetime(k.reporting_date)
    wide = k.pivot(index="publication_date", columns="reporting_date", values="cases").fillna(0)
    added = (wide - wide.shift(1).fillna(0)).iloc[1:]              # what each file added to each date
    cols = pd.date_range("2023-08-14", "2023-08-28")
    added = added.reindex(columns=cols).fillna(0)
    rows = added.index
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    show = np.where(added.to_numpy() > 0, added.to_numpy(), np.nan)
    ax.imshow(np.zeros_like(show), cmap=LinearSegmentedColormap.from_list("s", [SURFACE, SURFACE]), aspect="auto")
    ax.imshow(show, cmap=cmap, vmin=0, vmax=1200, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = added.iat[i, j]
            if v != 0:                  # a negative number is a removal
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=7.5, color=INK if v < 700 else SURFACE)
            elif cols[j] < rows[i]:
                ax.text(j, i, "·", ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(cols)), [f"{d:%a}\n{d:%d}" for d in cols], fontsize=8.5)
    ax.set_yticks(range(len(rows)), [f"file of {d:%a %d %b}" for d in rows], fontsize=8.5)
    for i, d in enumerate(rows):
        if added.iloc[i].abs().sum() == 0:
            ax.text(len(cols) - 0.3, i, "nothing new", va="center", ha="left", fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.set_xlabel("reporting date, August 2023")
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Cases each daily file added to each reporting date, 18 to 29 August 2023 (· = no change)")
    save(fig, "fig1_what_each_file_added.png")


def evaluate(rep: pd.DataFrame, final: pd.Series) -> pd.DataFrame:
    rep = rep.copy()
    rep["run_date"] = pd.to_datetime(rep.run_date)
    rep["as_of_publication"] = pd.to_datetime(rep.as_of_publication)
    rep["final_7d"] = [final.reindex(pd.date_range(p - pd.Timedelta(days=7), p - pd.Timedelta(days=1))).sum()
                       for p in rep.as_of_publication]
    rep["err"] = rep.nowcast_7d / rep.final_7d - 1
    return rep


def fig2_ten_mornings(reps: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(TEN))
    offsets = np.linspace(-0.27, 0.27, len(SHOWN))
    ax.axhline(0, color=AXIS, linewidth=1)
    for (v, label, color), off in zip(SHOWN, offsets):
        r = reps[v].set_index("run_date")
        y = [100 * r.err[pd.Timestamp(d)] for d in TEN]
        ax.vlines(x + off, 0, y, color=color, linewidth=2)
        ax.scatter(x + off, y, s=34, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=label)
    ax.set_xticks(x, [f"{pd.Timestamp(d):%a}\n{pd.Timestamp(d).day} {pd.Timestamp(d):%b}\n{pd.Timestamp(d):%Y}" for d in TEN],
                  fontsize=8.5)
    for (v, label, color), off in zip(SHOWN, offsets):        # values beyond the axis, written at its edge
        r = reps[v].set_index("run_date")
        for i, d in enumerate(TEN):
            y = 100 * r.err[pd.Timestamp(d)]
            if y > 140:
                ax.annotate(f"{y:+.0f}% \u2191", (i + off, 140), xytext=(4, -10), textcoords="offset points",
                            fontsize=8.5, color=INK2, ha="left")
    ax.set_ylabel("nowcast error against final (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2, fontsize=9)
    ax.set_ylim(-60, 140)
    ax.set_title("The ten mornings after a file with nothing new: the content check alone overshoots after holidays")
    save(fig, "fig2_ten_mornings.png")


def fig3_by_month(reps: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for v, label, color in (SHOWN[0], SHOWN[3]):
        r = reps[v]
        m = r.err.abs().groupby(r.run_date.dt.to_period("M")).mean()
        ax.plot(m.index.to_timestamp(), 100 * m.to_numpy(), color=color, marker="o", markersize=4, label=label)
    ax.set_ylabel("mean absolute error of the nowcast (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    ax.set_title("Monthly error of the nowcast, Part 4 against all three changes, June 2021 to August 2023")
    save(fig, "fig3_error_by_month.png")


def main() -> None:
    fig1_what_each_file_added()
    final = P.final_national()
    with ProcessPoolExecutor(max_workers=min(len(SHOWN), os.cpu_count() or 1)) as ex:
        reps = {v: evaluate(rep, final) for v, rep in ex.map(replay, [s[0] for s in SHOWN])}
    fig2_ten_mornings(reps)
    fig3_by_month(reps)


if __name__ == "__main__":
    main()
