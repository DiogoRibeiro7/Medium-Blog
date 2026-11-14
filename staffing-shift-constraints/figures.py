"""Figures for the article, from results/.

python figures.py --results results --out figures

fig1  the median test week: the hourly requirement and the cheapest roster under four shift regimes, perfect forecast
fig2  cost of each roster split into what the shift rules force and what the forecast adds, at k = 1 and k = 3
fig3  when calls arrive and which hours of the day bind the roster (descriptive, added after the results)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LABEL = {
    "R0": "hourly (no shifts)",
    "R1": "8-hour shifts, any start",
    "R2": "8-hour + 4-hour shifts",
    "R3": "three fixed shift starts",
    "R4": "weekly tours (5 days on)",
}
FORECAST = {"two_weeks": "same hour two weeks earlier", "mean4": "mean of four weeks", "gbm": "LightGBM"}
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog · data: NYC Open Data, 311 Call Center Inquiry"
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def credit(fig) -> None:
    fig.text(0.01, -0.015, CREDIT, fontsize=7.5, color=INK2, ha="left", va="top")


def fig1(res: Path, out: Path, week: str) -> None:
    rows = read_csv(res / "example_week.csv")
    need = np.array([float(r["need"]) for r in rows])
    h = np.arange(168)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.6), sharex=True, sharey=True)
    for ax, regime in zip(axes, ("R1", "R2", "R3", "R4"), strict=True):
        on = np.array([float(r[f"oracle_{regime}"]) for r in rows])
        ax.fill_between(h, 0, need, step="post", color=GRID, lw=0, label="agents needed (actual calls)")
        ax.fill_between(
            h,
            need,
            on,
            where=on > need,
            step="post",
            color=ORANGE,
            alpha=0.35,
            lw=0,
            label="agents on duty above the need",
        )
        ax.step(h, on, where="post", color=ORANGE, lw=1.4, label="agents on duty")
        idle = (on - need).sum() / need.sum()
        ax.set_title(f"{LABEL[regime]}: {100 * idle:.0f}% more agent-hours than needed this week", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for d in range(1, 7):
            ax.axvline(24 * d, color=INK2, lw=0.5, ls=":")
    axes[-1].set_xticks([24 * d + 12 for d in range(7)], DAYS)
    axes[-1].set_xlim(0, 168)
    for ax in axes:
        ax.set_ylabel("agents")
    axes[0].legend(frameon=False, loc="upper right", ncol=3, fontsize=8.5, bbox_to_anchor=(1, 1.32))
    fig.suptitle(
        f"Week of {week}, perfect forecast, 5-minute handle time", x=0.01, ha="left", fontsize=10, color=INK2, y=1.0
    )
    fig.tight_layout()
    credit(fig)
    fig.savefig(out / "fig1_week.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2(res: Path, out: Path) -> None:
    rows = [r for r in read_csv(res / "cells.csv") if float(r["aht"]) == 300.0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    regimes = ["R0", "R1", "R2", "R4", "R3"]
    names = ["two_weeks", "mean4", "gbm"]
    ypos, labels = [], []
    y = 0.0
    for regime in regimes:
        for n in names:
            ypos.append(y)
            labels.append(f"{LABEL[regime]} · {FORECAST[n]}")
            y += 1
        y += 0.8
    ypos = -np.array(ypos)
    for ax, k in zip(axes, ("1", "3"), strict=True):
        i = 0
        for regime in regimes:
            for n in names:
                r = next(x for x in rows if (x["regime"], x["forecast"], x["k"]) == (regime, n, k))
                cc, fc = float(r["constraint_cost"]), float(r["forecast_cost"])
                ax.barh(ypos[i], cc, color=ORANGE, height=0.78, edgecolor=SURFACE, lw=2)
                ax.barh(ypos[i], fc, left=cc, color=BLUE, height=0.78, edgecolor=SURFACE, lw=2)
                i += 1
        ax.set_title(f"shortfall weighted {k}x an idle agent-hour", loc="left")
        ax.set_xlabel("cost per agent-hour the calls needed")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ypos, labels, fontsize=8.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(
        handles,
        ["forced by the shift rules (perfect forecast)", "added by the forecast's errors"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.6, 1.04),
    )
    fig.tight_layout()
    credit(fig)
    fig.savefig(out / "fig2_split.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3(res: Path, out: Path) -> None:
    js = json.loads((res / "details.json").read_text(encoding="utf-8"))
    h = np.arange(24)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0]
    ax.bar(h, 100 * np.array(js["hour_profile_test"]), color=INK2, width=0.8)
    ax.set_ylabel("% of the day's calls")
    ax.set_title("When calls arrive (test weeks of 2019)", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax = axes[1]
    for off, regime, col in ((-0.2, "R1", BLUE), (0.2, "R4", ORANGE)):
        share = np.array(js["duals"][regime]["positive_weeks"]) / 7
        ax.bar(h + off, 100 * share, width=0.38, color=col, label=LABEL[regime], edgecolor=SURFACE, lw=1)
    ax.set_ylabel("% of days on which\nthe hour binds")
    ax.set_title("Which hours set the roster: positive shadow price, perfect forecast", loc="left")
    ax.set_xticks(range(0, 24, 2), [f"{x:02d}:00" for x in range(0, 24, 2)])
    ax.set_xlabel("hour of the day")
    ax.legend(frameon=False, loc="center right")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    credit(fig)
    fig.savefig(out / "fig3_binding.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = (a.results / "summary.txt").read_text(encoding="utf-8")
    week = summary.split("median week by calls: ")[1].split()[0]
    fig1(a.results, a.out, week)
    fig2(a.results, a.out)
    if (a.results / "details.json").exists():
        fig3(a.results, a.out)


if __name__ == "__main__":
    main()
