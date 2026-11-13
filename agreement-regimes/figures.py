"""Figures for the article, from ``results/`` and the input ratios.

    python figures.py

fig1  each pair's log(delivered / outside) with the registered split and the median of each segment
fig2  the later segment: observed smallest loss caught 90% of the time against the full-history and the segment
      predictions
fig3  the forward test: trailing-36-month prediction against the loss caught over the next 24 months
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RATIOS = HERE.parent / "second-count-prediction" / "results" / "corrected" / "ratios.csv"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc4"
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog · data: Citi Bike, NYC DOT, City of Chicago, NYC TLC"
GROUP = {"citibike": BLUE, "chicago": ORANGE, "TLC": VIOLET}
SHORT = {
    "citibike / operator reports": "Citi Bike vs operator reports",
    "citibike / DOT table": "Citi Bike vs NYC DOT table",
    "chicago / vehicles": "Chicago vs vehicle records",
    "chicago / drivers": "Chicago vs driver records",
    "TLC yellow / monthly report": "TLC yellow vs report",
    "TLC green / monthly report": "TLC green vs report",
    "TLC fhv / monthly report": "TLC FHV vs report",
    "TLC fhvhv / monthly report": "TLC high-volume vs report",
}

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


def color(name: str) -> str:
    return GROUP[name.split(" ")[0]]


def loss_label(x, _=None):
    return f"{100 * x:g}%"


def fig1(ratios: pd.DataFrame, pairs: dict) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 11), sharey=True)
    for ax, (name, p) in zip(axes.flat, pairs.items(), strict=True):
        d = ratios[(ratios.pair == name)]
        if name == "TLC fhv / monthly report":
            d = d[d.month != "2019-01"]
        t = pd.to_datetime(d.month + "-01")
        ax.plot(t, d.log_ratio.astype(float).map(math.expm1), color=color(name), lw=1.5)
        sp = p["split"]
        if sp["month"]:
            ts = pd.Timestamp(sp["month"] + "-01")
            ax.axvline(ts, color=INK2, lw=1, ls="--")
            ax.hlines(math.expm1(sp["median_left"]), t.min(), ts, color=INK, lw=2)
            ax.hlines(math.expm1(sp["median_right"]), ts, t.max(), color=INK, lw=2)
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.2)
        ax.set_ylim(-1.0, 1.3)
        ax.yaxis.set_major_locator(mticker.FixedLocator([-0.5, -0.1, -0.01, 0, 0.01, 0.1, 0.5]))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}%" if v else "0"))
        tag = "" if p["test"] else " (motivating)"
        ax.set_title(f"{SHORT[name]}{tag}: split {sp['month']}, -{100 * sp['reduction']:.0f}%", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.7)
    fig.suptitle(
        "Delivered / outside count - 1, the split at the largest level shift (dashed), each segment's median, and the\n"
        "share of the absolute deviation the split removes",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.99, 0.004, CREDIT, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    fig.savefig(FIG / "fig1_splits.png", dpi=150)
    plt.close(fig)


def band(ax, lo=0.0001, hi=1.0) -> None:
    xs = [lo, hi]
    ax.plot(xs, xs, color=INK2, lw=1)
    ax.fill_between(xs, [x / 2 for x in xs], [2 * x for x in xs], color=ORANGE, alpha=0.08, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(loss_label))
    ax.grid(color=GRID, lw=0.6)


OFFSET = {
    "chicago / vehicles": (8, -12),
    "TLC green / monthly report": (-118, 6),
    "chicago / drivers": (8, -12),
    "citibike / DOT table": (-140, 6),
}


def fig2(pairs: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    band(ax)
    for name, p in pairs.items():
        obs = p["observed_segment"] if p["observed_segment"] is not None else 0.6
        c = color(name)
        ax.annotate(
            "",
            xy=(p["pred_segment"], obs),
            xytext=(p["pred_full"], obs),
            arrowprops={"arrowstyle": "->", "color": c, "lw": 1.2},
        )
        ax.scatter(p["pred_full"], obs, s=40, color=SURFACE, edgecolor=c, lw=1.3, zorder=3)
        ax.scatter(p["pred_segment"], obs, s=70, color=c, edgecolor=c, zorder=4, marker="o" if p["test"] else "D")
        ax.annotate(
            SHORT[name],
            (p["pred_segment"], obs),
            xytext=OFFSET.get(name, (6, 5)),
            textcoords="offset points",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_xlabel("predicted smallest loss caught 90% of the time")
    ax.set_ylabel("observed on the later segment")
    ax.set_title(
        "Hollow: rule on the whole history; filled: rule on the later segment\n"
        "(diamond: the motivating pair; band: within a factor of two)",
        loc="left",
        fontsize=9.5,
    )
    fig.text(0.99, 0.005, CREDIT, ha="right", fontsize=6.3, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "fig2_segment.png", dpi=150)
    plt.close(fig)


def fig3(windows: pd.DataFrame, pairs: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    band(ax)
    for _, w in windows.iterrows():
        obs = w.observed if pd.notna(w.observed) else 0.6
        c = color(w.pair)
        ax.plot([w.pred_expanding, w.pred_trailing], [obs, obs], color=GRID, lw=1, zorder=1)
        ax.scatter(w.pred_expanding, obs, s=24, color=SURFACE, edgecolor=INK2, lw=0.9, zorder=2)
        ax.scatter(
            w.pred_trailing, obs, s=46, color=c if pairs[w.pair]["test"] else SURFACE, edgecolor=c, lw=1.3, zorder=3
        )
    for g, c in GROUP.items():
        ax.scatter([], [], s=46, color=c, label={"citibike": "Citi Bike", "chicago": "Chicago", "TLC": "NYC TLC"}[g])
    ax.scatter([], [], s=24, color=SURFACE, edgecolor=INK2, label="expanding history")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("predicted from the previous 36 months")
    ax.set_ylabel("observed over the next 24 months")
    ax.set_title("Forward windows, one per January (hollow coloured: the motivating pair)", loc="left", fontsize=9.5)
    fig.text(0.99, 0.005, CREDIT, ha="right", fontsize=6.3, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "fig3_forward.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    pairs = json.loads((RES / "pairs.json").read_text(encoding="utf-8"))
    fig1(pd.read_csv(RATIOS), pairs)
    fig2(pairs)
    fig3(pd.read_csv(RES / "windows.csv"), pairs)


if __name__ == "__main__":
    main()
