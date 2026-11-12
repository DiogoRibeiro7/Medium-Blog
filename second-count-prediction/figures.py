"""Figures for the article, from ``results/``.

    python figures.py

fig1  the two counts of each test pair, month by month: delivered / outside - 1
fig2  share of random losses caught, by loss, for the four test pairs, with the two predictions marked
fig3  predicted against observed smallest loss caught 90% of the time, test and calibration pairs
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
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc4"
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog · data: Citi Bike, NYC DOT, City of Chicago, NYC TLC"
TEST = ("citibike / operator reports", "citibike / DOT table", "chicago / vehicles", "chicago / drivers")
TITLE = {
    "citibike / operator reports": "Citi Bike files vs operator's monthly reports",
    "citibike / DOT table": "Citi Bike files vs NYC DOT monthly table",
    "chicago / vehicles": "Chicago rideshare trips vs vehicle records",
    "chicago / drivers": "Chicago rideshare trips vs driver records",
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
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)


def pct(x, _=None):
    return f"{100 * x:+.0f}%" if abs(x) >= 0.01 or x == 0 else f"{100 * x:+.1f}%"


def loss_label(x, _=None):
    return f"{100 * x:.2g}%"


def fig1(ratios: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=False)
    for ax, pair in zip(axes.flat, TEST, strict=True):
        d = ratios[ratios.pair == pair].copy()
        d["t"] = pd.to_datetime(d.month + "-01")
        y = (d.log_ratio.astype(float)).map(math.expm1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.plot(d.t, y, color=BLUE, lw=1.8, marker="o", ms=2.5)
        ax.set_title(TITLE[pair], loc="left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct))
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0, 0].set_ylabel("delivered vs outside count")
    axes[1, 0].set_ylabel("delivered vs outside count")
    fig.suptitle(
        "The two counts of each test pair, month by month (delivered / outside - 1)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.99, 0.005, CREDIT, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(FIG / "fig1_agreement.png", dpi=150)
    plt.close(fig)


def fig2(curves: pd.DataFrame, pairs: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
    for ax, pair in zip(axes, TEST, strict=True):
        d = curves[curves.pair == pair]
        p = pairs[pair]
        ax.axhline(0.9, color=INK2, lw=0.8, ls=":")
        ax.plot(d.loss, d.detected, color=BLUE, lw=2, marker="o", ms=3.5, label="caught (5% alarm budget)")
        ax.axvline(p["predicted_p95_rule"], color=ORANGE, lw=1.6, label="p95 rule")
        ax.axvspan(p["predicted_p95_rule"] / 2, min(2 * p["predicted_p95_rule"], 0.6), color=ORANGE, alpha=0.08)
        ax.axvline(p["predicted_gaussian_rule"], color=VIOLET, lw=1.4, ls="--", label="Gaussian rule")
        ax.set_xscale("log")
        ax.set_xlim(0.0009, 0.55)
        ax.xaxis.set_major_locator(mticker.FixedLocator([0.001, 0.01, 0.1, 0.5]))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(loss_label))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(TITLE[pair].replace(" vs ", "\nvs "), loc="left", fontsize=10)
        ax.set_xlabel("rows lost at random")
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("faulted months caught")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Random losses caught by the outside count, with the two predictions (shaded: within a factor of two "
        "of the p95 rule)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.99, 0.005, CREDIT, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(FIG / "fig2_detection.png", dpi=150)
    plt.close(fig)


def fig3(pairs: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lo, hi = 0.0003, 1.0
    xs = [lo, hi]
    ax.plot(xs, xs, color=INK2, lw=1)
    ax.fill_between(xs, [x / 2 for x in xs], [2 * x for x in xs], color=ORANGE, alpha=0.08, lw=0)
    for name, p in pairs.items():
        obs = p["observed_l90"] if p["observed_l90"] is not None else 0.6
        test = p["test"]
        ax.plot([p["predicted_gaussian_rule"], p["predicted_p95_rule"]], [obs, obs], color=GRID, lw=1, zorder=1)
        ax.scatter(
            p["predicted_p95_rule"], obs, s=60, zorder=3, color=ORANGE if test else SURFACE, edgecolor=ORANGE, lw=1.5
        )
        ax.scatter(
            p["predicted_gaussian_rule"],
            obs,
            s=40,
            zorder=2,
            marker="s",
            color=VIOLET if test else SURFACE,
            edgecolor=VIOLET,
            lw=1.2,
        )
        ax.annotate(
            name.replace(" / ", " vs ").replace("monthly report", "report"),
            (p["predicted_p95_rule"], obs),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(loss_label))
    ax.set_xlabel("predicted smallest loss caught 90% of the time")
    ax.set_ylabel("observed smallest loss caught 90% of the time")
    ax.grid(color=GRID, lw=0.6)
    ax.set_title(
        "Predicted against observed (circles: p95 rule; squares: Gaussian rule;\nfilled: test pairs; "
        "hollow: Part 4's pairs; band: within a factor of two)",
        loc="left",
        fontsize=9.5,
    )
    fig.text(0.99, 0.005, CREDIT, ha="right", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "fig3_predicted_observed.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    ratios = pd.read_csv(RES / "ratios.csv")
    curves = pd.read_csv(RES / "curves.csv")
    pairs = json.loads((RES / "pairs.json").read_text(encoding="utf-8"))
    fig1(ratios)
    fig2(curves, pairs)
    fig3(pairs)


if __name__ == "__main__":
    main()
