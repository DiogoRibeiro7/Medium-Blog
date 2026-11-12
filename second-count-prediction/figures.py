"""Figures for the article, from ``results/``.

    python figures.py

fig1  the two counts of each test pair, month by month: delivered / outside - 1 (Citi Bike as first counted,
      with the registered rule, and as counted again with the corrected rule)
fig2  detection against loss / predicted loss, and predicted against observed (post hoc counts for Citi Bike)
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


def loss_label(x, _=None):
    return f"{100 * x:g}%"


def fig1(reg: pd.DataFrame, cor: pd.DataFrame) -> None:
    """The two counts of each test pair, month by month, on a scale that is linear within +-1%."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharey=True)
    for ax, pair in zip(axes.flat, TEST, strict=True):
        for frame, color, lw, label, z in (
            (reg, INK2, 1.0, "files as first counted (registered rule)", 1),
            (cor, BLUE, 1.8, "files counted again (corrected rule)", 2),
        ):
            d = frame[frame.pair == pair]
            if pair.startswith("chicago") and frame is reg:
                continue
            t = pd.to_datetime(d.month + "-01")
            y = d.log_ratio.astype(float).map(math.expm1)
            ax.plot(t, y, color=color, lw=lw, ls="--" if frame is reg else "-", label=label, zorder=z)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.2)
        ax.set_ylim(-1.0, 1.3)
        ax.yaxis.set_major_locator(mticker.FixedLocator([-0.9, -0.5, -0.1, -0.01, 0, 0.01, 0.1, 0.5, 1.0]))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}%" if v else "0"))
        ax.set_title(TITLE[pair], loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("delivered / outside count - 1")
    fig.suptitle(
        "Delivered rows against a second count, month by month (linear within ±1%, logarithmic beyond)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.99, 0.005, CREDIT, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(FIG / "fig1_agreement.png", dpi=150)
    plt.close(fig)


def fig2(curves: pd.DataFrame, pairs: dict) -> None:
    """Left: detection against loss / predicted loss. Right: predicted against observed, both rules."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    colors = dict(zip(TEST, (BLUE, AQUA, ORANGE, VIOLET), strict=True))
    for name, p in pairs.items():
        d = curves[curves.pair == name]
        x = d.loss / p["predicted_p95_rule"]
        if p["test"]:
            ax.plot(x, d.detected, color=colors[name], lw=2, label=TITLE[name])
        else:
            ax.plot(x, d.detected, color=GRID, lw=1.2, zorder=0)
    ax.axvspan(0.5, 2, color=ORANGE, alpha=0.08, lw=0)
    ax.axvline(1, color=ORANGE, lw=1)
    ax.axhline(0.9, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 30)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("rows lost, as a multiple of the loss the p95 rule predicts")
    ax.set_ylabel("faulted months caught (5% alarm budget)")
    ax.set_title("Detection against the prediction (grey: Part 4's TLC pairs)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(color=GRID, lw=0.6)

    lo, hi = 0.0001, 1.0
    xs = [lo, hi]
    bx.plot(xs, xs, color=INK2, lw=1)
    bx.fill_between(xs, [x / 2 for x in xs], [2 * x for x in xs], color=ORANGE, alpha=0.08, lw=0)
    for name, p in pairs.items():
        obs = p["observed_l90"]
        c = colors.get(name, INK2)
        bx.plot([p["predicted_gaussian_rule"], p["predicted_p95_rule"]], [obs, obs], color=GRID, lw=1, zorder=1)
        bx.scatter(p["predicted_p95_rule"], obs, s=70, zorder=3, color=c if p["test"] else SURFACE, edgecolor=c, lw=1.5)
        bx.scatter(p["predicted_gaussian_rule"], obs, s=36, zorder=2, marker="s", color=SURFACE, edgecolor=c, lw=1.2)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(lo, hi)
    bx.set_ylim(lo, hi)
    for axis in (bx.xaxis, bx.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(loss_label))
    bx.set_xlabel("predicted smallest loss caught 90% of the time")
    bx.set_ylabel("observed")
    bx.set_title("Circles: p95 rule; squares: Gaussian rule; band: factor of two", loc="left", fontsize=10)
    bx.grid(color=GRID, lw=0.6)
    fig.suptitle(
        "Does agreement on clean months predict the smallest loss a second count catches?",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(0.99, 0.005, CREDIT, ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(FIG / "fig2_prediction.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    for old in ("fig1_agreement.png", "fig2_detection.png", "fig3_predicted_observed.png"):
        (FIG / old).unlink(missing_ok=True)
    cor_dir = RES / "corrected" if (RES / "corrected").exists() else RES
    reg = pd.read_csv(RES / "ratios.csv")
    cor = pd.read_csv(cor_dir / "ratios.csv")
    curves = pd.read_csv(cor_dir / "curves.csv")
    pairs = json.loads((cor_dir / "pairs.json").read_text(encoding="utf-8"))
    fig1(reg, cor)
    fig2(curves, pairs)


if __name__ == "__main__":
    main()
