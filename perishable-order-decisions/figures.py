"""Figures for the article, from results/.

python figures.py --results results --out figures

fig1  when products run out: hourly sales share and out-of-stock share, and daily sales against recovered demand
fig2  the 72 costs: four algorithms on each of six decision layers, at three cost ratios
fig3  how the main comparisons move with the strength of the recovery (descriptive, added after the results)
fig4  coverage of the orders: share of series-days with demand at or below the order
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
ALGO = {
    "naive7": ("#2a78d6", "o", "same weekday last week"),
    "ma4w": ("#eb6834", "s", "mean of last 4 same weekdays"),
    "lgbm_l2": ("#1baf7a", "^", "LightGBM, squared error"),
    "lgbm_tweedie": ("#eda100", "D", "LightGBM, Tweedie"),
}
LAYERS = [
    ("sales", "point"),
    ("sales", "pooled"),
    ("sales", "series"),
    ("recovered", "point"),
    ("recovered", "pooled"),
    ("recovered", "series"),
]
LAYER_LABEL = {
    ("sales", "point"): "sales · order the forecast (default)",
    ("sales", "pooled"): "sales · pooled quantile",
    ("sales", "series"): "sales · per-series quantile",
    ("recovered", "point"): "recovered · order the forecast",
    ("recovered", "pooled"): "recovered · pooled quantile",
    ("recovered", "series"): "recovered · per-series quantile (full)",
}
CREDIT = "Diogo Ribeiro · github.com/DiogoRibeiro7/Medium-Blog · data: FreshRetailNet-50K (Dingdong-Inc, CC BY 4.0)"

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


def fig1(res: Path, out: Path) -> None:
    hours = read_csv(res / "hours.csv")
    days = read_csv(res / "days.csv")
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.22)
    h = np.array([int(x["hour"]) for x in hours])
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(h, [100 * float(x["sales_share_clean_days"]) for x in hours], color="#2a78d6", width=0.8)
    ax.set_title("Share of a day's sales in each hour\n(fully in-stock days, fit window)", loc="left")
    ax.set_ylabel("% of daily sales")
    ax.set_xlabel("hour of the day")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(h, [100 * float(x["out_share_fit"]) for x in hours], color="#eb6834", width=0.8)
    ax.set_title("Share of series-days out of stock in each hour\n(fit window)", loc="left")
    ax.set_ylabel("% of series-days")
    ax.set_xlabel("hour of the day")
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax = fig.add_subplot(gs[1, :])
    t = np.arange(len(days))
    sales = np.array([float(x["sales"]) for x in days])
    rec = np.array([float(x["recovered"]) for x in days])
    scale = sales[: len(days) - 7].mean()
    ax.plot(t, sales / scale, color=INK2, lw=1.6, label="recorded sales")
    ax.plot(t, rec / scale, color="#1baf7a", lw=2.0, label="recovered demand")
    for name, lo, hi in (("fit", 0, 75), ("calibration", 76, 89), ("test", 90, 96)):
        if name == "test":
            ax.axvspan(lo - 0.5, hi + 0.5, color=GRID, alpha=0.8, lw=0)
        if name != "fit":
            ax.axvline(lo - 0.5, color=INK2, lw=0.8, ls=":")
        ax.text(
            (lo + hi) / 2,
            1.0,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=INK2,
            fontsize=9,
        )
    ticks = [0, 30, 60, 90]
    ax.set_xticks(ticks, [days[i]["date"] for i in ticks])
    ax.set_ylabel("total over 50,000 series\n(1 = mean daily sales, days 0-89)")
    ax.set_title("Daily totals: recorded sales and recovered demand", loc="left", pad=16)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylim(bottom=0)
    credit(fig)
    fig.savefig(out / "fig1_stockouts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2(res: Path, out: Path) -> None:
    rows = read_csv(res / "cells.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True, sharex=True)
    y = np.arange(len(LAYERS))[::-1]
    for ax, tau in zip(axes, ("0.5", "0.7", "0.9"), strict=True):
        for yi, layer in zip(y, LAYERS, strict=True):
            ax.axhline(yi, color=GRID, lw=0.6, zorder=0)
            for k, (a, (col, mk, _)) in enumerate(ALGO.items()):
                x = next(r for r in rows if (r["algo"], r["target"], r["rule"], r["tau"]) == (a, *layer, tau))
                yy = yi + (1.5 - k) * 0.12
                ax.scatter(float(x["cost"]), yy, color=col, marker=mk, s=40, edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.set_title(f"cost ratio {tau}", loc="left")
        ax.set_xlabel("cost per unit of demand")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [LAYER_LABEL[layer] for layer in LAYERS])
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markersize=7, label=lab) for c, m, lab in ALGO.values()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.04))
    fig.tight_layout()
    credit(fig)
    fig.savefig(out / "fig2_costs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3(res: Path, out: Path) -> None:
    js = json.loads((res / "details.json").read_text(encoding="utf-8"))
    series = [
        ("H1", "lgbm_l2 default - ma4w full", "LightGBM (default layer) minus\n4-week mean (full layer)", "#2a78d6"),
        (
            "H3",
            "lgbm_l2 series: sales - recovered",
            "LightGBM, per-series quantile:\ntrained on sales minus recovered",
            "#eb6834",
        ),
        (
            "H3",
            "lgbm_l2 point: sales - recovered",
            "LightGBM, order the forecast:\ntrained on sales minus recovered",
            "#1baf7a",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
    s = [e["s"] for e in js["strength"]]
    for ax, tau in zip(axes, (0.7, 0.9), strict=True):
        for h, label, text, col in series:
            pts = [
                next(
                    x for x in e["comparisons"] if x["hypothesis"] == h and x["comparison"] == label and x["tau"] == tau
                )
                for e in js["strength"]
            ]
            est = [p["estimate"] for p in pts]
            ax.fill_between(s, [p["lo"] for p in pts], [p["hi"] for p in pts], color=col, alpha=0.18, lw=0)
            ax.plot(s, est, color=col, lw=2, marker="o", markersize=4, label=text)
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_title(f"cost ratio {tau}", loc="left")
        ax.set_xlabel("strength of the recovery (0 = recorded sales are demand, 1 = registered)")
        ax.set_ylabel("difference in cost per unit of demand\n(above 0: the second option is cheaper)")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(s)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    credit(fig)
    fig.savefig(out / "fig3_strength.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4(res: Path, out: Path) -> None:
    rows = read_csv(res / "cells.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    x = np.arange(len(LAYERS))
    for ax, tau in zip(axes, ("0.7", "0.9"), strict=True):
        ax.axhline(100 * float(tau), color=INK2, lw=1, ls="--")
        ax.text(
            len(LAYERS) - 0.5,
            100 * float(tau) + 1,
            f"target {100 * float(tau):.0f}%",
            ha="right",
            color=INK2,
            fontsize=8.5,
        )
        for k, (a, (col, mk, _)) in enumerate(ALGO.items()):
            vals = [
                100
                * float(
                    next(r for r in rows if (r["algo"], r["target"], r["rule"], r["tau"]) == (a, *layer, tau))[
                        "coverage"
                    ]
                )
                for layer in LAYERS
            ]
            ax.scatter(x + (k - 1.5) * 0.12, vals, color=col, marker=mk, s=38, edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.set_xticks(
            x,
            [
                "sales\nforecast",
                "sales\npooled",
                "sales\nper-series",
                "recovered\nforecast",
                "recovered\npooled",
                "recovered\nper-series",
            ],
            fontsize=8.5,
        )
        ax.set_title(f"cost ratio {tau}", loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of test series-days with demand\nat or below the order")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markersize=7, label=lab) for c, m, lab in ALGO.values()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    credit(fig)
    fig.savefig(out / "fig4_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig1(a.results, a.out)
    fig2(a.results, a.out)
    if (a.results / "details.json").exists():
        fig3(a.results, a.out)
    fig4(a.results, a.out)


if __name__ == "__main__":
    main()
