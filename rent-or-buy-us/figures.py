"""Figures for the rent-or-buy-us article; reads data/ (through rentbuy_core) and results/.

python figures.py     # figures/fig1_*.png ... fig3_*.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

import rentbuy_core as C

HERE = Path(__file__).parent
DATA = Path(os.environ.get("RENTBUY_DATA", HERE / "data"))
RES = HERE / "results"
FIG = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE, MID = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f0efec"
SOURCE = (
    "Data: S&P CoreLogic Case-Shiller (Datahub mirror), BLS CPI rent of primary residence, Census ACS 2023, "
    "Freddie Mac PMMS, Damodaran (2026). Analysis: Diogo Ribeiro."
)

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
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2,
    }
)


def finish(fig, name: str, note: str = "") -> None:
    fig.text(
        0.01,
        0.01,
        (note + "\n" if note else "") + SOURCE,
        fontsize=7.5,
        color=INK2,
        ha="left",
        va="bottom",
    )
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1(d: pd.DataFrame, market: C.Market) -> None:
    """United States, 10-year hold: wealth at sale by purchase year, in 2025 dollars."""
    s = d[(d.area == "United States") & (d.horizon == 10)].set_index("year").sort_index()
    k = market.cpi[2025] / market.cpi.reindex(s.index + 10).to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(
        s.index,
        k * s.owner_disc / 1000,
        color=BLUE,
        marker="o",
        ms=4,
        label="owner (home equity after selling costs, plus any savings)",
    )
    ax.plot(
        s.index,
        k * s.renter_disc / 1000,
        color=ORANGE,
        marker="o",
        ms=4,
        label="renter who invests the whole difference",
    )
    ax.plot(
        s.index,
        k * s.renter_lazy / 1000,
        color=ORANGE,
        linestyle=(0, (4, 2)),
        linewidth=1.6,
        label="renter who invests only the down payment and spends the difference",
    )
    ax.set_ylabel("wealth after 10 years, thousands of 2025 dollars")
    ax.set_xlabel("year of purchase (sale ten years later)")
    ax.set_ylim(bottom=0)
    fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        fontsize=8.8,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.08),
    )
    ax.set_title(
        "United States, a typical home bought with 20% down, held for ten years",
        loc="left",
    )
    fig.suptitle(
        "The year of purchase moves the outcome far more than the renter's saving does",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    finish(
        fig,
        "fig1_us_by_purchase_year",
        "Case-Shiller national index and BLS national rent index, levels from ACS 2023; comparable rent = median contract rent x (rooms ratio)^0.5.",
    )


def fig2(d: pd.DataFrame) -> None:
    """Heat map: 16 metros by purchase year, 10-year hold, owner wealth over disciplined renter wealth."""
    s = d[(d.horizon == 10) & (d.area != "United States")].copy()
    s["v"] = np.log2(s.owner_disc.clip(lower=1) / s.renter_disc.clip(lower=1))
    order = s.groupby("area").v.mean().sort_values().index[::-1]
    grid = s.pivot(index="area", columns="year", values="v").reindex(order)
    cmap = LinearSegmentedColormap.from_list("div", ["#892b2a", "#d75853", "#f1aea8", MID, "#9ec5f4", "#3987e5", "#184f95"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    im = ax.imshow(
        np.ma.masked_invalid(grid.to_numpy()),
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(0, -2, 2),
    )
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xticks(range(0, len(grid.columns), 2), grid.columns[::2])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(
        ["renter 4x richer", "renter 2x", "even", "owner 2x", "owner 4x richer"],
        fontsize=8.5,
    )
    cb.outline.set_visible(False)
    ax.set_xlabel("year of purchase (sale ten years later)")
    ax.set_title(
        "Owner against a renter who invests the whole difference, ten-year hold",
        loc="left",
    )
    fig.suptitle(
        "Vertical stripes: across sixteen cities, the purchase year sets the winner",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    finish(
        fig,
        "fig2_metros_by_purchase_year",
        "Blank: no rent or price series for that year. Cities sorted by the average outcome.",
    )


def fig3(bracket: pd.DataFrame) -> None:
    """Category shares by horizon for the three comparable-rent rules."""
    cats = [
        ("owning wins either way", BLUE),
        ("saving decides", AQUA),
        ("renting wins either way", ORANGE),
        ("saving favours the owner", YELLOW),
    ]
    rules = [
        (0.0, "rent of the median rented home"),
        (0.5, "central: rent scaled by rooms^0.5"),
        (1.0, "rent scaled by rooms"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (e, title) in zip(axes, rules, strict=True):
        sub = bracket[bracket.e == e]
        left = np.zeros(3)
        for cat, col in cats:
            v = np.array([100 * (sub[sub.horizon == h].category == cat).mean() for h in (5, 10, 20)])
            ax.barh(
                ["5 years", "10 years", "20 years"],
                v,
                left=left,
                color=col,
                edgecolor=SURFACE,
                linewidth=2,
                height=0.62,
            )
            for i, (lv, vv) in enumerate(zip(left, v, strict=True)):
                if vv >= 8:
                    ax.text(
                        lv + vv / 2,
                        i,
                        f"{vv:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white" if col != YELLOW else INK,
                    )
            left += v
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        ax.set_xlabel("% of purchase years and cities")
    fig.legend(
        handles=[Patch(color=c, label=n) for n, c in cats],
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        fontsize=9,
    )
    fig.suptitle(
        "Saving decides the winner in a minority of cases, and only over long holds",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    finish(
        fig,
        "fig3_when_saving_decides",
        '16 metropolitan areas. "Saving decides": the owner beats a renter who spends the difference but not one who invests it.',
    )


def main() -> None:
    market = C.load_market(DATA)
    d = pd.read_csv(RES / "cohorts_central.csv")
    fig1(d, market)
    fig2(d)
    frames = []
    for e in (0.0, 0.5, 1.0):
        a = C.with_(C.Assumptions(), rent_elasticity=e)
        for name in C.METROS:
            area = C.load_area(name, DATA, a)
            for h in (5, 10, 20):
                frames.append(C.table(C.cohorts(area, market, h, a)).assign(e=e))
    fig3(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    main()
