"""Draw the figures from the outputs of ``spending.py``.

Run after ``spending.py``::

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, PURPLE, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d9a21b"
QUINTILE_COLOURS = {"QU1": ORANGE, "QU2": GOLD, "QU3": MUTED, "QU4": AQUA, "QU5": BLUE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def finish(fig: plt.Figure, name: str, source: str) -> None:
    fig.text(0.01, 0.01, source, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def core_share_by_quintile(hb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = {"QU1": "Poorest fifth", "QU2": "Second", "QU3": "Middle", "QU4": "Fourth", "QU5": "Richest fifth"}
    for q, colour in QUINTILE_COLOURS.items():
        s = hb.loc[q, "core_ex_imputed"].dropna()
        ax.plot(s.index, 100 * s.values, color=colour, marker="o", ms=4, label=labels[q])
    ax.set_title("Food, rent and household energy, % of consumption (imputed rent excluded)")
    ax.set_xticks([1988, 1994, 1999, 2005, 2010, 2015])
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    finish(fig, "fig1_core_share_by_quintile.png",
           "Source: Eurostat household budget survey (hbs_str_t223), income quintiles; author's calculations.")


def poorest_composition(hb: pd.DataFrame) -> None:
    q1 = hb.loc["QU1"].dropna(subset=["core_ex_imputed"])
    other_housing = q1["core_ex_imputed"] - q1["food_ex_imputed"] - q1["rent_actual_ex_imputed"] - q1["energy_ex_imputed"]
    parts = [("Food", q1["food_ex_imputed"], ORANGE), ("Actual rent", q1["rent_actual_ex_imputed"], PURPLE),
             ("Electricity, gas, fuels", q1["energy_ex_imputed"], GOLD),
             ("Water, dwelling repairs", other_housing, MUTED)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(q1.index))
    bottom = np.zeros(len(x))
    for label, values, colour in parts:
        ax.bar(x, 100 * values.values, bottom=bottom, color=colour, width=0.6, label=label)
        bottom += 100 * values.values
    ax.set_xticks(x, [str(y) for y in q1.index])
    ax.set_title("Poorest fifth: what the basic-needs share is made of, % of consumption")
    ax.set_ylim(0, 72)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    finish(fig, "fig2_poorest_composition.png",
           "Source: Eurostat household budget survey (hbs_str_t223), imputed rent excluded; author's calculations.")


def prices(price: pd.DataFrame) -> None:
    p = price[(price["start"] == 2015) & (price["wave"] == 2015)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(p.index, 100 * p["basic_needs"], color=ORANGE, label="Food, rent, household energy and services")
    ax.plot(p.index, 100 * p["all_items"], color=BLUE, label="All items")
    ax.set_title("Consumer prices in Portugal, 2015 = 100 (HICP)")
    ax.set_xlim(2015, 2025)
    ax.legend(loc="upper left", fontsize=8.5)
    finish(fig, "fig3_basic_needs_prices.png",
           "Source: Eurostat HICP (prc_hicp_aind), weights from the 2015 household budget survey; author's calculations.")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    hb = pd.read_csv(RESULTS / "hbs_shares_by_quintile.csv", index_col=["quintile", "year"])
    price = pd.read_csv(RESULTS / "basic_needs_price.csv", index_col="year")
    core_share_by_quintile(hb)
    poorest_composition(hb)
    prices(price)


if __name__ == "__main__":
    main()
