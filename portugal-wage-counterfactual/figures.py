"""Draw the figures from the outputs of ``wage_counterfactual.py``.

Run after ``wage_counterfactual.py``::

    python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SOURCE = "Sources: AMECO (2018 vintage) to 1995, Eurostat 1995–2024, World Bank CPI; author's calculations."
SOURCE_QP = ("Sources: AMECO (2018 vintage) to 1995, Eurostat 1995–2024, World Bank CPI, "
             "GEP/MTSSS Quadros de Pessoal; author's calculations.")

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
})


def finish(fig: plt.Figure, name: str, source: str = SOURCE) -> None:
    fig.text(0.01, 0.01, source, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def base_year_curve(sens: pd.DataFrame, actual: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sens["base"], actual * sens["cf_over_actual"], color=ORANGE,
            label="Counterfactual: base-year wage + CPI inflation + productivity per person")
    ax.plot(sens["base"], actual * sens["cf_over_actual_mult"], color=ORANGE, lw=1.0,
            ls="--", label="Same, multiplicative rule")
    ax.plot(sens["base"], actual * sens["cf_over_actual_hour"], color=ORANGE, lw=1.0,
            ls=":", label="Same, productivity per hour")
    ax.plot(sens["base"], actual * sens["cf_over_actual_no_cut"], color=AQUA,
            label="Same, but the wage never falls from one year to the next")
    ax.axhline(actual, color=BLUE, label=f"Actual (€{actual:,.0f})")
    ax.set_title("Mean monthly earnings in 2024 implied by each base year, €")
    ax.set_xlabel("Base year")
    ax.set_xlim(1975, 2023)
    ax.legend(loc="upper right", fontsize=8.5)
    finish(fig, "fig2_base_year_curve.png", SOURCE_QP)


def decomposition(dec: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.plot(dec["base"], dec["log_points_wage_share"], color=AQUA,
            label="Fall in the adjusted wage share since the base year")
    ax.plot(dec["base"], dec["log_points_price_wedge"], color=BLUE,
            label="CPI inflation minus GDP-deflator inflation")
    ax.plot(dec["base"], dec["log_points_total"], color=INK, lw=1.4, ls="--",
            label="Total: counterfactual over actual")
    ax.set_title("Where the gap comes from, by base year (log points, multiplicative rule)")
    ax.set_xlabel("Base year")
    ax.set_xlim(1975, 2023)
    ax.legend(loc="upper right", fontsize=8.5)
    finish(fig, "fig3_decomposition.png")


def path_from_1975(path: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(path.index, path["actual_eur_month"], color=BLUE, label="Actual")
    ax.plot(path.index, path["counterfactual_eur_month"], color=ORANGE,
            label="1975 level + CPI inflation + productivity per person")
    ax.set_yscale("log")
    ticks = [50, 100, 200, 500, 1000, 2000, 3000]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:,}" for v in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("Compensation per employee, € per month (annual ÷ 14), log scale")
    ax.set_xlim(1975, 2024)
    ax.legend(loc="lower right", fontsize=8.5)
    finish(fig, "fig1_path_from_1975.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    sens = pd.read_csv(RESULTS / "base_year_sensitivity.csv")
    dec = pd.read_csv(RESULTS / "decomposition_by_base_year.csv")
    path = pd.read_csv(RESULTS / "path_from_1975.csv", index_col="year")
    qp = pd.read_csv(HERE / "data" / "qp_earnings.csv", index_col="year")
    base_year_curve(sens, qp.loc[2024, "qp_mean_eur"])
    decomposition(dec)
    path_from_1975(path)


if __name__ == "__main__":
    main()
