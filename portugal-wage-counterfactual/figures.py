"""Draw the figure from the outputs of ``wage_counterfactual.py``.

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
BLUE, ORANGE = "#2a78d6", "#eb6834"

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


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    path = pd.read_csv(RESULTS / "path_from_1975.csv", index_col="year")
    sens = pd.read_csv(RESULTS / "base_year_sensitivity.csv")
    qp = pd.read_csv(HERE / "data" / "qp_earnings.csv", index_col="year")
    actual_2024 = qp.loc[2024, "qp_mean_eur"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(path.index, path["actual_eur_month"], color=BLUE, label="Actual")
    ax1.plot(path.index, path["counterfactual_eur_month"], color=ORANGE,
             label="1975 level + CPI inflation + productivity")
    ax1.set_yscale("log")
    ticks = [50, 100, 200, 500, 1000, 2000, 3000]
    ax1.yaxis.set_major_locator(FixedLocator(ticks))
    ax1.yaxis.set_major_formatter(FixedFormatter([f"{v:,}" for v in ticks]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_title("Compensation per employee, € per month (×14)")
    ax1.legend(loc="lower right")
    ax1.set_xlim(1975, 2024)

    ax2.plot(sens["base"], sens["qp_mean_cf"], color=ORANGE, label="Counterfactual 2024 mean")
    ax2.axhline(actual_2024, color=BLUE, label=f"Actual 2024 mean (€{actual_2024:,.0f})")
    ax2.set_title("2024 mean monthly earnings by base year, €")
    ax2.set_xlabel("Base year")
    ax2.legend(loc="upper right")
    ax2.set_xlim(1975, 2023)

    fig.text(0.01, 0.01,
             "Sources: AMECO (2018 vintage) to 1995, Eurostat 1995–2024, World Bank CPI, "
             "GEP/MTSSS Quadros de Pessoal; author's calculations.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "fig1_counterfactual_wages.png", dpi=200)


if __name__ == "__main__":
    main()
