"""The four figures of the article, drawn from data/ and results/ (run panel.py and run_analysis.py first).

python figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import transcribed as tr

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

INK, INK2, GRID = "#191919", "#666666", "#e3e3e3"
COLOURS = {"lisboa": "#257de0", "porto": "#f47230", "mst": "#00b894"}
LABELS = {"lisboa": "Metropolitano de Lisboa", "porto": "Metro do Porto", "mst": "Metro Sul do Tejo"}
SOURCE_INE = "Source: INE, Inquérito ao transporte por metropolitano. Chart: Diogo Ribeiro."

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def footer(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8, color=INK2)


def fig1_long_series() -> None:
    a = pd.read_csv(RESULTS / "annual_passengers.csv", index_col=0).loc[:2025]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for k in ("lisboa", "porto", "mst"):
        s = a[k].where(a[k] > 0)
        ax.plot(s.index, s.values, color=COLOURS[k], lw=2, marker="o", ms=4, label=LABELS[k])
    ax.axvspan(2019.5, 2021.5, color="#f2f2f2", zorder=0)
    ax.text(2020.5, 196, "pandemic", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.annotate(
        "2019: 183.1",
        xy=(2019, a.loc[2019, "lisboa"]),
        xytext=(2015.2, 196),
        fontsize=8.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
    )
    ax.annotate(
        "2025: 173.5",
        xy=(2025, a.loc[2025, "lisboa"]),
        xytext=(2022.6, 196),
        fontsize=8.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
    )
    ax.text(2025.3, a.loc[2025, "porto"], "94.7", va="center", fontsize=8.5, color=INK)
    ax.text(2025.3, a.loc[2025, "mst"], "20.0", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(2003.5, 2026.4)
    ax.set_ylim(0, 200)
    ax.set_ylabel("Passengers a year (millions)")
    ax.set_title("Passengers carried by Portugal's three metro systems, 2004–2025", loc="left", fontsize=12, pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=9)
    footer(fig, SOURCE_INE + " Metro Sul do Tejo opened in stages from 2007; INE's series starts in 2014.")
    fig.savefig(FIGURES / "fig1_three_systems.png")
    plt.close(fig)


def fig2_decomposition() -> None:
    d = pd.read_csv(RESULTS / "decomposition.csv")
    parts = [
        ("raw", "Change in passengers", "#9a9a9a"),
        ("calendar", "Calendar (weekdays, holidays)", COLOURS["lisboa"]),
        ("closures", "Days the network was shut", COLOURS["porto"]),
        ("remainder", "Change per day open", COLOURS["mst"]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.19
    for i, (_, row) in enumerate(d.iterrows()):
        for j, (col, lab, colour) in enumerate(parts):
            x = i + (j - 1.5) * (width + 0.02)
            v = row[col]
            ax.bar(x, v, width=width, color=colour, label=lab if i == 0 else None)
            ax.text(x, v + (0.12 if v >= 0 else -0.12), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(d)), ["2025 against 2024", "January–August 2026 against 2025"])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-5.6, 2.3)
    ax.set_ylabel("Percent / percentage points")
    ax.set_title("Metropolitano de Lisboa: how much of the fall the calendar and the closures explain", loc="left", fontsize=12, pad=12)
    ax.legend(ncol=2, loc="upper left", fontsize=8.5)
    footer(fig, SOURCE_INE + " Closures from operator and press notices; day weights estimated from 2015–2019 and 2023–2024.")
    fig.savefig(FIGURES / "fig2_calendar_closures.png")
    plt.close(fig)


def fig3_quarters_and_fleet() -> None:
    q = pd.read_csv(RESULTS / "quarterly_changes.csv")
    k = pd.read_csv(RESULTS / "kpi_quarters.csv")
    labels = [s.replace(" (Jul-Aug)", "*").replace(" Q", "\nQ") for s in q.quarter]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    x = range(len(q))
    for key in ("lisboa", "porto", "mst"):
        ax1.plot(x, q[key], color=COLOURS[key], lw=2, marker="o", ms=5, label=LABELS[key])
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_ylabel("Change on a year earlier, per day open (%)")
    ax1.set_title("Passengers by quarter, and the Lisbon fleet that was available to carry them", loc="left", fontsize=12, pad=12)
    ax1.legend(loc="upper right", fontsize=8.5)
    pos = {s: i for i, s in enumerate(q.quarter)}
    xs = [pos[s] for s in k.quarter]
    ax2.plot(xs, k.DMC, color=COLOURS["lisboa"], lw=2, marker="o", ms=5, label="Fleet availability")
    ax2.plot(xs, k.REG, color="#6a4ce2", lw=2, marker="s", ms=5, label="Trains within the scheduled interval")
    ax2.axhline(tr.KPI_REFERENCE["DMC"], color=COLOURS["lisboa"], lw=1, ls="--")
    ax2.axhline(tr.KPI_REFERENCE["REG"], color="#6a4ce2", lw=1, ls="--")
    ax2.text(-0.3, tr.KPI_REFERENCE["DMC"] - 0.5, "fleet availability, contract reference 85.89", fontsize=8, color=INK2, ha="left", va="top")
    ax2.text(-0.3, tr.KPI_REFERENCE["REG"] + 0.4, "interval, contract reference 80.00", fontsize=8, color=INK2, ha="left", va="bottom")
    ax2.set_ylim(72, 93)
    ax2.set_ylabel("Metropolitano de Lisboa (%)")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.set_xticks(list(x), labels)
    ax2.text(-0.3, 73, "the contract indicators start in 2025", fontsize=8, color=INK2, va="bottom")
    footer(
        fig,
        "Sources: INE (passengers); Metropolitano de Lisboa, contract reports for 2025 and Q1–Q2 2026 (indicators). "
        "* July–August only. Chart: Diogo Ribeiro.",
    )
    fig.savefig(FIGURES / "fig3_quarters_fleet.png")
    plt.close(fig)


def fig4_metro_area() -> None:
    c = pd.read_csv(RESULTS / "aml_changes_2025.csv").sort_values("change_m")
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    colours = [COLOURS["lisboa"] if v > 0 else COLOURS["porto"] for v in c.change_m]
    ax.barh(c["mode"], c.change_m, color=colours, height=0.6)
    for y, v in enumerate(c.change_m):
        ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-6, 19)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in trips, 2025 against 2024 (millions)")
    ax.set_title("Public transport in the Lisbon metro area, 2025: where trips were gained and lost", loc="left", fontsize=12, pad=12)
    footer(
        fig,
        "Sources: INE (buses by region of origin, metros, Tagus ferries); CP via 24notícias (suburban trains, validations). "
        "Units differ by survey. Chart: Diogo Ribeiro.",
    )
    fig.savefig(FIGURES / "fig4_metro_area_2025.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig1_long_series()
    fig2_decomposition()
    fig3_quarters_and_fleet()
    fig4_metro_area()
    print("wrote figures/fig1_three_systems.png, fig2_calendar_closures.png, fig3_quarters_fleet.png, fig4_metro_area_2025.png")


if __name__ == "__main__":
    main()
