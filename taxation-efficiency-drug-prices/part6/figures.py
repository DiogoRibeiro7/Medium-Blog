"""Figures for Part 6. Reads figure_data.json written by innovation_numbers.py.

    python figures.py

Writes two PNG files to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NEUTRAL = "#b9b8ae"

EU_NAMES = {"AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus", "CZ": "Czechia",
            "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France", "DE": "Germany", "EL": "Greece",
            "HU": "Hungary", "IS": "Iceland", "IE": "Ireland", "IT": "Italy", "LV": "Latvia", "LT": "Lithuania",
            "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
            "RO": "Romania", "RS": "Serbia", "SK": "Slovakia", "SI": "Slovenia", "ES": "Spain", "SE": "Sweden",
            "CH": "Switzerland", "TR": "Türkiye", "UK": "United Kingdom", "UA": "Ukraine"}


OFFSETS = {"PT": (0, -16, "center"), "AT": (4, -13, "left"), "ES": (6, 6, "left"), "FR": (-6, -10, "right"),
           "DE": (-6, 6, "right"), "PL": (6, 4, "left"), "RO": (6, 4, "left"), "IT": (-6, 5, "right"),
           "HU": (6, -2, "left"), "CY": (5, 2, "left"), "FI": (-6, 4, "right"), "MT": (6, 3, "left"),
           "NO": (6, 3, "left"), "TR": (6, -6, "left")}


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def note(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def fig_revenue(data: dict) -> None:
    comp = sorted(data["companies_2025"], key=lambda r: r["us_share"])
    tr = data["trend"]
    gdp = data["us_gdp_share"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.32})
    y = np.arange(len(comp))
    colors = [BLUE if r["hq"] == "US" else ORANGE for r in comp]
    a.barh(y, [100 * r["us_share"] for r in comp], height=0.66, color=colors, edgecolor=SURFACE, linewidth=2)
    for yi, r in zip(y, comp):
        a.text(100 * r["us_share"] + 1, yi, f"{100 * r['us_share']:.0f}%", va="center", fontsize=8.5, color=INK2)
    a.set_yticks(y, [f"{r['company']} ({r['hq']})" for r in comp], fontsize=8.5)
    a.tick_params(axis="y", length=0)
    a.grid(axis="y", visible=False)
    a.set_xlim(0, 88)
    a.set_ylim(-1.5, len(comp) - 0.4)
    a.axvline(100 * gdp["2025"], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    a.text(100 * gdp["2025"] + 0.8, -0.95, "US share of world GDP", fontsize=7.5, color=INK2, va="center")
    a.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    a.set_xlabel("Share of revenue earned in the United States, fiscal year 2025")
    a.set_title("By company (blue: headquartered in the US)")
    yrs = tr["years"]
    b.plot(yrs, [100 * v for v in tr["us_share"]], color=BLUE, marker="o", markersize=5, markeredgecolor=SURFACE,
           label=f"US share of revenue, {len(tr['firms'])} companies reporting it every year")
    b.plot(yrs, [100 * gdp[str(y)] for y in yrs], color=INK2, linestyle=(0, (4, 3)), linewidth=1.5,
           label="US share of world GDP")
    b.set_ylim(0, 80)
    b.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    b.set_xticks(yrs[::2])
    b.set_title("Over time, 2015 to 2025")
    b.legend(loc="lower left", fontsize=8.5)
    b.text(yrs[-1], 100 * tr["us_share"][-1] + 3, f"{100 * tr['us_share'][-1]:.0f}%", ha="center", fontsize=8.5, color=INK2)
    b.text(yrs[0], 100 * tr["us_share"][0] + 3, f"{100 * tr['us_share'][0]:.0f}%", ha="center", fontsize=8.5, color=INK2)
    fig.suptitle("Where research-based drug companies earn their revenue: the United States, more than half of it",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold", color=INK)
    note(fig, "Source: annual reports (10-K, 20-F) filed with the SEC, read from their XBRL data; revenue in the United States "
              "as tagged by each company. World Bank GDP in current dollars.\nThe right panel pools AbbVie, Amgen, Bristol-Myers "
              "Squibb, Eli Lilly, Gilead, Merck & Co and Pfizer.")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.17)
    fig.savefig(OUT / "fig1_us_share_of_revenue.png", dpi=200)
    plt.close(fig)


def fig_europe(data: dict) -> None:
    rows = data["europe"]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    label = {"PT", "DE", "FR", "IT", "ES", "UK", "CH", "BE", "DK", "EL", "SE", "NL", "IE", "PL", "BG", "SI", "AT", "RO",
             "MT", "NO", "TR"}
    for r in rows:
        x, y = 100 * r["market_gdp"], 100 * r["rd_gdp"]
        pt = r["iso2"] == "PT"
        ax.scatter(x, y, s=70 if pt else 42, color=ORANGE if pt else BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if r["iso2"] in label:
            dx, dy, ha = OFFSETS.get(r["iso2"], (6, 3, "left"))
            ax.annotate(EU_NAMES.get(r["iso2"], r["iso2"]), (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.5 if not pt else 9.5, color=INK if pt else INK2, ha=ha,
                        fontweight="semibold" if pt else "normal")
    ax.set_xlabel("Pharmaceutical market at ex-factory prices, % of GDP (2024)")
    ax.set_ylabel("Industry R&D carried out in the country, % of GDP (2024 or latest)")
    ax.set_xlim(0.4, 3.0)
    ax.set_ylim(-0.03, 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    ax.set_title("Europe: what each country spends on medicines and how much industry R&D it hosts are unrelated")
    note(fig, "Source: EFPIA, The Pharmaceutical Industry in Figures 2026 (market value and R&D by country, member associations); "
              "Eurostat and World Bank GDP. 27 countries with both figures.")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.13)
    fig.savefig(OUT / "fig2_market_and_research_by_country.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
    fig_revenue(data)
    fig_europe(data)


if __name__ == "__main__":
    main()
