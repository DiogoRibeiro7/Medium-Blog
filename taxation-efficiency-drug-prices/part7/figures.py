"""Figures for Part 7. Reads figure_data.json written by ledger_numbers.py.

    python figures.py

Writes three PNG files to figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1f9e76"

NAMES = {"AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czechia",
         "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France",
         "HU": "Hungary", "IS": "Iceland", "IT": "Italy", "LT": "Lithuania", "LV": "Latvia", "NL": "Netherlands",
         "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden", "SK": "Slovakia",
         "UK": "United Kingdom"}


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 10,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def note(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def fig_decomposition(data: dict) -> None:
    rows = [r for r in data["decomposition"] if r["r"] == max(x["r"] for x in data["decomposition"])]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        other = 100 * (r["borrowed"] + r["premium"])
        own = 100 * r["own"]
        ax.barh(yi, other, height=0.56, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.barh(yi, own, left=other, height=0.56, color=ORANGE, edgecolor=SURFACE, linewidth=2)
        ax.text(other / 2, yi, f"{other:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="semibold")
        ax.text(100 + 1.5, yi, f"{own:.0f}%", ha="left", va="center", fontsize=9, color=INK2)
        ax.text(-1.5, yi - 0.36, f"US price {r['us_pt_net']:.1f} times Portugal's net price", ha="right", va="center",
                fontsize=7.5, color=MUTED)
    ax.set_yticks(y, [r["benchmark"].replace(" / ", " against ") for r in rows], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Share of the gap between the US price and Portugal's net price (log scale of the price ratio)")
    ax.bar(0, 0, color=BLUE, label="Other countries' prices, carried into Portugal by the reference rule")
    ax.bar(0, 0, color=ORANGE, label="The SNS's own returns from industry (upper Infarmed estimate, 13.6% of spending)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=8.5)
    ax.set_title("Where Portugal's price advantage over the United States comes from, under four US benchmarks")
    note(fig, "Part 2 index (US Part D and French prices, single-source tablets and capsules, 2024); Part 3 rebuilt reference "
              "rule; RAND 2024 (US against 33 countries);\nInfarmed's estimate of industry returns for 2025. The rule sets "
              "Portugal's list price 3.7% above France's, which narrows the gap by 2 to 3 points;\nit is netted in the blue bars.")
    fig.subplots_adjust(left=0.25, right=0.96, top=0.9, bottom=0.4)
    fig.savefig(OUT / "fig1_where_the_price_gap_comes_from.png", dpi=200)
    plt.close(fig)


def fig_size(data: dict) -> None:
    pts = data["size"]["points"]
    b0, b_pop, b_aic = data["size"]["beta"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"wspace": 0.25})
    label_a = {"PT", "FR", "ES", "IT", "UK", "PL", "NL", "CH", "DK", "IS", "RO", "CY", "LV"}
    label_b = {"PT", "FR", "ES", "IT", "UK", "PL", "NL", "CH", "DK", "BE", "RO", "EL", "LV"}
    for p in pts:
        pt = p["geo"] == "PT"
        x1 = np.log10(p["pop"])
        a.scatter(p["pop"] / 1e6, 100 * np.exp(p["part_pop"] - np.mean([q["part_pop"] for q in pts])),
                  s=70 if pt else 40, color=ORANGE if pt else BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        b.scatter(p["aic"], 100 * np.exp(p["part_aic"] - np.mean([q["part_aic"] for q in pts])),
                  s=70 if pt else 40, color=ORANGE if pt else BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for ax, lab, xv, yv in ((a, label_a, p["pop"] / 1e6, p["part_pop"] - np.mean([q["part_pop"] for q in pts])),
                                (b, label_b, p["aic"], p["part_aic"] - np.mean([q["part_aic"] for q in pts]))):
            if p["geo"] in lab:
                ax.annotate(NAMES.get(p["geo"], p["geo"]), (xv, 100 * np.exp(yv)), xytext=(5, 3),
                            textcoords="offset points", fontsize=8 if not pt else 9, color=INK if pt else INK2,
                            fontweight="semibold" if pt else "normal")
        del x1
    mp = np.mean([q["part_pop"] for q in pts])
    ma = np.mean([q["part_aic"] for q in pts])
    xs = np.logspace(np.log10(0.3), np.log10(80), 50)
    a.plot(xs, 100 * np.exp(b_pop * np.log10(xs * 1e6) - mp), color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    xa = np.linspace(68, 128, 50)
    b.plot(xa, 100 * np.exp(b_aic * np.log(xa) - ma), color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    a.set_xscale("log")
    a.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
    a.xaxis.set_minor_locator(NullLocator())
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_xlabel("Population, millions (2025, log scale)")
    a.set_ylabel("Price level, other variable held fixed (mean = 100)")
    a.set_title(f"Size: a market ten times larger is {100 * (1 - np.exp(b_pop)):.0f}% cheaper")
    b.set_xlabel("Actual individual consumption per head, PPS (EU27 = 100, 2024)")
    b.set_title(f"Income: 10% richer, {100 * (np.exp(b_aic * np.log(1.1)) - 1):+.0f}% on the price level")
    fig.suptitle("European prices of medicines follow income more than the size of the buyer",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold", color=INK)
    note(fig, "EURIPID price level index of medicines, 2017, 25 countries (Part 4); Eurostat population and actual individual "
              "consumption. Each panel shows the price level after removing the\neffect of the other variable in a regression "
              "of the log price level on log population and log consumption per head; dashed lines are the fitted effects.")
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.19)
    fig.savefig(OUT / "fig2_price_level_size_and_income.png", dpi=200)
    plt.close(fig)


def fig_ledger(data: dict) -> None:
    rows = data["ledger"]
    pop = data["pop_pt"]
    colors = [INK2, BLUE, ORANGE, ORANGE, MUTED, GREEN, INK2]
    labels = ["Saving against US prices, medicines\nwithout generic competition",
              "  of which other countries' prices,\n  through the reference rule",
              "  of which the SNS's own returns",
              "Industry returns to the SNS,\nall medicines (Infarmed)",
              "What France's size would be worth\nto Portugal in list prices",
              "Spending deferred by the extra wait\n(upper end) = health forgone",
              "Administration: Portugal's share\nagainst the US share"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(rows))[::-1]
    for yi, r, c in zip(y, rows, colors):
        med, lo, hi = (1e6 * r[k] / pop for k in ("med", "lo", "hi"))
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=3, solid_capstyle="round", alpha=0.55)
        ax.scatter(med, yi, s=60, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.text(hi * 1.12, yi, f"€{med:,.0f}", va="center", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(8, 4000)
    ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000, 3000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Euros per resident a year, around 2025 (log scale): median and 5th-95th percentile of the inputs' range")
    ax.set_title("Portugal's ledger for medicines: the borrowed saving dwarfs everything the SNS does itself")
    note(fig, "Spending from the SNS transparency portal (2025); returns from Infarmed (EUR 500-600 million); US benchmarks from "
              "Part 2 and RAND 2024; the size effect from the regression of Figure 2;\nthe wait from Part 4; administration from "
              "the OECD health accounts (2024). The share of hospital spending without competition is an assumption (60-90%).")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.93, bottom=0.17)
    fig.savefig(OUT / "fig3_ledger_per_resident.png", dpi=200)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(exist_ok=True)
    data = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))
    fig_decomposition(data)
    fig_size(data)
    fig_ledger(data)


if __name__ == "__main__":
    main()
