"""Figures for the regional panel (reads regional_panel.csv and results_regional.json).

fig_regional_completions_per_1000.png    Completed dwellings per 1,000 residents, NUTS 3, 2011-2025
fig_regional_permits_and_completions.png Permitted and completed dwellings per 1,000 residents, NUTS 2, 2011-2025
fig_regional_population_growth.png       Annual population growth by NUTS 3, 2011-2020 and 2021-2025
fig_regional_growth_and_construction.png Population growth against completions per 1,000 residents, NUTS 3

Needs matplotlib (the analysis itself does not). Run after regional.py:
    python regional_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "figures"

BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#9a9993", "#e6e5e1", "#c9c8c2"
HIGHLIGHT = {"1A0": ("Grande Lisboa", BLUE), "1B0": ("Península de Setúbal", AQUA),
             "11A": ("Área Metropolitana do Porto", ORANGE), "PT": ("Portugal", INK)}
SOURCE = "Source: INE (Estatísticas das obras concluídas; Inquérito aos projetos de obras; Estimativas anuais da população residente), NUTS 2024. Diogo Ribeiro."

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 170,
    # Set here, not left to a site-wide matplotlibrc: hinting changes how text is rasterised, so a different
    # default (matplotlib's own is "default") moves text pixels in every figure. The release figures use none.
    "text.hinting": "no_hinting",
})


def read_panel() -> dict[str, dict[int, dict[str, str]]]:
    """region -> year -> row of regional_panel.csv."""
    panel: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with (HERE / "regional_panel.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            panel[row["region_code"]][int(row["year"])] = row
    return panel


def series(panel: dict[str, dict[int, dict[str, str]]], region: str, column: str) -> tuple[list[int], list[float]]:
    """Years and values of one column, skipping empty cells (gaps stay gaps)."""
    pairs = [(y, float(r[column])) for y, r in sorted(panel[region].items()) if r[column]]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def mark_vintages(ax: plt.Axes) -> None:
    """Shade the preliminary years and separate the two population bases."""
    ax.axvspan(2020.5, 2025.5, color=GRID, alpha=0.45, lw=0, zorder=0)


def completions_figure(panel: dict[str, dict[int, dict[str, str]]], results: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    mark_vintages(ax)
    nuts3 = [r for r, v in results["regions"].items() if v["level"] == "NUTS3"]
    for r in nuts3:
        if r not in HIGHLIGHT:
            x, y = series(panel, r, "completed_per_1000_residents")
            ax.plot(x, y, color=GREY, lw=0.9, zorder=2)
    ends = {}
    for r, (label, colour) in HIGHLIGHT.items():
        x, y = series(panel, r, "completed_per_1000_residents")
        ax.plot(x, y, color=colour, lw=2.3 if r != "PT" else 2.0, ls="-" if r != "PT" else "--", zorder=4)
        ends[r] = (label, x[-1], y[-1])
    placed: list[float] = []
    for r, (label, x_end, y_end) in sorted(ends.items(), key=lambda t: -t[1][2]):
        y_text = y_end
        while any(abs(y_text - p) < 0.13 for p in placed):
            y_text -= 0.05
        placed.append(y_text)
        ax.annotate(label, (x_end, y_end), xytext=(x_end + 0.35, y_text), textcoords="data", color=INK2, fontsize=8.5, va="center")
    ax.plot([], [], color=GREY, lw=0.9, label="other NUTS 3 regions")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(2011.2, ax.get_ylim()[1] * 0.97, "2011-2017: Amadora under-reported\n(lowers Grande Lisboa)", fontsize=7.8, color=INK2, va="top")
    ax.text(2020.7, ax.get_ylim()[1] * 0.97, "preliminary;\n2024-25 estimated", fontsize=7.8, color=INK2, va="top")
    ax.set_xlim(2010.5, 2028.5)
    ax.set_xticks(range(2011, 2026, 2))
    ax.set_ylabel("Completed dwellings per 1,000 residents")
    ax.set_title("Completed new family dwellings per 1,000 residents, NUTS 3 regions")
    fig.text(0.01, 0.01, SOURCE, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_regional_completions_per_1000.png")
    plt.close(fig)


def permits_figure(panel: dict[str, dict[int, dict[str, str]]], results: dict) -> None:
    nuts2 = [r for r, v in results["regions"].items() if v["level"] == "NUTS2"]
    order = sorted(nuts2, key=lambda r: -(results["regions"][r]["permits_per_1000_mean_2021_2025"] or 0))
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.6), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, order, strict=True):
        mark_vintages(ax)
        xp, yp = series(panel, r, "permitted_per_1000_residents")
        xc, yc = series(panel, r, "completed_per_1000_residents")
        ax.plot(xp, yp, color=BLUE, lw=1.9, label="permitted")
        ax.plot(xc, yc, color=ORANGE, lw=1.9, label="completed")
        xn, yn = series(panel, "PT", "completed_per_1000_residents")
        ax.plot(xn, yn, color=MUTED, lw=1.0, ls="--", label="completed, Portugal")
        ax.set_title(results["regions"][r]["name"], fontsize=9.3)
        ax.set_xticks([2011, 2015, 2020, 2025])
    axes.flat[0].legend(loc="upper left", fontsize=7.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("per 1,000 residents")
    fig.suptitle("Permitted and completed new family dwellings per 1,000 residents, NUTS 2 regions (2011-2025)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, SOURCE, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT / "fig_regional_permits_and_completions.png")
    plt.close(fig)


def growth_figure(results: dict) -> None:
    nuts3 = [r for r, v in results["regions"].items() if v["level"] == "NUTS3"]
    rows = []
    for r in nuts3:
        v = results["regions"][r]
        a, b = v["population_growth_2011_2020"], v["population_growth_2021_2025"]
        if a and b:
            rows.append((r, v["name"], a["annual_pct"], b["annual_pct"]))
    rows.sort(key=lambda t: t[3])
    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    for i, (r, name, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color=GRID, lw=2.2, zorder=1)
        ax.plot(a, i, "o", color=MUTED, ms=6, zorder=2)
        ax.plot(b, i, "o", color=BLUE if r not in HIGHLIGHT else ORANGE, ms=7, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for _, n, _, _ in rows], fontsize=8.3)
    for tick, (r, *_rest) in zip(ax.get_yticklabels(), rows, strict=True):
        if r in HIGHLIGHT:
            tick.set_fontweight("bold")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average annual population change (%)")
    ax.plot([], [], "o", color=MUTED, label="2011-2020 (census-based estimates)")
    ax.plot([], [], "o", color=BLUE, label="2021-2025 (administrative-based estimates)")
    ax.plot([], [], "o", color=ORANGE, label="2021-2025, Lisbon and Porto regions (bold)")
    ax.legend(loc="lower right", fontsize=8.3)
    ax.set_title("Population change by NUTS 3 region, two estimation periods")
    fig.text(0.01, 0.01, "Source: INE, Estimativas anuais da população residente, regrouped to NUTS 2024. "
             "The two periods use different estimation methods and are shown separately. Diogo Ribeiro.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(OUT / "fig_regional_population_growth.png")
    plt.close(fig)


def scatter_figure(results: dict) -> None:
    nuts3 = [r for r, v in results["regions"].items() if v["level"] == "NUTS3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (a, b) in zip(axes, ((2011, 2020), (2021, 2025)), strict=True):
        key_g = f"population_growth_{a}_{b}"
        pts = []
        for r in nuts3:
            v = results["regions"][r]
            if b == 2020:
                c1, c2 = v["completions_per_1000_mean_2011_2015"], v["completions_per_1000_mean_2016_2020"]
                c = (c1 + c2) / 2 if c1 is not None and c2 is not None else None
            else:
                c = v["completions_per_1000_mean_2021_2025"]
            if v[key_g] and c is not None:
                pts.append((r, v["name"], v[key_g]["annual_pct"], c))
        for r, name, x, y in pts:
            colour = HIGHLIGHT[r][1] if r in HIGHLIGHT else MUTED
            ax.plot(x, y, "o", color=colour, ms=7 if r in HIGHLIGHT else 5.5, zorder=3)
            if r in HIGHLIGHT or y == max(p[3] for p in pts) or x == max(p[2] for p in pts) or x == min(p[2] for p in pts):
                short = name.replace("Região Autónoma dos ", "").replace("Região Autónoma da ", "")
                ax.annotate(short, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7.6, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"Average annual population change {a}-{b} (%)")
        ax.set_title(f"{a}-{b}")
    axes[0].set_ylabel("Mean completed dwellings per 1,000 residents")
    fig.suptitle("Population change and completed dwellings per 1,000 residents, NUTS 3 regions (descriptive)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, SOURCE + " Construction intensity is not a measure of housing need.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "fig_regional_growth_and_construction.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panel = read_panel()
    results = json.loads((HERE / "results_regional.json").read_text(encoding="utf-8"))
    completions_figure(panel, results)
    permits_figure(panel, results)
    growth_figure(results)
    scatter_figure(results)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig_regional_*.png"))))


if __name__ == "__main__":
    main()
