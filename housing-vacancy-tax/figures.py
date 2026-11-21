"""Produce three PNG figures from the source census table and scenario CSVs.

Run ``python run.py`` first. Requires matplotlib (see requirements.txt).
All data are local snapshots and no network access is needed for recreation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vacancy_tax import load_census

ROOT = Path(__file__).resolve().parent


def read_table(path: Path) -> list[dict[str, str]]:
    """Read a generated CSV table without changing numeric precision."""
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def plot_census(root: Path = ROOT) -> None:
    """Observed 2021 counts by reason for vacancy and repair category."""
    cells = load_census(root / "data" / "census_2021_ihru_table11.csv")
    groups = ("For sale or rent", "Other reasons")
    categories = ("for_sale_or_rent", "other_reasons")
    sound = [next(c.dwellings for c in cells if c.market_status == m and c.repair_status == "no_or_minor_repairs") for m in categories]
    other = [next(c.dwellings for c in cells if c.market_status == m and c.repair_status == "medium_or_major_repairs") for m in categories]
    figure, ax = plt.subplots(figsize=(9, 5))
    x = [0, 1]
    ax.bar(x, [v / 1000 for v in sound], label="No or minor repairs")
    ax.bar(x, [v / 1000 for v in other], bottom=[v / 1000 for v in sound], label="Medium or major repairs")
    for index, (a, b) in enumerate(zip(sound, other)):
        ax.text(index, (a + b) / 1000 + 9, f"{a+b:,}", ha="center", fontsize=11)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Thousands of vacant dwellings")
    ax.set_ylim(0, 425)
    ax.set_title("Portugal's 2021 vacant housing stock is not one category")
    ax.legend(frameon=False)
    figure.text(0.1, 0.01, "Source: IHRU/OHARU (2024), Table 11; INE Census 2021. Not a count of legally taxable properties.", fontsize=8)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(root / "figures" / "fig01_vacant_stock_2021.png", dpi=170)
    plt.close(figure)


def plot_property_tax(root: Path = ROOT) -> None:
    """Modelled statutory IMI-multiplier arithmetic for an illustrative VPT."""
    rows = read_table(root / "results" / "tax_scenarios.csv")
    multipliers = [int(r["multiplier"]) for r in rows]
    liabilities = [float(r["annual_imi_eur"]) for r in rows]
    figure, ax = plt.subplots(figsize=(9, 5))
    positions = list(range(len(rows)))
    bars = ax.bar(positions, liabilities)
    ax.bar_label(bars, labels=[f"€{v:,.0f}" for v in liabilities], padding=3)
    ax.set_xticks(positions, [f"{n}× ordinary" for n in multipliers])
    ax.set_ylabel("Annual IMI for a hypothetical property (€)")
    ax.set_title("The cost of holding a taxable property vacant")
    ax.set_ylim(0, max(liabilities) * 1.15)
    figure.text(0.1, 0.01, "Assumptions: VPT €150,000; municipal IMI 0.35%. Multipliers are examples, not claims of actual assessment.", fontsize=8)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(root / "figures" / "fig02_imi_scenarios.png", dpi=170)
    plt.close(figure)


def plot_sensitivity(root: Path = ROOT) -> None:
    """Hypothetical released dwellings for two unobserved filter probabilities."""
    rows = read_table(root / "results" / "sensitivity.csv")
    eligible_shares = sorted({float(r["eligible_share"]) for r in rows})
    response_shares = sorted({float(r["response_share"]) for r in rows})
    values = {(float(r["eligible_share"]), float(r["response_share"])): float(r["hypothetically_released_dwellings"]) for r in rows}
    grid = [[values[(e, p)] for p in response_shares] for e in eligible_shares]
    figure, ax = plt.subplots(figsize=(9, 5.5))
    heat = ax.imshow(grid, aspect="auto")
    ax.set_xticks(range(len(response_shares)), [f"{100*x:.0f}%" for x in response_shares])
    ax.set_yticks(range(len(eligible_shares)), [f"{100*x:.0f}%" for x in eligible_shares])
    ax.set_xlabel("Hypothetical behavioural response among eligible dwellings")
    ax.set_ylabel("Hypothetically eligible share of 248,534 dwellings")
    ax.set_title("Sensitivity exercise, not a forecast of homes released")
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:,.0f}", ha="center", va="center", color="white" if value > max(max(g) for g in grid) / 2 else "black")
    figure.colorbar(heat, ax=ax, label="Hypothetical number of dwellings")
    figure.text(0.08, 0.01, "Baseline: 2021 Census, other reasons + no/minor repairs. Both shares are unobserved assumptions.", fontsize=8)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(root / "figures" / "fig03_sensitivity.png", dpi=170)
    plt.close(figure)


def main() -> None:
    ROOT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    plot_census()
    plot_property_tax()
    plot_sensitivity()


if __name__ == "__main__":
    main()
