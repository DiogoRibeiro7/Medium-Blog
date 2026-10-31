"""Reproduce the Gilded Age essay's source-based and illustrative figures.

The historical CSV files are small, explicitly sourced transcriptions. They
are not scraped, interpolated, or fitted. Model outputs use separate tables
and carry no claim about the observed behavior of nineteenth-century firms.

Python 3.10+ is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INK = "#1e2933"
BLUE = "#126c83"
COPPER = "#a85b2b"
GOLD = "#c69442"
RED = "#b6473e"
MUTED = "#596b79"
CREAM = "#fbfaf5"
PALE = "#e4ecee"

FIGURE_NAMES: tuple[str, ...] = (
    "fig1_railroad_expansion.png",
    "fig2_foreign_born_population.png",
    "fig3_growth_and_wages.png",
    "fig4_massachusetts_wealth.png",
    "fig5_labor_share_scenarios.png",
    "fig6_network_fixed_costs.png",
    "fig7_monopoly_surplus.png",
    "fig8_institutions_and_exclusion.png",
)


@dataclass(frozen=True)
class LinearMarket:
    """Illustrative inverse demand P(Q) = intercept - slope * Q and MC = cost."""

    intercept: float
    slope: float
    cost: float

    def __post_init__(self) -> None:
        for name in ("intercept", "slope", "cost"):
            value: float = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.slope <= 0 or self.cost < 0 or self.intercept <= self.cost:
            raise ValueError("Require positive slope and intercept > nonnegative cost")


@dataclass(frozen=True)
class MarketOutcome:
    """Competitive benchmark and single-price unregulated monopoly result."""

    competitive_quantity: float
    monopoly_quantity: float
    competitive_price: float
    monopoly_price: float
    deadweight_loss: float


def market_outcome(market: LinearMarket) -> MarketOutcome:
    """Return the efficient MC-pricing benchmark and profit-maximising monopoly.

    This deliberately assumes no fixed cost in the *static pricing* exercise.
    It is not a historical estimate and does not describe regulated railroads.
    """
    competitive_quantity: float = (market.intercept - market.cost) / market.slope
    monopoly_quantity: float = competitive_quantity / 2.0
    monopoly_price: float = market.intercept - market.slope * monopoly_quantity
    deadweight_loss: float = (
        0.5
        * (competitive_quantity - monopoly_quantity)
        * (monopoly_price - market.cost)
    )
    return MarketOutcome(
        competitive_quantity=competitive_quantity,
        monopoly_quantity=monopoly_quantity,
        competitive_price=market.cost,
        monopoly_price=monopoly_price,
        deadweight_loss=deadweight_loss,
    )


def average_cost(quantity: float | np.ndarray, fixed_cost: float, marginal_cost: float) -> np.ndarray:
    """Compute AC(Q)=F/Q+c for strictly positive output under fixed costs."""
    values: np.ndarray = np.asarray(quantity, dtype=float)
    for value, name in ((fixed_cost, "fixed_cost"), (marginal_cost, "marginal_cost")):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All quantities must be finite and strictly positive")
    return fixed_cost / values + marginal_cost


def compounded_index(annual_growth_percent: float, intervals: int) -> float:
    """Compound a published average annual rate; not observed annual levels."""
    if not math.isfinite(annual_growth_percent) or annual_growth_percent <= -100:
        raise ValueError("Annual growth must be finite and greater than -100%")
    if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 0:
        raise ValueError("Intervals must be a nonnegative integer")
    return 100.0 * (1.0 + annual_growth_percent / 100.0) ** intervals


def labor_share_path(
    productivity_growth: float,
    starting_share: float,
    ending_share: float,
    years: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Illustrate w_t/w_0=(A_t/A_0)*(s_t/s_0) under stipulated assumptions.

    Assumes output per worker is A_t, all output is assigned to labor or a
    residual, and the wage is the labor share times productivity. The share
    changes linearly between the supplied endpoints. Neither path is data.
    """
    if isinstance(years, bool) or not isinstance(years, int) or years < 1:
        raise ValueError("years must be a positive integer")
    if not math.isfinite(productivity_growth) or productivity_growth <= -1.0:
        raise ValueError("growth must be finite and greater than -100%")
    for share in (starting_share, ending_share):
        if not math.isfinite(share) or not 0.0 < share <= 1.0:
            raise ValueError("Shares must be in (0, 1]")
    times: np.ndarray = np.arange(years + 1, dtype=float)
    shares: np.ndarray = np.linspace(starting_share, ending_share, years + 1)
    wage_index: np.ndarray = 100.0 * ((1.0 + productivity_growth) ** times) * (shares / starting_share)
    return times, wage_index


def read_csv_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Read a transcribed CSV, checking header and nonempty rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader: csv.DictReader[str] = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"Unexpected columns in {path.name}")
        rows: list[dict[str, str]] = list(reader)
    if not rows or any(not all(row[col] for col in columns) for row in rows):
        raise ValueError(f"Missing records or cells in {path.name}")
    return rows


def railroad_data() -> tuple[np.ndarray, np.ndarray]:
    """Historical miles operated; 1904 Statistical Abstract Table 101."""
    rows = read_csv_rows(DATA / "railroad_mileage.csv", ("year", "miles_in_operation"))
    years = np.array([int(row["year"]) for row in rows], dtype=int)
    miles = np.array([int(row["miles_in_operation"]) for row in rows], dtype=float)
    if np.any(np.diff(years) <= 0) or np.any(miles <= 0):
        raise ValueError("Invalid railroad observations")
    return years, miles


def foreign_born_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decennial stocks and shares transcribed from Census Bureau presentation."""
    rows = read_csv_rows(
        DATA / "foreign_born_population.csv",
        ("year", "population_millions", "share_percent"),
    )
    years = np.array([int(row["year"]) for row in rows], dtype=int)
    persons = np.array([float(row["population_millions"]) for row in rows])
    shares = np.array([float(row["share_percent"]) for row in rows])
    if np.any(np.diff(years) <= 0) or np.any(persons <= 0) or np.any((shares <= 0) | (shares >= 100)):
        raise ValueError("Invalid foreign-born observations")
    return years, persons, shares


def wealth_data() -> tuple[np.ndarray, np.ndarray]:
    """Massachusetts male household heads' taxable wealth; Rockoff Table 2."""
    rows = read_csv_rows(
        DATA / "massachusetts_taxable_wealth.csv",
        ("year", "top_20_percent", "top_5_percent", "top_1_percent"),
    )
    years = np.array([int(row["year"]) for row in rows], dtype=int)
    columns = ("top_20_percent", "top_5_percent", "top_1_percent")
    values = np.array([[float(row[col]) for col in columns] for row in rows], dtype=float)
    if np.any(np.diff(years) <= 0) or np.any(values < 0) or np.any(values > 100):
        raise ValueError("Invalid wealth share observations")
    if np.any(np.diff(values, axis=1) > 1e-9):
        raise ValueError("Top 20 share must be >= top 5 >= top 1")
    return years, values


def growth_data() -> list[tuple[str, float, int]]:
    """Rockoff Table 1 annual rates, 1870–1899, with exact period intervals."""
    rows = read_csv_rows(
        DATA / "rockoff_growth_rates.csv",
        ("series", "annual_growth_percent", "start_year", "end_year"),
    )
    output: list[tuple[str, float, int]] = []
    for row in rows:
        start, end = int(row["start_year"]), int(row["end_year"])
        if end <= start:
            raise ValueError("Invalid historical period")
        rate: float = float(row["annual_growth_percent"])
        output.append((row["series"], rate, end - start))
    return output


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.23)
    plt.close(fig)


def figure1(path: Path) -> None:
    years, miles = railroad_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    ax.plot(years, miles / 1000, color=BLUE, lw=3.2, marker="o", ms=9)
    ax.fill_between(years, miles / 1000, color=BLUE, alpha=0.08)
    for year, val in zip(years, miles):
        ax.annotate(f"{val:,.0f}", (year, val / 1000), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(years)
    ax.set_ylim(0, 230)
    ax.set_ylabel("Railroad miles in operation (thousands)")
    ax.set_xlabel("Census-era benchmark years")
    ax.set_title("Railroad mileage grew about 3.7 times, 1870–1900", fontsize=16, pad=16)
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.01, -0.19, "Source: 1904 U.S. Statistical Abstract, Table 101; reproduces Poor's Railroad Manual.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _save(fig, path)


def figure2(path: Path) -> None:
    years, population, shares = foreign_born_data()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharex=True)
    axes[0].plot(years, population, color=BLUE, marker="o", lw=2.8)
    axes[0].set_ylabel("Foreign-born residents (millions)")
    axes[0].set_ylim(0, 16)
    axes[0].set_title("Number increased", fontsize=13)
    axes[1].plot(years, shares, color=COPPER, marker="s", lw=2.8)
    axes[1].set_ylabel("Share of total population (%)")
    axes[1].set_ylim(0, 18)
    axes[1].set_title("Share fluctuated", fontsize=13)
    for ax in axes:
        ax.set_xticks((1850, 1870, 1890, 1910))
        ax.grid(axis="y", alpha=0.23)
        ax.set_xlabel("Decennial census year")
    fig.suptitle("Immigration is not one number: stock versus population share", fontsize=16, y=1.03)
    fig.text(0.02, -0.06, "Source: U.S. Census Bureau (2022), rounded historical stock and percentage series.",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def figure4(path: Path) -> None:
    years, shares = wealth_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for idx, color, label in ((0, GOLD, "Top 20%"), (1, BLUE, "Top 5%"), (2, RED, "Top 1%")):
        ax.plot(years, shares[:, idx], lw=2.7, marker="o", ms=6, color=color, label=label)
    ax.axvspan(1870, 1900, color=PALE, alpha=0.5)
    ax.set_xlim(1816, 1914)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share of taxable wealth (%)")
    ax.set_xlabel("Observation year (1890 unavailable in this table)")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(loc="lower right", frameon=False, ncol=3)
    ax.set_title("Concentration in one historical source—not a U.S. wealth series", fontsize=15, pad=15)
    ax.text(0.01, -0.21,
            "Massachusetts male household heads. Rockoff (2008), Table 2; underlying Steckel–Moehling estimates.",
            color=MUTED, transform=ax.transAxes, fontsize=9)
    _save(fig, path)


def figure3(path: Path) -> None:
    entries = growth_data()
    names = [label.replace("Real ", "").replace(" per capita", "\nper capita") for label, _, _ in entries]
    rates = [rate for _, rate, _ in entries]
    indexes = [compounded_index(rate, periods) for _, rate, periods in entries]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    bars = ax.barh(names, indexes, color=[BLUE, GOLD, COPPER], height=0.57)
    ax.invert_yaxis()
    ax.axvline(100.0, color=MUTED, linestyle="--", lw=1.2)
    for bar, rate, index in zip(bars, rates, indexes):
        ax.text(index + 2.0, bar.get_y() + bar.get_height() / 2,
                f"{index:.0f}  ({rate:.2f}%/yr)", va="center", color=INK, fontsize=11)
    ax.set_xlim(0, max(indexes) + 75)
    ax.set_xlabel("1870 = 100; compounding published annual averages over 29 intervals")
    ax.set_title("Aggregate income and unskilled real wages both rose—at different rates", fontsize=15, pad=16)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)
    ax.text(0.0, -0.24, "Source: Rockoff (2008), Table 1. Constructed endpoints, not observed year-by-year series.",
            fontsize=9, color=MUTED, transform=ax.transAxes)
    _save(fig, path)


def figure6(path: Path) -> None:
    quantities = np.linspace(2, 30, 200)
    fixed_cost, marginal_cost = 40.0, 2.0
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].plot(quantities, average_cost(quantities, fixed_cost, marginal_cost), color=BLUE, lw=3,
                 label="Average cost (F/Q + c)")
    axes[0].axhline(marginal_cost, color=COPPER, linestyle="--", label="Marginal cost (c)")
    axes[0].set_ylim(0, 23)
    axes[0].set_ylabel("Cost per unit (illustrative units)")
    axes[0].set_xlabel("Output, Q")
    axes[0].legend(frameon=False)
    axes[0].set_title("Fixed costs create scale economies", fontsize=13)
    single = fixed_cost + marginal_cost * quantities
    duplicate = 2.0 * fixed_cost + marginal_cost * quantities
    axes[1].plot(quantities, single, label="One network", color=BLUE, lw=3)
    axes[1].plot(quantities, duplicate, label="Two duplicated networks", color=RED, lw=3)
    axes[1].set_ylabel("Total cost")
    axes[1].set_xlabel("Aggregate output, Q")
    axes[1].set_title("Duplicating infrastructure can waste F", fontsize=13)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Why large networks can be efficient without guaranteeing fair prices", fontsize=16, y=1.03)
    fig.text(0.01, -0.045, "Hypothetical model: F = 40, c = 2. No empirical calibration or price regulation.",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def figure7(path: Path) -> None:
    case = LinearMarket(intercept=12.0, slope=0.5, cost=2.0)
    outcome = market_outcome(case)
    quantities = np.linspace(0, 22, 200)
    prices = case.intercept - case.slope * quantities
    fig, ax = plt.subplots(figsize=(10.0, 6.1))
    ax.plot(quantities, prices, color=BLUE, lw=3, label="Inverse demand")
    ax.axhline(case.cost, color=COPPER, lw=2, label="Marginal cost")
    q_m, q_c = outcome.monopoly_quantity, outcome.competitive_quantity
    ax.fill_between(quantities, case.cost, prices, where=(quantities >= q_m) & (quantities <= q_c),
                    color=RED, alpha=0.25, label="Deadweight loss")
    ax.scatter([q_m, q_c], [outcome.monopoly_price, outcome.competitive_price],
               color=[RED, BLUE], s=100, zorder=3)
    ax.vlines(q_m, 0, outcome.monopoly_price, color=RED, ls=":")
    ax.vlines(q_c, 0, case.cost, color=BLUE, ls=":")
    ax.annotate("Monopoly", (q_m, outcome.monopoly_price), xytext=(-24, 15),
                textcoords="offset points", color=RED, ha="right", fontsize=11)
    ax.annotate("MC-pricing benchmark", (q_c, case.cost), xytext=(-8, 17),
                textcoords="offset points", color=BLUE, ha="right", fontsize=10)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 13)
    ax.set_xlabel("Quantity (hypothetical units)")
    ax.set_ylabel("Price (hypothetical units)")
    ax.set_title("Unregulated market power can reduce output even when scale is valuable", fontsize=15)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.14)
    ax.text(0.01, -0.16, "Assumptions: P=12–0.5Q, MC=2, no fixed cost in this pricing benchmark. Not historical data.",
            fontsize=9, color=MUTED, transform=ax.transAxes)
    _save(fig, path)


def figure5(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    scenarios: tuple[tuple[str, float, str], ...] = (
        ("Share rises to 66%", 0.66, BLUE),
        ("Share stays 60%", 0.60, GOLD),
        ("Share falls to 48%", 0.48, RED),
    )
    for label, end_share, color in scenarios:
        x, y = labor_share_path(0.025, 0.60, end_share, 30)
        ax.plot(x, y, color=color, lw=2.7, label=f"Real wage: {label}")
    ax.set_ylim(95, 260)
    ax.set_xlim(0, 30)
    ax.set_ylabel("Index, initial year = 100")
    ax.set_xlabel("Elapsed years (not historical calendar years)")
    ax.set_title("Productivity growth does not fix how its gains are divided", fontsize=16, pad=15)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.0, -0.19, "Hypothetical: productivity +2.5%/yr; initial labor share 60%; constant share tracks output.",
            color=MUTED, fontsize=9, transform=ax.transAxes)
    _save(fig, path)


def figure8(path: Path) -> None:
    events: tuple[tuple[int, str, str], ...] = (
        (1870, "Fifteenth Amendment", "Political inclusion"),
        (1877, "Great Railroad Strike", "Labor conflict"),
        (1882, "Chinese Exclusion Act", "Exclusion"),
        (1887, "Interstate Commerce Act / Dawes Act", "Regulation / dispossession"),
        (1890, "Sherman Antitrust Act", "Regulation"),
        (1892, "Homestead strike / People's Party platform", "Labor and agrarian politics"),
        (1894, "Pullman strike", "Labor conflict"),
        (1896, "Plessy v. Ferguson", "Legal segregation"),
    )
    fig, ax = plt.subplots(figsize=(11.4, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(events) + 1.2)
    ax.set_axis_off()
    ax.text(0.03, 8.8, "Year", fontsize=11, fontweight="bold", color=MUTED)
    ax.text(0.16, 8.8, "Event", fontsize=11, fontweight="bold", color=MUTED)
    ax.text(0.69, 8.8, "Institutional dimension", fontsize=11, fontweight="bold", color=MUTED)
    for idx, (year, event, category) in enumerate(events):
        y = 8.05 - idx
        if idx % 2 == 0:
            ax.add_patch(patches.Rectangle((0.018, y - 0.36), 0.966, 0.76,
                                           facecolor=PALE, edgecolor="none", alpha=0.58))
        tag = RED if "Exclusion" in category or "segregation" in category or "dispossession" in category else BLUE
        ax.text(0.035, y, str(year), va="center", fontsize=12, fontweight="bold", color=COPPER)
        ax.text(0.16, y, event, va="center", fontsize=11, color=INK)
        ax.text(0.70, y, category, va="center", fontsize=10, color=tag)
    ax.set_title("The same era expanded markets and contested democratic inclusion", fontsize=16, pad=20)
    ax.text(0.02, 0.20, "Selected documented events, not a quantitative chronology or a causal sequence.",
            fontsize=9, color=MUTED)
    _save(fig, path)


def _write_table(path: Path, fields: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def generate_tables(folder: Path) -> None:
    """Write derived source-based summaries separately from hypothetical outputs."""
    years, miles = railroad_data()
    _, foreign_population, _ = foreign_born_data()
    wealth_years, wealth = wealth_data()
    year_to_wealth = {int(year): wealth[idx, 2] for idx, year in enumerate(wealth_years)}
    summary: list[tuple[object, ...]] = [
        ("railroad_mileage_factor_1870_1900", float(miles[-1] / miles[0]), "ratio",
         "historical derived", "1904 US Statistical Abstract Table 101"),
        ("foreign_born_headcount_factor_1850_1910",
         float(foreign_population[-1] / foreign_population[0]), "ratio",
         "rounded historical derived", "US Census Bureau 2022"),
        ("massachusetts_top1_change_1870_1900",
         float(year_to_wealth[1900] - year_to_wealth[1870]), "percentage points",
         "historical derived; local sample", "Rockoff 2008 Table 2"),
    ]
    for name, rate, intervals in growth_data():
        summary.append((f"constructed_index_{name.lower().replace(' ', '_')}",
                        compounded_index(rate, intervals), "1870=100",
                        "compounded historic average, not observed series", "Rockoff 2008 Table 1"))
    _write_table(
        folder / "source_based_summary.csv",
        ("metric", "value", "unit", "evidence_class", "source"),
        summary,
    )
    market_rows: list[tuple[object, ...]] = []
    for slope in (0.30, 0.40, 0.50, 0.60, 0.80, 1.00):
        x = market_outcome(LinearMarket(intercept=12.0, slope=slope, cost=2.0))
        market_rows.append((slope, x.competitive_quantity, x.monopoly_quantity,
                            x.monopoly_price, x.deadweight_loss))
    _write_table(
        folder / "hypothetical_monopoly_sensitivity.csv",
        ("demand_slope", "competitive_quantity", "monopoly_quantity",
         "monopoly_price", "deadweight_loss"),
        market_rows,
    )
    scenario_rows: list[tuple[object, ...]] = []
    up = labor_share_path(0.025, 0.6, 0.66)[1]
    steady = labor_share_path(0.025, 0.6, 0.6)[1]
    down = labor_share_path(0.025, 0.6, 0.48)[1]
    for year in range(31):
        scenario_rows.append((year, up[year], steady[year], down[year]))
    _write_table(
        folder / "hypothetical_labor_share_paths.csv",
        ("elapsed_year", "wage_index_rising_share", "wage_index_constant_share",
         "wage_index_falling_share"),
        scenario_rows,
    )


def generate_all(output: Path) -> None:
    """Regenerate eight figures and three CSVs with no downloads or randomness."""
    figures = output / "figures"
    results = output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    _style()
    functions: tuple[Callable[[Path], None], ...] = (
        figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8
    )
    for name, method in zip(FIGURE_NAMES, functions):
        method(figures / name)
    generate_tables(results)
    print(f"Generated {len(FIGURE_NAMES)} figures and 3 tables in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT,
        help="Destination containing figures/ and results/ (default: article directory)"
    )
    args: argparse.Namespace = parser.parse_args()
    generate_all(args.output)


if __name__ == "__main__":
    main()
