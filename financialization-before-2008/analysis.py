"""Publication figures and bounded descriptive checks, United States 1990–2007.

All historical observations are frozen reproductions of public government
series. The eighth figure is an explicitly hypothetical household-equity
identity. Neither the charts nor the lead/lag correlations identify effects of
trade policy, interest rates, financial deregulation, or securitization.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURE_NAMES: tuple[str, ...] = (
    "figure_01_manufacturing_employment.png",
    "figure_02_trade_balance.png",
    "figure_03_income_growth_and_inequality.png",
    "figure_04_union_membership.png",
    "figure_05_house_price_income.png",
    "figure_06_household_debt.png",
    "figure_07_interest_rates.png",
    "figure_08_hypothetical_leverage.png",
)

INK = "#172F3C"
TEAL = "#007F83"
BLUE = "#467DB0"
AMBER = "#B36D20"
RED = "#A53A43"
GREY = "#697A83"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B3C0C6",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load_panel(path: Path = DATA / "annual_panel_1990_2007.csv") -> list[dict[str, float]]:
    """Load and validate exactly eighteen ordered records with finite values."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("Missing annual panel header")
        panel: list[dict[str, float]] = []
        for raw in reader:
            record: dict[str, float] = {}
            for field in reader.fieldnames:
                value = float(raw[field] or "nan")
                if not math.isfinite(value):
                    raise ValueError(f"Missing/non-finite panel value: {field}")
                record[field] = value
            panel.append(record)
    if [int(row["year"]) for row in panel] != list(range(1990, 2008)):
        raise ValueError("Unexpected panel year coverage")
    if any(not 0.0 < row["household_income_gini"] < 1.0 for row in panel):
        raise ValueError("Invalid household Gini")
    return panel


def vector(panel: Sequence[dict[str, float]], key: str) -> np.ndarray:
    """Return a finite, one-dimensional historical time-series array."""
    result = np.asarray([row[key] for row in panel], dtype=float)
    if result.ndim != 1 or len(result) != 18 or not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid historical vector: {key}")
    return result


def base_index(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalize positive levels to the first year's value (1990=100)."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("An index requires a positive finite series")
    return 100.0 * a / a[0]


def price_shock_equity(initial_ltv: float, decline_fraction: np.ndarray) -> np.ndarray:
    """Compute remaining equity as a fraction of the initial property value.

    Initial property value is 1, mortgage principal stays fixed at initial_ltv,
    and property value after a shock is 1 - decline_fraction. This is a static
    accounting scenario, not a default probability or a historical estimate.
    """
    if not math.isfinite(initial_ltv) or not 0.0 < initial_ltv < 1.0:
        raise ValueError("Initial loan-to-value must be between zero and one")
    shocks = np.asarray(decline_fraction, dtype=float)
    if shocks.ndim != 1 or not np.all(np.isfinite(shocks)) or np.any((shocks < 0.0) | (shocks >= 1.0)):
        raise ValueError("Price declines must lie in [0, 1)")
    return 1.0 - shocks - initial_ltv


def percentage_equity_lost(initial_ltv: float, decline_fraction: np.ndarray) -> np.ndarray:
    """Return loss relative to *initial equity*, which grows with leverage."""
    return 100.0 * decline_fraction / (1.0 - initial_ltv)


def lead_lag_correlations(panel: Sequence[dict[str, float]]) -> list[dict[str, float | int]]:
    """Exploratory lag correlations of first changes, never causal estimates.

    Positive lag means the housing-index annual log change is paired with a
    *later* percentage-point change in debt/output. With only 15–17 paired
    changes, autocorrelation and omitted confounders make inference fragile.
    """
    prices = vector(panel, "fhfa_house_price_index")
    debt = vector(panel, "debt_to_gdp_pct")
    price_growth = 100.0 * np.diff(np.log(prices))
    debt_change = np.diff(debt)
    output: list[dict[str, float | int]] = []
    for lag in range(-2, 3):
        if lag >= 0:
            x = price_growth[: len(price_growth) - lag]
            y = debt_change[lag:]
        else:
            x = price_growth[-lag:]
            y = debt_change[: len(debt_change) + lag]
        if len(x) != len(y) or len(x) < 10 or np.std(x) <= 0 or np.std(y) <= 0:
            raise ValueError(f"Invalid exploratory pair at lag {lag}")
        corr = float(np.corrcoef(x, y)[0, 1])
        if not math.isfinite(corr):
            raise ValueError(f"Non-finite exploratory correlation at lag {lag}")
        output.append({"lag_years": lag, "paired_years": len(x), "pearson_r_descriptive": corr})
    return output


def new_figure(title: str, note: str, *, two_panels: bool = False) -> tuple[Figure, list[Axes]]:
    """Create a consistent print-friendly figure with an explicit provenance line."""
    if two_panels:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.3), dpi=150)
        axlist = list(axes)
        fig.subplots_adjust(left=0.09, right=0.94, top=0.79, bottom=0.20, wspace=0.30)
    else:
        fig, ax = plt.subplots(figsize=(11.6, 6.3), dpi=150)
        axlist = [ax]
        fig.subplots_adjust(left=0.10, right=0.94, top=0.79, bottom=0.20)
    fig.suptitle(title, fontsize=18.5, fontweight="bold", x=0.09, y=0.95, ha="left", color=INK)
    fig.text(0.09, 0.055, note, ha="left", va="center", fontsize=8.2, color=GREY, wrap=True)
    for ax in axlist:
        ax.grid(axis="y", color="#DFE6E9", lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=6)
    return fig, axlist


def year_axis(ax: Axes, years: np.ndarray) -> None:
    """Use the same historic coverage and labeling in every data figure."""
    ax.set_xlim(1989.7, 2007.8)
    ax.set_xticks([1990, 1994, 1998, 2002, 2007])
    ax.set_xlabel("Year")


def legend(ax: Axes) -> None:
    """Place a compact, unobtrusive legend."""
    ax.legend(frameon=False, fontsize=9.3, loc="best")


def render_historical_figures(panel: Sequence[dict[str, float]], output: Path) -> None:
    """Render seven source-grounded figures, keeping distinct units separated."""
    years = vector(panel, "year")

    # Figure 1: distinguish the level of factory jobs from their share of jobs.
    fig, axes = new_figure(
        "Manufacturing employment in a growing labor market",
        "Source: BLS payroll employment via FRED MANEMP and PAYEMS. Annual averages, seasonally adjusted.",
        two_panels=True,
    )
    axes[0].plot(years, vector(panel, "manufacturing_employment_millions"), color=TEAL, lw=2.7)
    axes[0].set_ylabel("Manufacturing payroll jobs (millions)")
    axes[1].plot(years, vector(panel, "manufacturing_payroll_share_pct"), color=RED, lw=2.7)
    axes[1].set_ylabel("Manufacturing share of all payroll jobs (%)")
    for ax in axes:
        year_axis(ax, years)
    fig.savefig(output / FIGURE_NAMES[0], dpi=190)
    plt.close(fig)

    # Figure 2: trade flows share a common nominal GDP denominator.
    fig, axes = new_figure(
        "Trade flows grew; the trade gap widened",
        "Source: BEA GDP, exports, imports and net exports via FRED. Annual mean quarterly SAAR flows; % nominal GDP.",
        two_panels=True,
    )
    axes[0].plot(years, vector(panel, "exports_gdp_pct"), lw=2.5, color=TEAL, label="Exports")
    axes[0].plot(years, vector(panel, "imports_gdp_pct"), lw=2.5, color=AMBER, label="Imports")
    axes[0].set_ylabel("Share of nominal GDP (%)")
    legend(axes[0])
    axes[1].axhline(0, color=GREY, lw=1)
    axes[1].plot(years, vector(panel, "net_exports_gdp_pct"), lw=2.7, color=RED)
    axes[1].fill_between(years, vector(panel, "net_exports_gdp_pct"), 0, color=RED, alpha=0.10)
    axes[1].set_ylabel("Net exports / GDP (%)")
    for ax in axes:
        year_axis(ax, years)
    fig.savefig(output / FIGURE_NAMES[1], dpi=190)
    plt.close(fig)

    # Figure 3: normalized aggregate and median levels plus a separate Gini.
    fig, axes = new_figure(
        "Growth and the distribution of household income",
        "Sources: BEA real GDP/person, Census CPS ASEC real median household income and household Gini via FRED. Different populations; 1993 CPS break.",
        two_panels=True,
    )
    axes[0].plot(years, base_index(vector(panel, "real_gdp_per_capita_2017_usd")), color=BLUE, lw=2.6, label="Real GDP per capita")
    axes[0].plot(years, base_index(vector(panel, "real_median_household_income_2025_usd")), color=TEAL, lw=2.6, label="Real median HH income")
    axes[0].axhline(100, color=GREY, lw=1, ls="--")
    axes[0].set_ylabel("Separate indices, 1990 = 100")
    legend(axes[0])
    axes[1].plot(years, vector(panel, "household_income_gini"), color=RED, lw=2.6, marker="o", markersize=3.2)
    axes[1].axvline(1993, color=GREY, ls="--", lw=1, label="1993 survey change")
    axes[1].set_ylabel("Pre-tax household income Gini")
    axes[1].set_ylim(0.40, 0.49)
    legend(axes[1])
    for ax in axes:
        year_axis(ax, years)
    fig.savefig(output / FIGURE_NAMES[2], dpi=190)
    plt.close(fig)

    # Figure 4: total rate and member counts by sector use different scales.
    fig, axes = new_figure(
        "Union membership: shrinking rate, changing composition",
        "Source: U.S. Bureau of Labor Statistics, Union Membership Spotlight (2016); rounded annual estimates.",
        two_panels=True,
    )
    axes[0].plot(years, vector(panel, "union_membership_rate_pct"), lw=2.8, color=TEAL)
    axes[0].set_ylabel("Wage/salary workers belonging to union (%)")
    axes[1].plot(years, vector(panel, "private_union_members_thousands") / 1000, lw=2.5, color=RED, label="Private sector")
    axes[1].plot(years, vector(panel, "public_union_members_thousands") / 1000, lw=2.5, color=BLUE, label="Public sector")
    axes[1].set_ylabel("Union members (millions)")
    legend(axes[1])
    for ax in axes:
        year_axis(ax, years)
    fig.savefig(output / FIGURE_NAMES[3], dpi=190)
    plt.close(fig)

    # Figure 5: neither index is a house-price-to-income ratio in levels.
    fig, axes = new_figure(
        "House prices outpaced nominal median household income",
        "Sources: FHFA all-transactions HPI and Census nominal median HH income via FRED. Separate series indexed to 1990; not a payment-affordability measure.",
    )
    ax = axes[0]
    ax.plot(years, base_index(vector(panel, "fhfa_house_price_index")), lw=2.8, color=RED, label="FHFA U.S. house price index")
    ax.plot(years, base_index(vector(panel, "nominal_median_household_income_usd")), lw=2.8, color=BLUE, label="Median nominal household income")
    ax.set_ylabel("1990 = 100")
    year_axis(ax, years)
    legend(ax)
    fig.savefig(output / FIGURE_NAMES[4], dpi=190)
    plt.close(fig)

    # Figure 6: debt stocks include households *and* nonprofits, with Q4 GDP.
    fig, axes = new_figure(
        "Household-and-nonprofit liabilities grew relative to output",
        "Source: Federal Reserve Z.1 CMDEBT and HHMSDODNS; BEA GDP via FRED. Q4 debt stocks / Q4 annualized nominal GDP.",
        two_panels=True,
    )
    axes[0].plot(years, vector(panel, "debt_to_gdp_pct"), lw=2.8, color=RED, label="Total debt securities + loans")
    axes[0].plot(years, vector(panel, "mortgage_debt_to_gdp_pct"), lw=2.8, color=AMBER, label="1–4 family mortgage debt")
    axes[0].set_ylabel("Q4 debt / Q4 GDP annual rate (%)")
    legend(axes[0])
    axes[1].plot(years, vector(panel, "mortgage_share_of_debt_pct"), lw=2.8, color=TEAL)
    axes[1].set_ylabel("Mortgages as share of total debt (%)")
    for ax in axes:
        year_axis(ax, years)
    fig.savefig(output / FIGURE_NAMES[5], dpi=190)
    plt.close(fig)

    # Figure 7: separate short-policy and longer Treasury benchmark; no mortgage inference.
    fig, axes = new_figure(
        "Short and long U.S. benchmark interest rates",
        "Source: Federal Reserve H.15 and effective federal funds via FRED FEDFUNDS and GS10. Annual mean monthly rates; not mortgage loan rates.",
    )
    ax = axes[0]
    ax.plot(years, vector(panel, "fed_funds_rate_pct"), color=RED, lw=2.7, label="Effective federal funds")
    ax.plot(years, vector(panel, "ten_year_treasury_yield_pct"), color=BLUE, lw=2.7, label="10-year Treasury")
    ax.set_ylabel("Annual average interest rate (%)")
    year_axis(ax, years)
    legend(ax)
    fig.savefig(output / FIGURE_NAMES[6], dpi=190)
    plt.close(fig)


def render_hypothetical_figure(output: Path) -> None:
    """Illustrate the accounting arithmetic of fixed debt and falling collateral."""
    declines = np.linspace(0.0, 0.4, 201)
    fig, axes = new_figure(
        "When a property price shock consumes thin equity",
        "HYPOTHETICAL ONLY: initial property value = 100; fixed loan principal, no amortization, interest, fees, or default model. Negative equity is not a default estimate.",
        two_panels=True,
    )
    for ltv, color in ((0.60, TEAL), (0.80, BLUE), (0.95, RED)):
        eq = price_shock_equity(ltv, declines) * 100.0
        loss = percentage_equity_lost(ltv, declines)
        label = f"Initial LTV {ltv:.0%}"
        axes[0].plot(100 * declines, eq, lw=2.5, color=color, label=label)
        axes[1].plot(100 * declines, loss, lw=2.5, color=color, label=label)
    axes[0].axhline(0, color=GREY, ls="--", lw=1.2)
    axes[0].set_ylabel("Remaining equity / original house value (%)")
    axes[1].axhline(100, color=GREY, ls="--", lw=1.2, label="Initial equity exhausted")
    # The right panel zooms into the first 20% decline; the red curve reaches
    # 400% equity loss there. The left panel retains the full 40% shock range.
    axes[1].set_ylim(0, 420)
    axes[1].set_ylabel("Loss as % of initial equity")
    axes[0].set_xlim(0, 40)
    axes[0].set_xticks([0, 10, 20, 30, 40])
    axes[0].set_xlabel("House price decline (%)")
    axes[1].set_xlim(0, 20)
    axes[1].set_xticks([0, 5, 10, 15, 20])
    axes[1].set_xlabel("House price decline, detail view (%)")
    legend(axes[0])
    fig.savefig(output / FIGURE_NAMES[7], dpi=190)
    plt.close(fig)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, float | int | str]]) -> None:
    """Write stable, machine-readable derived result tables."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    key: str(record[key])
                    if isinstance(record[key], (str, int))
                    else f"{float(record[key]):.6f}"
                    for key in fieldnames
                }
            )


def write_results(panel: Sequence[dict[str, float]], output: Path) -> None:
    """Export endpoint comparisons, correlation sensitivity and toy shocks."""
    fields = (
        "manufacturing_employment_millions",
        "manufacturing_payroll_share_pct",
        "net_exports_gdp_pct",
        "real_gdp_per_capita_2017_usd",
        "real_median_household_income_2025_usd",
        "household_income_gini",
        "union_membership_rate_pct",
        "fhfa_house_price_index",
        "nominal_median_household_income_usd",
        "debt_to_gdp_pct",
        "mortgage_debt_to_gdp_pct",
        "mortgage_share_of_debt_pct",
    )
    endpoints: list[dict[str, float | int | str]] = []
    for field in fields:
        first, last = float(panel[0][field]), float(panel[-1][field])
        endpoints.append(
            {
                "metric": field,
                "start_1990": first,
                "end_2007": last,
                "absolute_change": last - first,
                "relative_change_pct": 100 * (last / first - 1) if first != 0 else float("nan"),
            }
        )
    write_csv(
        output / "endpoint_comparisons.csv",
        ("metric", "start_1990", "end_2007", "absolute_change", "relative_change_pct"),
        endpoints,
    )
    write_csv(
        output / "lead_lag_sensitivity.csv",
        ("lag_years", "paired_years", "pearson_r_descriptive"),
        lead_lag_correlations(panel),
    )
    scenarios: list[dict[str, float | int]] = []
    for ltv in (0.60, 0.80, 0.95):
        for decline in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
            d = np.array([decline], dtype=float)
            scenarios.append(
                {
                    "initial_ltv": ltv,
                    "house_price_decline_pct": 100.0 * decline,
                    "remaining_equity_pct_initial_price": float(price_shock_equity(ltv, d)[0] * 100.0),
                    "equity_loss_pct_initial_equity": float(percentage_equity_lost(ltv, d)[0]),
                }
            )
    write_csv(
        output / "hypothetical_leverage.csv",
        (
            "initial_ltv",
            "house_price_decline_pct",
            "remaining_equity_pct_initial_price",
            "equity_loss_pct_initial_equity",
        ),
        scenarios,
    )


def run(output_root: Path = ROOT) -> None:
    """Create exactly eight figures and three derived result tables."""
    if not isinstance(output_root, Path):
        raise TypeError("Output root must be a pathlib.Path")
    figures_dir = output_root / "figures"
    results_dir = output_root / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    render_historical_figures(panel, figures_dir)
    render_hypothetical_figure(figures_dir)
    write_results(panel, results_dir)
    print(f"Created {len(FIGURE_NAMES)} figures and 3 tables in {output_root}")


def main() -> None:
    """Command-line entry point used by local authors and CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT, help="alternate result/figure output root")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
