"""Eight original crisis-and-recovery figures and three audit tables.

Seven figures summarize historical government-source observations, while the
eighth is a hypothetical bank-capital accounting calculation. No figure
identifies a causal effect of any rescue, recession, institution, or policy.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from prepare_data import DATA, ROOT, build_panel, load_sources, race_rows

FIGURE_NAMES: tuple[str, ...] = (
    "figure_01_unemployment_and_recession.png",
    "figure_02_output_and_jobs_recovery.png",
    "figure_03_black_white_unemployment.png",
    "figure_04_housing_and_debt_service.png",
    "figure_05_income_and_inequality.png",
    "figure_06_rates_and_fed_balance_sheet.png",
    "figure_07_scf_racial_wealth.png",
    "figure_08_hypothetical_bank_deleveraging.png",
)
RESULT_NAMES: tuple[str, ...] = (
    "recovery_milestones.csv",
    "racial_outcomes.csv",
    "capital_scenarios.csv",
)
INK = "#193344"
BLUE = "#2C669B"
TEAL = "#007E85"
AMBER = "#B26B19"
RED = "#AE414A"
PURPLE = "#735B9E"
GREY = "#677883"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B9C5CC",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_wealth(path: Path = DATA / "scf_median_wealth_2022_dollars.csv") -> dict[int, dict[str, float]]:
    """Read a checked transcription of the Fed's accessible SCF median table.

    This is a triennial repeated cross-section, not an individual panel.
    Entries are the Federal Reserve's thousands-of-2022-dollars estimates.
    """
    output: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        if not rows.fieldnames or set(rows.fieldnames) != {
            "year", "group", "median_net_worth_thousands_2022_usd"
        }:
            raise ValueError("Unexpected SCF source table schema")
        for row in rows:
            year = int(row["year"])
            group = row["group"]
            amount = float(row["median_net_worth_thousands_2022_usd"])
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError(f"Invalid SCF wealth entry: {row}")
            group_values = output.setdefault(year, {})
            if group in group_values:
                raise ValueError(f"Duplicate SCF group-year: {group}, {year}")
            group_values[group] = amount
    if set(output) != {2007, 2010, 2013, 2016, 2019}:
        raise ValueError("SCF year set differs from source")
    if any(set(group_values) != {"White", "Black", "Hispanic"} for group_values in output.values()):
        raise ValueError("SCF group set differs from source")
    return output


def equity_constrained_sales(
    loss_fraction: float, initial_capital_fraction: float, required_capital_fraction: float = 0.08
) -> float:
    """Additional marked-down assets sold and debt repaid, fraction of A0.

    Starts with A0=1, E0=k0 and borrowed liabilities (1-k0). An exogenous
    mark-to-market loss l reduces both assets and equity by l. Selling assets
    at their new accounting value and repaying liabilities leaves equity
    unchanged. To restore E/A >= k_target, additional asset sales must be
    max(0, (1-l) - (k0-l)/k_target). No new capital or earnings are assumed.

    This arithmetic is NOT a behavioral credit supply or crisis model.
    """
    if not all(math.isfinite(x) for x in (loss_fraction, initial_capital_fraction, required_capital_fraction)):
        raise ValueError("All capital-model parameters must be finite")
    if not 0 <= loss_fraction < initial_capital_fraction < 1:
        raise ValueError("Loss must be nonnegative and smaller than initial equity")
    if not 0 < required_capital_fraction < 1:
        raise ValueError("Capital target must be strictly between 0 and 1")
    marked_assets = 1 - loss_fraction
    post_loss_equity = initial_capital_fraction - loss_fraction
    capacity = post_loss_equity / required_capital_fraction
    return max(0.0, marked_assets - capacity)


def first_recovered(
    observations: Mapping[date, float], baseline: float, after: date
) -> date | None:
    """First observed date at/above a baseline strictly after a cutoff."""
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("Baseline must be finite and positive")
    for day in sorted(observations):
        if day > after and observations[day] >= baseline:
            return day
    return None


def elapsed_months(start: date, end: date) -> int:
    """Calendar-month difference without false day precision."""
    return 12 * (end.year - start.year) + end.month - start.month


def milestones() -> list[dict[str, str | float | int]]:
    """Return distinct recovery clocks; none implies household recovery."""
    s = load_sources()
    panel = build_panel()
    income = {date(int(r["year"]), 1, 1): float(r["real_median_household_income_2025_usd"]) for r in panel}
    series = [
        ("Real GDP", s["GDPC1"], date(2007, 10, 1), date(2009, 4, 1), "Quarterly chained-dollar level"),
        ("Payroll jobs", s["PAYEMS"], date(2007, 12, 1), date(2009, 6, 1), "Monthly nonfarm payroll level"),
        ("FHFA house prices", s["USSTHPI"], date(2007, 10, 1), date(2009, 10, 1), "Quarterly all-transactions index; not seasonally adjusted"),
        ("Median real household income", income, date(2007, 1, 1), date(2009, 1, 1), "Annual Census median, 2025 dollars"),
    ]
    result: list[dict[str, str | float | int]] = []
    for label, values, base_day, cutoff, description in series:
        baseline = values[base_day]
        first = first_recovered(values, baseline, cutoff)
        if first is None:
            raise ValueError(f"Missing post-recession recovery milestone: {label}")
        result.append(
            {
                "indicator": label,
                "baseline_date": base_day.isoformat(),
                "baseline_value": baseline,
                "recovery_date": first.isoformat(),
                "months_from_dec2007": elapsed_months(date(2007, 12, 1), first),
                "measurement_note": description,
            }
        )
    return result


def racial_outcomes() -> list[dict[str, float | int]]:
    """SCF wealth medians and independent annual CPS unemployment gaps."""
    wealth = read_wealth()
    annual = {int(row["year"]): row for row in build_panel()}
    result: list[dict[str, float | int]] = []
    for year in sorted(wealth):
        values = wealth[year]
        result.append(
            {
                "year": year,
                "median_white_2022_usd_thousands": values["White"],
                "median_black_2022_usd_thousands": values["Black"],
                "median_hispanic_2022_usd_thousands": values["Hispanic"],
                "black_white_median_wealth_ratio_pct": 100 * values["Black"] / values["White"],
                "hispanic_white_median_wealth_ratio_pct": 100 * values["Hispanic"] / values["White"],
                "annual_black_white_unemployment_gap_pp": float(annual[year]["black_white_unemployment_gap_pp"]),
            }
        )
    return result


def capital_scenarios() -> list[dict[str, float]]:
    """Finite sensitivity grid; invented assets normalized to 100."""
    result: list[dict[str, float]] = []
    for k0 in (0.08, 0.10, 0.12):
        for half_point in range(0, 13):
            loss = half_point * 0.005
            extra_sales = equity_constrained_sales(loss, k0, 0.08)
            result.append(
                {
                    "loss_pct_initial_assets": 100 * loss,
                    "starting_equity_pct_initial_assets": 100 * k0,
                    "target_capital_ratio_pct": 8.0,
                    "marked_down_assets_pct_initial": 100 * (1 - loss),
                    "remaining_equity_pct_initial": 100 * (k0 - loss),
                    "additional_asset_sales_pct_initial": 100 * extra_sales,
                }
            )
    return result


def _figure(title: str, subtitle: str, panels: int = 1) -> tuple[Figure, Sequence[Axes]]:
    """Create publication-density layout with separately framed subtitles."""
    fig, axs = plt.subplots(panels, 1, figsize=(10.6, 4.7 if panels == 1 else 7.2), sharex=panels > 1)
    axes: list[Axes] = list(np.ravel(np.atleast_1d(axs)))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.76 if panels == 1 else 0.84, bottom=0.13, hspace=0.43)
    fig.suptitle(title, x=0.10, y=0.965, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.10, 0.900 if panels == 1 else 0.911, subtitle, ha="left", va="top", fontsize=9.1, color=GREY, wrap=True)
    return fig, axes


def _legend(ax: Axes) -> None:
    ax.legend(loc="best", frameon=False, fontsize=9)


def _time_axis(ax: Axes) -> None:
    ax.set_xlim(date(2007, 1, 1), date(2019, 12, 31))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#DCE5E9", linewidth=0.8)
    ax.tick_params(axis="x", labelbottom=True)


def _shade_recession(ax: Axes) -> None:
    ax.axvspan(date(2007, 12, 1), date(2009, 6, 30), alpha=0.13, color=GREY, label="NBER recession")


def _plot_series(ax: Axes, source: Mapping[date, float], *, color: str, label: str, scale: float = 1.0) -> None:
    dates = sorted(day for day in source if date(2007, 1, 1) <= day <= date(2019, 12, 31))
    ax.plot(dates, [source[d] * scale for d in dates], color=color, lw=2.4, label=label)


def make_figures(output: Path) -> None:
    """Generate seven historical figures and one clearly labeled toy model."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    s = load_sources()
    panel = build_panel()
    race = race_rows()
    wealth = read_wealth()

    # 1: NBER dating is activity dating, not the month of peak unemployment.
    fig, (ax,) = _figure("The recession ended before unemployment peaked", "BLS monthly unemployment; NBER contraction December 2007–June 2009. The shaded area marks business-cycle dating, not an estimate of policy effect.")
    _shade_recession(ax)
    _plot_series(ax, s["UNRATE"], color=RED, label="Unemployment rate")
    peak_day = max((day for day in s["UNRATE"] if 2007 <= day.year <= 2019), key=lambda day: s["UNRATE"][day])
    ax.scatter([peak_day], [s["UNRATE"][peak_day]], color=RED, zorder=5)
    ax.annotate(f"{peak_day:%b %Y}: {s['UNRATE'][peak_day]:.1f}%", (peak_day, s["UNRATE"][peak_day]), xytext=(14, -22), textcoords="offset points", color=RED)
    ax.set_ylabel("Unemployed / labor force (%)")
    _legend(ax)
    _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[0], dpi=180)
    plt.close(fig)

    # 2: levels normalized to their own Q4/December 2007 benchmarks.
    fig, axes = _figure("The GDP clock and the jobs clock disagreed", "BEA quarterly real GDP and BLS monthly nonfarm payroll jobs. Each index is normalized separately to late 2007 = 100; the denominator and sampling frequency differ.", panels=2)
    _shade_recession(axes[0])
    _plot_series(axes[0], s["GDPC1"], color=BLUE, label="Real GDP (2007Q4 = 100)", scale=100 / s["GDPC1"][date(2007, 10, 1)])
    axes[0].set_ylabel("Real output index")
    axes[0].axhline(100, color=GREY, ls="--", lw=1)
    _legend(axes[0])
    _shade_recession(axes[1])
    _plot_series(axes[1], s["PAYEMS"], color=TEAL, label="Payroll jobs (Dec 2007 = 100)", scale=100 / s["PAYEMS"][date(2007, 12, 1)])
    axes[1].set_ylabel("Nonfarm jobs index")
    axes[1].axhline(100, color=GREY, ls="--", lw=1)
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[1], dpi=180)
    plt.close(fig)

    # 3: Black/White CPS unemployment, not the ethnic categories in SCF.
    fig, axes = _figure("One labor market; unequal exposure", "BLS CPS seasonally adjusted monthly unemployment for Black and White workers. Their difference is descriptive, not an estimate of discriminatory treatment or any single policy.", panels=2)
    days = [date.fromisoformat(str(r["observation_date"])) for r in race]
    _shade_recession(axes[0])
    axes[0].plot(days, [float(r["black_unemployment_pct"]) for r in race], color=RED, lw=2.1, label="Black")
    axes[0].plot(days, [float(r["white_unemployment_pct"]) for r in race], color=BLUE, lw=2.1, label="White")
    axes[0].set_ylabel("Unemployment (%)")
    _legend(axes[0])
    axes[1].fill_between(days, [float(r["gap_percentage_points"]) for r in race], color=AMBER, alpha=0.22)
    axes[1].plot(days, [float(r["gap_percentage_points"]) for r in race], color=AMBER, lw=2.1, label="Black minus White")
    axes[1].set_ylabel("Gap (percentage points)")
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[2], dpi=180)
    plt.close(fig)

    # 4: FHFA nonseasonal index and Fed's aggregate debt-service ratio.
    fig, axes = _figure("Housing prices and debt burdens followed different paths", "FHFA quarterly all-transactions index rebased to 2007Q4; Fed quarterly household debt-service ratio (% of disposable personal income). Different measures; neither is median household equity.", panels=2)
    _shade_recession(axes[0])
    _plot_series(axes[0], s["USSTHPI"], color=AMBER, label="FHFA housing index (2007Q4 = 100)", scale=100 / s["USSTHPI"][date(2007, 10, 1)])
    axes[0].axhline(100, color=GREY, lw=1, ls="--")
    axes[0].set_ylabel("FHFA price index")
    _legend(axes[0])
    _plot_series(axes[1], s["TDSP"], color=PURPLE, label="Household debt-service ratio")
    axes[1].set_ylabel("Debt service / disposable income (%)")
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[3], dpi=180)
    plt.close(fig)

    # 5: real median household income and pre-tax money-income Gini.
    fig, axes = _figure("Median income recovered on its own timetable", "Census CPS ASEC via FRED: median household income in 2025 dollars (2007 = 100) and pre-tax money-income Gini. Survey processing changes complicate exact long-run comparisons.", panels=2)
    years = [int(r["year"]) for r in panel]
    axes[0].plot(years, [float(r["median_income_index_2007"]) for r in panel], lw=2.5, color=TEAL, marker="o", label="Real median household income")
    axes[0].axhline(100, color=GREY, lw=1, ls="--")
    axes[0].set_ylabel("Real income index, 2007 = 100")
    _legend(axes[0])
    axes[1].plot(years, [float(r["household_income_gini"]) for r in panel], color=RED, lw=2.4, marker="o", label="Household money-income Gini")
    axes[1].set_ylabel("Gini (0 = more equal)")
    axes[1].set_ylim(0.44, 0.51)
    _legend(axes[1])
    for ax in axes:
        ax.set_xlim(2007, 2019)
        ax.set_xticks(range(2007, 2020, 2))
        ax.grid(axis="y", color="#DCE5E9")
    fig.savefig(figures / FIGURE_NAMES[4], dpi=180)
    plt.close(fig)

    # 6: the Fed's monetary policy rate is distinct from its asset holdings.
    fig, axes = _figure("The price of overnight money fell; the Fed's balance sheet grew", "Fed effective funds rate (monthly) and total Fed assets, WALCL (weekly; million USD converted to trillion USD). Holdings are not fiscal expenditures or net household transfers.", panels=2)
    _shade_recession(axes[0])
    _plot_series(axes[0], s["FEDFUNDS"], color=BLUE, label="Effective federal funds rate")
    axes[0].set_ylabel("Annualized interest rate (%)")
    _legend(axes[0])
    _plot_series(axes[1], s["WALCL"], color=TEAL, label="Fed total assets", scale=1 / 1_000_000)
    axes[1].set_ylabel("Fed assets ($ trillion)")
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[5], dpi=180)
    plt.close(fig)

    # 7: SCF repeated-cross-section median net worth, not a panel of families.
    fig, axes = _figure("A recovery in aggregate wealth did not erase the racial wealth gap", "Federal Reserve SCF medians, thousands of 2022 dollars, 2007–2019. Different sampled families each survey; no confidence bands are represented. Right panel shows median ratios.", panels=2)
    survey_years = sorted(wealth)
    for group, color in (("White", BLUE), ("Black", RED), ("Hispanic", AMBER)):
        axes[0].plot(survey_years, [wealth[y][group] for y in survey_years], marker="o", lw=2.2, color=color, label=group)
    axes[0].set_ylabel("Median family net worth ($1,000, 2022)")
    _legend(axes[0])
    for group, color in (("Black", RED), ("Hispanic", AMBER)):
        axes[1].plot(survey_years, [100 * wealth[y][group] / wealth[y]["White"] for y in survey_years], marker="o", lw=2.2, color=color, label=f"{group} / White")
    axes[1].set_ylabel("Share of White median wealth (%)")
    axes[1].set_ylim(0, 30)
    _legend(axes[1])
    for ax in axes:
        ax.set_xticks(survey_years)
        ax.set_xlim(2006.3, 2019.7)
        ax.grid(axis="y", color="#DCE5E9")
    fig.savefig(figures / FIGURE_NAMES[6], dpi=180)
    plt.close(fig)

    # 8: entirely hypothetical capital arithmetic, no observed bank data.
    fig, (ax,) = _figure("HYPOTHETICAL: asset markdowns can force larger balance-sheet sales", "Invented bank with 100 initial asset units; initial equity 8%, 10%, or 12%, target capital/assets 8%, no recapitalization or earnings. Sales are in addition to the original asset markdown.")
    losses = np.linspace(0, 0.06, 61)
    for initial, color in ((0.08, RED), (0.10, AMBER), (0.12, BLUE)):
        ax.plot(100 * losses, [100 * equity_constrained_sales(float(x), initial) for x in losses], color=color, lw=2.4, label=f"Starting equity = {initial:.0%} of assets")
    ax.scatter([2], [100 * equity_constrained_sales(0.02, 0.08)], color=RED, zorder=5)
    ax.annotate("2% loss → 23% further sales", (2, 23), xytext=(10, 18), textcoords="offset points", color=RED)
    ax.set_xlabel("Exogenous asset markdown (% of initial assets)")
    ax.set_ylabel("Further asset sales (% of initial assets)")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", color="#DCE5E9")
    _legend(ax)
    fig.savefig(figures / FIGURE_NAMES[7], dpi=180)
    plt.close(fig)


def _write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    """Write deterministic source-linked result tables."""
    if not records:
        raise ValueError("Cannot write empty result table")
    fieldnames = tuple(records[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in records:
            if tuple(row) != fieldnames:
                raise ValueError("Result record field ordering changed")
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})


def make_results(output: Path) -> None:
    """Write the three numerical comparisons read by the article."""
    directory = output / "results"
    directory.mkdir(parents=True, exist_ok=True)
    for name, records in zip(RESULT_NAMES, (milestones(), racial_outcomes(), capital_scenarios()), strict=True):
        _write_csv(directory / name, records)


def main() -> None:
    """Regenerate all figures and results from frozen sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT, help="alternative artifact directory")
    args = parser.parse_args()
    make_figures(args.output)
    make_results(args.output)
    print(f"Generated {len(FIGURE_NAMES)} figures and {len(RESULT_NAMES)} result tables in {args.output}")


if __name__ == "__main__":
    main()
