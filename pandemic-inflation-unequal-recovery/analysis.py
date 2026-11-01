"""Seven sourced historical graphics, one hypothetical model, three audit tables.

All historical series are observations, not estimates of causal policy effects.
The model is an explicitly hypothetical arithmetic selection example.
"""

from __future__ import annotations

import argparse
import csv
import io
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

from prepare_data import ROOT, build_panel, monthly_rows, read_fred_sources

FIGURE_NAMES: tuple[str, ...] = (
    "figure_01_pandemic_unemployment.png",
    "figure_02_gdp_and_jobs.png",
    "figure_03_inflation_vs_price_level.png",
    "figure_04_nominal_and_real_earnings.png",
    "figure_05_median_income_and_gini.png",
    "figure_06_fed_policy_and_balance_sheet.png",
    "figure_07_housing_prices_and_shelter.png",
    "figure_08_hypothetical_wage_selection.png",
)
RESULT_NAMES: tuple[str, ...] = (
    "recovery_milestones.csv",
    "endpoint_comparisons.csv",
    "composition_scenarios.csv",
)
INK = "#193344"
BLUE = "#2C669B"
TEAL = "#007E85"
AMBER = "#B26B19"
RED = "#AE414A"
PURPLE = "#735B9E"
GREY = "#677883"
GRID = "#DEE7EB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
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


def wage_selection_mean(
    low_wage: float, high_wage: float, low_job_loss_fraction: float, initial_low_share: float = 0.5
) -> float:
    """Observed average after layoffs, with both individual wages unchanged.

    Initial workers have low-share s at pay L and (1-s) at pay H.
    A fraction p of the low-wage group disappears from employment.
    Composition alone moves the remaining mean to
    [s*(1-p)*L + (1-s)*H] / [s*(1-p) + (1-s)].
    This is bookkeeping, never an empirical layoff estimate.
    """
    inputs = (low_wage, high_wage, low_job_loss_fraction, initial_low_share)
    if any(not math.isfinite(float(value)) for value in inputs):
        raise ValueError("Non-finite composition input")
    if not 0 < low_wage < high_wage:
        raise ValueError("Pay levels must be positive and ordered")
    if not 0 <= low_job_loss_fraction <= 1 or not 0 < initial_low_share < 1:
        raise ValueError("Invalid population share or layoff fraction")
    remaining_low = initial_low_share * (1 - low_job_loss_fraction)
    remaining_high = 1 - initial_low_share
    return (remaining_low * low_wage + remaining_high * high_wage) / (remaining_low + remaining_high)


def first_recovered(observations: Mapping[date, float], baseline: float, after: date) -> date | None:
    """First post-shock observation at or above an earlier level."""
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("Invalid recovery baseline")
    for day in sorted(observations):
        if day > after and observations[day] >= baseline:
            return day
    return None


def month_distance(begin: date, end: date) -> int:
    """Number of calendar months, without spurious daily precision."""
    return 12 * (end.year - begin.year) + end.month - begin.month


def recovery_milestones() -> list[dict[str, str | int | float]]:
    """Disjoint GDP, jobs and pretax median-income recovery clocks."""
    fred = read_fred_sources()
    income = {
        date(int(row["year"]), 1, 1): float(row["real_median_household_income_2025_usd"])
        for row in build_panel()
    }
    series = (
        ("Real GDP", fred["GDPC1"], date(2019, 10, 1), date(2020, 4, 1), "Quarterly; BEA; 2019Q4 level"),
        ("Payroll employment", fred["PAYEMS"], date(2020, 2, 1), date(2020, 4, 1), "Monthly; BLS CES; February 2020 jobs"),
        ("Real median household income", income, date(2019, 1, 1), date(2020, 1, 1), "Annual; Census CPS ASEC; 2025 USD"),
    )
    rows: list[dict[str, str | int | float]] = []
    for name, values, baseline_day, cutoff, note in series:
        baseline = values[baseline_day]
        first = first_recovered(values, baseline, cutoff)
        if first is None:
            raise ValueError(f"No observed return to benchmark by end of 2025: {name}")
        rows.append(
            {
                "indicator": name,
                "baseline_date": baseline_day.isoformat(),
                "baseline_value": baseline,
                "first_at_or_above_date": first.isoformat(),
                "months_after_feb2020": month_distance(date(2020, 2, 1), first),
                "note": note,
            }
        )
    return rows


def endpoint_comparisons() -> list[dict[str, str | float]]:
    """Heterogeneous before/after ratios with explicit distinct benchmarks."""
    fred = read_fred_sources()
    monthly = {date.fromisoformat(str(row["observation_date"])): row for row in monthly_rows()}
    annual = {int(row["year"]): row for row in build_panel()}
    end = monthly[date(2025, 12, 1)]
    items: tuple[tuple[str, str, str, float, float, str], ...] = (
        ("CPI-U all-items", "Dec 2019", "Dec 2025", 100, float(end["cpi_sa_index_dec2019"]), "BLS seasonally adjusted index; consumption basket"),
        ("CPI-U shelter", "Dec 2019", "Dec 2025", 100, float(end["shelter_sa_index_dec2019"]), "BLS shelter services index, not asking rent"),
        ("Average hourly earnings, nominal", "Dec 2019", "Dec 2025", 100, float(end["nominal_earnings_index_dec2019"]), "BLS CES total private; worker composition changes"),
        ("Average hourly earnings, CPI-adjusted", "Dec 2019", "Dec 2025", 100, float(end["real_earnings_index_dec2019"]), "Nominal BLS CES divided by all-item CPI-U"),
        ("FHFA all-transactions house prices", "2019Q4", "2025Q4", fred["USSTHPI"][date(2019, 10, 1)], fred["USSTHPI"][date(2025, 10, 1)], "Not seasonally adjusted; refinance and sales data"),
        ("Real GDP", "2019Q4", "2025Q4", fred["GDPC1"][date(2019, 10, 1)], fred["GDPC1"][date(2025, 10, 1)], "BEA quarterly chained-dollar GDP"),
        ("Nonfarm payroll employment", "Feb 2020", "Dec 2025", fred["PAYEMS"][date(2020, 2, 1)], fred["PAYEMS"][date(2025, 12, 1)], "BLS CES, thousands of payroll jobs"),
        ("Real median household money income", "2019", "2025", float(annual[2019]["real_median_household_income_2025_usd"]), float(annual[2025]["real_median_household_income_2025_usd"]), "Census CPS ASEC, 2025 inflation-adjusted dollars"),
    )
    return [
        {
            "indicator": name,
            "baseline_period": start,
            "endpoint_period": finish,
            "baseline_value": baseline,
            "endpoint_value": last,
            "percent_change": 100 * (last / baseline - 1),
            "measurement_note": note,
        }
        for name, start, finish, baseline, last, note in items
    ]


def composition_scenarios() -> list[dict[str, float]]:
    """Fixed $15 and $35 wages, 50/50 starting workforce, 0–100% low-paid layoffs."""
    scenarios: list[dict[str, float]] = []
    initial_average = wage_selection_mean(15, 35, 0)
    for step in range(11):
        lost = step / 10
        observed_mean = wage_selection_mean(15, 35, lost)
        scenarios.append(
            {
                "low_wage_group_jobs_lost_pct": 100 * lost,
                "remaining_employment_pct_initial": 100 * (1 - 0.5 * lost),
                "observed_average_hourly_wage_usd": observed_mean,
                "observed_wage_index_initial_100": 100 * observed_mean / initial_average,
                "individual_wage_growth_pct": 0.0,
            }
        )
    return scenarios


def _figure(title: str, subtitle: str, panels: int = 1) -> tuple[Figure, list[Axes]]:
    """Publication-style figure with legible title and subtitle space."""
    fig, axis = plt.subplots(panels, 1, figsize=(10.4, 5 if panels == 1 else 7.5), sharex=panels > 1)
    axs: list[Axes] = list(np.ravel(np.atleast_1d(axis)))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.77 if panels == 1 else 0.84, bottom=0.13, hspace=0.48)
    fig.suptitle(title, x=0.11, y=0.967, ha="left", fontsize=16.7, fontweight="bold", color=INK)
    fig.text(0.11, 0.903 if panels == 1 else 0.913, subtitle, ha="left", va="top", fontsize=9.1, color=GREY, wrap=True)
    return fig, axs


def _legend(ax: Axes) -> None:
    ax.legend(loc="best", frameon=False, fontsize=9)


def _time_axis(ax: Axes) -> None:
    ax.set_xlim(date(2019, 1, 1), date(2025, 12, 31))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(axis="x", labelbottom=True)


def _pandemic_recession(ax: Axes) -> None:
    ax.axvspan(date(2020, 2, 1), date(2020, 4, 30), color=GREY, alpha=0.12, label="NBER recession")


def _values(rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    """Translate official missing values to matplotlib gaps, not estimates."""
    return [math.nan if row[field] is None else float(row[field]) for row in rows]


def _plot_fred(ax: Axes, series: Mapping[date, float], color: str, label: str, baseline: float = 1) -> None:
    dates = sorted(day for day in series if 2019 <= day.year <= 2025)
    ax.plot(dates, [100 * series[d] / baseline if baseline != 1 else series[d] for d in dates], color=color, lw=2.4, label=label)


def make_figures(output: Path) -> None:
    """Produce exactly eight figures; history and toy model never conflated."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fred = read_fred_sources()
    monthly = monthly_rows()
    annual = build_panel()
    dates = [date.fromisoformat(str(row["observation_date"])) for row in monthly]
    years = [int(row["year"]) for row in annual]

    # 1. The dramatic two-month recession and unequal levels of unemployment.
    fig, axes = _figure("A short recession, a long unequal labor shock", "BLS CPS, seasonally adjusted; NBER February–April 2020 recession. October 2025 is unavailable; April 2020 was affected by survey misclassification.", panels=2)
    for ax in axes:
        _pandemic_recession(ax)
    axes[0].plot(dates, _values(monthly, "total_unemployment_pct"), color=RED, lw=2.4, label="Total unemployment")
    axes[0].set_ylabel("Unemployment (%)")
    _legend(axes[0])
    axes[1].plot(dates, _values(monthly, "black_unemployment_pct"), color=PURPLE, lw=2.2, label="Black unemployment")
    axes[1].plot(dates, _values(monthly, "white_unemployment_pct"), color=BLUE, lw=2.2, label="White unemployment")
    axes[1].set_ylabel("Unemployment (%)")
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[0], dpi=170)
    plt.close(fig)

    # 2. Output and aggregate jobs: different units and recovery timing.
    fig, axes = _figure("The output clock and the jobs clock", "BEA quarterly real GDP (2019Q4 = 100) and BLS monthly payroll jobs (February 2020 = 100), each rebased to its own benchmark.", panels=2)
    _pandemic_recession(axes[0])
    _plot_fred(axes[0], fred["GDPC1"], BLUE, "Real GDP", fred["GDPC1"][date(2019, 10, 1)])
    axes[0].set_ylabel("GDP index")
    axes[0].axhline(100, color=GREY, ls="--", lw=0.9)
    _legend(axes[0])
    _pandemic_recession(axes[1])
    _plot_fred(axes[1], fred["PAYEMS"], TEAL, "Nonfarm payroll jobs", fred["PAYEMS"][date(2020, 2, 1)])
    axes[1].set_ylabel("Jobs index")
    axes[1].axhline(100, color=GREY, ls="--", lw=0.9)
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[1], dpi=170)
    plt.close(fig)

    # 3. The central conceptual distinction: annual rate versus accumulated level.
    fig, axes = _figure("Inflation cooled; the price level stayed higher", "BLS all-items CPI-U: seasonally adjusted level rebased to December 2019, versus the not-seasonally-adjusted 12-month inflation rate. Missing October 2025 retained.", panels=2)
    axes[0].plot(dates, _values(monthly, "cpi_sa_index_dec2019"), color=RED, lw=2.7, label="Accumulated price level")
    axes[0].axhline(100, color=GREY, ls="--", lw=0.9)
    axes[0].set_ylabel("Dec 2019 = 100")
    _legend(axes[0])
    axes[1].plot(dates, _values(monthly, "cpi_nsa_12m_inflation_pct"), color=BLUE, lw=2.5, label="12-month inflation")
    axes[1].axhline(2, color=GREY, ls="--", lw=0.9, label="2% reference; not the Fed's PCE metric")
    axes[1].scatter([date(2022, 6, 1)], [9.1], color=BLUE, zorder=3)
    axes[1].annotate("June 2022: 9.1%", (date(2022, 6, 1), 9.1), xytext=(13, -21), textcoords="offset points", color=BLUE)
    axes[1].set_ylabel("12-month rate (%)")
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[2], dpi=170)
    plt.close(fig)

    # 4. All private workers, not a balanced panel of the same individuals.
    fig, (ax,) = _figure("Bigger paychecks did not always buy more", "BLS total-private average hourly earnings, divided by headline CPI-U (seasonally adjusted). Both series indexed December 2019 = 100; sample composition shifts in 2020.")
    ax.plot(dates, _values(monthly, "nominal_earnings_index_dec2019"), color=TEAL, lw=2.7, label="Nominal average hourly earnings")
    ax.plot(dates, _values(monthly, "real_earnings_index_dec2019"), color=RED, lw=2.7, label="CPI-deflated average hourly earnings")
    ax.axhline(100, color=GREY, lw=0.9, ls="--")
    ax.set_ylabel("Dec 2019 = 100")
    _legend(ax)
    _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[3], dpi=170)
    plt.close(fig)

    # 5. Annual income is not individual wage and pretax Gini is not posttax.
    fig, axes = _figure("Aggregate recovery did not settle distribution", "Census CPS ASEC annual estimates via FRED: real median household money income in 2025 dollars, and pretax household money-income Gini. Breaks and survey sampling matter.", panels=2)
    axes[0].plot(years, [float(r["median_income_index_2019"]) for r in annual], color=TEAL, lw=2.4, marker="o", label="Median real household income")
    axes[0].axhline(100, color=GREY, ls="--", lw=0.9)
    axes[0].set_ylabel("2019 = 100")
    _legend(axes[0])
    axes[1].plot(years, [float(r["household_income_gini"]) for r in annual], color=RED, lw=2.4, marker="o", label="Household money-income Gini")
    axes[1].set_ylabel("Gini (0–1)")
    axes[1].set_ylim(0.46, 0.51)
    _legend(axes[1])
    for ax in axes:
        ax.set_xticks(years)
        ax.set_xlim(2019, 2025)
        ax.grid(axis="y", color=GRID)
    fig.savefig(figures / FIGURE_NAMES[4], dpi=170)
    plt.close(fig)

    # 6. Interest rates and the balance sheet are different policy measures.
    fig, axes = _figure("Emergency easing followed by monetary tightening", "Federal Reserve effective federal funds rate (monthly) and total Fed assets (weekly WALCL, millions converted to trillions USD). Not transfers to households.", panels=2)
    _plot_fred(axes[0], fred["FEDFUNDS"], BLUE, "Effective federal funds rate")
    axes[0].set_ylabel("Annualized rate (%)")
    _legend(axes[0])
    weekly_days = sorted(day for day in fred["WALCL"] if 2019 <= day.year <= 2025)
    axes[1].plot(weekly_days, [fred["WALCL"][day] / 1_000_000 for day in weekly_days], color=PURPLE, lw=2.4, label="Federal Reserve assets")
    axes[1].set_ylabel("Assets ($ trillion)")
    _legend(axes[1])
    for ax in axes:
        _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[5], dpi=170)
    plt.close(fig)

    # 7. Ownership asset prices versus rental/owner-equivalent consumption costs.
    fig, (ax,) = _figure("Housing was two different inflation stories", "FHFA all-transactions U.S. home-price index (quarterly, 2019Q4 = 100) and BLS CPI-U shelter (monthly, December 2019 = 100): distinct baskets, methods and lags.")
    _plot_fred(ax, fred["USSTHPI"], AMBER, "FHFA house prices", fred["USSTHPI"][date(2019, 10, 1)])
    ax.plot(dates, _values(monthly, "shelter_sa_index_dec2019"), color=PURPLE, lw=2.5, label="CPI shelter services")
    ax.axhline(100, color=GREY, lw=0.9, ls="--")
    ax.set_ylabel("Own pre-pandemic base = 100")
    _legend(ax)
    _time_axis(ax)
    fig.savefig(figures / FIGURE_NAMES[6], dpi=170)
    plt.close(fig)

    # 8. Clearly counterfactual arithmetic; $15/$35 individual wages never rise.
    fig, axes = _figure("Hypothetical: a pay average can rise when jobs vanish", "Invented 50/50 workforce at fixed hourly pay of USD 15 and USD 35. Only low-paid layoffs vary. NOT measured pandemic layoffs or an estimate of their contribution.", panels=2)
    rows = composition_scenarios()
    x = [r["low_wage_group_jobs_lost_pct"] for r in rows]
    axes[0].plot(x, [r["observed_average_hourly_wage_usd"] for r in rows], color=TEAL, lw=2.7, marker="o", label="Average among remaining employed")
    axes[0].axhline(25, color=GREY, ls="--", lw=1, label="Starting average = $25")
    axes[0].set_ylabel("Observed mean pay ($/hour)")
    _legend(axes[0])
    axes[1].plot(x, [r["remaining_employment_pct_initial"] for r in rows], color=RED, lw=2.7, marker="o", label="Jobs still employed")
    axes[1].set_ylabel("Initial headcount = 100%")
    axes[1].set_xlabel("Fraction of initial low-paid jobs eliminated (%)")
    _legend(axes[1])
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.grid(axis="y", color=GRID)
    fig.savefig(figures / FIGURE_NAMES[7], dpi=170)
    plt.close(fig)


def render_csv(rows: Sequence[Mapping[str, object]]) -> str:
    """Audit tables with stable header ordering and six-decimal numeric values."""
    if not rows:
        raise ValueError("Cannot render an empty table")
    columns = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()})
    return buffer.getvalue()


def result_outputs(output: Path = ROOT) -> dict[Path, str]:
    """All analytical results derived strictly from checked-in source files."""
    return {
        output / "results" / RESULT_NAMES[0]: render_csv(recovery_milestones()),
        output / "results" / RESULT_NAMES[1]: render_csv(endpoint_comparisons()),
        output / "results" / RESULT_NAMES[2]: render_csv(composition_scenarios()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check CSVs and PNG existence without writing")
    args = parser.parse_args()
    expected = result_outputs()
    for path, contents in expected.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != contents:
                raise SystemExit(f"Analytical table drift: {path}")
            print(f"Verified {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            print(f"Generated {path.relative_to(ROOT)}")
    if args.check:
        for name in FIGURE_NAMES:
            figure = ROOT / "figures" / name
            if not figure.is_file() or figure.stat().st_size < 20000:
                raise SystemExit(f"Missing or empty figure: {name}")
        print("Verified all eight figure files")
    else:
        make_figures(ROOT)
        print(f"Generated {len(FIGURE_NAMES)} figures")


if __name__ == "__main__":
    main()
