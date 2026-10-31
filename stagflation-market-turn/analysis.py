"""Eight reproducible figures on the U.S. political economy, 1970–1990.

Figures 1–7 plot descriptive transcriptions of original BLS, Federal Reserve,
Census Bureau, and Office of Management and Budget series. Figure 8 is a
deterministic, explicitly hypothetical real-wage indexing mechanism.

These figures do not identify effects of Volcker, Reagan, deregulation, or
any other historical policy. Dates, source vintages and transformations are
documented in README.md. No network access is needed to reproduce results.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURE_NAMES = (
    "figure_01_cpi_inflation.png",
    "figure_02_unemployment.png",
    "figure_03_federal_funds.png",
    "figure_04_racial_unemployment.png",
    "figure_05_household_gini.png",
    "figure_06_union_membership.png",
    "figure_07_fiscal_balance.png",
    "figure_08_hypothetical_wage_indexation.png",
)
INK, TEAL, BLUE = "#172D3B", "#047D83", "#4279A5"
RED, GOLD, PURPLE, GREY = "#BB5960", "#D38B37", "#746CA5", "#617780"
PAPER = "#FCFCFA"


def read_numeric_csv(
    filename: str,
    columns: Iterable[str],
    year_key: str = "year",
) -> dict[str, np.ndarray]:
    """Read ordered source transcriptions, preserving blank values as NaN.

    A blank value means genuinely unavailable for that named source series,
    never zero. All non-year columns are numeric when present.
    """
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty data file: {filename}")
    result: dict[str, np.ndarray] = {}
    for key in (year_key, *columns):
        try:
            result[key] = np.array(
                [float(row[key]) if row[key].strip() else np.nan for row in rows],
                dtype=float,
            )
        except (KeyError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid column {key!r} in {filename}") from exc
    years = result[year_key]
    if np.any(~np.isfinite(years)) or not np.all(np.diff(years) > 0):
        raise ValueError("All year identifiers must be finite and strictly ascending")
    if np.any(years != np.floor(years)):
        raise ValueError("Year identifiers must be integers")
    for key in columns:
        if np.any(np.isinf(result[key])):
            raise ValueError("Source values cannot be infinite")
    return result


def macro_data() -> dict[str, np.ndarray]:
    """Annual means of 12 monthly FRED observations, 1969–1990.

    CPIAUCNS is non-seasonally adjusted; the three labor and policy rates
    use the source series' own adjustment conventions. The 1969 CPI value
    is retained to calculate 1970 annual-average inflation.
    """
    return read_numeric_csv(
        "fred_annual_macro.csv",
        ("cpi_u_annual_average_index", "unemployment_annual_average_percent",
         "fed_funds_annual_average_percent",
         "black_unemployment_annual_average_percent",
         "white_unemployment_annual_average_percent"),
    )


def gini_data() -> dict[str, np.ndarray]:
    """Census household money-income Gini, 1967–1990, series GINIALLRH."""
    return read_numeric_csv("census_household_gini.csv",
                            ("household_money_income_gini",))


def union_data() -> dict[str, np.ndarray]:
    """BLS union membership/representation comparable series, 1983–1990."""
    return read_numeric_csv(
        "bls_union_membership.csv",
        ("union_membership_rate_percent", "union_members_thousands",
         "employed_wage_salary_workers_thousands",
         "represented_by_unions_rate_percent"),
    )


def fiscal_data() -> dict[str, np.ndarray]:
    """OMB unified fiscal balance as a percent of fiscal-year GDP."""
    return read_numeric_csv(
        "omb_fiscal_balance.csv",
        ("unified_surplus_deficit_percent_of_fiscal_gdp",),
        year_key="fiscal_year",
    )


def annual_average_inflation(cpi_index: np.ndarray) -> np.ndarray:
    """Calculate annual-average CPI inflation, 100*(CPI_t/CPI_{t-1}-1).

    Input annual CPI must contain the preceding base year; output is one
    observation shorter. This is NOT December-to-December CPI inflation.
    """
    cpi = np.asarray(cpi_index, dtype=float)
    if cpi.ndim != 1 or cpi.size < 2:
        raise ValueError("Annual CPI series must be one-dimensional, >=2 values")
    if np.any(~np.isfinite(cpi)) or np.any(cpi <= 0):
        raise ValueError("CPI indexes must be positive and finite")
    return 100 * (cpi[1:] / cpi[:-1] - 1)


def racial_unemployment_gap(black: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Black minus White unemployment rates in percentage points.

    A missing Black rate remains missing. No interpolation is attempted.
    These descriptive rate differences do not identify policy effects.
    """
    b, w = np.asarray(black, dtype=float), np.asarray(white, dtype=float)
    if b.ndim != 1 or b.shape != w.shape or b.size == 0:
        raise ValueError("Rates must be equal-shaped, nonempty 1D series")
    for series in (b, w):
        present = series[np.isfinite(series)]
        if np.any((present < 0) | (present > 100)) or np.any(np.isinf(series)):
            raise ValueError("Unemployment rates must be percentages in [0,100]")
    return b - w


def _fraction(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return float(value)


def illustrative_real_wage(
    periods: np.ndarray,
    baseline_nominal_growth: float = .03,
    annual_inflation: float = .08,
    inflation_indexation: float = 0,
) -> np.ndarray:
    """Invented real-wage index under fixed inflation and partial indexation.

    The nominal growth assumption is (1-a)*g + a*pi for indexation a.
    At t years, purchasing power is 100*((1+nominal)/(1+pi))**t.
    It is NOT a model of real U.S. wages, bargaining equilibrium, productivity,
    labor demand, or actual contracts in the 1970s or 1980s.
    """
    years = np.asarray(periods, dtype=float)
    if years.ndim != 1 or years.size == 0 or np.any(~np.isfinite(years)):
        raise ValueError("Periods must form a nonempty finite 1D array")
    if np.any(years < 0) or np.any(years != np.floor(years)):
        raise ValueError("Periods must be nonnegative whole years")
    g = _fraction(baseline_nominal_growth, "baseline_nominal_growth")
    pi = _fraction(annual_inflation, "annual_inflation")
    a = _fraction(inflation_indexation, "inflation_indexation")
    nominal_growth = (1 - a) * g + a * pi
    return 100 * ((1 + nominal_growth) / (1 + pi)) ** years


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "savefig.dpi": 180,
        "font.size": 11, "axes.titlesize": 16,
        "axes.titleweight": "bold", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#BBC6CC",
    })


def _format(ax: plt.Axes, xlabel: str = "Year") -> None:
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#9CAEB8", alpha=.22)


def _save(fig: plt.Figure, path: Path, note: str) -> None:
    fig.subplots_adjust(bottom=.19)
    fig.text(.065, .02, note, fontsize=8.1, color=GREY)
    fig.savefig(path, bbox_inches="tight", pad_inches=.22)
    plt.close(fig)


def _inflation(path: Path) -> None:
    d = macro_data()
    years = d["year"][1:]
    inflation = annual_average_inflation(d["cpi_u_annual_average_index"])
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    ax.plot(years, inflation, color=RED, lw=2.8, marker="o", ms=5.5)
    ax.axhline(0, color=GREY, lw=1)
    ax.annotate(f"1980: {inflation[10]:.1f}%", (1980, inflation[10]),
                (1981.8, 15), fontsize=11, weight="bold",
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(title="Inflation returned as a central political problem",
           ylabel="Annual-average CPI-U inflation (%)",
           xlim=(1969.4, 1990.6), ylim=(0, 17))
    ax.set_xticks([1970, 1974, 1978, 1982, 1986, 1990])
    _format(ax)
    _save(fig, path, "HISTORICAL • BLS CPIAUCNS via FRED; annual mean CPI versus preceding annual mean (not Dec/Dec).")


def _unemployment(path: Path) -> None:
    d = macro_data()
    mask = d["year"] >= 1970
    years = d["year"][mask]
    rates = d["unemployment_annual_average_percent"][mask]
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    ax.plot(years, rates, color=BLUE, lw=3, marker="o", ms=5.5)
    ax.axvspan(1981.0, 1982.9, color=RED, alpha=.07)
    ax.annotate("1982: 9.7% (annual mean)", (1982, rates[12]),
                (1977.4, 11.0), fontsize=10.2,
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(title="The disinflation period included severe unemployment",
           ylabel="Civilian unemployment rate (%)",
           xlim=(1969.4, 1990.6), ylim=(2, 12.7))
    ax.set_xticks([1970, 1974, 1978, 1982, 1986, 1990])
    _format(ax)
    _save(fig, path, "HISTORICAL • BLS UNRATE via FRED, mean of 12 seasonally adjusted monthly rates.")


def _funds(path: Path) -> None:
    d = macro_data()
    mask = d["year"] >= 1970
    years, rates = d["year"][mask], d["fed_funds_annual_average_percent"][mask]
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    ax.plot(years, rates, color=TEAL, lw=3, marker="o", ms=5.5)
    ax.axvline(1979, color=GOLD, lw=1.3, ls="--", label="Volcker became Fed chair")
    ax.annotate("1981 annual mean: 16.4%", (1981, rates[11]), (1982.6, 18.2),
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(title="Monetary restraint reached extraordinary nominal rates",
           ylabel="Effective federal funds rate (%)",
           xlim=(1969.4, 1990.6), ylim=(0, 20))
    ax.set_xticks([1970, 1974, 1978, 1982, 1986, 1990])
    _format(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    _save(fig, path, "HISTORICAL • Federal Reserve FEDFUNDS via FRED, mean of 12 monthly rates; not the announced target.")


def _racial(path: Path) -> None:
    d = macro_data()
    mask = d["year"] >= 1972
    y = d["year"][mask]
    black = d["black_unemployment_annual_average_percent"][mask]
    white = d["white_unemployment_annual_average_percent"][mask]
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    ax.plot(y, black, color=RED, lw=2.9, marker="s", ms=5, label="Black")
    ax.plot(y, white, color=BLUE, lw=2.9, marker="o", ms=5, label="White")
    ax.fill_between(y, white, black, color=GOLD, alpha=.15)
    ax.annotate("1983 Black: 19.5%", (1983, 19.5), (1985, 19.8),
                fontsize=10, arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(title="The same labor market did not impose equal risks",
           ylabel="Unemployment rate (%)", xlim=(1971.4, 1990.6),
           ylim=(2, 22))
    ax.set_xticks([1972, 1976, 1980, 1984, 1988, 1990])
    _format(ax)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path, "HISTORICAL • BLS LNS14000006 (Black) and LNS14000003 (White), annual means of monthly SA rates.")


def _gini(path: Path) -> None:
    d = gini_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    ax.plot(d["year"], d["household_money_income_gini"], color=PURPLE,
            lw=3, marker="o", ms=5.2)
    ax.axvspan(1981, 1988, color=GOLD, alpha=.075)
    ax.annotate("1980: 0.403", (1980, .403), (1974.0, .428),
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.annotate("1990: 0.428", (1990, .428), (1985.0, .447),
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(title="Household money-income inequality increased",
           ylabel="Household money-income Gini ratio",
           xlim=(1966.6, 1990.7), ylim=(.355, .459))
    ax.set_xticks([1967, 1971, 1975, 1979, 1983, 1987, 1990])
    _format(ax)
    _save(fig, path, "HISTORICAL • Census GINIALLRH via FRED; household money income, not wealth or causal tax incidence.")


def _union(path: Path) -> None:
    d = union_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    ax.plot(d["year"], d["union_membership_rate_percent"], color=RED,
            marker="o", lw=2.9, label="Union members")
    ax.plot(d["year"], d["represented_by_unions_rate_percent"], color=TEAL,
            marker="s", lw=2.9, label="Represented by unions")
    ax.set(title="Union membership declined in the comparable 1983–90 series",
           ylabel="Share of employed wage/salary workers (%)",
           xlim=(1982.6, 1990.4), ylim=(14, 25.5))
    ax.set_xticks(list(range(1983, 1991)))
    _format(ax)
    ax.legend(frameon=False)
    _save(fig, path, "HISTORICAL • BLS CPS union affiliation Table 35; comparable series begins 1983, not 1970.")


def _fiscal(path: Path) -> None:
    d = fiscal_data()
    y = d["fiscal_year"]
    values = d["unified_surplus_deficit_percent_of_fiscal_gdp"]
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    ax.bar(y, values, color=[RED if yr >= 1981 else BLUE for yr in y], width=.72)
    ax.axhline(0, color=INK, lw=1.2)
    ax.annotate("FY 1983: −5.88% of GDP", (1983, values[13]),
                (1976, -6.5), fontsize=10.3,
                arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(title="Federal budget deficits widened in the early 1980s",
           ylabel="Unified fiscal balance (% of fiscal-year GDP)",
           xlim=(1969.4, 1990.6), ylim=(-7.6, .65))
    ax.set_xticks([1970, 1974, 1978, 1982, 1986, 1990])
    _format(ax, xlabel="Fiscal year (not calendar year)")
    _save(fig, path, "HISTORICAL • OMB via FRED FYFSDFYGDP; 1977 changes fiscal-year ending month (June to September).")


def _hypothetical(path: Path) -> None:
    periods = np.arange(0, 21)
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    for alpha, color, label in (
        (0.0, RED, "No inflation indexation"),
        (.5, GOLD, "50% indexation"),
        (1.0, TEAL, "Full indexation"),
    ):
        ax.plot(periods, illustrative_real_wage(periods, inflation_indexation=alpha),
                color=color, lw=3, label=label)
    ax.axhline(100, color=GREY, ls=":", lw=1)
    ax.set(title="The protection in a wage rule changes real purchasing power",
           xlabel="Years in hypothetical constant-inflation scenario",
           ylabel="Real-wage index (initial year = 100)",
           ylim=(30, 110), xlim=(0, 20))
    _format(ax, xlabel="Years")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, path, "HYPOTHETICAL • 8% fixed inflation, 3% base nominal wage growth; 0/50/100% invented indexation.")


def _write_csv(filename: Path, fields: tuple[str, ...],
               records: Iterable[tuple[object, ...]]) -> None:
    with filename.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(records)


def _generate_tables(result_directory: Path) -> None:
    result_directory.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[object, ...]] = []
    files: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
        ("fred_annual_macro.csv", "year",
         ("cpi_u_annual_average_index", "unemployment_annual_average_percent",
          "fed_funds_annual_average_percent",
          "black_unemployment_annual_average_percent",
          "white_unemployment_annual_average_percent"), "FRED BLS/Federal Reserve"),
        ("census_household_gini.csv", "year", ("household_money_income_gini",),
         "U.S. Census GINIALLRH"),
        ("bls_union_membership.csv", "year",
         ("union_membership_rate_percent", "union_members_thousands",
          "employed_wage_salary_workers_thousands",
          "represented_by_unions_rate_percent"), "BLS CPS Table 35"),
        ("omb_fiscal_balance.csv", "fiscal_year",
         ("unified_surplus_deficit_percent_of_fiscal_gdp",), "OMB FYFSDFYGDP"),
    )
    for filename, time_key, cols, source in files:
        data = read_numeric_csv(filename, cols, year_key=time_key)
        for i, year in enumerate(data[time_key]):
            for key in cols:
                value = data[key][i]
                rows.append((int(year), time_key, key, "" if np.isnan(value)
                             else f"{value:.6f}",
                             "missing" if np.isnan(value) else "historical_descriptive",
                             source))
    _write_csv(result_directory / "source_based_summary.csv",
               ("year", "time_basis", "series", "value", "evidence_class", "source"),
               rows)

    d = macro_data()
    inflation = annual_average_inflation(d["cpi_u_annual_average_index"])
    gap = racial_unemployment_gap(
        d["black_unemployment_annual_average_percent"],
        d["white_unemployment_annual_average_percent"],
    )
    result_rows = []
    for i in range(1, len(d["year"])):
        year = int(d["year"][i])
        result_rows.append((year, "cpi_annual_average_inflation_percent",
                            f"{inflation[i - 1]:.6f}", "derived_descriptive"))
        if np.isfinite(gap[i]):
            result_rows.append((year, "black_minus_white_unemployment_pp",
                                f"{gap[i]:.6f}", "derived_descriptive"))
    _write_csv(result_directory / "descriptive_derived_metrics.csv",
               ("year", "metric", "value", "evidence_class"), result_rows)

    periods = np.arange(21)
    _write_csv(result_directory / "hypothetical_real_wage_paths.csv",
               ("years", "inflation_indexation_share", "real_wage_index",
                "evidence_class"),
               ((int(year), f"{alpha:.1f}", f"{value:.6f}", "hypothetical")
                for alpha in (0.0, .5, 1.0)
                for year, value in zip(
                    periods, illustrative_real_wage(periods, inflation_indexation=alpha),
                )))


def build_all(output: Path | None = None) -> list[Path]:
    """Regenerate eight PNGs and three inspectable result CSVs."""
    base = ROOT if output is None else Path(output)
    figure_dir = base / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _style()
    builders: tuple[Callable[[Path], None], ...] = (
        _inflation, _unemployment, _funds, _racial,
        _gini, _union, _fiscal, _hypothetical,
    )
    figures = [figure_dir / name for name in FIGURE_NAMES]
    for draw, destination in zip(builders, figures):
        draw(destination)
    _generate_tables(base / "results")
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Root directory for regenerated figures and result tables")
    options = parser.parse_args()
    for figure in build_all(options.output):
        print(figure)
