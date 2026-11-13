"""Render publication-size figures from the verified descriptive research panel.

The graphs do not estimate causal effects. Their axes are kept separate wherever
the underlying units or survey methods differ.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from analysis import (
    COUNTRIES,
    DATA_DIR,
    LABOUR_VARIANTS,
    RESULTS_DIR,
    ROOT,
    read_labour_shares,
    read_us_history,
    write_or_check_results,
)

FIGURES_DIR = ROOT / "figures"
COUNTRY_NAMES: dict[str, str] = {
    "PT": "Portugal",
    "ES": "Spain",
    "FR": "France",
    "DE": "Germany",
    "EU27_2020": "EU27 aggregate",
}
BLUE, RED, GREEN, GOLD = "#23466d", "#b34c41", "#267d75", "#a77725"


def _style() -> None:
    """Set a consistent, readable, colour-blind-conscious editorial style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#343a40",
            "text.color": "#1d2935",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output: Path) -> None:
    """Save a high-resolution PNG without displaying an interactive window."""

    fig.savefig(
        output,
        dpi=185,
        bbox_inches="tight",
        metadata={"Title": output.stem, "Software": "neoliberalism-evidence-synthesis"},
    )
    plt.close(fig)


def _housing_panel(path: Path) -> dict[str, list[dict[str, float]]]:
    """Read and type-check the deterministic, verified annual derived panel."""

    out: dict[str, list[dict[str, float]]] = {country: [] for country in COUNTRIES}
    numeric = (
        "year",
        "house_price_index_2015_100",
        "nominal_fte_salary_index_2015_100",
        "house_price_to_fte_salary_index_2015_100",
        "official_price_income_index_2015_100",
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            country = row["geo"]
            if country not in out:
                raise ValueError(f"Unexpected country: {country}")
            out[country].append({key: float(row[key]) for key in numeric})
    if any(len(series) != 11 for series in out.values()):
        raise ValueError("Expected an 11-year annual panel for every geography")
    return out


def figure_house_prices_vs_salaries(
    panel: dict[str, list[dict[str, float]]], destination: Path
) -> None:
    """Compare nominal housing prices and nominal FTE salary index, 2015=100."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, geo in zip(axes.flat, ("PT", "ES", "FR", "DE")):
        entries = panel[geo]
        years = [item["year"] for item in entries]
        hpi = [item["house_price_index_2015_100"] for item in entries]
        wage = [item["nominal_fte_salary_index_2015_100"] for item in entries]
        ax.plot(years, hpi, color=BLUE, linewidth=2.6, label="House price index")
        ax.plot(
            years,
            wage,
            color=RED,
            linewidth=2.6,
            linestyle="--",
            label="Average FTE salary",
        )
        ax.axhline(100, linewidth=0.8, color="#87909b", alpha=0.5)
        ax.set_title(COUNTRY_NAMES[geo], loc="left", pad=8)
        ax.set_xticks((2015, 2018, 2021, 2025))
        ax.grid(axis="y", color="#d3d8de", alpha=0.5)
        ax.annotate(
            f"{hpi[-1]:.0f}",
            (years[-1], hpi[-1]),
            xytext=(-5, 7),
            textcoords="offset points",
            color=BLUE,
            ha="right",
        )
        ax.annotate(
            f"{wage[-1]:.0f}",
            (years[-1], wage[-1]),
            xytext=(-5, -14),
            textcoords="offset points",
            color=RED,
            ha="right",
        )
    fig.suptitle(
        "National house prices outpaced average employee salaries",
        fontsize=17,
        fontweight="bold",
        x=0.08,
        ha="left",
    )
    fig.supxlabel("Calendar year")
    fig.supylabel("Index (2015 = 100)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.025),
        frameon=False,
    )
    fig.text(
        0.08,
        -0.035,
        "Eurostat: prc_hpi_q, nama_10_fte. Nominal national indices; not median wages or a causal estimate.",
        fontsize=9,
        color="#525963",
    )
    fig.tight_layout(rect=(0.025, 0.08, 1, 0.94))
    _save(fig, destination)


def figure_two_denominators(summary: dict[str, object], destination: Path) -> None:
    """Show how changing the income denominator changes the measured pressure."""

    endpoints = summary["housing_wage"]["endpoints"]  # type: ignore[index]
    geos = ["PT", "ES", "EU27_2020", "DE", "FR"]
    fte = [
        float(endpoints[geo]["house_price_to_fte_salary_growth_pct"]) for geo in geos
    ]
    income = [float(endpoints[geo]["official_price_income_growth_pct"]) for geo in geos]
    fig, ax = plt.subplots(figsize=(11, 6.1))
    positions = list(range(len(geos)))
    width = 0.35
    bars1 = ax.bar(
        [p - width / 2 for p in positions],
        fte,
        width=width,
        color=BLUE,
        label="House prices / average FTE salary",
    )
    bars2 = ax.bar(
        [p + width / 2 for p in positions],
        income,
        width=width,
        color=GREEN,
        label="Eurostat house price / disposable income",
    )
    ax.bar_label(bars1, labels=[f"{v:+.1f}%" for v in fte], padding=3, fontsize=9)
    ax.bar_label(bars2, labels=[f"{v:+.1f}%" for v in income], padding=3, fontsize=9)
    ax.axhline(0, color="#7d858e", linewidth=0.9)
    ax.set_xticks(positions, [COUNTRY_NAMES[c] for c in geos])
    ax.set_ylim(min(-14, min(income) - 5), max(fte) + 18)
    ax.set_ylabel("Change in national price/income index, 2015–2025 (%)")
    ax.set_title(
        "The housing-income gap depends on what income means",
        loc="left",
        fontsize=16,
        pad=15,
    )
    ax.grid(axis="y", color="#d3d8de", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    fig.text(
        0.10,
        0.005,
        "EU27 overlaps the countries. The official denominator is disposable income per capita; the salary measure is gross FTE average.",
        fontsize=8.7,
        color="#525963",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, destination)


def figure_labour_share(data_dir: Path, destination: Path) -> None:
    """Show four defensible definitions of Portugal's GDP labour share."""

    share = read_labour_shares(data_dir)
    years = list(range(1995, 2025))
    colours = ("#56657a", BLUE, RED, GREEN)
    labels = {
        "raw": "Employees only (no self-employed income)",
        "employment": "Self-employed imputed by headcount",
        "hours": "Self-employed imputed by hours",
        "mixed_all": "All self-employed mixed income assigned to labour",
    }
    fig, ax = plt.subplots(figsize=(12, 6.1))
    for name, colour in zip(LABOUR_VARIANTS, colours):
        values = [share[(year, name, "market_prices")] for year in years]
        ax.plot(years, values, color=colour, linewidth=2.2, label=labels[name])
        ax.annotate(
            f"{values[-1]:.1f}%",
            (years[-1], values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
            color=colour,
        )
    ax.set_xlim(1995, 2027)
    ax.set_ylabel("Share of GDP at market prices (%)")
    ax.set_xlabel("Calendar year")
    ax.set_title(
        "Portugal's labour share: the definition changes the answer",
        loc="left",
        fontsize=16,
        pad=14,
    )
    ax.grid(axis="y", color="#d3d8de", alpha=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(
        0.10,
        0.01,
        "Eurostat national accounts, 1995–2024. Headcount, hours and income allocation assumptions are not equivalent.",
        fontsize=9,
        color="#525963",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, destination)


def _axis_grid(ax: Axes) -> None:
    """Add a quiet horizontal grid to a historical plot."""

    ax.grid(axis="y", color="#d3d8de", alpha=0.5)
    ax.set_axisbelow(True)


def figure_us_context(data_dir: Path, destination: Path) -> None:
    """Separate tax-rate and survey-inequality panels without dual-axis correlation."""

    history = read_us_history(data_dir)
    tax_years = list(history["tax"])
    gini_years = list(history["gini"])
    fig, (tax_ax, inequality_ax) = plt.subplots(2, 1, figsize=(11, 8.1), sharex=True)
    tax_ax.plot(
        tax_years, [history["tax"][y][0] for y in tax_years], color=BLUE, linewidth=2.4
    )
    tax_ax.axvline(1981, linestyle=":", color="#87909b", linewidth=1.1)
    tax_ax.set_ylabel("Highest statutory rate (%)")
    tax_ax.set_title("Top U.S. marginal income-tax rate", loc="left", pad=8)
    _axis_grid(tax_ax)
    inequality_ax.plot(
        gini_years,
        [history["gini"][y][0] for y in gini_years],
        color=RED,
        linewidth=2.4,
    )
    inequality_ax.axvline(1993, linestyle="--", color="#7e6c80", linewidth=1.3)
    inequality_ax.text(
        1993.5, 0.401, "1993 survey redesign", fontsize=9, color="#6d5c70"
    )
    inequality_ax.set_ylabel("Household money-income Gini")
    inequality_ax.set_xlabel("Calendar year")
    inequality_ax.set_ylim(0.38, 0.51)
    inequality_ax.set_title(
        "U.S. household income inequality (survey series)", loc="left", pad=8
    )
    _axis_grid(inequality_ax)
    fig.suptitle(
        "A historical context, not a causal estimate",
        fontsize=17,
        fontweight="bold",
        x=0.085,
        ha="left",
    )
    fig.text(
        0.085,
        0.01,
        "IRS/FRED statutory rate; U.S. Census household money-income Gini. 1993 survey redesign affects comparability.",
        fontsize=9,
        color="#525963",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    _save(fig, destination)


def figure_us_macro(data_dir: Path, destination: Path) -> None:
    """Plot historical aggregate growth and investment without period attribution."""

    history = read_us_history(data_dir)["macro"]
    years = list(history)
    growth = [history[year][0] for year in years]
    investment = [history[year][1] for year in years]
    fig, (growth_ax, investment_ax) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    growth_ax.plot(years, growth, color=BLUE, linewidth=1.8)
    growth_ax.axhline(0, color="#7d858e", linewidth=0.9)
    growth_ax.set_ylabel("Annual real GDP growth (%)")
    growth_ax.set_title("Economic growth remains cyclical", loc="left", pad=8)
    _axis_grid(growth_ax)

    investment_ax.plot(years, investment, color=GREEN, linewidth=2)
    investment_ax.set_ylabel("Private investment / nominal GDP (%)")
    investment_ax.set_xlabel("Calendar year")
    investment_ax.set_title(
        "Private investment's GDP share also varies", loc="left", pad=8
    )
    _axis_grid(investment_ax)
    for ax in (growth_ax, investment_ax):
        ax.axvline(1981, color="#87909b", linestyle=":", linewidth=1.1)
    fig.suptitle(
        "U.S. investment and growth: neither is a tax-policy counterfactual",
        fontsize=16,
        fontweight="bold",
        x=0.085,
        ha="left",
    )
    fig.text(
        0.085,
        0.01,
        "BEA aggregate national accounts, 1964–2018. Business cycles and simultaneous policies are not held constant.",
        fontsize=9,
        color="#525963",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    _save(fig, destination)


def render_all(
    output_dir: Path = FIGURES_DIR,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
) -> list[Path]:
    """Validate the analytical outputs, then regenerate all five offline figures."""

    summary = write_or_check_results(
        check=True, output_dir=results_dir, data_dir=data_dir
    )
    panel = _housing_panel(results_dir / "housing_wage_panel.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    targets = [
        output_dir / "fig01_nominal_house_prices_vs_salaries.png",
        output_dir / "fig02_housing_income_denominator_sensitivity.png",
        output_dir / "fig03_portugal_labour_share_definitions.png",
        output_dir / "fig04_us_tax_and_inequality_context.png",
        output_dir / "fig05_us_growth_and_investment_context.png",
    ]
    figure_house_prices_vs_salaries(panel, targets[0])
    figure_two_denominators(summary, targets[1])
    figure_labour_share(data_dir, targets[2])
    figure_us_context(data_dir, targets[3])
    figure_us_macro(data_dir, targets[4])
    return targets


def main() -> None:
    """CLI to create (but not infer causal interpretations from) five figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    for file in render_all(output_dir=args.output):
        print(file)


if __name__ == "__main__":
    main()
