"""Reproduce descriptive European housing figures from frozen official-data extracts.

This module deliberately does not estimate the causal price effect of building
homes. The Bulgaria exhibit mixes annual completion totals with Q2 annual HPI
growth; the European comparative predictor is *permits*, not completions.

Run ``python analysis.py --output figures``. Outputs are deterministic for a
fixed environment and include a source/data checksum manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent
DATA: Path = ROOT / "data"
COUNTRIES: tuple[str, ...] = ("PT", "ES", "DE", "FR")
PALETTE: dict[str, str] = {
    "PT": "#c2413b",
    "ES": "#d09528",
    "DE": "#486d94",
    "FR": "#398575",
}
NAMES: dict[str, str] = {
    "PT": "Portugal",
    "ES": "Spain",
    "DE": "Germany",
    "FR": "France",
}


@dataclass(frozen=True)
class Inputs:
    """Validated, frozen data used exclusively for descriptive visualisations."""

    bulgaria: pd.DataFrame
    portugal_builds: pd.DataFrame
    portugal_occupancy: pd.DataFrame
    housing: pd.DataFrame
    permits: pd.DataFrame
    annual: pd.DataFrame
    associations: pd.DataFrame


def read_data(filename: str, required_columns: set[str]) -> pd.DataFrame:
    """Read an exact snapshot and reject missing fields, not silently impute."""

    data_path: Path = DATA / filename
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing frozen input: {data_path}")
    frame: pd.DataFrame = pd.read_csv(data_path)
    missing_columns: set[str] = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{filename} missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"{filename} contains no observations")
    return frame


def load_inputs() -> Inputs:
    """Load and perform cross-file consistency checks on the auditable data."""

    bg: pd.DataFrame = read_data(
        "bulgaria_chart_audit_2021_2025.csv",
        {"year", "completed_dwellings", "hpi_q2_yoy_pct"},
    )
    pt: pd.DataFrame = read_data(
        "portugal_new_builds_1995_2025.csv",
        {"year", "completed_new_dwellings", "provisional"},
    )
    occ: pd.DataFrame = read_data(
        "portugal_housing_2021.csv", {"occupation", "dwellings"}
    )
    hp: pd.DataFrame = read_data(
        "annual_housing_indices_2015_2025.csv",
        {"geo", "year", "real_hpi_quarter_mean", "price_income_change_from_2015_pct"},
    )
    permits: pd.DataFrame = read_data(
        "annual_supply_intensity_2015_2025.csv",
        {"geo", "year", "permitted_dwellings_per_1000_households"},
    )
    annual: pd.DataFrame = read_data(
        "annual_supply_price_diagnostics_2015_2025.csv",
        {"geo", "year", "real_hpi_yoy_pct", "permits_per_1000_households_t"},
    )
    assoc: pd.DataFrame = read_data(
        "annual_supply_price_diagnostics_2015_2025_associations.csv",
        {
            "geo",
            "permit_lead_years",
            "observed_pairs",
            "pearson_r",
            "unflagged_pearson_r",
        },
    )

    if bg.year.tolist() != [2021, 2022, 2023, 2024, 2025]:
        raise ValueError("Bulgarian Q2 audit requires exactly five chronological years")
    if bg.completed_dwellings.tolist() != [17868, 19847, 22649, 20998, 37571]:
        raise ValueError("Bulgarian completion values differ from audited NSI snapshot")
    if not np.allclose(bg.hpi_q2_yoy_pct, [9.1, 14.6, 10.7, 15.1, 15.5]):
        raise ValueError("Bulgarian Q2 HPI values differ from audited source")
    if int(occ.dwellings.sum()) != 5_970_677:
        raise ValueError("Portugal Census occupation categories do not reconcile")
    if int(pt.loc[pt.year == 2006, "completed_new_dwellings"].iloc[0]) != 68_764:
        raise ValueError("Portugal 2006 completion anchor differs from INE/Pordata")
    if int(pt.loc[pt.year == 2025, "completed_new_dwellings"].iloc[0]) != 27_301:
        raise ValueError(
            "Portugal 2025 preliminary completion anchor differs from INE/Pordata"
        )
    for country in COUNTRIES:
        for frame in (hp, permits, annual):
            rows: pd.DataFrame = frame.loc[frame.geo == country]
            if rows.year.tolist() != list(range(2015, 2026)):
                raise ValueError(
                    f"Unmatched 2015–2025 country-year panel for {country}"
                )
    if not set(COUNTRIES).issubset(set(assoc.geo)):
        raise ValueError("Missing one or more countries in correlation sensitivity")

    return Inputs(bg, pt, occ, hp, permits, annual, assoc)


def set_style() -> None:
    """Use consistent, accessible figure defaults suitable for Medium."""

    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "figure.dpi": 130,
            "savefig.dpi": 170,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.19,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
        }
    )


def finish(fig: plt.Figure, path: Path, note: str) -> None:
    """Add a data/identification footer and save an editorial image."""

    fig.text(0.03, 0.018, note, fontsize=8, ha="left", color="#596579")
    fig.tight_layout(rect=(0, 0.055, 1, 0.98))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_01_bulgaria(inputs: Inputs, output: Path) -> None:
    """Show the original chart without combining incompatible y-axis units."""

    bg = inputs.bulgaria
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(bg.year, bg.completed_dwellings, color="#4476a2", width=0.65)
    axes[0].set_ylabel("Dwellings completed / calendar year")
    axes[0].set_title(
        "Bulgaria: the two observations are true; the causal story is not identified"
    )
    for year, value in zip(bg.year, bg.completed_dwellings, strict=True):
        axes[0].text(year, value + 650, f"{value:,}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 43_000)
    axes[1].plot(bg.year, bg.hpi_q2_yoy_pct, marker="o", linewidth=2.7, color="#bf4b4c")
    axes[1].set_ylabel("Q2 house-price growth, year-on-year (%)")
    axes[1].set_xlabel("Calendar year (annual completions; price outcome in Q2)")
    axes[1].set_xticks(bg.year)
    axes[1].set_ylim(0, 19)
    for year, growth in zip(bg.year, bg.hpi_q2_yoy_pct, strict=True):
        axes[1].annotate(
            f"{growth:.1f}%",
            (year, growth),
            (0, 10),
            textcoords="offset points",
            ha="center",
        )
    finish(
        fig,
        output,
        "NSI Bulgaria; Eurostat HPI. Q2 2022 begins a new NSI HPI data-source regime; annual/Q2 periods differ.",
    )


def figure_02_stock_flow(inputs: Inputs, output: Path) -> None:
    """Express the construction *flow* relative to the known 2025 stock only."""

    bg = inputs.bulgaria
    stock_2025: int = 4_360_805
    shares: pd.Series = 100 * bg.completed_dwellings / stock_2025
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(bg.year, shares, color="#398575", width=0.62)
    ax.axhline(1.0, linestyle="--", color="#7d8794", linewidth=1.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Annual completed dwellings / fixed 2025 housing stock (%)")
    ax.set_xticks(bg.year)
    ax.set_title(
        "A 110% rise in the construction flow is still below 1% of housing stock"
    )
    for year, share in zip(bg.year, shares, strict=True):
        ax.text(year, share + 0.035, f"{share:.2f}%", ha="center")
    finish(
        fig,
        output,
        "Illustrative fixed-denominator normalisation: NSI 2025 stock 4,360,805, NOT historic annual stock levels.",
    )


def figure_03_portugal_completions(inputs: Inputs, output: Path) -> None:
    """Display the long construction contraction and incomplete recovery."""

    pt = inputs.portugal_builds
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(pt.year, pt.completed_new_dwellings, color="#c6d8e8", alpha=0.9)
    ax.plot(pt.year, pt.completed_new_dwellings, color="#33678d", linewidth=2.6)
    for year in (2002, 2006, 2015, 2025):
        val: int = int(pt.loc[pt.year == year, "completed_new_dwellings"].iloc[0])
        ax.plot(year, val, marker="o", color="#c2413b")
        ax.annotate(
            f"{year}: {val:,}",
            (year, val),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
        )
    ax.set_ylabel("New family dwellings completed / year")
    ax.set_xlabel("Year")
    ax.set_title(
        "Portugal: construction recovered, but remains far below its early-2000s peak"
    )
    finish(
        fig,
        output,
        "INE via Pordata, extraction/update 17 July 2026. Years 2021–2025 preliminary. Refurbishments excluded.",
    )


def figure_04_occupation(inputs: Inputs, output: Path) -> None:
    """Decompose total Portuguese housing stock without declaring it all marketable."""

    occ = inputs.portugal_occupancy
    colors: list[str] = ["#398575", "#d09528", "#c2413b"]
    labels: list[str] = list(occ.occupation)
    values: list[int] = [int(v) for v in occ.dwellings]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    wedges = ax.pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.38, "edgecolor": "white"},
    )[0]
    ax.legend(
        wedges,
        [
            f"{label}: {v / 1e6:.3f} million ({v / sum(values):.1%})"
            for label, v in zip(labels, values, strict=True)
        ],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
    )
    ax.set_title(
        "Portugal 2021: a large stock is not the same as an available home", pad=18
    )
    ax.text(0, 0.08, "5.97m", ha="center", va="center", fontsize=22, weight="bold")
    ax.text(0, -0.15, "all dwellings", ha="center", va="center", fontsize=11)
    finish(
        fig,
        output,
        "INE/LNEC Census 2021 categories. Vacant does NOT mean habitable, offered, accessible or where demand lies.",
    )


def figure_05_permits(inputs: Inputs, output: Path) -> None:
    """Compare annual residential permits, explicitly not completed units."""

    fig, ax = plt.subplots(figsize=(11.5, 6))
    for geo in COUNTRIES:
        dat = inputs.permits.loc[inputs.permits.geo == geo]
        ax.plot(
            dat.year,
            dat.permitted_dwellings_per_1000_households,
            label=NAMES[geo],
            linewidth=2.4,
            marker="o",
            markersize=3.2,
            color=PALETTE[geo],
        )
    ax.set_title("Europe's supply pipeline: authorised dwellings per 1,000 households")
    ax.set_ylabel("Permitted dwellings / 1,000 households")
    ax.set_xlabel("Year")
    ax.legend(ncol=4, loc="upper center")
    finish(
        fig,
        output,
        "Eurostat sts_cobp_a and lfst_hhnhtych; country survey breaks and provisional marks in archived CSV.",
    )


def figure_06_real_prices(inputs: Inputs, output: Path) -> None:
    """Show inflation-adjusted price-index divergence, not affordability itself."""

    fig, ax = plt.subplots(figsize=(11.5, 6))
    for geo in COUNTRIES:
        dat = inputs.housing.loc[inputs.housing.geo == geo]
        base: float = float(dat.loc[dat.year == 2015, "real_hpi_quarter_mean"].iloc[0])
        ax.plot(
            dat.year,
            100 * dat.real_hpi_quarter_mean / base,
            label=NAMES[geo],
            linewidth=2.5,
            color=PALETTE[geo],
        )
    ax.axhline(100, color="#8490a0", linewidth=1, linestyle="--")
    ax.set_title("Deflated house prices: four different paths, 2015 = 100")
    ax.set_ylabel("Eurostat real HPI, rebased to 2015 = 100")
    ax.set_xlabel("Year")
    ax.legend(ncol=4, loc="upper left")
    finish(
        fig,
        output,
        "Eurostat tipsho30 annual mean of four quarters. CPI/HPI deflation is NOT a measure of housing-cost burden.",
    )


def figure_07_portugal_scatter(inputs: Inputs, output: Path) -> None:
    """Illustrate association and reverse causality using actual PT observations."""

    pt = inputs.annual.loc[(inputs.annual.geo == "PT") & (inputs.annual.year >= 2016)]
    fig, ax = plt.subplots(figsize=(10, 6))
    points = ax.scatter(
        pt.permits_per_1000_households_t,
        pt.real_hpi_yoy_pct,
        c=pt.year,
        cmap="viridis",
        s=88,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    for _, row in pt.iterrows():
        ax.annotate(
            str(int(row.year)),
            (row.permits_per_1000_households_t, row.real_hpi_yoy_pct),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    fig.colorbar(points, ax=ax, label="Observation year")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xlabel("Portugal permits / 1,000 surveyed households, same year")
    ax.set_ylabel("Annual real house-price-index growth (%)")
    ax.set_title("Portugal: association is not an effect of completed construction")
    finish(
        fig,
        output,
        "10 national observations, 2016–2025. Same-year permits are endogenous; provisional and survey-break flags matter.",
    )


def figure_08_lags(inputs: Inputs, output: Path) -> None:
    """Compare non-causal Pearson associations and sample sizes by permit lag."""

    data = inputs.associations.loc[
        (inputs.associations.geo.isin(COUNTRIES))
        & (inputs.associations.permit_lead_years.isin([0, 2]))
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6))
    xpos: np.ndarray = np.arange(len(COUNTRIES), dtype=float)
    width: float = 0.34
    for idx, lag in enumerate((0, 2)):
        sub = (
            data.loc[data.permit_lead_years == lag]
            .set_index("geo")
            .loc[list(COUNTRIES)]
        )
        positions = xpos + (idx - 0.5) * width
        bars = ax.bar(
            positions,
            sub.pearson_r,
            width=width,
            label="Same year" if lag == 0 else "Permits 2 years earlier",
            color="#4476a2" if lag == 0 else "#cf9278",
        )
        for bar, n in zip(bars, sub.observed_pairs, strict=True):
            top: float = float(bar.get_height())
            ax.text(
                bar.get_x() + width / 2,
                top + (0.025 if top >= 0 else -0.07),
                f"n={int(n)}",
                ha="center",
                fontsize=9,
            )
    ax.axhline(0, color="#687381", linewidth=1.3)
    ax.set_xticks(xpos, [NAMES[g] for g in COUNTRIES])
    ax.set_ylim(-0.48, 0.9)
    ax.set_ylabel("Within-country Pearson r (not causal elasticity)")
    ax.set_title("Timing the permits does not resolve confounding or reverse causality")
    ax.legend()
    finish(
        fig,
        output,
        "Eurostat frozen diagnostics, at most 10 year-pairs/country. Excludes overlapping EU27 aggregate.",
    )


def figure_09_price_income(inputs: Inputs, output: Path) -> None:
    """Compare within-country changes in the standardised price/income ratio."""

    frame: pd.DataFrame = inputs.housing.loc[inputs.housing.geo.isin(COUNTRIES)]
    fig, ax = plt.subplots(figsize=(11.5, 6))
    for geo in COUNTRIES:
        country = frame.loc[frame.geo == geo]
        ax.plot(
            country.year,
            country.price_income_change_from_2015_pct,
            label=NAMES[geo],
            color=PALETTE[geo],
            linewidth=2.3,
        )
    ax.axhline(0, linestyle="--", color="#81909a", linewidth=1)
    ax.set_ylabel("Change versus own 2015 ratio (%)")
    ax.set_xlabel("Year")
    ax.set_title("Housing prices relative to income: changes within countries")
    ax.legend(ncol=4, loc="upper left")
    finish(
        fig,
        output,
        "Eurostat tipsho60. National standardised ratios have different baselines; plotted changes are not level comparisons.",
    )


def counterfactual_price_growth_pct(
    demand_log_shock: float, supply_log_shock: float, eta: float
) -> float:
    """Illustrative inverse-demand calculation, **not fitted to observations**.

    ``log(P_1/P_0) = demand_log_shock - eta*supply_log_shock``.
    All inputs are dimensionless log changes. ``eta`` is an assumed inverse
    supply-price sensitivity, not an identified estimate for any country.
    """

    if not all(math.isfinite(v) for v in (demand_log_shock, supply_log_shock, eta)):
        raise ValueError("Scenario parameters must be finite")
    if eta <= 0 or supply_log_shock < 0:
        raise ValueError("Require positive assumed eta and nonnegative supply growth")
    return 100 * math.expm1(demand_log_shock - eta * supply_log_shock)


def figure_10_counterfactual(inputs: Inputs, output: Path) -> None:
    """Make the missing counterfactual in the Facebook chart explicit."""

    del inputs  # This figure is deliberately not fitted to empirical inputs.
    supply_pct: np.ndarray = np.linspace(0, 4, 101)
    eta: float = 1.5
    fig, ax = plt.subplots(figsize=(10.8, 6))
    for demand_log_shock, color in (
        (0.06, "#398575"),
        (0.12, "#d09528"),
        (0.18, "#c2413b"),
    ):
        changes: list[float] = [
            counterfactual_price_growth_pct(
                demand_log_shock, math.log1p(float(x) / 100), eta
            )
            for x in supply_pct
        ]
        ax.plot(
            supply_pct,
            changes,
            linewidth=2.5,
            color=color,
            label=f"Assumed demand log-shock {100 * demand_log_shock:.0f}%",
        )
    ax.axhline(0, linestyle="--", color="#8090a2")
    ax.set_xlabel("Assumed percentage increase in effective stock")
    ax.set_ylabel("Implied model house-price change (%)")
    ax.set_title(
        "A hypothetical demand shock can raise prices even while supply expands"
    )
    ax.legend(loc="upper right")
    finish(
        fig,
        output,
        "SYNTHETIC SCENARIO ONLY. Log inverse-demand form, assumed eta=1.5. Not a Bulgaria/Portugal estimate.",
    )


FIGURES: tuple[tuple[str, Callable[[Inputs, Path], None]], ...] = (
    ("fig01_bulgaria_observed.png", figure_01_bulgaria),
    ("fig02_bulgaria_stock_flow.png", figure_02_stock_flow),
    ("fig03_portugal_new_completions.png", figure_03_portugal_completions),
    ("fig04_portugal_occupancy.png", figure_04_occupation),
    ("fig05_european_permit_intensity.png", figure_05_permits),
    ("fig06_european_real_hpi.png", figure_06_real_prices),
    ("fig07_portugal_permits_prices.png", figure_07_portugal_scatter),
    ("fig08_correlations_by_lag.png", figure_08_lags),
    ("fig09_price_income_relative.png", figure_09_price_income),
    ("fig10_synthetic_counterfactual.png", figure_10_counterfactual),
)


def sha256(path: Path) -> str:
    """Return the exact file digest to track frozen bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> dict[str, object]:
    """Create all figures and an audit summary from checked local inputs."""

    inputs: Inputs = load_inputs()
    output.mkdir(parents=True, exist_ok=True)
    set_style()
    for filename, plot in FIGURES:
        plot(inputs, output / filename)

    bg: pd.DataFrame = inputs.bulgaria
    pt_assoc: pd.DataFrame = inputs.associations.loc[
        (inputs.associations.geo == "PT") & (inputs.associations.permit_lead_years == 0)
    ]
    reference_row: pd.Series = pt_assoc.iloc[0]
    result: dict[str, object] = {
        "analysis_type": "descriptive_and_explicitly_synthetic_scenario",
        "causal_effect_estimated": False,
        "bulgaria_2021_2025_completion_growth_pct": round(
            100
            * (
                float(bg.completed_dwellings.iloc[-1])
                / float(bg.completed_dwellings.iloc[0])
                - 1
            ),
            3,
        ),
        "bulgaria_2025_completions_pct_fixed_2025_stock": round(
            100 * int(bg.completed_dwellings.iloc[-1]) / 4_360_805, 4
        ),
        "portugal_2006_completions": 68_764,
        "portugal_2025_preliminary_completions": 27_301,
        "portugal_2021_habitual_stock": 4_142_581,
        "portugal_2021_vacant_stock": 723_215,
        "portugal_contemporaneous_permits_real_hpi_pearson_r": round(
            float(reference_row.pearson_r), 6
        ),
        "portugal_contemporaneous_unflagged_r": round(
            float(reference_row.unflagged_pearson_r), 6
        ),
        "portugal_unflagged_n": int(reference_row.unflagged_pairs),
        "figure_files": [f for f, _ in FIGURES],
        "source_sha256": {p.name: sha256(p) for p in sorted(DATA.glob("*"))},
        "figure_sha256": {
            filename: sha256(output / filename) for filename, _ in FIGURES
        },
        "limitations": [
            "Bulgaria completions annual vs house-price growth in Q2",
            "Bulgarian HPI data-source break beginning Q1 2022",
            "Eurostat country panels count permitted dwellings, not completed dwellings",
            "Only four independent country series and at most ten usable annual pairs each",
            "EU27 is an overlapping aggregate, not another independent country",
            "No investment-demand, effective vacancy, municipal market or causal supply shocks in this panel",
            "The hypothetical price curves use uncalibrated assumptions",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Command-line entry point to regenerate publication figures locally."""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "figures", help="Figure output directory"
    )
    args: argparse.Namespace = parser.parse_args()
    result: dict[str, object] = run(args.output)
    generated_files: object = result["figure_files"]
    if not isinstance(generated_files, list):
        raise TypeError("Expected a list of generated figure paths")
    print(
        f"Generated {len(generated_files)} descriptive/synthetic figures in {args.output}"
    )


if __name__ == "__main__":
    main()
