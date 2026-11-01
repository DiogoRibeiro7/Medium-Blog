"""Compare new-contract rent medians with municipal median gross fiscal income.

The two medians describe *different statistical populations*. A fixed floor
area turns the cross-statistic ratio into a synthetic screening proxy, not the
rent burden of a tenant or the median of household-level rent/income ratios.
The common 2020–2024 window is mandatory: 2019 new-contract rents and 2025
fiscal income are unpublished in the fixed municipal panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import (
    MUNICIPALITIES,
    ROOT,
    SOURCE_BLOB_SHA,
    SOURCE_COMMIT,
    SUBSET_SHA256,
    Observation,
    load_panel,
    require_float,
)


YEARS = tuple(range(2020, 2025))
AREAS_M2 = (50, 70, 90)
FIGURES = (
    "fig20_synthetic_rent_fiscal_income_70m2.png",
    "fig21_rent_vs_fiscal_income_growth_2020_2024.png",
)
RESULTS = (
    "fiscal_income_rent_annual_municipal_proxy.csv",
    "fiscal_income_rent_municipal_endpoints.csv",
    "fiscal_income_rent_annual_summary.csv",
    "fiscal_income_rent_provenance.json",
)


def _positive_number(value: float, label: str) -> float:
    """Validate a strictly positive, finite real input, excluding booleans."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def synthetic_rent_income_percent(
    rent_eur_m2_month: float, gross_fiscal_income_eur_year: float, area_m2: float,
) -> float:
    """Compute 100 × 12 × area × median new-rent / median fiscal income.

    The quotient of two aggregate medians is not a household-level burden,
    including when it happens to exceed commonly cited burden thresholds.
    """
    rent = _positive_number(rent_eur_m2_month, "New-contract rent median")
    income = _positive_number(gross_fiscal_income_eur_year, "Gross fiscal-income median")
    area = _positive_number(area_m2, "Illustrative floor area")
    return 100.0 * 12.0 * area * rent / income


def relative_proxy_growth_percent(
    baseline_rent: float, current_rent: float,
    baseline_income: float, current_income: float,
) -> float:
    """Growth in a fixed-area rent/income quotient; floor area cancels."""
    old_rent = _positive_number(baseline_rent, "Baseline rent")
    new_rent = _positive_number(current_rent, "Current rent")
    old_income = _positive_number(baseline_income, "Baseline income")
    new_income = _positive_number(current_income, "Current income")
    return 100.0 * ((new_rent / old_rent) / (new_income / old_income) - 1.0)


def _pair(panel: dict[tuple[str, int], Observation], name: str, year: int) -> tuple[float, float]:
    """Require a published rent and gross fiscal median in the valid window."""
    if year not in YEARS or name not in MUNICIPALITIES:
        raise ValueError("Unsupported municipality or incomplete fiscal-income/rent year")
    observation = panel[name, year]
    return (_positive_number(require_float(observation.rent_m2, "rent_m2"), "rent_m2"),
            _positive_number(require_float(observation.income, "income"), "income"))


def _validate_names(names: Sequence[str]) -> list[str]:
    """Reject partial, duplicated or out-of-area samples."""
    if len(names) != len(MUNICIPALITIES) or set(names) != MUNICIPALITIES:
        raise ValueError("Expected each of the 17 AMP municipalities exactly once")
    return sorted(names)


def annual_table(
    panel: dict[tuple[str, int], Observation], names: Sequence[str],
) -> list[dict[str, str | int]]:
    """Emit all 85 matched municipality-years with no imputed income."""
    rows: list[dict[str, str | int]] = []
    for municipality in _validate_names(names):
        rent_0, income_0 = _pair(panel, municipality, YEARS[0])
        for year in YEARS:
            rent, income = _pair(panel, municipality, year)
            record: dict[str, str | int] = {
                "municipality": municipality,
                "year": year,
                "new_contract_rent_median_eur_m2_month": f"{rent:.6f}",
                "gross_declared_fiscal_income_median_eur_year": f"{income:.6f}",
                "rent_median_growth_from_2020_pct": f"{100 * (rent / rent_0 - 1):.6f}",
                "fiscal_income_median_growth_from_2020_pct": f"{100 * (income / income_0 - 1):.6f}",
                "synthetic_rent_income_proxy_growth_from_2020_pct":
                    f"{relative_proxy_growth_percent(rent_0, rent, income_0, income):.6f}",
            }
            for area in AREAS_M2:
                record[f"synthetic_{area}m2_annual_rent_over_gross_fiscal_income_pct"] = (
                    f"{synthetic_rent_income_percent(rent, income, float(area)):.6f}"
                )
            rows.append(record)
    return rows


def endpoint_table(annual: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Pair exact 2020 and 2024 observations without pooling municipalities."""
    by_key = {(str(r["municipality"]), int(r["year"])): r for r in annual}
    if (len(annual) != len(MUNICIPALITIES) * len(YEARS)
            or set(by_key) != {(name, year) for name in MUNICIPALITIES for year in YEARS}):
        raise ValueError("Annual rent/fiscal proxy grid is incomplete or duplicated")
    output: list[dict[str, str | int]] = []
    for name in sorted(MUNICIPALITIES):
        before, after = by_key[name, YEARS[0]], by_key[name, YEARS[-1]]
        field = "synthetic_70m2_annual_rent_over_gross_fiscal_income_pct"
        original, latest = float(before[field]), float(after[field])
        output.append({
            "municipality": name,
            "baseline_year": YEARS[0],
            "endpoint_year": YEARS[-1],
            "new_contract_rent_growth_pct": after["rent_median_growth_from_2020_pct"],
            "gross_declared_fiscal_income_growth_pct": after["fiscal_income_median_growth_from_2020_pct"],
            "synthetic_70m2_rent_over_gross_fiscal_income_2020_pct": before[field],
            "synthetic_70m2_rent_over_gross_fiscal_income_2024_pct": after[field],
            "synthetic_70m2_proxy_change_percentage_points": f"{latest - original:.6f}",
            "synthetic_proxy_relative_growth_pct": after["synthetic_rent_income_proxy_growth_from_2020_pct"],
        })
    return output


def summary_table(annual: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Report unweighted cross-municipality medians, never pooled tax data."""
    output: list[dict[str, str | int]] = []
    for year in YEARS:
        rows = [r for r in annual if int(r["year"]) == year]
        if len(rows) != len(MUNICIPALITIES) or {r["municipality"] for r in rows} != MUNICIPALITIES:
            raise ValueError("Missing annual municipality in fiscal-income summary")
        growths = [float(r["synthetic_rent_income_proxy_growth_from_2020_pct"]) for r in rows]
        ratios = [float(r["synthetic_70m2_annual_rent_over_gross_fiscal_income_pct"]) for r in rows]
        output.append({
            "year": year,
            "municipalities": len(rows),
            "median_municipal_new_rent_growth_from_2020_pct":
                f"{median(float(r['rent_median_growth_from_2020_pct']) for r in rows):.6f}",
            "median_municipal_fiscal_income_growth_from_2020_pct":
                f"{median(float(r['fiscal_income_median_growth_from_2020_pct']) for r in rows):.6f}",
            "median_municipal_synthetic_proxy_relative_growth_pct": f"{median(growths):.6f}",
            "median_municipal_synthetic_70m2_proxy_level_pct": f"{median(ratios):.6f}",
            "synthetic_70m2_proxy_level_q25_pct": f"{np.quantile(ratios, 0.25):.6f}",
            "synthetic_70m2_proxy_level_q75_pct": f"{np.quantile(ratios, 0.75):.6f}",
            "municipalities_proxy_worse_than_2020": sum(value > 1e-9 for value in growths),
            "municipalities_proxy_better_than_2020": sum(value < -1e-9 for value in growths),
        })
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    """Write a fixed-column CSV with stable row ordering and line endings."""
    if not rows:
        raise ValueError("Cannot write an empty fiscal-income table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_proxy_levels(endpoints: Sequence[dict[str, str | int]], path: Path) -> None:
    """Compare matched-period synthetic levels for every municipality."""
    ordered = sorted(endpoints, key=lambda r: float(r["synthetic_70m2_rent_over_gross_fiscal_income_2024_pct"]))
    fig, ax = plt.subplots(figsize=(11.1, 8.6))
    positions = np.arange(len(ordered))
    before = [float(r["synthetic_70m2_rent_over_gross_fiscal_income_2020_pct"]) for r in ordered]
    after = [float(r["synthetic_70m2_rent_over_gross_fiscal_income_2024_pct"]) for r in ordered]
    for y, a, b in zip(positions, before, after):
        ax.plot([a, b], [y, y], color="#c5d0d6", linewidth=2, zorder=1)
    ax.scatter(before, positions, color="#2f78a4", label="2020", zorder=3, s=45)
    ax.scatter(after, positions, color="#b4553e", label="2024", marker="D", zorder=3, s=45)
    ax.set_yticks(positions, [str(r["municipality"]) for r in ordered], fontsize=9.5)
    ax.set_xlabel("Synthetic annual new rent / annual gross fiscal-income median (%)")
    ax.set_title("70 m² illustrative rental proxy, 2020–2024", fontsize=14)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "Porto":
            tick.set_fontweight("bold")
    fig.text(0.015, 0.021,
             "12 × 70 m² × municipality median new-contract €/m²/month ÷ municipality median gross IRS-declared fiscal income."
             " Ratio of two medians, NOT an observed household rent burden.", fontsize=8.3, color="#526371")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.11, top=0.91)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_growth_comparison(endpoints: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show whether new-contract rent growth exceeds gross fiscal-income growth."""
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    x = [float(r["gross_declared_fiscal_income_growth_pct"]) for r in endpoints]
    y = [float(r["new_contract_rent_growth_pct"]) for r in endpoints]
    diagonal = max(max(x), max(y)) + 4
    ax.plot([0, diagonal], [0, diagonal], color="#75828d", linestyle="--", linewidth=1.3,
            label="Equal percentage growth")
    ax.scatter(x, y, color="#2f78a4", alpha=0.85, s=60, edgecolors="white", linewidths=0.5)
    porto = next(r for r in endpoints if r["municipality"] == "Porto")
    px, py = float(porto["gross_declared_fiscal_income_growth_pct"]), float(porto["new_contract_rent_growth_pct"])
    ax.scatter([px], [py], s=105, color="#b4553e", edgecolors="white", linewidths=0.7, zorder=4)
    ax.annotate("Porto", (px, py), xytext=(px + 1.0, py - 2.4), fontsize=10, weight="bold")
    ax.set_xlim(0, diagonal)
    ax.set_ylim(0, diagonal)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Municipal gross fiscal-income median growth, 2020–2024 (%)")
    ax.set_ylabel("Municipal new-contract rent median growth, 2020–2024 (%)")
    ax.set_title("Rents outpaced the fiscal-income statistic in all 17 municipalities", fontsize=11.6)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.03, 0.025,
             "Matched municipal medians; not a tenant cohort or a household-level rent-income comparison."
             " Both series are nominal EUR.", fontsize=8.7, color="#526371")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.90)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate two figures, three tables and a bounded source ledger."""
    names = sorted(MUNICIPALITIES)
    annual = annual_table(load_panel(), names)
    endpoints = endpoint_table(annual)
    summary = summary_table(annual)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_proxy_levels(endpoints, figure_dir / FIGURES[0])
    plot_growth_comparison(endpoints, figure_dir / FIGURES[1])
    for filename, rows in zip(RESULTS[:3], (annual, endpoints, summary)):
        _write_csv(result_dir / filename, rows)
    porto = next(r for r in endpoints if r["municipality"] == "Porto")
    report: dict[str, object] = {
        "evidence_class": "derived_cross_statistic_municipality_median_ratio_not_household_burden",
        "source_project": "DiogoRibeiro7/portugal-housing-generations",
        "source_commit": SOURCE_COMMIT,
        "source_blob_sha": SOURCE_BLOB_SHA,
        "fixed_municipal_panel_sha256": SUBSET_SHA256,
        "official_income_methodology": "https://www.ine.pt/documentos/municipios/rnlocal/ano2022/1B0.pdf",
        "official_income_2024_release": "https://webinq.ine.pt/home.aspx",
        "source_variables": {
            "rent_m2": "median_new_contract_housing_rent_EUR_m2_month",
            "income": "median_gross_declared_IRS_income_per_fiscal_household_EUR_year",
        },
        "matched_years": list(YEARS),
        "municipalities": len(names),
        "municipality_years": len(annual),
        "illustrative_floor_areas_m2": list(AREAS_M2),
        "formula": "100 * (12 * floor_area_m2 * municipal_rent_median_eur_m2_month) / municipal_gross_fiscal_income_median_eur_year",
        "growth_formula": "100 * ((rent_t/rent_2020)/(income_t/income_2020)-1); floor area cancels",
        "final_year_summary": summary[-1],
        "porto_proxy_relative_growth_pct": porto["synthetic_proxy_relative_growth_pct"],
        "limits": [
            "Both source statistics are reported in nominal EUR; fiscal-income and rent growth are nominal",
            "Annual summaries are unweighted medians of 17 municipal changes, not pooled contracts or tax households",
            "The rent statistic covers newly signed tenancy contracts, not all existing tenants",
            "Gross declared fiscal income covers an IRS filing population, not every resident or actual new renter",
            "Tax households and tenants need not be the same set of people or household definition",
            "Ratio of two municipal medians is not a median of household-level rent/income burdens",
            "Dwelling sizes 50, 70 and 90 m2 are assumptions, not observed tenancy floor areas",
            "Gross fiscal income is before IRS and other deductions; it is not disposable household income",
            "No income observations for 2025 and no published new-contract rent medians for 2019",
            "No repeated-property constant-quality adjustment or displacement, migration and spillover identification",
        ],
    }
    (result_dir / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Reproduced {report['municipality_years']} matched municipality-years of fiscal/rental pressure proxies")


if __name__ == "__main__":
    main()
