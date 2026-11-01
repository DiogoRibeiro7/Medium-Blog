"""Describe Porto-to-other-municipality new-contract rent price gradients.

These are hypothetical *location comparisons* based on independent
municipality-year medians, not matched dwellings, available rentals, household
searches, or actual relocation costs. National HICP is used to express
hypothetical euro differences in constant 2020 consumer-price euros.
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

from analysis import MUNICIPALITIES, ROOT, SOURCE_COMMIT, SUBSET_SHA256, Observation, load_panel, require_float
from extract_hicp import DOI, SOURCE_SHA256, load_hicp


BASE_YEAR = 2020
END_YEAR = 2025
YEARS = tuple(range(BASE_YEAR, END_YEAR + 1))
COMPARATORS = tuple(sorted(MUNICIPALITIES - {"Porto"}))
AREAS_M2 = (50, 70, 90)
FIGURES = (
    "fig22_porto_relative_rent_discounts_2020_2025.png",
    "fig23_porto_rent_discount_and_real_savings_annual.png",
)
RESULTS = (
    "porto_rent_options_annual_municipal_comparisons.csv",
    "porto_rent_options_municipal_endpoints.csv",
    "porto_rent_options_annual_summary.csv",
    "porto_rent_options_provenance.json",
)


def _positive(value: float, label: str) -> float:
    """Validate a finite, strictly positive numeric median or input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be positive and finite")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def rent_option(
    porto_rent_eur_m2_month: float,
    alternative_rent_eur_m2_month: float,
    hicp_target: float,
    hicp_base: float,
    *,
    area_m2: float,
) -> tuple[float, float, float]:
    """Return percent discount, nominal monthly difference, 2020-EUR difference.

    Negative differences are valid: the comparator can be more expensive.
    HICP does not enter the percentage discount because both rents are
    compared in the *same* publication year.
    """
    porto = _positive(porto_rent_eur_m2_month, "Porto new-rent median")
    alternative = _positive(alternative_rent_eur_m2_month, "Alternative new-rent median")
    target = _positive(hicp_target, "Target HICP")
    baseline = _positive(hicp_base, "Baseline HICP")
    area = _positive(area_m2, "Illustrative floor area")
    nominal_difference = area * (porto - alternative)
    return (
        100.0 * (1.0 - alternative / porto),
        nominal_difference,
        nominal_difference * baseline / target,
    )


def _rent(
    panel: dict[tuple[str, int], Observation], municipality: str, year: int,
) -> float:
    """Require an actually published positive new-lease median."""
    if municipality not in MUNICIPALITIES or year not in YEARS:
        raise ValueError("Unsupported municipality or new-contract publication year")
    return _positive(require_float(panel[municipality, year].rent_m2, "rent_m2"), "rent_m2")


def _hicp(hicp: dict[int, float], year: int) -> float:
    """Require the exact annual HICP year and exclude invalid values."""
    if year not in YEARS:
        raise ValueError("Unsupported HICP comparison year")
    try:
        return _positive(hicp[year], "HICP")
    except KeyError as exc:
        raise ValueError(f"Missing annual Portugal HICP for {year}") from exc


def annual_table(
    panel: dict[tuple[str, int], Observation], hicp: dict[int, float],
) -> list[dict[str, str | int]]:
    """Emit 96 area-sensitive Porto comparisons, without residence flows."""
    baseline_hicp = _hicp(hicp, BASE_YEAR)
    base_porto = _rent(panel, "Porto", BASE_YEAR)
    rows: list[dict[str, str | int]] = []
    for municipality in COMPARATORS:
        base_alt = _rent(panel, municipality, BASE_YEAR)
        baseline_discount, _, baseline_real = rent_option(
            base_porto, base_alt, baseline_hicp, baseline_hicp, area_m2=70,
        )
        for year in YEARS:
            porto, other, index = _rent(panel, "Porto", year), _rent(panel, municipality, year), _hicp(hicp, year)
            discount, nominal, real = rent_option(porto, other, index, baseline_hicp, area_m2=70)
            row: dict[str, str | int] = {
                "alternative_municipality": municipality,
                "year": year,
                "porto_new_rent_median_eur_m2_month": f"{porto:.6f}",
                "alternative_new_rent_median_eur_m2_month": f"{other:.6f}",
                "portugal_hicp_annual_average_2025_100": f"{index:.2f}",
                "porto_minus_alternative_eur_m2_month": f"{porto - other:.6f}",
                "alternative_relative_rent_discount_vs_porto_pct": f"{discount:.6f}",
                "relative_rent_discount_change_since_2020_pp": f"{discount - baseline_discount:.6f}",
                "synthetic_70m2_nominal_monthly_saving_eur": f"{nominal:.6f}",
                "synthetic_70m2_real_2020_eur_monthly_saving": f"{real:.6f}",
                "synthetic_70m2_real_2020_eur_saving_change_since_2020": f"{real - baseline_real:.6f}",
            }
            for area in AREAS_M2:
                if area == 70:
                    # The 70 m² primary scenario is already in the record.
                    continue
                _, nominal_area, real_area = rent_option(porto, other, index, baseline_hicp, area_m2=area)
                row[f"synthetic_{area}m2_nominal_monthly_saving_eur"] = f"{nominal_area:.6f}"
                row[f"synthetic_{area}m2_real_2020_eur_monthly_saving"] = f"{real_area:.6f}"
            rows.append(row)
    return rows


def _keyed_annual(
    annual: Sequence[dict[str, str | int]],
) -> dict[tuple[str, int], dict[str, str | int]]:
    """Reject omitted, duplicate or out-of-sample comparator-year records."""
    keyed = {(str(row["alternative_municipality"]), int(row["year"])): row for row in annual}
    expected = {(name, year) for name in COMPARATORS for year in YEARS}
    if len(annual) != len(expected) or set(keyed) != expected:
        raise ValueError("Incomplete or duplicated 16 x 6 Porto comparator grid")
    return keyed


def endpoint_table(
    annual: Sequence[dict[str, str | int]],
) -> list[dict[str, str | int]]:
    """Pair 2020 and 2025 nominal/real hypothetical savings per comparator."""
    by_key = _keyed_annual(annual)
    rows: list[dict[str, str | int]] = []
    for name in COMPARATORS:
        baseline, target = by_key[name, BASE_YEAR], by_key[name, END_YEAR]
        rows.append({
            "alternative_municipality": name,
            "baseline_year": BASE_YEAR,
            "endpoint_year": END_YEAR,
            "relative_rent_discount_2020_pct": baseline["alternative_relative_rent_discount_vs_porto_pct"],
            "relative_rent_discount_2025_pct": target["alternative_relative_rent_discount_vs_porto_pct"],
            "relative_discount_change_percentage_points": target["relative_rent_discount_change_since_2020_pp"],
            "synthetic_70m2_nominal_saving_2020_eur_month": baseline["synthetic_70m2_nominal_monthly_saving_eur"],
            "synthetic_70m2_nominal_saving_2025_eur_month": target["synthetic_70m2_nominal_monthly_saving_eur"],
            "synthetic_70m2_real_saving_2025_in_2020_eur_month": target["synthetic_70m2_real_2020_eur_monthly_saving"],
            "synthetic_70m2_real_saving_change_2020_to_2025_eur_month": (
                target["synthetic_70m2_real_2020_eur_saving_change_since_2020"]
            ),
        })
    return rows


def summary_table(
    annual: Sequence[dict[str, str | int]],
) -> list[dict[str, str | int]]:
    """Show independent unweighted medians and quartiles of the 16 options."""
    keyed = _keyed_annual(annual)
    rows: list[dict[str, str | int]] = []
    for year in YEARS:
        block = [keyed[name, year] for name in COMPARATORS]
        discounts = [float(r["alternative_relative_rent_discount_vs_porto_pct"]) for r in block]
        nominal = [float(r["synthetic_70m2_nominal_monthly_saving_eur"]) for r in block]
        real = [float(r["synthetic_70m2_real_2020_eur_monthly_saving"]) for r in block]
        rows.append({
            "year": year,
            "comparison_municipalities": len(block),
            "median_relative_discount_pct": f"{median(discounts):.6f}",
            "relative_discount_q25_pct": f"{np.quantile(discounts, 0.25):.6f}",
            "relative_discount_q75_pct": f"{np.quantile(discounts, 0.75):.6f}",
            "median_synthetic_70m2_nominal_monthly_saving_eur": f"{median(nominal):.6f}",
            "median_synthetic_70m2_real_2020_eur_monthly_saving": f"{median(real):.6f}",
            "synthetic_70m2_real_saving_q25_2020_eur": f"{np.quantile(real, 0.25):.6f}",
            "synthetic_70m2_real_saving_q75_2020_eur": f"{np.quantile(real, 0.75):.6f}",
            "alternative_municipalities_cheaper_than_porto": sum(x > 0 for x in discounts),
            "alternative_municipalities_with_smaller_discount_than_2020": sum(
                float(r["relative_rent_discount_change_since_2020_pp"]) < -1e-9 for r in block
            ),
            "alternative_municipalities_with_higher_real_savings_than_2020": sum(
                float(r["synthetic_70m2_real_2020_eur_saving_change_since_2020"]) > 1e-9 for r in block
            ),
        })
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    """Write stable UTF-8 CSV with a single fixed schema."""
    if not rows:
        raise ValueError("Empty Porto comparison table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_relative_discounts(endpoints: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show endpoints of each municipality's Porto-relative new-rent discount."""
    rows = sorted(endpoints, key=lambda r: float(r["relative_rent_discount_2025_pct"]))
    fig, ax = plt.subplots(figsize=(11.4, 8.7))
    positions = np.arange(len(rows))
    before = [float(r["relative_rent_discount_2020_pct"]) for r in rows]
    after = [float(r["relative_rent_discount_2025_pct"]) for r in rows]
    for y, a, b in zip(positions, before, after):
        ax.plot([a, b], [y, y], lw=2, color="#a9bbc5", zorder=1)
    ax.scatter(before, positions, label="2020", s=52, color="#2f78a4", zorder=3)
    ax.scatter(after, positions, label="2025", marker="D", s=48, color="#bd6743", zorder=3)
    ax.set_yticks(positions, [str(r["alternative_municipality"]) for r in rows], fontsize=9.2)
    ax.set_xlabel("100 × (1 − other municipality rent median / Porto rent median), %")
    ax.set_title("Relative new-contract rent discount to Porto, 2020 vs 2025", fontsize=13)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.023,
             "13 of 16 discounts narrowed. All remain positive. Independent annual municipal new-lease medians, "
             "not matched homes or guaranteed available options.", fontsize=8.7, color="#526371")
    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.12, top=0.90)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_annual_contrast(summary: Sequence[dict[str, str | int]], path: Path) -> None:
    """Contrast relative discounts and 2020-EUR hypothetical saving levels."""
    years = [int(r["year"]) for r in summary]
    fields = [
        ("median_relative_discount_pct", "relative_discount_q25_pct", "relative_discount_q75_pct",
         "Median Porto-relative discount (%)", "#2f78a4"),
        ("median_synthetic_70m2_real_2020_eur_monthly_saving",
         "synthetic_70m2_real_saving_q25_2020_eur", "synthetic_70m2_real_saving_q75_2020_eur",
         "Median monthly 70 m² saving (2020 EUR)", "#b4553e"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (middle, low, high, label, color) in zip(axes, fields):
        center = [float(r[middle]) for r in summary]
        lower = [float(r[low]) for r in summary]
        upper = [float(r[high]) for r in summary]
        ax.plot(years, center, marker="o", color=color, lw=2.2, label="Unweighted median")
        ax.fill_between(years, lower, upper, alpha=0.19, color=color, label="Across-municipality IQR")
        ax.set_xticks(years)
        ax.set_xlabel("Publication year")
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("A smaller percentage discount can coexist with a larger real euro gap", fontsize=13)
    fig.text(0.025, 0.028,
             "16 municipal new-contract median comparisons with Porto; 70 m² and zero added travel costs are assumptions. "
             "National Eurostat HICP expressed in 2020 consumer-price euros.", fontsize=8.6, color="#526371")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.85, wspace=0.28)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate two diagnostic figures, three tables and source provenance."""
    annual = annual_table(load_panel(), load_hicp())
    endpoints = endpoint_table(annual)
    summary = summary_table(annual)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_relative_discounts(endpoints, figure_dir / FIGURES[0])
    plot_annual_contrast(summary, figure_dir / FIGURES[1])
    for filename, rows in zip(RESULTS[:3], (annual, endpoints, summary)):
        _write_csv(result_dir / filename, rows)
    latest = summary[-1]
    report: dict[str, object] = {
        "evidence_class": "descriptive_comparison_of_independent_municipal_new_contract_rent_medians",
        "housing_source_upstream_commit": SOURCE_COMMIT,
        "housing_source_fixed_subset_sha256": SUBSET_SHA256,
        "hicp_official_dataset_doi": DOI,
        "hicp_source_response_sha256": SOURCE_SHA256,
        "observed_years": list(YEARS),
        "porto_reference": "Porto",
        "comparison_municipalities": list(COMPARATORS),
        "municipality_year_comparisons": len(annual),
        "hypothetical_dwelling_sizes_m2": list(AREAS_M2),
        "nominal_saving_formula": "area_m2 * (Porto median new-rent EUR/m2/month - comparator median new-rent EUR/m2/month)",
        "real_saving_formula": "nominal_saving * (Portugal annual average HICP 2020 / HICP current year)",
        "relative_discount_formula": "100 * (1 - comparator median new-rent EUR/m2/month / Porto median new-rent EUR/m2/month)",
        "2025_summary": latest,
        "limits": [
            "Independent annual municipal medians are not matched apartments or constant-quality new-rent indices",
            "The 16 other AMP municipalities are administrative comparators, not validated substitutable suburbs",
            "70 m2 is an assumed same-size apartment; 50/90 m2 scenarios scale cash amounts, not observed rentals",
            "No unit availability, vacancy, contract mix, location within municipality or household sorting is measured",
            "Nominal and HICP-deflated savings include no actual transport cost, time, moving cost or household tax/income",
            "HICP is the Portugal national all-items deflator, not a municipal housing-cost index",
            "A positive monthly rent difference cannot prove a feasible or welfare-improving household move",
            "Cross-municipal rent growth and relative discount changes cannot identify directional housing-price contagion",
            "Medians and quartiles summarize 16 equally weighted municipality comparisons; not a pooled tenancy estimate",
        ],
    }
    (result_dir / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Regenerated {report['municipality_year_comparisons']} Porto/other AMP rent comparisons")


if __name__ == "__main__":
    main()
