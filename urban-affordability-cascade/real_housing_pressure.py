"""Compare nominal and HICP-deflated AMP municipal housing median changes.

This calculation isolates the contribution of national *all-items* consumer
price inflation to the *arithmetic* of a reported nominal housing median.
It cannot identify real housing affordability, individual rents or moves,
house quality, or an outward causal transmission mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import Observation, ROOT, SUBSET_SHA256, load_panel, require_float
from commuting_network import load_weights
from extract_hicp import PROVENANCE as HICP_PROVENANCE, load_hicp
from sequential_pressure import METRICS, first_passage


FIGURES = (
    "fig18_nominal_vs_hicp_deflated_housing_growth.png",
    "fig19_nominal_vs_real_housing_first_passage.png",
)
RESULTS = (
    "real_housing_annual_municipal_medians.csv",
    "real_housing_first_passage.csv",
    "real_housing_municipal_endpoints.csv",
    "real_housing_provenance.json",
)


def _housing_values(
    panel: dict[tuple[str, int], Observation], municipality: str,
    field: str, base_year: int, target_year: int,
) -> tuple[float, float]:
    """Read two published EUR/m² medians, rejecting non-positive inputs."""
    if field not in ("price_m2", "rent_m2") or not 2019 <= base_year <= target_year <= 2025:
        raise ValueError("Invalid housing field or baseline/target calendar years")
    earlier = require_float(getattr(panel[(municipality, base_year)], field), field)
    later = require_float(getattr(panel[(municipality, target_year)], field), field)
    if not all(math.isfinite(value) and value > 0 for value in (earlier, later)):
        raise ValueError("Housing median must be positive and finite")
    return earlier, later


def nominal_and_deflated_growth(
    panel: dict[tuple[str, int], Observation], hicp: dict[int, float], *,
    municipality: str, field: str, base_year: int, target_year: int,
) -> tuple[float, float, float]:
    """Return nominal, HICP-deflated and all-items inflation changes (%).

    Formula: real ratio = (P_target / P_base) / (HICP_target / HICP_base).
    Both housing medians are for their published transaction/contract mix.
    """
    start, end = _housing_values(panel, municipality, field, base_year, target_year)
    try:
        earlier_index, later_index = hicp[base_year], hicp[target_year]
    except KeyError as exc:
        raise ValueError("Missing exact HICP reference year") from exc
    if not all(isinstance(index, (int, float)) and not isinstance(index, bool)
               and math.isfinite(index) and index > 0 for index in (earlier_index, later_index)):
        raise ValueError("Non-positive or invalid national consumer price index")
    nominal_ratio = end / start
    index_ratio = later_index / earlier_index
    return (100.0 * (nominal_ratio - 1),
            100.0 * (nominal_ratio / index_ratio - 1),
            100.0 * (index_ratio - 1))


def real_first_passage(
    panel: dict[tuple[str, int], Observation], hicp: dict[int, float], *,
    municipality: str, field: str, base_year: int, threshold_pct: float,
) -> int | None:
    """First *observed year* whose HICP-deflated change crosses a threshold."""
    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("Threshold must be positive and finite")
    if base_year not in (2019, 2020):
        raise ValueError("Unsupported baseline year")
    for year in range(base_year + 1, 2026):
        _, real, _ = nominal_and_deflated_growth(
            panel, hicp, municipality=municipality, field=field,
            base_year=base_year, target_year=year,
        )
        if real >= threshold_pct:
            return year
    return None


def first_passage_table(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], hicp: dict[int, float],
) -> list[dict[str, str | int]]:
    """Place nominal and general-HICP-deflated thresholds side by side."""
    output: list[dict[str, str | int]] = []
    for municipality in names:
        record: dict[str, str | int] = {"municipality": municipality}
        for outcome, (field, base, thresholds) in METRICS.items():
            for cutoff in thresholds:
                nominal = first_passage(panel, municipality=municipality, field=field,
                                        base_year=base, threshold_pct=float(cutoff))
                real = real_first_passage(panel, hicp, municipality=municipality, field=field,
                                          base_year=base, threshold_pct=float(cutoff))
                record[f"{outcome}_plus_{cutoff}_nominal_year"] = "" if nominal is None else nominal
                record[f"{outcome}_plus_{cutoff}_hicp_deflated_year"] = "" if real is None else real
        output.append(record)
    return output


def annual_table(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], hicp: dict[int, float],
) -> list[dict[str, str | int]]:
    """Unweighted medians of 17 municipality growth rates (never pooled prices)."""
    output: list[dict[str, str | int]] = []
    for outcome, (field, base, _) in METRICS.items():
        for year in range(base, 2026):
            triples = [nominal_and_deflated_growth(panel, hicp, municipality=m,
                       field=field, base_year=base, target_year=year) for m in names]
            values = np.array(triples)
            output.append({
                "outcome": outcome, "year": year, "baseline_year": base,
                "municipalities": len(names),
                "portugal_hicp_annual_average_index": f"{hicp[year]:.2f}",
                "national_general_inflation_since_baseline_pct": f"{values[0, 2]:.6f}",
                "median_municipality_nominal_growth_pct": f"{np.median(values[:, 0]):.6f}",
                "median_municipality_hicp_deflated_growth_pct": f"{np.median(values[:, 1]):.6f}",
            })
    return output


def endpoint_table(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], hicp: dict[int, float],
) -> list[dict[str, str | int]]:
    """Show exactly what changes for each observed municipality median."""
    output: list[dict[str, str | int]] = []
    for outcome, (field, base, _) in METRICS.items():
        for municipality in names:
            start, end = _housing_values(panel, municipality, field, base, 2025)
            nominal, real, inflation = nominal_and_deflated_growth(
                panel, hicp, municipality=municipality, field=field,
                base_year=base, target_year=2025,
            )
            output.append({
                "outcome": outcome, "municipality": municipality,
                "baseline_year": base, "endpoint_year": 2025,
                "units": "EUR_m2" if field == "price_m2" else "EUR_m2_month",
                "observed_baseline_median": f"{start:.6f}",
                "observed_2025_median": f"{end:.6f}",
                "nominal_growth_pct": f"{nominal:.6f}",
                "national_general_inflation_pct": f"{inflation:.6f}",
                "hicp_deflated_growth_pct": f"{real:.6f}",
            })
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Empty housing/inflation result")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_growth(rows: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show real and nominal growth alongside the national HICP adjustment."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), sharey=True)
    for ax, outcome, title in zip(axes, METRICS, ("Sold-price medians", "New-contract rent medians")):
        block = [r for r in rows if r["outcome"] == outcome]
        years = [int(r["year"]) for r in block]
        ax.plot(years, [float(r["median_municipality_nominal_growth_pct"]) for r in block],
                marker="o", lw=2.1, color="#2f78a4", label="Nominal municipal median")
        ax.plot(years, [float(r["median_municipality_hicp_deflated_growth_pct"]) for r in block],
                marker="s", lw=2.1, color="#ba704c", label="HICP-deflated median")
        ax.plot(years, [float(r["national_general_inflation_since_baseline_pct"]) for r in block],
                lw=1.5, linestyle="--", color="#708080", label="National all-items HICP")
        ax.set_title(title)
        ax.set_xticks(years)
        ax.set_xlabel("Published year")
        ax.grid(alpha=0.17)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Cumulative change from each outcome's baseline (%)")
    axes[0].legend(loc="upper left", fontsize=8.3, frameon=False)
    fig.suptitle("AMP housing median changes: nominal and general-inflation adjusted", fontsize=13)
    fig.text(0.02, 0.022,
             "Median of 17 municipal percentage changes (not pooled transaction medians). "
             "National Eurostat HICP; sale baseline 2019, new-rent baseline 2020.", fontsize=8.4, color="#566777")
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.15, top=0.85, wspace=0.10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_thresholds(rows: Sequence[dict[str, str | int]], path: Path) -> None:
    """Compare observed first-passage years and right-censored cells."""
    fields = (("sold_price_plus_50_nominal_year", "Sale +50%\nnominal"),
              ("sold_price_plus_50_hicp_deflated_year", "Sale +50%\nHICP-deflated"),
              ("new_contract_rent_plus_25_nominal_year", "New rent +25%\nnominal"),
              ("new_contract_rent_plus_25_hicp_deflated_year", "New rent +25%\nHICP-deflated"))
    matrix = np.array([[float(row[field]) if row[field] != "" else np.nan for field, _ in fields]
                       for row in rows])
    fig, ax = plt.subplots(figsize=(11.2, 8.25))
    palette = plt.get_cmap("YlGnBu").copy()
    palette.set_bad("#e1e5e8")
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=palette, vmin=2021, vmax=2025, aspect="auto")
    ax.set_xticks(range(len(fields)), [label for _, label in fields], fontsize=10.2)
    ax.set_yticks(range(len(rows)), [str(r["municipality"]) for r in rows], fontsize=9)
    for i, row in enumerate(rows):
        for j, (field, _) in enumerate(fields):
            year = row[field]
            ax.text(j, i, str(year) if year != "" else "not yet", va="center", ha="center",
                    fontsize=8.8, color="white" if year != "" and int(year) >= 2024 else "#203548")
        if row["municipality"] == "Porto":
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_title("First annual threshold years before and after general inflation adjustment", fontsize=13, pad=15)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_ticks([2021, 2022, 2023, 2024, 2025])
    colorbar.set_label("First published crossing year")
    fig.text(0.022, 0.023,
             "National annual average HICP, 2025=100. The deflated series is not a rent-burden or fixed-quality index; "
             "'not yet' = unpassed by 2025.", fontsize=8.5, color="#566777")
    fig.subplots_adjust(left=0.23, right=0.93, bottom=0.12, top=0.90)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Reproduce inflation comparisons from two separately checksummed inputs."""
    panel = load_panel()
    names, _ = load_weights()
    hicp = load_hicp()
    annual = annual_table(panel, names, hicp)
    crossings = first_passage_table(panel, names, hicp)
    endpoints = endpoint_table(panel, names, hicp)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_growth(annual, figure_dir / FIGURES[0])
    plot_thresholds(crossings, figure_dir / FIGURES[1])
    for filename, rows in zip(RESULTS[:3], (annual, crossings, endpoints)):
        _write_csv(result_dir / filename, rows)
    source = json.loads((ROOT / "data" / HICP_PROVENANCE).read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "evidence_classes": ["observed_nominal_housing_median", "observed_official_national_hicp",
                             "derived_hicp_deflated_municipal_median_comparison"],
        "input_housing_panel_sha256": SUBSET_SHA256,
        "hicp_input_source_sha256": source["source_response_sha256"],
        "hicp_derived_csv_sha256": source["derived_csv_sha256"],
        "hicp_source_query": source["source_query"],
        "hicp_source_dataset_updated": source["source_dataset_updated"],
        "hicp_reference": "2025=100_national_all_items_annual_average",
        "housing_outcomes": {k: {"field": field, "baseline_year": base, "thresholds_pct": list(thresholds)}
                             for k, (field, base, thresholds) in METRICS.items()},
        "municipality_count": len(names),
        "endpoint_summary": [r for r in annual if int(r["year"]) == 2025],
        "crossing_counts": {
            outcome: {
                f"plus_{threshold}": {
                    basis: sum(row[f"{outcome}_plus_{threshold}_{basis}_year"] != "" for row in crossings)
                    for basis in ("nominal", "hicp_deflated")
                }
                for threshold in thresholds
            }
            for outcome, (_, _, thresholds) in METRICS.items()
        },
        "limits": [
            "HICP is national all-items consumer inflation, not a municipal cost-of-living deflator",
            "HICP excludes imputed owner-occupied housing costs and is not a house-price index",
            "The result is real relative to an all-items consumer basket, not real income-based affordability",
            "Sold-price and new-contract rent annual medians reflect transaction and contract composition",
            "The annual first-passage comparison cannot determine intrayear sequence or household relocation",
            "No directed residential migration OD, tenancy panel, or causal spillover is identified",
        ],
    }
    (result_dir / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = run(args.output)
    print(f"Reproduced nominal and HICP-deflated housing comparisons for {result['municipality_count']} municipalities")


if __name__ == "__main__":
    main()
