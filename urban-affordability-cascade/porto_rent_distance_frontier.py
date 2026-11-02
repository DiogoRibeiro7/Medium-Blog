"""Screen municipal new-rent medians against official centroid distance.

The 2020/2025 Pareto sets are *mathematical comparisons of municipal
summaries*, not actual housing options, route accessibility, commuting
times, neighbourhood rankings or welfare conclusions. Geography uses the
same checksummed CAOP2025 derivative for both years.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import ROOT, SOURCE_COMMIT, SUBSET_SHA256, load_panel, require_float
from extract_caop import PROVENANCE as CAOP_PROVENANCE
from porto_distance_sensitivity import load_centroids
from porto_rent_options import COMPARATORS


YEARS = (2020, 2025)
AREA_M2 = 70
DISTANCE_MARGINS_KM = (0., 2., 5.)
RENT_MARGINS_EUR_M2 = (0., .5, 1.)
FIGURES = ("fig32_porto_rent_distance_pareto_2020_2025.png",
           "fig33_rent_distance_dominance_margin_sensitivity.png")
RESULTS = ("porto_rent_distance_frontier_municipal_years.csv",
           "porto_rent_distance_frontier_margin_grid.csv",
           "porto_rent_distance_frontier_changes.csv",
           "porto_rent_distance_frontier_provenance.json")


def _nonnegative_finite(value: float, name: str) -> float:
    """Reject invalid numeric margins, including booleans and NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class MunicipalOption:
    """Administrative rent/distance point, *not* a dwelling for rent."""

    municipality: str
    year: int
    distance_km: float
    new_contract_rent_eur_m2: float

    def __post_init__(self) -> None:
        if not isinstance(self.municipality, str) or not self.municipality.strip() or "|" in self.municipality:
            raise ValueError("A named municipality without delimiters is required")
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise ValueError("Municipality year must be an integer")
        if _nonnegative_finite(self.distance_km, "centroid separation") <= 0:
            raise ValueError("Centroid separation must be positive")
        if _nonnegative_finite(self.new_contract_rent_eur_m2, "new-lease median") <= 0:
            raise ValueError("New-lease median must be positive")


def municipal_options() -> dict[int, list[MunicipalOption]]:
    """Join the fixed 2020/2025 rent panel to checksummed CAOP2025 centroids."""
    panel = load_panel()
    centroids = load_centroids()
    px, py = centroids["Porto"]
    output: dict[int, list[MunicipalOption]] = {}
    for year in YEARS:
        options = []
        for municipality in COMPARATORS:
            x, y = centroids[municipality]
            distance = math.hypot(x - px, y - py) / 1000
            rent = require_float(panel[municipality, year].rent_m2,
                                 f"{municipality} {year} new-contract rent median")
            options.append(MunicipalOption(municipality, year, distance, rent))
        if len(options) != 16 or len({o.municipality for o in options}) != 16:
            raise ValueError("Exactly 16 distinct AMP municipal comparators required")
        output[year] = options
    return output


def pareto_dominators(
    options: Sequence[MunicipalOption],
    minimum_distance_advantage_km: float = 0,
    minimum_rent_advantage_eur_m2: float = 0,
) -> dict[str, tuple[str, ...]]:
    """Return every alternative weakly closer/cheaper, with one strict gain.

    Positive margins require *at least* that much additional advantage
    on both dimensions. These margins are deterministic stress scenarios,
    not standard errors, confidence regions or rent-estimation bounds.
    """
    distance_margin = _nonnegative_finite(minimum_distance_advantage_km, "distance margin")
    rent_margin = _nonnegative_finite(minimum_rent_advantage_eur_m2, "rent margin")
    points = tuple(options)
    if len(points) < 2 or not all(isinstance(p, MunicipalOption) for p in points):
        raise ValueError("At least two validated municipal points required")
    if len({p.municipality for p in points}) != len(points):
        raise ValueError("Duplicate municipal point")
    if len({p.year for p in points}) != 1:
        raise ValueError("A Pareto comparison requires one common rent year")
    output: dict[str, tuple[str, ...]] = {}
    for candidate in points:
        dominators = [other for other in points if other.municipality != candidate.municipality
                      and other.distance_km <= candidate.distance_km - distance_margin
                      and other.new_contract_rent_eur_m2 <= candidate.new_contract_rent_eur_m2 - rent_margin
                      and (other.distance_km < candidate.distance_km
                           or other.new_contract_rent_eur_m2 < candidate.new_contract_rent_eur_m2)]
        # A deterministic, low-distance-first display order, with all
        # qualifying alternatives retained rather than one cherry-picked pair.
        dominators.sort(key=lambda p: (p.distance_km, p.new_contract_rent_eur_m2, p.municipality))
        output[candidate.municipality] = tuple(p.municipality for p in dominators)
    return output


def _require_year_grids(options_by_year: dict[int, list[MunicipalOption]]) -> None:
    """Reject missing, duplicated or misdated administrative comparison rows."""
    if set(options_by_year) != set(YEARS):
        raise ValueError("Both designated comparison years required")
    for year in YEARS:
        options = options_by_year[year]
        if len(options) != 16 or {o.municipality for o in options} != set(COMPARATORS):
            raise ValueError("Expected all 16 named municipalities in each rent year")
        if any(o.year != year for o in options):
            raise ValueError("Rent year mismatches the municipal comparison grid")


def municipal_table(options_by_year: dict[int, list[MunicipalOption]]) -> list[dict[str, str | int]]:
    """Write all 32 cross-sectional price/distance comparisons."""
    _require_year_grids(options_by_year)
    panel = load_panel()
    table: list[dict[str, str | int]] = []
    for year in YEARS:
        options = options_by_year[year]
        dominators = pareto_dominators(options)
        porto_rent = require_float(panel["Porto", year].rent_m2, f"Porto {year} new rent median")
        if porto_rent <= 0:
            raise ValueError("Porto rent must be positive")
        for option in options:
            better = dominators[option.municipality]
            table.append({
                "municipality": option.municipality,
                "rent_year": year,
                "geography_reference": "DGT CAOP2025 EPSG:3763 municipal area centroid",
                "centroid_to_porto_euclidean_km": f"{option.distance_km:.6f}",
                "new_contract_median_rent_eur_m2_month": f"{option.new_contract_rent_eur_m2:.6f}",
                "porto_new_contract_median_rent_eur_m2_month": f"{porto_rent:.6f}",
                "synthetic_70m2_monthly_rent_eur": f"{AREA_M2 * option.new_contract_rent_eur_m2:.6f}",
                "synthetic_70m2_rent_saving_vs_porto_eur_month": (
                    f"{AREA_M2 * (porto_rent - option.new_contract_rent_eur_m2):.6f}"
                ),
                "pareto_nondominated_in_two_municipal_summary_dimensions": str(not better).lower(),
                "number_of_municipal_summary_dominators": len(better),
                "closer_and_no_more_expensive_summary_municipalities": "|".join(better),
            })
    if len(table) != 2 * len(COMPARATORS):
        raise ValueError("Expected 32 municipal-year comparison records")
    return table


def margin_table(options_by_year: dict[int, list[MunicipalOption]]) -> list[dict[str, str | int]]:
    """Audit sensitivity to non-statistical minimum material gains."""
    _require_year_grids(options_by_year)
    output: list[dict[str, str | int]] = []
    for year in YEARS:
        options = options_by_year[year]
        for distance_margin in DISTANCE_MARGINS_KM:
            for rent_margin in RENT_MARGINS_EUR_M2:
                better = pareto_dominators(options, distance_margin, rent_margin)
                dominated = sorted(name for name, others in better.items() if others)
                output.append({
                    "rent_year": year,
                    "minimum_centroid_distance_gain_km": f"{distance_margin:.2f}",
                    "minimum_new_rent_gain_eur_m2_month": f"{rent_margin:.2f}",
                    "municipal_comparisons": len(options),
                    "mathematically_dominated_municipal_summaries": len(dominated),
                    "comparisons_without_a_dominator_meeting_margins": len(options) - len(dominated),
                    "dominated_municipality_names": "|".join(dominated),
                })
    return output


def status_changes(table: Sequence[dict[str, str | int]]) -> list[dict[str, str]]:
    """Identify changes in the *summary-defined* efficient set."""
    lookup = {(str(r["municipality"]), int(r["rent_year"])): r for r in table}
    expected = {(name, year) for name in COMPARATORS for year in YEARS}
    if len(table) != len(expected) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicated municipal frontier pairs")
    output: list[dict[str, str]] = []
    for name in COMPARATORS:
        old, new = lookup[name, YEARS[0]], lookup[name, YEARS[1]]
        if any(str(r["pareto_nondominated_in_two_municipal_summary_dimensions"]) not in ("true", "false")
               for r in (old, new)):
            raise ValueError("Unexpected boolean status in municipal frontier table")
        before = str(old["pareto_nondominated_in_two_municipal_summary_dimensions"]) == "true"
        after = str(new["pareto_nondominated_in_two_municipal_summary_dimensions"]) == "true"
        outcome = "entered" if after and not before else "left" if before and not after else "remained" if after else "never_on_frontier"
        output.append({
            "municipality": name,
            "nondominated_2020": str(before).lower(),
            "nondominated_2025": str(after).lower(),
            "summary_frontier_status_2020_to_2025": outcome,
            "centroid_to_porto_euclidean_km": str(new["centroid_to_porto_euclidean_km"]),
        })
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Cannot save empty municipal summary table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_frontier(options_by_year: dict[int, list[MunicipalOption]], path: Path) -> None:
    """Compare level-based efficient sets, not observed residential choices."""
    _require_year_grids(options_by_year)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.5), sharex=True, sharey=True)
    for year, ax in zip(YEARS, axes):
        options = options_by_year[year]
        better = pareto_dominators(options)
        efficient = sorted((o for o in options if not better[o.municipality]), key=lambda o: o.distance_km)
        inefficient = [o for o in options if better[o.municipality]]
        ax.scatter([o.distance_km for o in inefficient], [o.new_contract_rent_eur_m2 for o in inefficient],
                   s=63, color="#b3bdc2", edgecolor="white", linewidth=.6, zorder=3, label="Dominated summary")
        ax.plot([o.distance_km for o in efficient], [o.new_contract_rent_eur_m2 for o in efficient],
                color="#137c73", lw=1.5, alpha=.76, zorder=2)
        ax.scatter([o.distance_km for o in efficient], [o.new_contract_rent_eur_m2 for o in efficient],
                   s=70, color="#137c73", edgecolor="white", linewidth=.6, zorder=4, label="Nondominated summary")
        for o in efficient:
            ax.annotate(o.municipality, (o.distance_km, o.new_contract_rent_eur_m2),
                        xytext=(5, 6), textcoords="offset points", fontsize=8, color="#1b4242")
        ax.set_title(f"{year}: {len(efficient)}/16 nondominated", fontsize=12)
        ax.set_xlabel("CAOP2025 Porto centroid separation (km)", fontsize=10)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Municipal median new-contract rent (EUR/m²/month)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Municipal price–distance frontiers: the set changes across rent years", fontsize=14, y=.97)
    fig.text(.02, .02, "Mathematical ranking of two administrative summaries only. Centroid separation is not travel time; rental availability is unknown.",
             fontsize=9, color="#526371")
    fig.subplots_adjust(left=.09, right=.985, bottom=.16, top=.86, wspace=.16)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_margins(rows: Sequence[dict[str, str | int]], path: Path) -> None:
    """Present all predeclared price/distance advantage cutoffs."""
    lookup = {(int(r["rent_year"]), float(r["minimum_centroid_distance_gain_km"]),
               float(r["minimum_new_rent_gain_eur_m2_month"])): r for r in rows}
    expected = {(year, d, rent) for year in YEARS
                for d in DISTANCE_MARGINS_KM for rent in RENT_MARGINS_EUR_M2}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Incomplete or duplicated municipal dominance margin grid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), sharey=True)
    for year, ax in zip(YEARS, axes):
        matrix = np.array([[int(lookup[year, d, price]["mathematically_dominated_municipal_summaries"])
                            for price in RENT_MARGINS_EUR_M2] for d in DISTANCE_MARGINS_KM])
        heat = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=16, aspect="auto")
        ax.set_xticks(range(len(RENT_MARGINS_EUR_M2)), [f"€{p:g}/m²" for p in RENT_MARGINS_EUR_M2])
        ax.set_yticks(range(len(DISTANCE_MARGINS_KM)), [f"{d:g} km" for d in DISTANCE_MARGINS_KM])
        ax.set_xlabel("Minimum cheaper-rent margin")
        ax.set_title(str(year), fontsize=13)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", weight="bold", fontsize=14,
                        color="white" if matrix[i, j] >= 9 else "#263947")
    axes[0].set_ylabel("Minimum shorter centroid distance")
    fig.suptitle("Dominated municipal summaries out of 16, under stricter margins", fontsize=13.5, y=.98)
    fig.colorbar(heat, ax=axes, fraction=.03, pad=.03, label="Count of comparisons")
    fig.text(.02, .02, "Deterministic thresholds, not confidence intervals. At zero margins, at least one dimension must improve strictly.",
             fontsize=8.7, color="#526371")
    fig.subplots_adjust(left=.1, right=.88, bottom=.2, top=.85, wspace=.15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Rebuild derived tables, figures and provenance without network access."""
    points = municipal_options()
    table = municipal_table(points)
    margins = margin_table(points)
    changes = status_changes(table)
    figures, results = output / "figures", output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plot_frontier(points, figures / FIGURES[0])
    plot_margins(margins, figures / FIGURES[1])
    for filename, rows in zip(RESULTS[:3], (table, margins, changes)):
        _write_csv(results / filename, rows)
    source = json.loads(CAOP_PROVENANCE.read_text(encoding="utf-8"))
    frontier = {str(year): sorted(name for name, members in pareto_dominators(points[year]).items() if not members)
                for year in YEARS}
    report: dict[str, object] = {
        "evidence_class": "source_pinned_municipal_new_lease_rent_medians_and_administrative_centroid_pareto_screen",
        "housing_upstream_commit": SOURCE_COMMIT,
        "housing_panel_sha256": SUBSET_SHA256,
        "geography_publisher": "DGT, CAOP2025 administrative municipal boundaries; CC-BY 4.0",
        "geography_source_response_sha256": source["source_response_sha256"],
        "geography_derived_centroids_sha256": source["derived_csv_sha256"],
        "geography_epsg": 3763,
        "rent_reference_years": list(YEARS),
        "assumed_dwelling_area_m2_for_euro_display": AREA_M2,
        "minimum_centroid_distance_gain_km": list(DISTANCE_MARGINS_KM),
        "minimum_new_rent_gain_eur_m2_month": list(RENT_MARGINS_EUR_M2),
        "municipal_year_rows": len(table),
        "margin_scenarios": len(margins),
        "comparison_year_changes": len(changes),
        "nondominated_municipal_summaries_by_year": frontier,
        "limits": [
            "Municipal new-contract median rents do not identify available equivalent homes or the rent faced by any given household",
            "Both 2020 and 2025 rent slices use official CAOP2025 administrative area-weighted centroids, not year-specific historic boundaries",
            "Straight-line centroid distance from Porto is not a route, time, job access, public transport or household travel burden",
            "Dominance considers only two summaries: lower municipal median new rent and shorter centroid separation",
            "Rent magnitudes refer to different changing pools of new leases with no quality or type matching",
            "Dominated mathematical summaries can still be preferred by households for reasons outside the two measured dimensions",
            "Material-improvement margins are hypothetical cutoffs, not confidence intervals, uncertainty estimates or measurement corrections",
            "The 70 square metre multiplication is synthetic; no market listing, rental stock or dwelling availability is checked",
            "The 16 comparisons are municipalities, not independent search alternatives, people, flows or observed movers",
            "No spatial price contagion, causal migration mechanism, displacement, supply response or welfare effect is identified",
        ],
    }
    (results / RESULTS[3]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Reproduced {report['municipal_year_rows']} municipal rent–distance comparison rows")


if __name__ == "__main__":
    main()
