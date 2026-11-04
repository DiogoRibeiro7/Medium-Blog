"""Audit geographic confounding in all municipal network-reference controls.

The fixed 2021 symmetric work/study network is not a residential-migration
network. The CAOP2025 area-weighted centroids supply a purely geometric
control, not travel times or household access. Partial Spearman coefficients
are conditional rank associations, not identified effects. Matched
reference comparisons condition on *different* reference-specific distances.
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

from analysis import MUNICIPALITIES, ROOT, SOURCE_COMMIT, SUBSET_SHA256
from extract_caop import PROVENANCE as CAOP_PROVENANCE
from extract_commuting import PROVENANCE as COMMUTING_PROVENANCE
from municipal_reference_placebos import REFERENCE
from municipal_reference_window_sensitivity import (
    MATERIAL_RHO_DIFFERENCE, OUTCOMES, _require_window_grid, _window_vectors,
    window_pairs, window_ranks,
)
from porto_distance_sensitivity import load_centroids, partial_rank_correlation
from porto_network_gradient import END_YEAR, MODES, WINDOW_STARTS, spearman_rho


HIGH_RANK_DISTANCE_CORRELATION = 0.90  # Diagnostic flag, never a significance cutoff.
FIGURES = (
    "fig38_reference_exposure_distance_collinearity.png",
    "fig39_municipal_reference_distance_adjusted_ranks.png",
)
RESULTS = (
    "municipal_reference_geography_pair_distances.csv",
    "municipal_reference_geography_partial_grid.csv",
    "municipal_reference_geography_matched_partial.csv",
    "municipal_reference_geography_porto_influence.csv",
    "municipal_reference_geography_primary_summary.csv",
    "municipal_reference_geography_provenance.json",
)

Row = dict[str, str | int]
DistanceIndex = dict[tuple[str, str], float]


def distance_pairs(
    centroids: dict[str, tuple[float, float]] | None = None,
) -> list[Row]:
    """Derive every ordered reference pair from the checksummed CAOP snapshot.

    The optional coordinates facilitate synthetic tests; normal execution
    always checks the pinned official CAOP derivative via load_centroids().
    """
    coords = load_centroids() if centroids is None else centroids
    if set(coords) != set(MUNICIPALITIES):
        raise ValueError("Geography must contain exactly the 17 AMP municipalities")
    rows: list[Row] = []
    for reference in sorted(MUNICIPALITIES):
        x, y = coords[reference]
        for comparator in sorted(MUNICIPALITIES - {reference}):
            a, b = coords[comparator]
            if not all(math.isfinite(float(v)) for v in (x, y, a, b)):
                raise ValueError("Nonfinite CAOP projected coordinates")
            km = math.hypot(float(a) - float(x), float(b) - float(y)) / 1_000
            if not math.isfinite(km) or km <= 0:
                raise ValueError("Positive distance required between different centroids")
            rows.append({
                "reference_municipality": reference,
                "comparator_municipality": comparator,
                "caop_year": 2025,
                "distance_definition": "caop2025_epsg3763_area_weighted_centroid_euclidean_km",
                "centroid_distance_km": f"{km:.9f}",
            })
    _distance_index(rows)
    return rows


def _distance_index(rows: Sequence[Row]) -> DistanceIndex:
    """Require all 272 directed views and symmetric, positive distances."""
    expected = {(a, b) for a in MUNICIPALITIES for b in MUNICIPALITIES if a != b}
    seen: DistanceIndex = {}
    for row in rows:
        key = (str(row["reference_municipality"]), str(row["comparator_municipality"]))
        km = float(row["centroid_distance_km"])
        if key not in expected or key in seen or not math.isfinite(km) or km <= 0:
            raise ValueError("Invalid, duplicated or non-positive CAOP pair distance")
        if int(row["caop_year"]) != 2025:
            raise ValueError("CAOP pair source year must remain 2025")
        seen[key] = km
    if len(rows) != 272 or set(seen) != expected:
        raise ValueError("Incomplete CAOP reference-pair distance grid")
    if any(abs(km - seen[b, a]) > 1e-8 for (a, b), km in seen.items()):
        raise ValueError("Centroid distances must be symmetric")
    return seen


def _distances(rows: Sequence[Row], indexed: DistanceIndex) -> np.ndarray:
    """Align the correct reference-to-comparator distance to rent observations."""
    values = np.array([
        indexed[str(r["reference_municipality"]), str(r["comparator_municipality"])]
        for r in rows
    ], dtype=float)
    if len(values) < 5 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("At least five positive reference-specific distances required")
    return values


def full_reference_grid(window_rows: Sequence[Row], distance_rows: Sequence[Row]) -> list[Row]:
    """Full 4×17×3×2 unadjusted and distance-conditional rank grid."""
    _require_window_grid(window_rows)
    distances = _distance_index(distance_rows)
    earlier = {
        (int(r["window_start_year"]), str(r["reference_municipality"]),
         str(r["exposure_mode"]), str(r["outcome"])): r
        for r in window_ranks(window_rows)
    }
    grouped: dict[tuple[int, str], list[Row]] = {}
    for row in window_rows:
        grouped.setdefault((int(row["window_start_year"]),
                            str(row["reference_municipality"])), []).append(row)
    output: list[Row] = []
    for start in WINDOW_STARTS:
        for reference in sorted(MUNICIPALITIES):
            selected = grouped[start, reference]
            if len(selected) != 16:
                raise ValueError("Exactly 16 comparator municipalities are required")
            d = _distances(selected, distances)
            for mode in MODES:
                for outcome in OUTCOMES:
                    x, y = _window_vectors(selected, mode, outcome)
                    rho = spearman_rho(x, y)
                    previous = earlier[start, reference, mode, outcome]
                    if abs(rho - float(previous["spearman_rank_rho"])) > 5e-9:
                        raise ValueError("Unadjusted coefficient disagrees with previous rank audit")
                    exposure_distance_rho = spearman_rho(x, d)
                    conditional = partial_rank_correlation(x, y, d)
                    output.append({
                        "window_start_year": start,
                        "window_end_year": END_YEAR,
                        "commuting_network_year": 2021,
                        "caop_centroid_year": 2025,
                        "reference_municipality": reference,
                        "exposure_mode": mode,
                        "outcome": outcome,
                        "comparators": 16,
                        "unadjusted_spearman_rho": f"{rho:.9f}",
                        "conditional_on_centroid_distance_partial_spearman_rho": f"{conditional:.9f}",
                        "exposure_vs_centroid_distance_spearman_rho": f"{exposure_distance_rho:.9f}",
                        "outcome_vs_centroid_distance_spearman_rho": f"{spearman_rho(y, d):.9f}",
                        "rank_distance_vif_diagnostic": f"{1 / (1 - exposure_distance_rho ** 2):.6f}",
                        "strong_exposure_distance_collinearity_flag": int(
                            abs(exposure_distance_rho) >= HIGH_RANK_DISTANCE_CORRELATION
                        ),
                        "unadjusted_absolute_rank_among_17": previous["absolute_rho_rank_among_17"],
                        "conditional_absolute_rank_among_17": 0,
                    })
    expected = len(WINDOW_STARTS) * len(MUNICIPALITIES) * len(MODES) * len(OUTCOMES)
    if len(output) != expected:
        raise ValueError("Incomplete CAOP-conditioned reference grid")
    for start in WINDOW_STARTS:
        for mode in MODES:
            for outcome in OUTCOMES:
                subset = [r for r in output if r["window_start_year"] == start
                          and r["exposure_mode"] == mode and r["outcome"] == outcome]
                absolute = [
                    abs(float(r["conditional_on_centroid_distance_partial_spearman_rho"]))
                    for r in subset
                ]
                for row, value in zip(subset, absolute):
                    row["conditional_absolute_rank_among_17"] = 1 + sum(
                        other > value + 1e-10 for other in absolute
                    )
    return output


def matched_reference_controls(window_rows: Sequence[Row], distance_rows: Sequence[Row]) -> list[Row]:
    """Compare Porto and each alternative on the same 15 own-growth rents.

    Each reference has its own distance-to-reference control vector.
    Thus this comparison controls for distinct geometries: the difference
    is not solely attributable to replacing the network reference.
    """
    _require_window_grid(window_rows)
    distances = _distance_index(distance_rows)
    lookup = {
        (int(r["window_start_year"]), str(r["reference_municipality"]),
         str(r["comparator_municipality"])): r for r in window_rows
    }
    output: list[Row] = []
    for start in WINDOW_STARTS:
        for alternative in sorted(MUNICIPALITIES - {REFERENCE}):
            names = sorted(MUNICIPALITIES - {REFERENCE, alternative})
            porto = [lookup[start, REFERENCE, name] for name in names]
            alt = [lookup[start, alternative, name] for name in names]
            dp, da = _distances(porto, distances), _distances(alt, distances)
            for mode in MODES:
                xp, yp = _window_vectors(porto, mode, OUTCOMES[0])
                xa, ya = _window_vectors(alt, mode, OUTCOMES[0])
                if not np.array_equal(yp, ya):
                    raise ValueError("Matched own-growth outcomes must remain exactly the same")
                raw_p, raw_a = spearman_rho(xp, yp), spearman_rho(xa, ya)
                partial_p = partial_rank_correlation(xp, yp, dp)
                partial_a = partial_rank_correlation(xa, ya, da)
                output.append({
                    "window_start_year": start,
                    "window_end_year": END_YEAR,
                    "alternative_reference": alternative,
                    "baseline_reference": REFERENCE,
                    "exposure_mode": mode,
                    "outcome": OUTCOMES[0],
                    "shared_comparators": len(names),
                    "shared_municipality_names": "|".join(names),
                    "porto_raw_rho_on_same_15": f"{raw_p:.9f}",
                    "alternative_raw_rho_on_same_15": f"{raw_a:.9f}",
                    "raw_alternative_minus_porto_rho": f"{raw_a - raw_p:.9f}",
                    "porto_distance_conditional_rho_on_same_15": f"{partial_p:.9f}",
                    "alternative_distance_conditional_rho_on_same_15": f"{partial_a:.9f}",
                    "conditional_alternative_minus_porto_rho": f"{partial_a - partial_p:.9f}",
                    "porto_exposure_distance_rho_on_same_15": f"{spearman_rho(xp, dp):.9f}",
                    "alternative_exposure_distance_rho_on_same_15": f"{spearman_rho(xa, da):.9f}",
                    "conditioning_distance_differs_by_reference": 1,
                })
    if len(output) != len(WINDOW_STARTS) * 16 * len(MODES):
        raise ValueError("Incomplete matched reference-geography grid")
    return output


def porto_influence(window_rows: Sequence[Row], distance_rows: Sequence[Row]) -> list[Row]:
    """Single-comparator deletions for the Porto per-resident own-growth rank."""
    _require_window_grid(window_rows)
    distances = _distance_index(distance_rows)
    output: list[Row] = []
    for start in WINDOW_STARTS:
        rows = sorted((r for r in window_rows if r["window_start_year"] == start
                       and r["reference_municipality"] == REFERENCE),
                      key=lambda r: str(r["comparator_municipality"]))
        if len(rows) != 16:
            raise ValueError("Porto must have 16 comparator municipalities")
        x, y = _window_vectors(rows, "per_1000_residents", OUTCOMES[0])
        d = _distances(rows, distances)
        raw = spearman_rho(x, y)
        conditional = partial_rank_correlation(x, y, d)
        for i, row in enumerate(rows):
            mask = np.arange(len(rows)) != i
            adjusted = partial_rank_correlation(x[mask], y[mask], d[mask])
            output.append({
                "window_start_year": start,
                "window_end_year": END_YEAR,
                "reference_municipality": REFERENCE,
                "exposure_mode": "per_1000_residents",
                "outcome": OUTCOMES[0],
                "removed_comparator": row["comparator_municipality"],
                "remaining_comparators": 15,
                "full_16_unadjusted_rho": f"{raw:.9f}",
                "full_16_conditional_rho": f"{conditional:.9f}",
                "leave_one_out_15_conditional_rho": f"{adjusted:.9f}",
                "conditional_sign_changed": int(
                    (adjusted > 0 > conditional) or (adjusted < 0 < conditional)
                ),
            })
    if len(output) != len(WINDOW_STARTS) * 16:
        raise ValueError("Incomplete Porto influence grid")
    return output


def primary_summary(full: Sequence[Row], matched: Sequence[Row],
                    influence: Sequence[Row]) -> list[Row]:
    """Report all prespecified Porto windows, no best-year selection."""
    expected_full = {(y, r, m, o) for y in WINDOW_STARTS for r in MUNICIPALITIES
                     for m in MODES for o in OUTCOMES}
    indexed = {(int(r["window_start_year"]), str(r["reference_municipality"]),
                str(r["exposure_mode"]), str(r["outcome"])): r for r in full}
    if len(full) != len(expected_full) or set(indexed) != expected_full:
        raise ValueError("Missing or duplicated geography-control rank results")
    expected_matched = {(y, r, m) for y in WINDOW_STARTS
                        for r in MUNICIPALITIES - {REFERENCE} for m in MODES}
    matched_index = {(int(r["window_start_year"]), str(r["alternative_reference"]),
                      str(r["exposure_mode"])): r for r in matched}
    if len(matched) != len(expected_matched) or set(matched_index) != expected_matched:
        raise ValueError("Missing or duplicated matched geography controls")
    expected_loo = {(y, r) for y in WINDOW_STARTS for r in MUNICIPALITIES - {REFERENCE}}
    influence_index = {(int(r["window_start_year"]), str(r["removed_comparator"])): r
                       for r in influence}
    if len(influence) != len(expected_loo) or set(influence_index) != expected_loo:
        raise ValueError("Missing or duplicated Porto geography influence rows")
    output: list[Row] = []
    for start in WINDOW_STARTS:
        for outcome in OUTCOMES:
            porto = indexed[start, REFERENCE, "per_1000_residents", outcome]
            cells = [indexed[start, ref, "per_1000_residents", outcome]
                     for ref in sorted(MUNICIPALITIES)]
            lower, threshold = [], []
            if outcome == OUTCOMES[0]:
                lower = [ref for ref in sorted(MUNICIPALITIES - {REFERENCE})
                         if float(matched_index[start, ref, "per_1000_residents"][
                             "conditional_alternative_minus_porto_rho"]) < 0]
                threshold = [ref for ref in sorted(MUNICIPALITIES - {REFERENCE})
                             if float(matched_index[start, ref, "per_1000_residents"][
                                 "conditional_alternative_minus_porto_rho"]) <= -MATERIAL_RHO_DIFFERENCE]
            loo = [influence_index[start, ref] for ref in sorted(MUNICIPALITIES - {REFERENCE})]
            output.append({
                "window_start_year": start,
                "window_end_year": END_YEAR,
                "outcome": outcome,
                "exposure_mode": "per_1000_residents",
                "porto_unadjusted_rho": porto["unadjusted_spearman_rho"],
                "porto_conditional_rho": porto["conditional_on_centroid_distance_partial_spearman_rho"],
                "porto_unadjusted_absolute_rank_among_17": porto["unadjusted_absolute_rank_among_17"],
                "porto_conditional_absolute_rank_among_17": porto["conditional_absolute_rank_among_17"],
                "porto_exposure_distance_rho": porto["exposure_vs_centroid_distance_spearman_rho"],
                "high_collinearity_references_out_of_17": sum(
                    int(r["strong_exposure_distance_collinearity_flag"]) for r in cells
                ),
                "matched_alternatives_more_negative_conditional_count": len(lower) if outcome == OUTCOMES[0] else "",
                "matched_alternatives_more_negative_conditional_names": "|".join(lower) if outcome == OUTCOMES[0] else "",
                "matched_alternatives_at_least_0_05_more_negative_count": (
                    len(threshold) if outcome == OUTCOMES[0] else ""
                ),
                "matched_alternatives_at_least_0_05_more_negative_names": (
                    "|".join(threshold) if outcome == OUTCOMES[0] else ""
                ),
                "porto_primary_loo_conditional_min": (
                    f"{min(float(r['leave_one_out_15_conditional_rho']) for r in loo):.9f}"
                    if outcome == OUTCOMES[0] else ""
                ),
                "porto_primary_loo_conditional_max": (
                    f"{max(float(r['leave_one_out_15_conditional_rho']) for r in loo):.9f}"
                    if outcome == OUTCOMES[0] else ""
                ),
                "porto_primary_loo_sign_changes": (
                    sum(int(r["conditional_sign_changed"]) for r in loo)
                    if outcome == OUTCOMES[0] else ""
                ),
            })
    return output


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    """Write an ordered, self-describing UTF-8 CSV result."""
    if not rows:
        raise ValueError("Cannot serialize an empty geography-control table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_conditioned_grids(full: Sequence[Row], path: Path) -> None:
    """Show Porto among every reference, before and after distance conditioning."""
    selected = [r for r in full if r["exposure_mode"] == "per_1000_residents"
                and r["outcome"] == OUTCOMES[0]]
    lookup = {(int(r["window_start_year"]), str(r["reference_municipality"])): r
              for r in selected}
    references = sorted(MUNICIPALITIES, key=lambda name: float(
        lookup[2020, name]["unadjusted_spearman_rho"]))
    fig, axes = plt.subplots(1, 2, figsize=(13.1, 8), sharex=True, sharey=True)
    for ax, key, label in zip(
        axes,
        ("unadjusted_spearman_rho", "conditional_on_centroid_distance_partial_spearman_rho"),
        ("Unadjusted Spearman ρ", "Conditional on CAOP centroid distance"),
    ):
        matrix = np.array([[float(lookup[year, ref][key]) for year in WINDOW_STARTS]
                           for ref in references])
        heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(4), [f"{year}–25" for year in WINDOW_STARTS])
        ax.set_yticks(range(17), references, fontsize=8.6)
        ax.set_title(label, fontsize=11.5)
        for i in range(17):
            for j in range(4):
                ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center",
                        fontsize=8.3, color="white" if abs(matrix[i, j]) >= .60 else "#203747")
        ax.add_patch(plt.Rectangle((-.5, references.index(REFERENCE) - .5), 4, 1,
                                   fill=False, linewidth=2, edgecolor="#126b5f"))
    color_ax = fig.add_axes([.905, .19, .018, .57])
    fig.colorbar(heat, cax=color_ax, label="Descriptive rank coefficient")
    fig.suptitle("All reference-node rent-growth ranks: geography matters", fontsize=13.4, y=.978)
    fig.text(.02, .018, "Own nominal new-rent growth, 2020/21/22/23–2025. 2021 work/study pairs per 1,000 residents; 2025 CAOP area centroids.",
             fontsize=8.3, color="#536574")
    fig.subplots_adjust(left=.19, right=.88, bottom=.12, top=.88, wspace=.11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rank_collinearity(full: Sequence[Row], path: Path) -> None:
    """Show rank overlap between work/study exposure and centroid distance."""
    subset = [r for r in full if r["window_start_year"] == 2020
              and r["outcome"] == OUTCOMES[0]]
    lookup = {(str(r["reference_municipality"]), str(r["exposure_mode"])): r
              for r in subset}
    references = sorted(MUNICIPALITIES, key=lambda ref: float(
        lookup[ref, "per_1000_residents"]["exposure_vs_centroid_distance_spearman_rho"]))
    matrix = np.array([[
        float(lookup[ref, mode]["exposure_vs_centroid_distance_spearman_rho"])
        for mode in MODES
    ] for ref in references])
    if matrix.shape != (17, 3):
        raise ValueError("Full 17×3 reference-distance collinearity panel required")
    fig, ax = plt.subplots(figsize=(9.4, 8.4))
    heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["Raw pairs", "Pairs per 1,000 residents", "Within-AMP link share"])
    ax.set_yticks(range(17), references, fontsize=9)
    ax.set_title("Work/study connectivity tracks geometric proximity", fontsize=12.4, pad=12)
    for i in range(17):
        for j in range(3):
            warning = abs(matrix[i, j]) >= HIGH_RANK_DISTANCE_CORRELATION
            ax.text(j, i, f"{matrix[i,j]:+.2f}{'*' if warning else ''}", ha="center", va="center",
                    fontsize=9.3, color="white" if abs(matrix[i, j]) >= .70 else "#243747",
                    weight="bold" if warning else "normal")
    ax.add_patch(plt.Rectangle((-.5, references.index(REFERENCE) - .5), 3, 1,
                               fill=False, linewidth=2.0, edgecolor="#126b5f"))
    fig.colorbar(heat, ax=ax, fraction=.035, pad=.035, label="Spearman ρ(exposure, centroid distance)")
    fig.text(.02, .025, "* |ρ| ≥ 0.90 on unrounded values: flagged collinearity, not a test. Centroid distances are not travel costs.",
             fontsize=8.4, color="#536574")
    fig.subplots_adjust(left=.25, right=.92, bottom=.12, top=.89)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Reproduce geography-conditioned reference controls from local pins."""
    windows = window_pairs()
    pairs = distance_pairs()
    full = full_reference_grid(windows, pairs)
    matched = matched_reference_controls(windows, pairs)
    influence = porto_influence(windows, pairs)
    summary = primary_summary(full, matched, influence)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_rank_collinearity(full, figure_dir / FIGURES[0])
    plot_conditioned_grids(full, figure_dir / FIGURES[1])
    for filename, table in zip(RESULTS[:5], (pairs, full, matched, influence, summary)):
        _write_csv(result_dir / filename, table)
    ca = json.loads(CAOP_PROVENANCE.read_text(encoding="utf-8"))
    commute = json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "evidence_class": "noncausal_reference_specific_geometry_conditional_rank_negative_control",
        "housing_upstream_commit": SOURCE_COMMIT,
        "housing_panel_sha256": SUBSET_SHA256,
        "commuting_2021_source_status": commute["source_type"],
        "commuting_source_workbook_sha256": commute["source_workbook_sha256"],
        "commuting_derived_pair_csv_sha256": commute["derived_csv_sha256"],
        "caop2025_source_response_sha256": ca["source_response_sha256"],
        "caop2025_derived_csv_sha256": ca["derived_csv_sha256"],
        "caop2025_source_license": ca["source_license"],
        "centroid_crs_epsg": 3763,
        "commuting_network_year": 2021,
        "rent_window_starts": list(WINDOW_STARTS),
        "common_rent_endpoint": END_YEAR,
        "exposure_modes": list(MODES),
        "outcomes": list(OUTCOMES),
        "strong_exposure_distance_rho_absolute_threshold": HIGH_RANK_DISTANCE_CORRELATION,
        "matched_conditional_rho_difference_descriptive_threshold": MATERIAL_RHO_DIFFERENCE,
        "distance_pair_rows": len(pairs),
        "full_reference_rows": len(full),
        "matched_support_rows": len(matched),
        "porto_leave_one_out_rows": len(influence),
        "primary_summary_rows": len(summary),
        "primary_summary": summary,
        "limits": [
            "Area-weighted CAOP2025 municipal polygon centroids are not population centers, commuting times, or housing access",
            "2025 administrative geometry is not reconstructed at each 2020/2021/2022/2023 baseline",
            "Rank distance conditioning only removes a one-variable monotone rank association, not omitted local factors",
            "High exposure-distance collinearity and samples of only 15 or 16 make partial coefficients unstable",
            "The 17 reference groups and nested rent windows overlap; partial rank coefficients have no reported p-values",
            "Matched own-growth outcomes are invariant but reference-specific geographic control vectors differ",
            "Commuting links are symmetric work/study interactions, not observed directed residential migration",
            "The third-party Census-derived commuting workbook has not been revalidated against official INE API",
            "The 2021 network does not predate the 2020 baseline and is not an exogenous shock for later windows",
            "New-contract rent medians do not track the same rental units or tenants across years",
            "No causally identified household displacement, urban price contagion, transport burden or supply elasticity",
        ],
    }
    (result_dir / RESULTS[5]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Regenerated {report['full_reference_rows']} geography-conditional rank comparisons")


if __name__ == "__main__":
    main()
