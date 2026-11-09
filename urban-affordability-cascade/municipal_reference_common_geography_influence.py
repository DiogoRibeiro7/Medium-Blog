"""Leave-one-municipality-out audit of matched shared-geography rank contrasts.

This is a *descriptive* deletion sensitivity exercise. Each matched
Porto-versus-alternative comparison starts with 15 municipalities; removing
one leaves 14. Crucially, all exposure, outcome and both geographic-control
ranks are rebuilt on each 14-municipality sample before residualisation.

The 2021 undirected work/study network is not a migration origin/destination
matrix. Deletion ranges are neither jackknife confidence intervals nor
independent replications, and no causal spillover is identified.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis import MUNICIPALITIES, ROOT, SOURCE_COMMIT, SUBSET_SHA256
from extract_caop import PROVENANCE as CAOP_PROVENANCE
from extract_commuting import PROVENANCE as COMMUTING_PROVENANCE
from municipal_reference_common_geography import (
    PRIMARY_MODE,
    matched_common_controls,
    rank_residual_correlation,
)
from municipal_reference_geography_controls import _distance_index, distance_pairs
from municipal_reference_placebos import REFERENCE
from municipal_reference_window_sensitivity import (
    MATERIAL_RHO_DIFFERENCE,
    OUTCOMES,
    _require_window_grid,
    _window_vectors,
    window_pairs,
)
from porto_network_gradient import END_YEAR, MODES, WINDOW_STARTS

Row = dict[str, str | int]
FIGURES = (
    "fig42_matched_reference_deletion_sign_reversals.svg",
    "fig43_matched_reference_deletion_window_stability.svg",
)
RESULTS = (
    "municipal_reference_common_geography_influence_deletions.csv",
    "municipal_reference_common_geography_influence_pairs.csv",
    "municipal_reference_common_geography_influence_windows.csv",
    "municipal_reference_common_geography_influence_provenance.json",
)


def _numeric(value: float) -> str:
    """Use the same nine-decimal numeric reporting convention as the parent."""
    if not np.isfinite(value):
        raise ValueError("Influence statistic is not finite")
    return f"{float(value):.9f}"


def deletion_grid(
    windows: Sequence[Row],
    distances: Sequence[Row],
    matched: Sequence[Row],
) -> list[Row]:
    """Re-rank all 2,880 deletion samples on the symmetric two-distance design.

    Parameters are complete, pinned upstream data tables, not already ranked
    vectors. The parent 576-row table is an independently computed baseline
    oracle; it must agree before deletion results can be reported.
    """
    _require_window_grid(windows)
    index = _distance_index(distances)
    baseline = {
        (int(r["window_start_year"]), str(r["alternative_reference"]),
         str(r["exposure_mode"])): r
        for r in matched if r["shared_control_set"] == "both_distances"
    }
    expected_baselines = len(WINDOW_STARTS) * (len(MUNICIPALITIES) - 1) * len(MODES)
    if len(matched) != expected_baselines * 3 or len(baseline) != expected_baselines:
        raise ValueError("Expected the complete parent 576-row shared-geography grid")
    window_index = {
        (int(r["window_start_year"]), str(r["reference_municipality"]),
         str(r["comparator_municipality"])): r for r in windows
    }
    output: list[Row] = []
    for year in WINDOW_STARTS:
        for alternative in sorted(MUNICIPALITIES - {REFERENCE}):
            names = sorted(MUNICIPALITIES - {REFERENCE, alternative})
            if len(names) != 15:
                raise ValueError("Reference comparison must have 15 municipalities")
            porto = [window_index[year, REFERENCE, name] for name in names]
            other = [window_index[year, alternative, name] for name in names]
            controls = np.column_stack((
                [index[REFERENCE, name] for name in names],
                [index[alternative, name] for name in names],
            ))
            for mode in MODES:
                x_porto, y_porto = _window_vectors(porto, mode, OUTCOMES[0])
                x_other, y_other = _window_vectors(other, mode, OUTCOMES[0])
                if not np.array_equal(y_porto, y_other):
                    raise ValueError("References must share identical own-rent outcomes")
                parent = baseline[year, alternative, mode]
                rho_porto, _, _ = rank_residual_correlation(
                    x_porto, y_porto, controls)
                rho_other, _, _ = rank_residual_correlation(
                    x_other, y_porto, controls)
                baseline_delta = rho_other - rho_porto
                if abs(baseline_delta - float(parent[
                    "shared_alternative_minus_porto_rho"
                ])) > 5e-8:
                    raise ValueError("Recomputed 15-municipality baseline disagrees with parent")
                for omitted_index, omitted in enumerate(names):
                    mask = np.ones(len(names), dtype=bool)
                    mask[omitted_index] = False
                    # This calls average_ranks inside rank_residual_correlation
                    # on the *14-observation vectors*, not precomputed 15-ranks.
                    p, p_remain, y_remain = rank_residual_correlation(
                        x_porto[mask], y_porto[mask], controls[mask, :])
                    a, a_remain, y_other_remain = rank_residual_correlation(
                        x_other[mask], y_other[mask], controls[mask, :])
                    if abs(y_remain - y_other_remain) > 1e-10:
                        raise ValueError("Deleted samples have different outcome controls")
                    delta = a - p
                    output.append({
                        "window_start_year": year,
                        "window_end_year": END_YEAR,
                        "alternative_reference": alternative,
                        "baseline_reference": REFERENCE,
                        "exposure_mode": mode,
                        "shared_control_set": "both_distances",
                        "omitted_municipality": omitted,
                        "remaining_comparators": 14,
                        "baseline_shared_alternative_minus_porto_rho": _numeric(baseline_delta),
                        "deleted_shared_porto_rho": _numeric(p),
                        "deleted_shared_alternative_rho": _numeric(a),
                        "deleted_shared_alternative_minus_porto_rho": _numeric(delta),
                        "deletion_minus_baseline_delta": _numeric(delta - baseline_delta),
                        "sign_reversal": int((delta < 0) != (baseline_delta < 0)),
                        "material_margin_crossing": int(
                            (delta <= -MATERIAL_RHO_DIFFERENCE)
                            != (baseline_delta <= -MATERIAL_RHO_DIFFERENCE)
                        ),
                        "porto_exposure_remaining_rank_variance_fraction": _numeric(p_remain),
                        "alternative_exposure_remaining_rank_variance_fraction": _numeric(a_remain),
                        "outcome_remaining_rank_variance_fraction": _numeric(y_remain),
                    })
    expected = expected_baselines * 15
    keys = {
        (int(r["window_start_year"]), str(r["alternative_reference"]),
         str(r["exposure_mode"]), str(r["omitted_municipality"])) for r in output
    }
    if len(output) != expected or len(keys) != expected:
        raise ValueError("Missing or duplicated 2,880 municipality deletion cases")
    return output


def pair_summary(rows: Sequence[Row]) -> list[Row]:
    """Summarise all 15 deletions per year, alternative and exposure mode."""
    expected = {
        (year, alternative, mode, omitted)
        for year in WINDOW_STARTS
        for alternative in MUNICIPALITIES - {REFERENCE}
        for mode in MODES
        for omitted in MUNICIPALITIES - {REFERENCE, alternative}
    }
    keys = {
        (int(r["window_start_year"]), str(r["alternative_reference"]),
         str(r["exposure_mode"]), str(r["omitted_municipality"])) for r in rows
    }
    if len(rows) != len(expected) or keys != expected:
        raise ValueError("Pair summary requires all 2,880 unique deletion cases")
    groups: dict[tuple[int, str, str], list[Row]] = {}
    for row in rows:
        key = (int(row["window_start_year"]), str(row["alternative_reference"]),
               str(row["exposure_mode"]))
        groups.setdefault(key, []).append(row)
    result: list[Row] = []
    for year in WINDOW_STARTS:
        for alternative in sorted(MUNICIPALITIES - {REFERENCE}):
            for mode in MODES:
                group = groups[year, alternative, mode]
                base = {str(r["baseline_shared_alternative_minus_porto_rho"]) for r in group}
                if len(group) != 15 or len(base) != 1:
                    raise ValueError("Inconsistent baseline or missing deletion comparator")
                deltas = np.array([float(r["deleted_shared_alternative_minus_porto_rho"])
                                   for r in group])
                shifts = np.array([float(r["deletion_minus_baseline_delta"])
                                   for r in group])
                influential = sorted(
                    group,
                    key=lambda r: (-abs(float(r["deletion_minus_baseline_delta"])),
                                   str(r["omitted_municipality"])),
                )[0]
                min_exposure_variance = min(
                    float(r[k]) for r in group for k in (
                        "porto_exposure_remaining_rank_variance_fraction",
                        "alternative_exposure_remaining_rank_variance_fraction",
                    )
                )
                result.append({
                    "window_start_year": year,
                    "alternative_reference": alternative,
                    "exposure_mode": mode,
                    "baseline_shared_alternative_minus_porto_rho": next(iter(base)),
                    "deletion_samples": len(group),
                    "sign_reversals_over_15": sum(int(r["sign_reversal"]) for r in group),
                    "material_margin_crossings_over_15": sum(
                        int(r["material_margin_crossing"]) for r in group),
                    "minimum_deleted_delta": _numeric(float(deltas.min())),
                    "maximum_deleted_delta": _numeric(float(deltas.max())),
                    "largest_absolute_delta_shift": _numeric(float(np.abs(shifts).max())),
                    "most_influential_omitted_municipality": influential["omitted_municipality"],
                    "minimum_exposure_remaining_rank_variance_fraction": _numeric(
                        min_exposure_variance),
                })
    if len(result) != 192:
        raise ValueError("Expected 192 pair-level deletion summaries")
    return result


def window_summary(pairs: Sequence[Row]) -> list[Row]:
    """Count fragile references without claiming independent observations."""
    expected = {
        (year, alternative, mode)
        for year in WINDOW_STARTS
        for alternative in MUNICIPALITIES - {REFERENCE}
        for mode in MODES
    }
    keys = {(int(r["window_start_year"]), str(r["alternative_reference"]),
             str(r["exposure_mode"])) for r in pairs}
    if len(pairs) != len(expected) or keys != expected:
        raise ValueError("Window summary requires all 192 pair summaries")
    result: list[Row] = []
    for year in WINDOW_STARTS:
        for mode in MODES:
            group = sorted(
                (r for r in pairs if int(r["window_start_year"]) == year
                 and r["exposure_mode"] == mode),
                key=lambda r: str(r["alternative_reference"]),
            )
            flipped = [str(r["alternative_reference"]) for r in group
                       if int(r["sign_reversals_over_15"]) > 0]
            crossed = [str(r["alternative_reference"]) for r in group
                       if int(r["material_margin_crossings_over_15"]) > 0]
            result.append({
                "window_start_year": year,
                "window_end_year": END_YEAR,
                "exposure_mode": mode,
                "alternatives": len(group),
                "deletion_cases": len(group) * 15,
                "alternatives_with_any_sign_reversal": len(flipped),
                "sign_reversal_alternative_names": "|".join(flipped),
                "alternatives_with_any_margin_crossing": len(crossed),
                "margin_crossing_alternative_names": "|".join(crossed),
                "minimum_exposure_remaining_rank_variance_fraction": _numeric(
                    min(float(r["minimum_exposure_remaining_rank_variance_fraction"])
                        for r in group)),
                "largest_absolute_delta_shift": _numeric(
                    max(float(r["largest_absolute_delta_shift"]) for r in group)),
            })
    if len(result) != 12:
        raise ValueError("Expected 12 year-by-exposure summaries")
    return result


def heatmap_svg(pairs: Sequence[Row]) -> str:
    """Draw a plain SVG with all 16 references and all four rent windows."""
    selection = {(int(r["window_start_year"]), str(r["alternative_reference"])): r
                 for r in pairs if r["exposure_mode"] == PRIMARY_MODE}
    alternatives = sorted(MUNICIPALITIES - {REFERENCE})
    if len(selection) != 64:
        raise ValueError("Heatmap requires the complete 16 x 4 primary grid")
    colors = ("#edf5f9", "#d2e6f2", "#abd0e6", "#6eafd1", "#24668d")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="810" viewBox="0 0 1020 810">',
        '<rect width="1020" height="810" fill="#ffffff"/>',
        '<text x="34" y="48" font-size="24" font-family="sans-serif" fill="#183248">Municipality deletion can reverse reference rankings</text>',
        '<text x="34" y="78" font-size="15" font-family="sans-serif" fill="#43596c">Number of sign reversals across 15 leave-one-out samples · shared two-distance controls</text>',
    ]
    for col, year in enumerate(WINDOW_STARTS):
        lines.append(
            f'<text x="{335 + col * 153}" y="134" font-size="16" text-anchor="middle" '
            f'font-family="sans-serif" fill="#183248">{year}–25</text>'
        )
    for row_index, name in enumerate(alternatives):
        center = 158 + row_index * 35
        lines.append(
            f'<text x="250" y="{center + 20}" font-size="14" text-anchor="end" '
            f'font-family="sans-serif" fill="#183248">{html.escape(name)}</text>'
        )
        for col, year in enumerate(WINDOW_STARTS):
            count = int(selection[year, name]["sign_reversals_over_15"])
            if not 0 <= count <= 15:
                raise ValueError("Invalid deletion sign-reversal count")
            fill = colors[min(count, 4)]
            color = "#ffffff" if count >= 4 else "#183248"
            lines.append(
                f'<rect x="{263 + col * 153}" y="{center}" width="143" height="30" '
                f'fill="{fill}" rx="3"/>'
            )
            lines.append(
                f'<text x="{335 + col * 153}" y="{center + 21}" font-size="15" '
                f'text-anchor="middle" font-family="sans-serif" fill="{color}">{count}</text>'
            )
    lines.extend([
        '<text x="34" y="758" font-size="13" font-family="sans-serif" fill="#43596c">2021 undirected work/study pairs per 1,000 comparator residents; 2025 administrative centroids.</text>',
        '<text x="34" y="783" font-size="13" font-family="sans-serif" fill="#43596c">Descriptive deletion sensitivity, not a significance test, independent replication or residential-migration effect.</text>',
        '</svg>',
    ])
    return "\n".join(lines) + "\n"


def stability_svg(windows: Sequence[Row]) -> str:
    """Plot two declared instability indicators for the four rent baselines."""
    chosen = {int(r["window_start_year"]): r for r in windows
              if r["exposure_mode"] == PRIMARY_MODE}
    if set(chosen) != set(WINDOW_STARTS):
        raise ValueError("Stability plot requires all four primary windows")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="560" viewBox="0 0 1020 560">',
        '<rect width="1020" height="560" fill="#ffffff"/>',
        '<text x="35" y="48" font-size="24" font-family="sans-serif" fill="#183248">Matched reference sensitivity by rent baseline</text>',
        '<text x="35" y="77" font-size="15" font-family="sans-serif" fill="#43596c">Alternative references affected by at least one of 15 deletions (out of 16)</text>',
        '<rect x="40" y="103" width="17" height="17" fill="#2b6e94"/>',
        '<text x="65" y="117" font-size="14" font-family="sans-serif" fill="#183248">Any sign reversal</text>',
        '<rect x="286" y="103" width="17" height="17" fill="#c58a48"/>',
        '<text x="311" y="117" font-size="14" font-family="sans-serif" fill="#183248">Any crossing of descriptive −0.05 margin</text>',
    ]
    for year_index, year in enumerate(WINDOW_STARTS):
        item = chosen[year]
        values = (
            int(item["alternatives_with_any_sign_reversal"]),
            int(item["alternatives_with_any_margin_crossing"]),
        )
        for position, (count, fill) in enumerate(zip(values, ("#2b6e94", "#c58a48"))):
            if not 0 <= count <= 16:
                raise ValueError("Invalid affected-alternative count")
            x = 118 + year_index * 220 + position * 65
            height = count * 16
            top = 419 - height
            lines.append(
                f'<rect x="{x}" y="{top}" width="57" height="{height}" fill="{fill}"/>'
            )
            lines.append(
                f'<text x="{x + 28}" y="{top - 12}" font-size="17" text-anchor="middle" '
                f'font-family="sans-serif" fill="#183248">{count}</text>'
            )
        lines.append(
            f'<text x="{150 + year_index * 220}" y="454" font-size="17" '
            f'text-anchor="middle" font-family="sans-serif" fill="#183248">{year}–25</text>'
        )
    lines.extend([
        '<line x1="55" y1="420" x2="970" y2="420" stroke="#7890a0" stroke-width="1"/>',
        '<text x="35" y="505" font-size="13" font-family="sans-serif" fill="#43596c">Shared 2025 rent endpoint; overlapping windows are dependent and not independent confirmations.</text>',
        '<text x="35" y="531" font-size="13" font-family="sans-serif" fill="#43596c">Changing the sample changes ranks and collinearity; neither series estimates causal displacement.</text>',
        '</svg>',
    ])
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    """Write deterministic, complete CSV tables."""
    if not rows:
        raise ValueError("Cannot write an empty influence table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate all deletion evidence and source-checked provenance offline."""
    windows, distances = window_pairs(), distance_pairs()
    parent = matched_common_controls(windows, distances)
    deletions = deletion_grid(windows, distances, parent)
    pairs = pair_summary(deletions)
    summaries = window_summary(pairs)
    result_dir, figure_dir = output / "results", output / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(RESULTS[:3], (deletions, pairs, summaries)):
        _write_csv(result_dir / name, rows)
    (figure_dir / FIGURES[0]).write_text(heatmap_svg(pairs), encoding="utf-8")
    (figure_dir / FIGURES[1]).write_text(stability_svg(summaries), encoding="utf-8")
    ca = json.loads(CAOP_PROVENANCE.read_text(encoding="utf-8"))
    commute = json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "evidence_class": "noncausal_matched_reference_leave_one_municipality_out_rank_sensitivity",
        "housing_upstream_commit": SOURCE_COMMIT,
        "housing_panel_sha256": SUBSET_SHA256,
        "commuting_2021_source_type": commute["source_type"],
        "commuting_source_workbook_sha256": commute["source_workbook_sha256"],
        "commuting_derived_pair_csv_sha256": commute["derived_csv_sha256"],
        "caop2025_source_response_sha256": ca["source_response_sha256"],
        "caop2025_derived_csv_sha256": ca["derived_csv_sha256"],
        "centroid_crs_epsg": 3763,
        "commuting_network_year": 2021,
        "rent_window_starts": list(WINDOW_STARTS),
        "rent_window_end": END_YEAR,
        "exposure_modes": list(MODES),
        "outcome": OUTCOMES[0],
        "shared_control_set": "both_distances",
        "deletion_rank_method": "independently_recompute_average_ranks_and_residuals_on_each_14_unit_sample",
        "matched_reference_comparators": 15,
        "comparators_after_deletion": 14,
        "matched_reference_mode_year_cases": len(pairs),
        "municipality_deletion_cases": len(deletions),
        "window_mode_summary_rows": len(summaries),
        "descriptive_negative_delta_threshold": MATERIAL_RHO_DIFFERENCE,
        "primary_mode_summary": [
            r for r in summaries if r["exposure_mode"] == PRIMARY_MODE
        ],
        "limitations": [
            "Leave-one-out ranges are deterministic influence diagnostics, not jackknife confidence intervals",
            "The 15 original comparator municipalities are spatially dependent and differ across alternatives",
            "Deleting a comparator changes all ranks and sometimes leaves little independent exposure rank variation",
            "Four rent windows share the 2025 outcome endpoint and do not provide independent replications",
            "The fixed 2021 work/study matrix is undirected and does not measure residential migration",
            "The 2021 commuting matrix is a secondary Census-derived extract not directly source-reconciled",
            "CAOP2025 administrative centroid distance is not travel time or accessible housing supply",
            "New-contract rent medians are not fixed-dwelling or household rent trajectories",
            "No significance tests, directed displacement, causal contagion or policy effect is identified",
        ],
    }
    (result_dir / RESULTS[3]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline reproducibility entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Regenerated {report['municipality_deletion_cases']} matched deletion cases")


if __name__ == "__main__":
    main()
