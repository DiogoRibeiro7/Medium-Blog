"""Compare matched work/study network references under identical geography controls.

Each of the 16 alternatives to Porto is compared on exactly the same 15
municipalities and the same own-new-rent-growth outcomes. Unlike the preceding
audit, both exposure vectors are residualised against the *same* ranked
CAOP2025 distance covariates: distance to Porto, distance to the alternative,
or both. This is a descriptive falsification check, not causal identification.
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
from municipal_reference_geography_controls import (
    _distance_index,
    distance_pairs,
    matched_reference_controls,
)
from municipal_reference_placebos import REFERENCE
from municipal_reference_window_sensitivity import (
    MATERIAL_RHO_DIFFERENCE,
    OUTCOMES,
    _require_window_grid,
    _window_vectors,
    window_pairs,
)
from porto_network_gradient import (
    END_YEAR,
    MODES,
    WINDOW_STARTS,
    average_ranks,
    spearman_rho,
)


CONTROLS = ("porto_distance", "alternative_distance", "both_distances")
PRIMARY_MODE = "per_1000_residents"
FIGURES = (
    "fig40_matched_shared_geography_rank_contrasts.png",
    "fig41_shared_geography_reference_robustness.png",
)
RESULTS = (
    "municipal_reference_common_geography_matched.csv",
    "municipal_reference_common_geography_summary.csv",
    "municipal_reference_common_geography_provenance.json",
)
Row = dict[str, str | int]


def rank_residual_correlation(
    exposure: np.ndarray, outcome: np.ndarray, controls: np.ndarray,
) -> tuple[float, float, float]:
    """Correlate residual *average ranks* after one/two shared controls.

    Return (partial rank correlation, residual exposure rank variance share,
    residual outcome rank variance share). An intercept and all control rank
    columns are included in both projections. Exact rank deficiency or near-
    complete residual removal is rejected instead of producing a coefficient.
    These residual variance shares are diagnostics, not variance explained by
    a causal model. This function does not estimate uncertainty or p-values.
    """
    x, y, z = (np.asarray(a, dtype=float) for a in (exposure, outcome, controls))
    if (x.ndim != 1 or y.ndim != 1 or z.ndim != 2 or x.shape != y.shape
            or z.shape[0] != x.size or z.shape[1] not in (1, 2)
            or x.size < z.shape[1] + 5):
        raise ValueError("Expect matched ranks with one or two controls and at least k+5 rows")
    if not all(np.all(np.isfinite(v)) for v in (x, y, z)):
        raise ValueError("Matched ranks and geographic controls must be finite")

    ranked_x, ranked_y = average_ranks(x), average_ranks(y)
    design = np.column_stack([np.ones(x.size), *(
        average_ranks(z[:, i]) for i in range(z.shape[1])
    )])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Rank-deficient shared geographic controls")

    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    variation_x = float(np.dot(residual_x, residual_x))
    variation_y = float(np.dot(residual_y, residual_y))
    total_x = float(np.dot(ranked_x - ranked_x.mean(), ranked_x - ranked_x.mean()))
    total_y = float(np.dot(ranked_y - ranked_y.mean(), ranked_y - ranked_y.mean()))
    share_x, share_y = variation_x / total_x, variation_y / total_y
    if not all(math.isfinite(v) and v >= 1e-8 for v in (share_x, share_y)):
        raise ValueError("Geography removes almost all exposure or outcome rank variance")
    rho = float(np.clip(np.dot(residual_x, residual_y) /
                        math.sqrt(variation_x * variation_y), -1.0, 1.0))
    return rho, share_x, share_y


def matched_common_controls(
    windows: Sequence[Row], distances: Sequence[Row],
) -> list[Row]:
    """Run the full 4 years x 16 alternatives x 3 modes x 3 common controls.

    Check every unadjusted and earlier *different-control* result against the
    previous module, retaining them only as benchmarks for specification
    sensitivity. No year, alternative, or exposure mode is selected by fit.
    """
    _require_window_grid(windows)
    index = _distance_index(distances)
    prior = {
        (int(r["window_start_year"]), str(r["alternative_reference"]),
         str(r["exposure_mode"])): r
        for r in matched_reference_controls(windows, distances)
    }
    by_key = {
        (int(r["window_start_year"]), str(r["reference_municipality"]),
         str(r["comparator_municipality"])): r for r in windows
    }
    output: list[Row] = []
    for year in WINDOW_STARTS:
        for alternative in sorted(MUNICIPALITIES - {REFERENCE}):
            names = sorted(MUNICIPALITIES - {REFERENCE, alternative})
            if len(names) != 15:
                raise ValueError("Matched design requires exactly 15 municipalities")
            porto = [by_key[year, REFERENCE, name] for name in names]
            other = [by_key[year, alternative, name] for name in names]
            distance_p = np.array([index[REFERENCE, name] for name in names])
            distance_a = np.array([index[alternative, name] for name in names])
            distance_rank_rho = spearman_rho(distance_p, distance_a)
            control_vectors = {
                "porto_distance": distance_p[:, None],
                "alternative_distance": distance_a[:, None],
                "both_distances": np.column_stack((distance_p, distance_a)),
            }
            for mode in MODES:
                exposure_p, outcome_p = _window_vectors(porto, mode, OUTCOMES[0])
                exposure_a, outcome_a = _window_vectors(other, mode, OUTCOMES[0])
                if not np.array_equal(outcome_p, outcome_a):
                    raise ValueError("The two network references must share identical own-rent outcomes")
                old = prior[year, alternative, mode]
                raw_p, raw_a = spearman_rho(exposure_p, outcome_p), spearman_rho(exposure_a, outcome_a)
                if (abs(raw_p - float(old["porto_raw_rho_on_same_15"])) > 5e-9
                        or abs(raw_a - float(old["alternative_raw_rho_on_same_15"])) > 5e-9):
                    raise ValueError("Unadjusted ranks disagree with prior matched comparison")
                for control_name in CONTROLS:
                    common = control_vectors[control_name]
                    p, p_remaining, y_remaining = rank_residual_correlation(
                        exposure_p, outcome_p, common)
                    a, a_remaining, y_other_remaining = rank_residual_correlation(
                        exposure_a, outcome_a, common)
                    if abs(y_remaining - y_other_remaining) > 1e-10:
                        raise ValueError("Common controls did not remove the same outcome rank component")
                    # A one-control, same-reference projection must reproduce
                    # the previously reported separate-control coefficient.
                    if control_name == CONTROLS[0] and abs(
                        p - float(old["porto_distance_conditional_rho_on_same_15"])
                    ) > 5e-8:
                        raise ValueError("Porto-only geometry fails previous distance audit")
                    if control_name == CONTROLS[1] and abs(
                        a - float(old["alternative_distance_conditional_rho_on_same_15"])
                    ) > 5e-8:
                        raise ValueError("Alternative-only geometry fails previous distance audit")
                    delta = a - p
                    prior_delta = float(old["conditional_alternative_minus_porto_rho"])
                    output.append({
                        "window_start_year": year,
                        "window_end_year": END_YEAR,
                        "alternative_reference": alternative,
                        "baseline_reference": REFERENCE,
                        "exposure_mode": mode,
                        "outcome": OUTCOMES[0],
                        "shared_control_set": control_name,
                        "shared_comparators": len(names),
                        "shared_municipality_names": "|".join(names),
                        "raw_porto_rho": f"{raw_p:.9f}",
                        "raw_alternative_rho": f"{raw_a:.9f}",
                        "raw_alternative_minus_porto_rho": f"{raw_a - raw_p:.9f}",
                        "prior_different_controls_alternative_minus_porto_rho": f"{prior_delta:.9f}",
                        "shared_porto_conditional_rho": f"{p:.9f}",
                        "shared_alternative_conditional_rho": f"{a:.9f}",
                        "shared_alternative_minus_porto_rho": f"{delta:.9f}",
                        "shared_minus_different_controls_delta": f"{delta - prior_delta:.9f}",
                        "porto_exposure_remaining_rank_variance_fraction": f"{p_remaining:.9f}",
                        "alternative_exposure_remaining_rank_variance_fraction": f"{a_remaining:.9f}",
                        "outcome_remaining_rank_variance_fraction": f"{y_remaining:.9f}",
                        "porto_to_alternative_centroid_distance_rank_rho": f"{distance_rank_rho:.9f}",
                    })
    expected = len(WINDOW_STARTS) * (len(MUNICIPALITIES) - 1) * len(MODES) * len(CONTROLS)
    keys = {(r["window_start_year"], r["alternative_reference"],
             r["exposure_mode"], r["shared_control_set"]) for r in output}
    if len(output) != expected or len(keys) != expected:
        raise ValueError("Incomplete or duplicated matched common-geography grid")
    return output


def summary_grid(matched: Sequence[Row]) -> list[Row]:
    """Summarise every window, mode, and shared-control set without selection."""
    expected = {(year, mode, control, alternative)
                for year in WINDOW_STARTS for mode in MODES for control in CONTROLS
                for alternative in MUNICIPALITIES - {REFERENCE}}
    keys = {(int(r["window_start_year"]), str(r["exposure_mode"]),
             str(r["shared_control_set"]), str(r["alternative_reference"]))
            for r in matched}
    if len(matched) != len(expected) or keys != expected:
        raise ValueError("All 576 matched common-control rows are required")
    result: list[Row] = []
    for year in WINDOW_STARTS:
        for mode in MODES:
            for control in CONTROLS:
                rows = sorted((r for r in matched if r["window_start_year"] == year
                               and r["exposure_mode"] == mode
                               and r["shared_control_set"] == control),
                              key=lambda r: str(r["alternative_reference"]))
                negative = [str(r["alternative_reference"]) for r in rows if
                            float(r["shared_alternative_minus_porto_rho"]) < 0]
                material = [str(r["alternative_reference"]) for r in rows if
                            float(r["shared_alternative_minus_porto_rho"]) <= -MATERIAL_RHO_DIFFERENCE]
                flips = [str(r["alternative_reference"]) for r in rows if
                         (float(r["shared_alternative_minus_porto_rho"]) < 0) !=
                         (float(r["prior_different_controls_alternative_minus_porto_rho"]) < 0)]
                deltas = np.array([float(r["shared_alternative_minus_porto_rho"]) for r in rows])
                result.append({
                    "window_start_year": year,
                    "window_end_year": END_YEAR,
                    "exposure_mode": mode,
                    "shared_control_set": control,
                    "alternatives": len(rows),
                    "alternatives_more_negative_than_porto": len(negative),
                    "more_negative_alternative_names": "|".join(negative),
                    "alternatives_at_least_0_05_more_negative": len(material),
                    "at_least_0_05_more_negative_names": "|".join(material),
                    "sign_changes_vs_prior_different_controls": len(flips),
                    "sign_change_alternative_names": "|".join(flips),
                    "median_alternative_minus_porto_rho": f"{float(np.median(deltas)):.9f}",
                    "minimum_alternative_minus_porto_rho": f"{float(deltas.min()):.9f}",
                    "maximum_alternative_minus_porto_rho": f"{float(deltas.max()):.9f}",
                    "minimum_exposure_remaining_rank_variance_fraction": f"{min(min(float(r[k]) for k in (
                        'porto_exposure_remaining_rank_variance_fraction',
                        'alternative_exposure_remaining_rank_variance_fraction')) for r in rows):.9f}",
                })
    if len(result) != len(WINDOW_STARTS) * len(MODES) * len(CONTROLS):
        raise ValueError("Incomplete summary of common-controls sensitivity")
    return result


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    """Write stable UTF-8 result files with every declared field present."""
    if not rows:
        raise ValueError("Cannot write an empty shared-geography table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_shared_deltas(matched: Sequence[Row], path: Path) -> None:
    """Show all alternatives for both-distance conditioning, not winners only."""
    selected = [r for r in matched if r["exposure_mode"] == PRIMARY_MODE
                and r["shared_control_set"] == "both_distances"]
    lookup = {(int(r["window_start_year"]), str(r["alternative_reference"])): r
              for r in selected}
    alternatives = sorted(MUNICIPALITIES - {REFERENCE})
    values = np.array([[float(lookup[year, name]["shared_alternative_minus_porto_rho"])
                        for year in WINDOW_STARTS] for name in alternatives])
    if values.shape != (16, 4):
        raise ValueError("Full 16-by-4 symmetric common-control matrix required")
    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    heat = ax.imshow(values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_xticks(range(4), [f"{year}–25" for year in WINDOW_STARTS])
    ax.set_yticks(range(16), alternatives, fontsize=9)
    ax.set_title("Alternative minus Porto, same 15 municipalities and two distances", fontsize=11.7)
    for row in range(16):
        for col in range(4):
            ax.text(col, row, f"{values[row, col]:+.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(values[row, col]) >= 0.7 else "#263f52")
    fig.colorbar(heat, ax=ax, fraction=0.033, pad=0.035,
                 label="Partial Spearman contrast (alternative − Porto)")
    fig.text(0.03, 0.025, "2021 work/study pairs per 1,000 residents; 2025 CAOP centroids; both exposures conditioned on both distances.",
             fontsize=8.3, color="#526779")
    fig.subplots_adjust(left=.24, right=.91, bottom=.12, top=.91)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_control_sensitivity(summary: Sequence[Row], path: Path) -> None:
    """Compare all three common-control choices to the earlier separate case."""
    chosen = {(int(r["window_start_year"]), str(r["shared_control_set"])): r for r in summary
              if r["exposure_mode"] == PRIMARY_MODE}
    specs = (
        ("porto_distance", "Shared Porto distance"),
        ("alternative_distance", "Shared alternative distance"),
        ("both_distances", "Shared pair of distances"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), sharey=True)
    palette = ("#2a7189", "#ca8549", "#6c5497")
    x = np.arange(4)
    for ax, key, label in zip(
        axes,
        ("alternatives_more_negative_than_porto", "alternatives_at_least_0_05_more_negative"),
        ("Any negative alternative-minus-Porto contrast", "Contrast ≤ −0.05 (descriptive margin)"),
    ):
        for index, (control, description) in enumerate(specs):
            values = [int(chosen[year, control][key]) for year in WINDOW_STARTS]
            ax.bar(x + (index - 1) * .24, values, width=.22,
                   color=palette[index], label=description)
        ax.set_xticks(x, [f"{year}–25" for year in WINDOW_STARTS])
        ax.set_ylim(0, 16.9)
        ax.set_yticks(range(0, 17, 2))
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
        ax.set_title(label, fontsize=11)
    axes[0].set_ylabel("Alternatives out of 16")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(.5, .955))
    fig.suptitle("Matched-reference conclusions depend on the common geography control",
                 fontsize=13.0, y=1.01)
    fig.text(.025, .018, "All controls applied identically to both network exposures; rank collinearity and overlapping rent windows remain.",
             fontsize=8.4, color="#526779")
    fig.subplots_adjust(left=.075, right=.98, bottom=.14, top=.79, wspace=.12)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Reproduce the exact common-geography audit with no external requests."""
    windows, distances = window_pairs(), distance_pairs()
    matched = matched_common_controls(windows, distances)
    summary = summary_grid(matched)
    figures, results = output / "figures", output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plot_shared_deltas(matched, figures / FIGURES[0])
    plot_control_sensitivity(summary, figures / FIGURES[1])
    _write_csv(results / RESULTS[0], matched)
    _write_csv(results / RESULTS[1], summary)
    ca = json.loads(CAOP_PROVENANCE.read_text(encoding="utf-8"))
    commute = json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "evidence_class": "noncausal_matched_comparator_shared_geography_rank_sensitivity",
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
        "shared_control_sets": list(CONTROLS),
        "matched_municipalities_per_reference_pair": 15,
        "descriptive_negative_delta_threshold": MATERIAL_RHO_DIFFERENCE,
        "matched_rows": len(matched),
        "summary_rows": len(summary),
        "primary_mode_summary": [r for r in summary if r["exposure_mode"] == PRIMARY_MODE],
        "limitations": [
            "Each pair uses the same 15 municipalities, own-rent outcomes and common control rank columns",
            "One-distance controls privilege either Porto or the alternative; the both-distances design is symmetric",
            "Two correlated distance rank controls can leave very little independent work/study variation",
            "Partial rank coefficients with only 15 observations have no defensible causal or population inference here",
            "Overlapping rent windows share the 2025 endpoint and are not independent validation periods",
            "The 2021 symmetric work/study matrix does not identify 2019-2025 residential moves or treatment",
            "The commuting matrix is attributed to INE via a third-party extract, not independently revalidated",
            "CAOP2025 polygon-centroid distance is not residence-weighted distance, public transport access or travel time",
            "New-contract rent medians do not follow fixed tenants, buildings or constant-quality dwellings",
            "The matched 15-person municipality support varies by alternative; contrasts are dependent",
            "No uncertainty interval, statistical significance, moving decisions or housing-price contagion is identified",
        ],
    }
    (results / RESULTS[2]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Regenerated {report['matched_rows']} matched shared-geography controls")


if __name__ == "__main__":
    main()
