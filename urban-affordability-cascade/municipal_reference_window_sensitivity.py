"""Temporal negative controls for changing the municipal network reference.

The 2021 symmetric work/study network is held fixed while the rental
median growth baseline moves from 2020 to 2023. Windows overlap and
share the 2025 endpoint. These rank comparisons are not independent,
not observed residential moves, and not causal spillover estimators.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import MUNICIPALITIES, ROOT, SOURCE_COMMIT, SUBSET_SHA256, load_panel, require_float
from extract_commuting import PROVENANCE as COMMUTING_PROVENANCE
from municipal_reference_placebos import MODE_COLUMNS, REFERENCE, pair_table
from porto_network_gradient import END_YEAR, MODES, WINDOW_STARTS, spearman_rho


OUTCOMES = ("own_new_rent_growth_window_pct", "reference_discount_compression_window_pp")
MATERIAL_RHO_DIFFERENCE = .05  # Descriptive comparison margin, not a statistical significance threshold.
FIGURES = ("fig36_municipal_reference_rank_window_sensitivity.png",
           "fig37_matched_porto_reference_window_differences.png")
RESULTS = ("municipal_reference_window_pair_years.csv",
           "municipal_reference_window_rank_grid.csv",
           "municipal_reference_window_matched_support.csv",
           "municipal_reference_window_primary_summary.csv",
           "municipal_reference_window_provenance.json")


def _require_window_grid(rows: Sequence[dict[str, str | int]]) -> None:
    """Enforce 17 references × 16 comparators × four baseline years."""
    expected = {(reference, comparator, year)
                for reference in MUNICIPALITIES for comparator in MUNICIPALITIES
                for year in WINDOW_STARTS if comparator != reference}
    observed = {(str(r["reference_municipality"]), str(r["comparator_municipality"]),
                 int(r["window_start_year"])) for r in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("Missing or duplicated municipal reference × comparator × year observations")


def window_pairs() -> list[dict[str, str | int]]:
    """Build all overlap-aware rent windows from the pinned panel."""
    baseline_pairs = pair_table()  # Includes verified work/study weights and 2021 denominator.
    panel = load_panel()
    rents: dict[tuple[str, int], float] = {
        (municipality, year): require_float(panel[municipality, year].rent_m2,
                                            f"{municipality} {year} new-contract rent")
        for municipality in MUNICIPALITIES for year in (*WINDOW_STARTS, END_YEAR)
    }
    if any(value <= 0 for value in rents.values()):
        raise ValueError("A municipal new-lease rent median must be positive")
    rows: list[dict[str, str | int]] = []
    for source in baseline_pairs:
        reference = str(source["reference_municipality"])
        comparator = str(source["comparator_municipality"])
        for start in WINDOW_STARTS:
            a, b = rents[reference, start], rents[comparator, start]
            a_end, b_end = rents[reference, END_YEAR], rents[comparator, END_YEAR]
            rows.append({
                "reference_municipality": reference,
                "comparator_municipality": comparator,
                "window_start_year": start,
                "window_end_year": END_YEAR,
                "reference_start_new_rent_eur_m2_month": f"{a:.6f}",
                "comparator_start_new_rent_eur_m2_month": f"{b:.6f}",
                "undirected_work_study_pair_interactions_2021": source["undirected_work_study_pair_interactions_2021"],
                "interactions_per_1000_comparator_residents_2021": source["interactions_per_1000_comparator_residents_2021"],
                "pair_share_of_comparator_amp_interactions_2021": source["pair_share_of_comparator_amp_interactions_2021"],
                "own_new_rent_growth_window_pct": f"{100 * (b_end / b - 1):.9f}",
                "reference_discount_compression_window_pp": (
                    f"{100 * (b_end / a_end - b / a):.9f}"
                ),
            })
    _require_window_grid(rows)
    return rows


def _window_vectors(
    rows: Sequence[dict[str, str | int]], mode: str, outcome: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Require finite and nonconstant inputs for the declared window fields."""
    if mode not in MODES or mode not in MODE_COLUMNS or outcome not in OUTCOMES:
        raise ValueError("Unknown temporal reference exposure mode or outcome")
    x = np.array([float(r[MODE_COLUMNS[mode]]) for r in rows], dtype=float)
    y = np.array([float(r[outcome]) for r in rows], dtype=float)
    if len(rows) < 5 or x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("At least five paired finite temporal reference values required")
    if np.any(x <= 0):
        raise ValueError("Interaction exposure must be positive")
    return x, y


def window_ranks(rows: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """All 4×17×3×2 reference/window/mode/outcome coefficients."""
    _require_window_grid(rows)
    grouped: dict[tuple[int, str], list[dict[str, str | int]]] = {}
    for row in rows:
        key = (int(row["window_start_year"]), str(row["reference_municipality"]))
        grouped.setdefault(key, []).append(row)
    table: list[dict[str, str | int]] = []
    for year in WINDOW_STARTS:
        for reference in sorted(MUNICIPALITIES):
            selected = grouped[year, reference]
            if len(selected) != 16:
                raise ValueError("Expected 16 comparator municipalities for each window/reference")
            for mode in MODES:
                for outcome in OUTCOMES:
                    x, y = _window_vectors(selected, mode, outcome)
                    table.append({
                        "window_start_year": year,
                        "window_end_year": END_YEAR,
                        "network_year": 2021,
                        "reference_municipality": reference,
                        "exposure_mode": mode,
                        "outcome": outcome,
                        "comparator_municipalities": len(selected),
                        "spearman_rank_rho": f"{spearman_rho(x, y):.9f}",
                        "absolute_rho_rank_among_17": 0,
                    })
    for year in WINDOW_STARTS:
        for mode in MODES:
            for outcome in OUTCOMES:
                selected = [r for r in table if r["window_start_year"] == year
                            and r["exposure_mode"] == mode and r["outcome"] == outcome]
                magnitudes = [abs(float(r["spearman_rank_rho"])) for r in selected]
                for r, magnitude in zip(selected, magnitudes):
                    r["absolute_rho_rank_among_17"] = 1 + sum(
                        other > magnitude + 1e-10 for other in magnitudes
                    )
    if len(table) != len(WINDOW_STARTS) * 17 * len(MODES) * len(OUTCOMES):
        raise ValueError("Incomplete temporal reference rank grid")
    return table


def matched_window_ranks(rows: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Use a shared set of 15 outcomes for Porto and each other reference.

    Only the reference-invariant *own rent growth* outcome is used.
    Relative-discount compression depends on the reference rent, so
    its changes would combine a denominator shift with network choice.
    """
    _require_window_grid(rows)
    lookup = {(int(r["window_start_year"]), str(r["reference_municipality"]),
               str(r["comparator_municipality"])): r for r in rows}
    table: list[dict[str, str | int]] = []
    for year in WINDOW_STARTS:
        for alternative in sorted(set(MUNICIPALITIES) - {REFERENCE}):
            names = sorted(set(MUNICIPALITIES) - {REFERENCE, alternative})
            porto = [lookup[year, REFERENCE, name] for name in names]
            other = [lookup[year, alternative, name] for name in names]
            for mode in MODES:
                xp, yp = _window_vectors(porto, mode, OUTCOMES[0])
                xa, ya = _window_vectors(other, mode, OUTCOMES[0])
                if not np.array_equal(yp, ya):
                    raise ValueError("Own-rent-growth outcome changed between matched reference nodes")
                rho_p, rho_a = spearman_rho(xp, yp), spearman_rho(xa, ya)
                table.append({
                    "window_start_year": year,
                    "window_end_year": END_YEAR,
                    "alternative_reference": alternative,
                    "baseline_reference": REFERENCE,
                    "exposure_mode": mode,
                    "outcome": OUTCOMES[0],
                    "shared_comparators": len(names),
                    "shared_municipality_names": "|".join(names),
                    "porto_rho_on_same_15": f"{rho_p:.9f}",
                    "alternative_rho_on_same_15": f"{rho_a:.9f}",
                    "alternative_minus_porto_rho": f"{rho_a - rho_p:.9f}",
                })
    if len(table) != len(WINDOW_STARTS) * 16 * len(MODES):
        raise ValueError("Incomplete matched 15-comparator temporal rank grid")
    return table


def primary_summary(
    ranks: Sequence[dict[str, str | int]],
    matched: Sequence[dict[str, str | int]],
) -> list[dict[str, str | int]]:
    """Report Porto's actual sensitivity without selecting a preferred year."""
    rank_lookup = {(int(r["window_start_year"]), str(r["reference_municipality"]),
                    str(r["exposure_mode"]), str(r["outcome"])): r for r in ranks}
    matched_lookup = {(int(r["window_start_year"]), str(r["alternative_reference"]),
                       str(r["exposure_mode"])): r for r in matched}
    expected_ranks = {(year, ref, mode, outcome) for year in WINDOW_STARTS for ref in MUNICIPALITIES
                      for mode in MODES for outcome in OUTCOMES}
    expected_matched = {(year, ref, mode) for year in WINDOW_STARTS
                        for ref in MUNICIPALITIES - {REFERENCE} for mode in MODES}
    if len(ranks) != len(expected_ranks) or set(rank_lookup) != expected_ranks:
        raise ValueError("Missing or duplicate temporal reference ranks")
    if len(matched) != len(expected_matched) or set(matched_lookup) != expected_matched:
        raise ValueError("Missing or duplicate matched reference contrasts")
    summary: list[dict[str, str | int]] = []
    for year in WINDOW_STARTS:
        for outcome in OUTCOMES:
            all_ranks = [rank_lookup[year, ref, "per_1000_residents", outcome] for ref in sorted(MUNICIPALITIES)]
            porto = rank_lookup[year, REFERENCE, "per_1000_residents", outcome]
            lowest = min(all_ranks, key=lambda r: float(r["spearman_rank_rho"]))
            stronger = []
            materially_stronger = []
            if outcome == OUTCOMES[0]:
                stronger = sorted(ref for ref in MUNICIPALITIES - {REFERENCE}
                                  if float(matched_lookup[year, ref, "per_1000_residents"]["alternative_minus_porto_rho"]) < 0)
                materially_stronger = sorted(ref for ref in MUNICIPALITIES - {REFERENCE}
                                             if float(matched_lookup[year, ref, "per_1000_residents"]["alternative_minus_porto_rho"])
                                             <= -MATERIAL_RHO_DIFFERENCE)
            summary.append({
                "window_start_year": year,
                "window_end_year": END_YEAR,
                "outcome": outcome,
                "exposure_mode": "per_1000_residents",
                "porto_rho": porto["spearman_rank_rho"],
                "porto_absolute_rank_among_17": porto["absolute_rho_rank_among_17"],
                "negative_reference_coefficients_out_of_17": sum(float(r["spearman_rank_rho"]) < 0 for r in all_ranks),
                "most_negative_reference": lowest["reference_municipality"],
                "matched_alternatives_more_negative_than_porto_count": len(stronger) if outcome == OUTCOMES[0] else "",
                "matched_alternatives_more_negative_names": "|".join(stronger) if outcome == OUTCOMES[0] else "",
                "matched_alternatives_at_least_0_05_more_negative_count": (
                    len(materially_stronger) if outcome == OUTCOMES[0] else ""
                ),
                "matched_alternatives_at_least_0_05_more_negative_names": (
                    "|".join(materially_stronger) if outcome == OUTCOMES[0] else ""
                ),
            })
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Cannot save empty temporal reference table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_temporal_references(ranks: Sequence[dict[str, str | int]], path: Path) -> None:
    """All 17×4 primary per-resident outcomes, not confidence bands."""
    lookup = {(int(r["window_start_year"]), str(r["reference_municipality"]), str(r["outcome"])): r
              for r in ranks if r["exposure_mode"] == "per_1000_residents"}
    references = sorted(MUNICIPALITIES, key=lambda ref: float(
        lookup[2020, ref, OUTCOMES[0]]["spearman_rank_rho"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, outcome, title in zip(axes, OUTCOMES, ("Own nominal new-rent growth", "Reference-relative discount compression")):
        matrix = np.array([[float(lookup[year, ref, outcome]["spearman_rank_rho"])
                            for year in WINDOW_STARTS] for ref in references])
        heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-.8, vmax=.8, aspect="auto")
        ax.set_xticks(range(len(WINDOW_STARTS)), [f"{year}–25" for year in WINDOW_STARTS])
        ax.set_yticks(range(len(references)), references, fontsize=8.5)
        ax.set_title(title, fontsize=11.5)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8.2,
                        color="white" if abs(matrix[i, j]) > .55 else "#263844")
        porto_row = references.index(REFERENCE)
        ax.add_patch(plt.Rectangle((-.5, porto_row - .5), len(WINDOW_STARTS), 1,
                                   fill=False, linewidth=1.9, edgecolor="#0d655c"))
    # Use a dedicated colour-bar axis; the default automatic placement can
    # obscure the right-hand panel's final-year annotations.
    color_axis = fig.add_axes([.895, .19, .018, .58])
    fig.colorbar(heat, cax=color_axis, label="Spearman ρ")
    fig.suptitle("Reference-node rank patterns depend on the rent-growth window", fontsize=13.5, y=.975)
    fig.text(.02, .017, "2021 undirected work/study links, per 1,000 comparator residents; Porto row outlined. Nested windows share the 2025 endpoint.",
             fontsize=8.7, color="#586a74")
    fig.subplots_adjust(left=.18, right=.875, bottom=.12, top=.87, wspace=.10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_matched_windows(matched: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show all 16 alternatives on matched 15-place own-growth outcomes."""
    subset = [r for r in matched if r["exposure_mode"] == "per_1000_residents"]
    lookup = {(int(r["window_start_year"]), str(r["alternative_reference"])): r for r in subset}
    alternatives = sorted(MUNICIPALITIES - {REFERENCE}, key=lambda ref: float(
        lookup[2020, ref]["alternative_minus_porto_rho"]))
    matrix = np.array([[float(lookup[year, ref]["alternative_minus_porto_rho"])
                        for year in WINDOW_STARTS] for ref in alternatives])
    if matrix.shape != (16, len(WINDOW_STARTS)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Incomplete 15-comparator time-window contrast matrix")
    limit = max(.8, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(10.1, 8.4))
    heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(WINDOW_STARTS)), [f"{year}–2025" for year in WINDOW_STARTS])
    ax.set_yticks(range(len(alternatives)), alternatives, fontsize=9)
    ax.set_title("Matched alternatives minus Porto: four overlapping rent windows", fontsize=12.4, pad=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(matrix[i, j]) > limit * .66 else "#223744", weight="bold")
    fig.colorbar(heat, ax=ax, fraction=.035, pad=.025, label="Alternative minus Porto Spearman ρ")
    fig.text(.02, .024, "Same 15 comparator rents within each pair; negative means alternative is more negative. Neither causal nor independent windows.",
             fontsize=8.3, color="#586a74")
    fig.subplots_adjust(left=.24, right=.9, bottom=.145, top=.89)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate all time-window diagnostics from fixed offline inputs."""
    pairs = window_pairs()
    ranks = window_ranks(pairs)
    matched = matched_window_ranks(pairs)
    summary = primary_summary(ranks, matched)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_temporal_references(ranks, figure_dir / FIGURES[0])
    plot_matched_windows(matched, figure_dir / FIGURES[1])
    for name, rows in zip(RESULTS[:4], (pairs, ranks, matched, summary)):
        _write_csv(result_dir / name, rows)
    source = json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "evidence_class": "fixed_2021_undirected_commuting_network_descriptive_overlapping_rent_window_negative_controls",
        "housing_upstream_commit": SOURCE_COMMIT,
        "housing_panel_sha256": SUBSET_SHA256,
        "commuting_source_status": source["source_type"],
        "commuting_source_workbook_sha256": source["source_workbook_sha256"],
        "commuting_derived_pair_csv_sha256": source["derived_csv_sha256"],
        "commuting_reference_year": 2021,
        "rent_window_starts": list(WINDOW_STARTS),
        "common_rent_window_end": END_YEAR,
        "exposure_modes": list(MODES),
        "outcomes": list(OUTCOMES),
        "pair_window_rows": len(pairs),
        "reference_rank_rows": len(ranks),
        "matched_support_rows": len(matched),
        "primary_summary_rows": len(summary),
        "material_matched_rho_difference_threshold": MATERIAL_RHO_DIFFERENCE,
        "porto_primary_window_summary": [r for r in summary],
        "limits": [
            "2021 work/study interaction pairs are undirected and cannot identify residential migration or forced moves",
            "Third-party Census-derived commuting source has not been independently reconciled with direct official INE API counts",
            "Network reference nodes are not residential migration origins and rotating them is not a randomized placebo",
            "2020 and 2021 baseline rent windows are not strictly subsequent to the 2021 commuting snapshot",
            "2022 and 2023 baseline windows follow the 2021 snapshot but do not make the network exogenous",
            "2020/2021/2022/2023 to 2025 rent growth windows overlap and share the 2025 endpoint",
            "Every comparison uses cross-sectional new-lease medians, not repeated equivalent dwellings or observed household rents",
            "Reference-relative discount compression changes with the reference rent denominator, unlike own rent growth",
            "Spearman coefficients and matched 15-municipality contrasts are dependent descriptive diagnostics, not p-values",
            "A difference of 0.05 in rho is a declared materiality screen, not a confidence or significance threshold",
            "No migration, rent-price contagion, causal treatment effect, welfare difference or supply elasticity is identified",
        ],
    }
    (result_dir / RESULTS[4]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Regenerated {report['reference_rank_rows']} temporal rank comparisons")


if __name__ == "__main__":
    main()
