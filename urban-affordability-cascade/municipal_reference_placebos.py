"""Reference-node falsification checks for Porto-centred rent-network claims.

Rotating a reference municipality over an undirected 2021 work/study
network is a *descriptive negative-control design*. It is neither a
residential-migration origin, an exogenous treatment, a random placebo
assignment, a time-preceding exposure nor an inferential p-value.
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

from analysis import MUNICIPALITIES, ROOT, SOURCE_COMMIT, SUBSET_SHA256, load_panel, require_float
from commuting_network import load_weights
from extract_commuting import PROVENANCE as COMMUTING_PROVENANCE
from porto_network_gradient import MODES, spearman_rho


BASE_YEAR = 2020
END_YEAR = 2025
COMMUTING_YEAR = 2021
REFERENCE = "Porto"
OUTCOMES = ("own_new_rent_growth_2020_2025_pct", "reference_discount_compression_2020_2025_pp")
MODE_COLUMNS = {
    "raw_interactions": "undirected_work_study_pair_interactions_2021",
    "per_1000_residents": "interactions_per_1000_comparator_residents_2021",
    "share_of_amp_interactions": "pair_share_of_comparator_amp_interactions_2021",
}
FIGURES = ("fig34_rent_growth_rank_by_commuting_reference.png",
           "fig35_matched_porto_alternative_reference_rank_differences.png")
RESULTS = ("municipal_reference_placebos_all_pairs.csv",
           "municipal_reference_placebos_rank_associations.csv",
           "municipal_reference_placebos_matched_support.csv",
           "municipal_reference_placebos_leave_one_out.csv",
           "municipal_reference_placebos_provenance.json")


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    """Reject non-real/bool/NaN values before computing rank comparisons."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"{label} must be strictly positive")
    return number


def pair_table() -> list[dict[str, str | int]]:
    """Create 17×16 distinct reference-comparator views of fixed records."""
    names, weights = load_weights()
    if len(names) != 17 or set(names) != set(MUNICIPALITIES) or weights.shape != (17, 17):
        raise ValueError("The 17 AMP municipality identifiers are required")
    panel = load_panel()
    rents = {(name, year): _finite_number(
        require_float(panel[name, year].rent_m2, f"{name} {year} new-contract median"),
        "new-contract rent", positive=True)
        for name in names for year in (BASE_YEAR, END_YEAR)}
    populations = {name: _finite_number(
        require_float(panel[name, COMMUTING_YEAR].population, f"{name} resident population"),
        "municipal resident population", positive=True) for name in names}
    strengths = weights.sum(axis=1)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or np.any(~np.isfinite(strengths)):
        raise ValueError("Invalid work/study interaction matrix")
    output: list[dict[str, str | int]] = []
    for i, reference in enumerate(names):
        for j, comparator in enumerate(names):
            if i == j:
                continue  # Source diagonal is zero: no within-municipality counterpart.
            pair = _finite_number(float(weights[i, j]), "undirected work/study pair", positive=True)
            strength = _finite_number(float(strengths[j]), "comparison municipality strength", positive=True)
            ref_2020, ref_2025 = rents[reference, BASE_YEAR], rents[reference, END_YEAR]
            alt_2020, alt_2025 = rents[comparator, BASE_YEAR], rents[comparator, END_YEAR]
            own_growth = 100 * (alt_2025 / alt_2020 - 1)
            # Reference-relative discount at year t = 100*(1-r_alt_t/r_ref_t).
            # Compression is 2020 discount less 2025 discount, in pp.
            compression = 100 * ((1 - alt_2020 / ref_2020) - (1 - alt_2025 / ref_2025))
            output.append({
                "reference_municipality": reference,
                "comparator_municipality": comparator,
                "reference_2020_new_rent_eur_m2_month": f"{ref_2020:.6f}",
                "reference_2025_new_rent_eur_m2_month": f"{ref_2025:.6f}",
                "comparator_2020_new_rent_eur_m2_month": f"{alt_2020:.6f}",
                "comparator_2025_new_rent_eur_m2_month": f"{alt_2025:.6f}",
                "comparator_residents_2021": f"{populations[comparator]:.6f}",
                "undirected_work_study_pair_interactions_2021": int(pair),
                "interactions_per_1000_comparator_residents_2021": f"{1000 * pair / populations[comparator]:.9f}",
                "pair_share_of_comparator_amp_interactions_2021": f"{pair / strength:.12f}",
                "own_new_rent_growth_2020_2025_pct": f"{own_growth:.9f}",
                "reference_discount_compression_2020_2025_pp": f"{compression:.9f}",
            })
    _validate_pair_grid(output)
    return output


def _validate_pair_grid(rows: Sequence[dict[str, str | int]]) -> None:
    """Reject any dropped, repeated or self-reference cell in the 272 grid."""
    expected = {(a, b) for a in MUNICIPALITIES for b in MUNICIPALITIES if a != b}
    observed = {(str(r["reference_municipality"]), str(r["comparator_municipality"])) for r in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("Expected exactly 17×16 distinct commuting reference pairs")


def _vectors(rows: Sequence[dict[str, str | int]], mode: str, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    if mode not in MODES or mode not in MODE_COLUMNS or outcome not in OUTCOMES:
        raise ValueError("Unknown prespecified mode or outcome")
    x = np.array([float(r[MODE_COLUMNS[mode]]) for r in rows], dtype=float)
    y = np.array([float(r[outcome]) for r in rows], dtype=float)
    if len(rows) < 5 or x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("At least five finite paired reference comparisons required")
    if np.any(x <= 0):
        raise ValueError("Non-positive interaction intensity")
    return x, y


def rank_associations(rows: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """All 17 reference nodes × three modes × two rent outcomes."""
    _validate_pair_grid(rows)
    by_reference = {name: [r for r in rows if r["reference_municipality"] == name]
                    for name in sorted(MUNICIPALITIES)}
    table: list[dict[str, str | int]] = []
    for reference, selected in by_reference.items():
        if len(selected) != 16:
            raise ValueError("Each reference node must have 16 comparators")
        for mode in MODES:
            for outcome in OUTCOMES:
                x, y = _vectors(selected, mode, outcome)
                table.append({
                    "reference_municipality": reference,
                    "exposure_mode": mode,
                    "outcome": outcome,
                    "comparator_municipalities": len(selected),
                    "spearman_rank_rho": f"{spearman_rho(x, y):.9f}",
                    "absolute_rho_rank_among_17_references": 0,
                })
    # A rank of 1 means the largest absolute descriptive coefficient in
    # that mode/outcome. The references are dependent: not a p-value.
    for mode in MODES:
        for outcome in OUTCOMES:
            subset = [r for r in table if r["exposure_mode"] == mode and r["outcome"] == outcome]
            magnitudes = [abs(float(r["spearman_rank_rho"])) for r in subset]
            for r, value in zip(subset, magnitudes):
                r["absolute_rho_rank_among_17_references"] = 1 + sum(other > value + 1e-10
                                                                         for other in magnitudes)
    if len(table) != 17 * len(MODES) * len(OUTCOMES):
        raise ValueError("Incomplete reference-node diagnostic grid")
    return table


def matched_support(rows: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Change the reference node holding the 15 comparator names fixed.

    The own-rent-growth outcome is identical between the paired
    references. The relative-discount outcome is *not*: its numerator
    changes with the rent of the named reference municipality.
    """
    _validate_pair_grid(rows)
    lookup = {(str(r["reference_municipality"]), str(r["comparator_municipality"])): r for r in rows}
    table: list[dict[str, str | int]] = []
    for alternative in sorted(set(MUNICIPALITIES) - {REFERENCE}):
        shared = sorted(set(MUNICIPALITIES) - {REFERENCE, alternative})
        porto_rows = [lookup[REFERENCE, name] for name in shared]
        alternative_rows = [lookup[alternative, name] for name in shared]
        for mode in MODES:
            for outcome in OUTCOMES:
                xp, yp = _vectors(porto_rows, mode, outcome)
                xa, ya = _vectors(alternative_rows, mode, outcome)
                if outcome == OUTCOMES[0] and not np.array_equal(yp, ya):
                    raise ValueError("Municipal own-growth outcomes must be reference-invariant")
                rho_porto, rho_other = spearman_rho(xp, yp), spearman_rho(xa, ya)
                table.append({
                    "alternative_reference": alternative,
                    "baseline_reference": REFERENCE,
                    "exposure_mode": mode,
                    "outcome": outcome,
                    "shared_comparators": len(shared),
                    "shared_municipality_names": "|".join(shared),
                    "outcome_reference_invariant": str(outcome == OUTCOMES[0]).lower(),
                    "porto_reference_rho_on_same_15": f"{rho_porto:.9f}",
                    "alternative_reference_rho_on_same_15": f"{rho_other:.9f}",
                    "alternative_minus_porto_rho": f"{rho_other - rho_porto:.9f}",
                })
    if len(table) != 16 * len(MODES) * len(OUTCOMES):
        raise ValueError("Incomplete 15-comparator matched-support contrast grid")
    return table


def leave_one_out(rows: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """All 17×16 single-municipality deletions for primary reference screen."""
    _validate_pair_grid(rows)
    table: list[dict[str, str | int]] = []
    for reference in sorted(MUNICIPALITIES):
        selected = sorted((r for r in rows if r["reference_municipality"] == reference),
                          key=lambda r: str(r["comparator_municipality"]))
        x, y = _vectors(selected, "per_1000_residents", OUTCOMES[0])
        full = spearman_rho(x, y)
        for i, omitted in enumerate(selected):
            dropped = spearman_rho(np.delete(x, i), np.delete(y, i))
            table.append({
                "reference_municipality": reference,
                "omitted_comparator": str(omitted["comparator_municipality"]),
                "remaining_comparators": len(selected) - 1,
                "exposure_mode": "per_1000_residents",
                "outcome": OUTCOMES[0],
                "full_16_rho": f"{full:.9f}",
                "leave_one_out_15_rho": f"{dropped:.9f}",
            })
    return table


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty reference diagnostic")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_reference_rhos(table: Sequence[dict[str, str | int]], path: Path) -> None:
    """Compare two outcomes for the primary per-resident exposure mode."""
    lookup = {(str(r["reference_municipality"]), str(r["exposure_mode"]), str(r["outcome"])): r
              for r in table}
    selected = sorted(MUNICIPALITIES, key=lambda name: float(
        lookup[name, "per_1000_residents", OUTCOMES[0]]["spearman_rank_rho"]))
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(10.3, 8.3))
    ax.axvline(0, color="#687b84", lw=1.2, ls="--")
    for offset, outcome, color, label in ((-.16, OUTCOMES[0], "#2b739f", "Own nominal rent growth"),
                                          (.16, OUTCOMES[1], "#bd7734", "Reference discount compression")):
        x = [float(lookup[name, "per_1000_residents", outcome]["spearman_rank_rho"]) for name in selected]
        ax.scatter(x, y + offset, s=53, alpha=.93, color=color, zorder=3, label=label)
    ix = selected.index(REFERENCE)
    ax.axhspan(ix - .47, ix + .47, color="#e2f1ee", alpha=.8, zorder=0)
    ax.set_yticks(y, selected, fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ of 2021 link intensity vs 2020–2025 rent outcome")
    ax.set_title("Change the commuting reference: a descriptive falsification check", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .025, "Reference nodes are not migration origins; each reference excludes itself (16 comparators). Porto row shaded.",
             fontsize=9, color="#586a74")
    fig.subplots_adjust(left=.21, right=.96, bottom=.12, top=.90)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_matched_contrasts(table: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show matched-15 differences for the reference-invariant growth outcome."""
    subset = [r for r in table if r["outcome"] == OUTCOMES[0]]
    lookup = {(str(r["alternative_reference"]), str(r["exposure_mode"])): r for r in subset}
    alternatives = sorted(set(MUNICIPALITIES) - {REFERENCE}, key=lambda n: float(
        lookup[n, "per_1000_residents"]["alternative_minus_porto_rho"]))
    modes = ("raw_interactions", "per_1000_residents", "share_of_amp_interactions")
    matrix = np.array([[float(lookup[n, mode]["alternative_minus_porto_rho"]) for mode in modes]
                       for n in alternatives])
    if matrix.shape != (16, len(modes)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Incomplete matched-support reference contrast matrix")
    fig, ax = plt.subplots(figsize=(10.3, 8.4))
    limit = max(.8, float(np.max(np.abs(matrix))))
    heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(alternatives)), alternatives, fontsize=9)
    ax.set_xticks(range(len(modes)), ["Raw interactions", "Per 1,000 residents", "Share of comparator links"],
                  fontsize=9)
    ax.set_title("Alternative minus Porto rank coefficient, on the same 15 places", fontsize=12.5, pad=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    weight="bold", color="white" if abs(matrix[i, j]) > limit * .66 else "#223744")
    fig.colorbar(heat, ax=ax, fraction=.035, pad=.03, label="Difference in Spearman ρ")
    fig.text(.02, .025, "Own-rent growth outcome is fixed. Negative cells: other reference is more negative than Porto; not an effect.",
             fontsize=8.6, color="#586a74")
    fig.subplots_adjust(left=.24, right=.90, bottom=.15, top=.87)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate reference screens, matched support, deletions and ledger."""
    pairs = pair_table()
    ranks = rank_associations(pairs)
    matched = matched_support(pairs)
    deletions = leave_one_out(pairs)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_reference_rhos(ranks, figure_dir / FIGURES[0])
    plot_matched_contrasts(matched, figure_dir / FIGURES[1])
    for filename, rows in zip(RESULTS[:4], (pairs, ranks, matched, deletions)):
        _write_csv(result_dir / filename, rows)
    source = json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8"))
    primary = [r for r in ranks if r["exposure_mode"] == "per_1000_residents"]
    porto = {r["outcome"]: {"rho": r["spearman_rank_rho"],
                           "absolute_rank_among_17": r["absolute_rho_rank_among_17_references"]}
             for r in primary if r["reference_municipality"] == REFERENCE}
    matched_primary = [r for r in matched if r["exposure_mode"] == "per_1000_residents"
                       and r["outcome"] == OUTCOMES[0]]
    more_negative = sorted(str(r["alternative_reference"]) for r in matched_primary
                           if float(r["alternative_minus_porto_rho"]) < 0)
    report: dict[str, object] = {
        "evidence_class": "descriptive_reference_node_negative_control_undirected_2021_work_study_and_municipal_new_lease_rents",
        "housing_upstream_commit": SOURCE_COMMIT,
        "housing_panel_sha256": SUBSET_SHA256,
        "commuting_source_status": source["source_type"],
        "commuting_source_workbook_sha256": source["source_workbook_sha256"],
        "commuting_derived_pairs_sha256": source["derived_csv_sha256"],
        "commuting_reference_year": COMMUTING_YEAR,
        "rent_comparison_years": [BASE_YEAR, END_YEAR],
        "outcomes": list(OUTCOMES),
        "exposure_modes": list(MODES),
        "reference_nodes": len(MUNICIPALITIES),
        "comparators_per_reference": len(MUNICIPALITIES) - 1,
        "reference_comparator_rows": len(pairs),
        "full_reference_rank_rows": len(ranks),
        "matched_porto_alternative_contrasts": len(matched),
        "primary_leave_one_out_rows": len(deletions),
        "porto_primary_reference": porto,
        "alternative_references_more_negative_than_porto_on_same_15_own_growth": more_negative,
        "limits": [
            "2021 Census-derived undirected work/study pair counts are not directional moves or residential displacement",
            "Third-party commuting workbook is attributed to INE but not independently reconciled with a fresh official API response",
            "2021 commuting exposure occurs after the 2020 rent baseline and is not an exogenous pretreatment measure",
            "Changing a network reference is a descriptive negative control, not a randomized placebo or inferential p-value",
            "Each full reference excludes itself; matched comparisons use identical sets of 15 other municipalities",
            "Own municipal rent growth is reference-invariant; reference-relative rent discount compression changes with the reference rent denominator",
            "Rent medians describe changing sets of newly signed contracts and are not quality-adjusted unit-level prices",
            "Municipalities share national and metropolitan shocks and are not independent draws or exchangeable labels",
            "Differences in correlation signs or magnitudes do not identify spillover transmission, migration flows or timing",
            "Rank correlations, counts and leave-one-out variation cannot establish a causal relationship or welfare consequence",
        ],
    }
    (result_dir / RESULTS[4]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Reproduced {report['full_reference_rank_rows']} reference-node rent-network comparisons")


if __name__ == "__main__":
    main()
