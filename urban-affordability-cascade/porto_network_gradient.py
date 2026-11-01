"""Test descriptive Porto-link intensity versus rent-discount compression.

The 2021 pair counts are undirected work/study interactions from a frozen
third-party Census-derived workbook attributed to INE. They are not observed
residential origin-destination flows, nor a pre-2020 exposure instrument.
Random-label shuffles are an exploratory assignment benchmark, not a
sampling p-value or a causal hypothesis test.
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

from analysis import ROOT, SOURCE_COMMIT, SUBSET_SHA256, load_panel, require_float
from commuting_network import load_weights
from extract_commuting import PROVENANCE as COMMUTING_PROVENANCE
from extract_hicp import SOURCE_SHA256, load_hicp
from porto_rent_options import COMPARATORS, YEARS, annual_table


WINDOW_STARTS = (2020, 2021, 2022, 2023)
END_YEAR = 2025
MODES = ("raw_interactions", "per_1000_residents", "share_of_amp_interactions")
SEED = 20261101
PERMUTATIONS = 9999
FIGURES = (
    "fig24_porto_link_intensity_vs_rent_discount_compression.png",
    "fig25_porto_link_rank_sensitivity_windows.png",
)
RESULTS = (
    "porto_network_gradient_municipal_exposures.csv",
    "porto_network_gradient_rank_benchmarks.csv",
    "porto_network_gradient_leave_one_out.csv",
    "porto_network_gradient_provenance.json",
)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign deterministic average ranks to exact ties without SciPy."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.all(np.isfinite(x)):
        raise ValueError("At least four finite one-dimensional values are required")
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Constant input has undefined rank correlation")
    start = np.cumsum(counts) - counts + 1
    average = start + (counts - 1) / 2.0
    return average[inverse].astype(float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Correlate two finite, nonconstant arrays of equal length."""
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Rank vectors must have identical one-dimensional shapes")
    a, b = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Rank variance must be positive")
    return float(np.clip((a @ b) / denominator, -1.0, 1.0))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, including correct average-rank ties."""
    return _pearson(average_ranks(x), average_ranks(y))


def random_label_benchmark(
    x: np.ndarray, y: np.ndarray, *, seed: int = SEED, permutations: int = PERMUTATIONS,
) -> tuple[float, float]:
    """Return rho and two-sided shuffled-label tail fraction with plus-one.

    This is a *descriptive fixed-network assignment benchmark*. Municipal
    exchangeability is implausible. Do not call this an inferential p-value.
    """
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 99:
        raise ValueError("At least 99 integer label shuffles are required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    ranks_x, ranks_y = average_ranks(x), average_ranks(y)
    observed = _pearson(ranks_x, ranks_y)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        shuffled = _pearson(ranks_x, rng.permutation(ranks_y))
        hits += abs(shuffled) >= abs(observed) - 1e-12
    return observed, (hits + 1) / (permutations + 1)


def leave_one_out_rhos(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Report every one-municipality deletion, without selective trimming."""
    if x.ndim != 1 or y.shape != x.shape or len(x) < 5:
        raise ValueError("Need at least five paired municipalities")
    return [
        spearman_rho(np.delete(x, i), np.delete(y, i))
        for i in range(len(x))
    ]


def exposure_table() -> list[dict[str, str]]:
    """Join the pinned 2021 undirected links with published municipal rents."""
    names, weights = load_weights()
    if set(names) != set(COMPARATORS) | {"Porto"} or weights.shape != (17, 17):
        raise ValueError("Expected the 17-municipality fixed commuting pair matrix")
    observations = load_panel()
    annual = annual_table(observations, load_hicp())
    rents = {(str(r["alternative_municipality"]), int(r["year"])): r for r in annual}
    if len(rents) != len(COMPARATORS) * len(YEARS):
        raise ValueError("Missing Porto-relative annual rent comparisons")
    porto_index = names.index("Porto")
    rows: list[dict[str, str]] = []
    for municipality in COMPARATORS:
        index = names.index(municipality)
        pair = float(weights[porto_index, index])
        total = float(weights[index, :].sum())
        population = require_float(observations[municipality, 2021].population, "population")
        if not all(math.isfinite(z) and z > 0 for z in (pair, total, population)) or pair > total:
            raise ValueError("Non-positive or inconsistent population/commuting denominator")
        record: dict[str, str] = {
            "municipality": municipality,
            "porto_undirected_work_study_interactions_2021": str(int(pair)),
            "municipal_population_2021": f"{population:.6f}",
            "all_amp_undirected_interaction_strength_2021": f"{total:.6f}",
            "porto_link_interactions_per_1000_municipal_residents": f"{1000 * pair / population:.6f}",
            "porto_share_of_within_amp_undirected_interaction_strength": f"{pair / total:.9f}",
            "relative_rent_discount_2020_pct": str(rents[municipality, 2020]["alternative_relative_rent_discount_vs_porto_pct"]),
            "relative_rent_discount_2025_pct": str(rents[municipality, 2025]["alternative_relative_rent_discount_vs_porto_pct"]),
        }
        ending = float(record["relative_rent_discount_2025_pct"])
        for year in WINDOW_STARTS:
            start = float(rents[municipality, year]["alternative_relative_rent_discount_vs_porto_pct"])
            record[f"discount_compression_{year}_2025_percentage_points"] = f"{start - ending:.6f}"
        rows.append(record)
    return rows


def _mode_values(rows: Sequence[dict[str, str]], mode: str) -> np.ndarray:
    """Read the three predeclared fixed-network exposure conventions."""
    fields = {
        "raw_interactions": "porto_undirected_work_study_interactions_2021",
        "per_1000_residents": "porto_link_interactions_per_1000_municipal_residents",
        "share_of_amp_interactions": "porto_share_of_within_amp_undirected_interaction_strength",
    }
    if mode not in fields:
        raise ValueError("Unknown Porto-link exposure mode")
    values = np.array([float(r[fields[mode]]) for r in rows], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Missing or non-positive Porto-link exposures")
    return values


def _compression(rows: Sequence[dict[str, str]], baseline_year: int) -> np.ndarray:
    if baseline_year not in WINDOW_STARTS:
        raise ValueError("Unsupported rent-discount compression baseline")
    values = np.array([float(r[f"discount_compression_{baseline_year}_2025_percentage_points"]) for r in rows])
    if not np.all(np.isfinite(values)):
        raise ValueError("Missing rent-discount comparison")
    return values


def diagnostic_tables(
    exposures: Sequence[dict[str, str]], *, permutations: int = PERMUTATIONS, seed: int = SEED,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Compute 12 declared rank benchmarks and all 16 primary deletions."""
    if len(exposures) != 16 or {r["municipality"] for r in exposures} != set(COMPARATORS):
        raise ValueError("Exactly 16 unique comparator municipalities are required")
    stats: list[dict[str, str | int]] = []
    primary_deletions: list[dict[str, str | int]] = []
    for year in WINDOW_STARTS:
        y = _compression(exposures, year)
        for mode in MODES:
            x = _mode_values(exposures, mode)
            rho, fraction = random_label_benchmark(x, y, permutations=permutations, seed=seed)
            deletions = leave_one_out_rhos(x, y)
            stats.append({
                "window_start_year": year, "window_end_year": END_YEAR,
                "exposure_mode": mode, "municipalities": len(exposures),
                "spearman_rho": f"{rho:.6f}",
                "shuffled_label_two_sided_tail_fraction_not_p_value": f"{fraction:.6f}",
                "leave_one_out_min_rho": f"{min(deletions):.6f}",
                "leave_one_out_max_rho": f"{max(deletions):.6f}",
                "shuffles": permutations, "seed": seed,
            })
            if year == 2020 and mode == "per_1000_residents":
                for row, deletion in zip(exposures, deletions):
                    primary_deletions.append({
                        "removed_municipality": str(row["municipality"]),
                        "full_16_spearman_rho": f"{rho:.6f}",
                        "leave_one_out_15_spearman_rho": f"{deletion:.6f}",
                        "leave_one_out_minus_full_rho": f"{deletion - rho:.6f}",
                    })
    return stats, primary_deletions


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    """Write stable UTF-8 text with a homogeneous column order."""
    if not rows:
        raise ValueError("Empty Porto network-gradient table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_primary_scatter(
    exposures: Sequence[dict[str, str]], benchmarks: Sequence[dict[str, str | int]], path: Path,
) -> None:
    """Show the primary per-resident 2021 link against 2020–2025 compression."""
    x = _mode_values(exposures, "per_1000_residents")
    y = _compression(exposures, 2020)
    primary = next(r for r in benchmarks if r["window_start_year"] == 2020
                   and r["exposure_mode"] == "per_1000_residents")
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.axhline(0, color="#73828c", linestyle="--", linewidth=1.1)
    ax.scatter(x, y, color="#2f78a4", s=76, alpha=0.87, edgecolor="white", linewidths=0.6)
    highlighted = {"Vila Nova de Gaia", "Matosinhos", "Valongo", "Espinho", "Arouca"}
    for row, a, b in zip(exposures, x, y):
        if row["municipality"] in highlighted:
            ax.annotate(str(row["municipality"]), (a, b),
                        xytext=(4, 5), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("2021 Porto work/study pair interactions per 1,000 municipal residents")
    ax.set_ylabel("Porto-relative new-rent discount compression, 2020–2025 (pp)")
    ax.set_title("No positive rank relationship in 16 municipal comparisons", fontsize=13)
    ax.text(0.03, 0.96, f"Spearman ρ = {float(primary['spearman_rho']):+.2f}\n"
            f"Random-label tail fraction = {float(primary['shuffled_label_two_sided_tail_fraction_not_p_value']):.2f}",
            transform=ax.transAxes, va="top", fontsize=9.3, color="#304a60")
    ax.grid(alpha=0.17)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.02,
             "Pair counts are symmetric, not directed movers. Per-resident scaling is not a commuter share; "
             "the shuffle fraction is not a causal or sampling p-value.", color="#526371", fontsize=8.5)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.89)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_window_sensitivity(
    benchmarks: Sequence[dict[str, str | int]], path: Path,
) -> None:
    """Show how the 2025 endpoint association depends on baseline and unit."""
    matrix = np.array([
        [float(next(r["spearman_rho"] for r in benchmarks
                    if r["exposure_mode"] == mode and r["window_start_year"] == year))
         for year in WINDOW_STARTS]
        for mode in MODES
    ])
    labels = ["Raw Porto pair count", "Per 1,000 residents", "Porto share of AMP links"]
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(WINDOW_STARTS)), [f"{year}–2025" for year in WINDOW_STARTS])
    ax.set_yticks(range(len(MODES)), labels)
    ax.set_title("Spearman association is sensitive to the time window", fontsize=13, pad=13)
    for i in range(len(MODES)):
        for j in range(len(WINDOW_STARTS)):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center",
                    fontsize=12, color="#172f43", weight="bold")
    cb = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("Spearman rank correlation")
    fig.text(0.02, 0.026,
             "2021 undirected work/study interactions; exploratory 16-unit comparisons. "
             "Neither baseline selection nor exposure normalization identifies moving households.",
             color="#526371", fontsize=8.2)
    fig.subplots_adjust(left=0.28, right=0.91, bottom=0.17, top=0.85)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT, *, permutations: int = PERMUTATIONS) -> dict[str, object]:
    """Reproduce the fixed-source network/rent gradient association audit."""
    exposures = exposure_table()
    stats, influence = diagnostic_tables(exposures, permutations=permutations)
    figures, results = output / "figures", output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plot_primary_scatter(exposures, stats, figures / FIGURES[0])
    plot_window_sensitivity(stats, figures / FIGURES[1])
    for filename, rows in zip(RESULTS[:3], (exposures, stats, influence)):
        _write_csv(results / filename, rows)
    commuting_source = json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8"))
    raw = _mode_values(exposures, "raw_interactions")
    population = np.array([float(row["municipal_population_2021"]) for row in exposures])
    report: dict[str, object] = {
        "evidence_class": "exploratory_noncausal_fixed_commuting_pair_rent_discount_rank_association",
        "housing_upstream_commit": SOURCE_COMMIT,
        "housing_fixed_panel_sha256": SUBSET_SHA256,
        "commuting_derived_csv_sha256": commuting_source["derived_csv_sha256"],
        "commuting_source_workbook_sha256": commuting_source["source_workbook_sha256"],
        "commuting_interpretation": commuting_source["semantics"],
        "hicp_official_source_sha256": SOURCE_SHA256,
        "municipalities": len(exposures),
        "windows": [[year, END_YEAR] for year in WINDOW_STARTS],
        "exposure_modes": list(MODES),
        "shuffles": permutations,
        "seed": SEED,
        "raw_porto_pair_count_vs_2021_population_spearman_rho": f"{spearman_rho(raw, population):.6f}",
        "primary_per_1000_2020_2025": next(r for r in stats
            if r["exposure_mode"] == "per_1000_residents" and r["window_start_year"] == 2020),
        "limits": [
            "Original 2021 commuting interaction counts are symmetric and have no direction; they are not residential moves",
            "2021 work/study interactions are a third-party workbook extract not independently verified by a direct INE API response",
            "The network overlaps 2020-2025 housing outcomes and cannot act as an exogenous pre-treatment exposure",
            "Porto pair intensity per 1,000 residents is a mechanical normalization, not an individual commuter probability",
            "The within-AMP share denominator excludes all interactions outside the 17-municipality subset",
            "The outcome is a difference of independent new-contract rent median ratios, not a quality-adjusted rent change",
            "Random-label benchmark fractions are not inferential p-values: municipal exchangeability is implausible",
            "Three exposure conventions and four timing windows are exploratory, not a post-selection causal model",
            "Only 16 dependent municipality comparisons share the same Porto reference; no directional price transmission identified",
            "No household displacement, residential OD flow, real renter burden or travel-time accessibility is measured",
        ],
    }
    (results / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--shuffles", type=int, default=PERMUTATIONS)
    args = parser.parse_args(argv)
    report = run(args.output, permutations=args.shuffles)
    print(f"Reproduced {report['municipalities']} Porto-link rank diagnostics with {report['shuffles']} shuffles")


if __name__ == "__main__":
    main()
