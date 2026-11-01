"""Compare Porto commuter links with a source-pinned geography baseline.

Uses area-weighted CAOP2025 municipal polygon centroids in EPSG:3763.
Euclidean centroid distances do not represent road distance, travel time,
housing availability or population-weighted access. Partial rank correlations
are descriptive adjustments, not causal or inferential estimators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import MUNICIPALITIES, ROOT
from extract_caop import FIELDS as CAOP_FIELDS, PROVENANCE as CAOP_PROVENANCE, SNAPSHOT as CAOP_SNAPSHOT
from extract_commuting import municipality_codes_from_panel
from porto_network_gradient import MODES, WINDOW_STARTS, _compression, _mode_values, exposure_table, spearman_rho


FIGURES = ("fig26_porto_geographic_distance_vs_rent_discount_compression.png",
           "fig27_porto_commuting_gradient_distance_adjustment.png")
RESULTS = ("porto_distance_sensitivity_municipal.csv",
           "porto_distance_sensitivity_rank_diagnostics.csv",
           "porto_distance_sensitivity_leave_one_out.csv",
           "porto_distance_sensitivity_provenance.json")


def load_centroids(path: Path = CAOP_SNAPSHOT, provenance: Path = CAOP_PROVENANCE) -> dict[str, tuple[float, float]]:
    """Load exactly 17 official geographic derivatives, checking the pin."""
    meta = json.loads(provenance.read_text(encoding="utf-8"))
    if meta.get("source_epsg") != 3763 or meta.get("records") != 17 or "DGT" not in meta.get("source_name", ""):
        raise ValueError("Unexpected CAOP provenance scope or coordinate reference")
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["derived_csv_sha256"]:
        raise ValueError("CAOP centroid CSV SHA-256 mismatch")
    code_to_name = municipality_codes_from_panel()
    result: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CAOP_FIELDS):
            raise ValueError("Unexpected CAOP-derived CSV columns")
        for row in reader:
            name, code = row["municipality"], row["municipality_code"]
            if name not in MUNICIPALITIES or code_to_name.get(code) != name or name in result:
                raise ValueError("Duplicate or mismatched official municipality identity")
            east, north = float(row["centroid_easting_epsg3763_m"]), float(row["centroid_northing_epsg3763_m"])
            area = float(row["official_area_ha"])
            if not all(math.isfinite(v) for v in (east, north, area)) or area <= 0:
                raise ValueError("Invalid projected official centroid")
            result[name] = (east, north)
    if set(result) != set(MUNICIPALITIES):
        raise ValueError("Missing 17-municipality CAOP2025 coverage")
    return result


def partial_rank_correlation(x: np.ndarray, y: np.ndarray, distance: np.ndarray) -> float:
    """Pearson residual correlation after linear projection of average ranks.

    This is the usual one-covariate partial Spearman coefficient. Rejects
    near-perfect rank collinearity rather than assigning a spurious value.
    """
    x, y, distance = (np.asarray(a, dtype=float) for a in (x, y, distance))
    if x.ndim != 1 or x.shape != y.shape or x.shape != distance.shape or len(x) < 5:
        raise ValueError("At least five equal-length rank observations required")
    a, b, c = spearman_rho(x, y), spearman_rho(x, distance), spearman_rho(y, distance)
    denominator = math.sqrt(max(0., (1 - b * b) * (1 - c * c)))
    if denominator < 1e-9:
        raise ValueError("Rank collinearity prevents distance adjustment")
    return float(np.clip((a - b * c) / denominator, -1, 1))


def comparison_table() -> list[dict[str, str]]:
    """Join the prior pinned comparisons to official administrative distances."""
    coordinates = load_centroids()
    px, py = coordinates["Porto"]
    table = []
    for record in exposure_table():
        name = record["municipality"]
        east, north = coordinates[name]
        km = math.hypot(east - px, north - py) / 1000
        if not math.isfinite(km) or km <= 0:
            raise ValueError("Non-positive centroid-to-Porto distance")
        table.append({**record, "caop2025_centroid_easting_epsg3763_m": f"{east:.3f}",
                      "caop2025_centroid_northing_epsg3763_m": f"{north:.3f}",
                      "caop2025_porto_centroid_euclidean_distance_km": f"{km:.6f}"})
    return table


def _distance(rows: Sequence[dict[str, str]]) -> np.ndarray:
    values = np.array([float(r["caop2025_porto_centroid_euclidean_distance_km"]) for r in rows])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Invalid centroid distance")
    return values


def diagnostics(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str | int]], list[dict[str, str]]]:
    """Report all combinations plus all 16 primary partial-rank deletions."""
    if len(rows) != 16 or len({r["municipality"] for r in rows}) != 16:
        raise ValueError("Exactly 16 unique comparators required")
    distance = _distance(rows)
    stats: list[dict[str, str | int]] = []
    deletions: list[dict[str, str]] = []
    for year in WINDOW_STARTS:
        outcome = _compression(rows, year)
        distance_rho = spearman_rho(distance, outcome)
        stats.append({"start_year": year, "end_year": 2025,
                      "exposure": "centroid_distance_km", "municipalities": 16,
                      "unadjusted_spearman_rho": f"{distance_rho:.6f}",
                      "partial_spearman_controlling_centroid_distance": "",
                      "exposure_vs_distance_spearman_rho": "1.000000"})
        for mode in MODES:
            x = _mode_values(rows, mode)
            raw = spearman_rho(x, outcome)
            adjusted = partial_rank_correlation(x, outcome, distance)
            stats.append({"start_year": year, "end_year": 2025, "exposure": mode,
                          "municipalities": 16, "unadjusted_spearman_rho": f"{raw:.6f}",
                          "partial_spearman_controlling_centroid_distance": f"{adjusted:.6f}",
                          "exposure_vs_distance_spearman_rho": f"{spearman_rho(x, distance):.6f}"})
            if year == 2020 and mode == "per_1000_residents":
                for i, record in enumerate(rows):
                    mask = np.arange(len(rows)) != i
                    leave = partial_rank_correlation(x[mask], outcome[mask], distance[mask])
                    deletions.append({"removed_municipality": record["municipality"],
                                      "full_16_partial_spearman": f"{adjusted:.6f}",
                                      "leave_one_out_15_partial_spearman": f"{leave:.6f}"})
    return stats, deletions


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty diagnostic table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_distance(rows: Sequence[dict[str, str]], stats: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show unadjusted distance gradient, retaining every municipality."""
    distance, outcome = _distance(rows), _compression(rows, 2020)
    corr = next(r for r in stats if r["exposure"] == "centroid_distance_km" and r["start_year"] == 2020)
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    ax.axhline(0, color="#687a86", ls="--", lw=1)
    ax.scatter(distance, outcome, s=74, color="#2b779a", alpha=.86, edgecolor="white", linewidth=.6)
    labels = {"Arouca", "Matosinhos", "Vila Nova de Gaia", "Valongo", "Póvoa de Varzim", "Espinho"}
    for r, x, y in zip(rows, distance, outcome):
        if r["municipality"] in labels:
            ax.annotate(r["municipality"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8.6)
    ax.set_xlabel("CAOP2025 municipal centroid-to-Porto centroid straight-line distance (km)")
    ax.set_ylabel("Porto-relative new-rent discount compression, 2020–2025 (pp)")
    ax.set_title("Geographic proximity is another imperfect exposure baseline", fontsize=13)
    ax.text(.98, .95, f"Spearman ρ = {float(corr['unadjusted_spearman_rho']):+.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#27445c")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .02, "Centroids weight municipal land area, not households. Straight-line distance is not travel time or access.",
             fontsize=8.5, color="#526371")
    fig.subplots_adjust(left=.14, right=.98, bottom=.16, top=.89)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_partial(stats: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show raw versus distance-adjusted rank associations over four windows."""
    modes = ("raw_interactions", "per_1000_residents", "share_of_amp_interactions")
    matrix = np.array([
        [float(next(r[field] for r in stats if r["start_year"] == year and r["exposure"] == mode))
         for year in WINDOW_STARTS]
        for mode in modes for field in ("unadjusted_spearman_rho",
                                       "partial_spearman_controlling_centroid_distance")
    ])
    fig, ax = plt.subplots(figsize=(9.3, 6))
    heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [f"{year}–2025" for year in WINDOW_STARTS])
    ax.set_yticks(range(6), ["Raw pairs", "Raw, conditional", "Per 1,000", "Per 1,000, conditional",
                                  "Within-AMP share", "Share, conditional"])
    ax.set_title("Distance adjustment does not identify a commuting effect", fontsize=13, pad=12)
    for i in range(6):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center", fontsize=11, color="#172b3b", weight="bold")
    fig.colorbar(heat, ax=ax, fraction=.034, pad=.03, label="Spearman rank coefficient")
    fig.text(.02, .025, "Conditional = partial Spearman controlling for CAOP2025 centroid distance; n=16, no causal interpretation.",
             fontsize=8.3, color="#526371")
    fig.subplots_adjust(left=.27, right=.92, bottom=.14, top=.88)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate four geography sensitivity files and two figures offline."""
    rows = comparison_table()
    stats, deletions = diagnostics(rows)
    figdir, resultdir = output / "figures", output / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resultdir.mkdir(parents=True, exist_ok=True)
    plot_distance(rows, stats, figdir / FIGURES[0])
    plot_partial(stats, figdir / FIGURES[1])
    for name, data in zip(RESULTS[:3], (rows, stats, deletions)):
        _write_csv(resultdir / name, data)
    caop = json.loads(CAOP_PROVENANCE.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "evidence_class": "descriptive_noncausal_caop2025_centroid_distance_sensitivity",
        "caop2025_source_url": caop["source_url"],
        "caop2025_source_response_sha256": caop["source_response_sha256"],
        "caop2025_derived_csv_sha256": caop["derived_csv_sha256"],
        "caop2025_license": caop["source_license"],
        "exposure_source": "Pinned 2021 undirected work/study Census-derived third-party pair workbook",
        "municipalities": len(rows), "windows": [[y, 2025] for y in WINDOW_STARTS],
        "diagnostic_rows": len(stats), "leave_one_out_rows": len(deletions),
        "primary_distance_spearman": next(r["unadjusted_spearman_rho"] for r in stats
                                         if r["exposure"] == "centroid_distance_km" and r["start_year"] == 2020),
        "primary_per_resident_partial_spearman": next(r["partial_spearman_controlling_centroid_distance"]
                                                     for r in stats if r["exposure"] == "per_1000_residents"
                                                     and r["start_year"] == 2020),
        "limits": [
            "CAOP2025 municipal centroids are area-weighted polygon centers, not housing or population centers",
            "Straight-line projected distances are not drive time, public-transport access or realistic search costs",
            "CAOP2025 boundaries are not geography reconstructed separately for each housing observation year",
            "Only 16 municipality outcomes share the Porto rent-reference denominator and strong regional dependence",
            "Partial Spearman rank adjustment leaves unmeasured confounding, collinearity and nonlinear selection",
            "The 2021 commuting network overlaps outcomes and represents undirected work/study interaction, not residential moves",
            "Third-party Census-derived commuting interactions have not been independently confirmed against official INE API",
            "Changing newly signed rental dwellings and publication granularity prevent quality-adjusted price comparisons",
            "Neither direction of migration nor housing price propagation or causal displacement can be inferred",
        ],
    }
    (resultdir / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Reproduced {report['municipalities']} municipal distance comparisons and 16 rank diagnostics")


if __name__ == "__main__":
    main()
