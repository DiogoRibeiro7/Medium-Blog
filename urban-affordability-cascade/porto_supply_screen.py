"""Screen municipal dwelling-construction flows against Porto rent pressure.

Uses the fixed 17×7 INE-derived panel. Completions are new family-housing
dwellings completed, permits are new family-housing dwellings licensed.
The numbers are not net additions to the accessible rental stock. Since
construction and rent windows overlap, all associations are descriptive.
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

from analysis import INPUT_PANEL, MUNICIPALITIES, ROOT, SUBSET_SHA256, load_panel, require_float
from extract_commuting import municipality_codes_from_panel
from porto_network_gradient import leave_one_out_rhos, spearman_rho


WINDOW_ENDS = (2023, 2024, 2025)
WINDOW_START = 2020
END_RENT_YEAR = 2025
MODES = ("completed_dwellings", "licensed_dwellings")
NORMALIZATIONS = ("raw_count", "per_1000_population_2020", "per_1000_population_2021")
OUTCOMES = ("porto_discount_compression_2020_2025_pp", "own_nominal_rent_growth_2020_2025_pct")
FIGURES = ("fig28_municipal_construction_vs_porto_rent_pressure.png",
           "fig29_construction_rent_rank_window_sensitivity.png")
RESULTS = ("porto_supply_screen_municipal_exposures.csv",
           "porto_supply_screen_rank_windows.csv",
           "porto_supply_screen_leave_one_out.csv",
           "porto_supply_screen_provenance.json")


def construction_panel(path: Path = INPUT_PANEL) -> dict[tuple[str, int], dict[str, int | str]]:
    """Audit pinned construction-unit counts and preliminary/estimate flags."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != SUBSET_SHA256:
        raise ValueError("Fixed housing panel source checksum differs")
    codes = municipality_codes_from_panel(path)
    records: dict[tuple[str, int], dict[str, int | str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"dico", "municipality", "year", "completions", "permits", "flags"}.issubset(reader.fieldnames):
            raise ValueError("Construction panel is missing required columns")
        for row in reader:
            name = row["municipality"]
            year = int(row["year"])
            if (name not in MUNICIPALITIES or codes.get(row["dico"]) != name
                    or year not in range(2019, 2026) or (name, year) in records):
                raise ValueError("Invalid or duplicate construction municipality-year identity")
            values: dict[str, int | str] = {"flags": row["flags"]}
            for col in ("completions", "permits"):
                raw = row[col]
                if not raw.isdecimal() or (len(raw) > 1 and raw[0] == "0"):
                    raise ValueError(f"Invalid whole-number {col} at {name}, {year}")
                values[col] = int(raw)
            flags = set(row["flags"].split(";"))
            if ("construction_preliminary" in flags) != (year >= 2021):
                raise ValueError("Unexpected construction preliminary flag period")
            if ("completions_estimated" in flags) != (year >= 2024):
                raise ValueError("Unexpected estimated-completions flag period")
            records[name, year] = values
    if set(records) != {(name, year) for name in MUNICIPALITIES for year in range(2019, 2026)}:
        raise ValueError("Expected complete 17 × 7 construction-count panel")
    observations = load_panel(path)
    for key, record in records.items():
        if int(record["completions"]) != require_float(observations[key].completions, "completions"):
            raise ValueError("Construction counts disagree with observed panel reader")
    return records


def municipal_exposures() -> list[dict[str, str]]:
    """Join published supply flows, population and two outcome conventions."""
    records = construction_panel()
    observations = load_panel()
    porto_2020 = require_float(observations["Porto", 2020].rent_m2, "Porto 2020 rent")
    porto_2025 = require_float(observations["Porto", 2025].rent_m2, "Porto 2025 rent")
    result: list[dict[str, str]] = []
    for name in sorted(MUNICIPALITIES - {"Porto"}):
        p20 = require_float(observations[name, 2020].population, "2020 population")
        p21 = require_float(observations[name, 2021].population, "2021 population")
        p25 = require_float(observations[name, 2025].population, "2025 population")
        r20 = require_float(observations[name, 2020].rent_m2, "2020 rent")
        r25 = require_float(observations[name, 2025].rent_m2, "2025 rent")
        if min(p20, p21, p25, r20, r25) <= 0:
            raise ValueError("Non-positive construction-exposure denominator")
        record = {
            "municipality": name,
            "municipal_population_2020": f"{p20:.0f}",
            "municipal_population_2021": f"{p21:.0f}",
            "municipal_population_2025": f"{p25:.0f}",
            "population_change_2020_2025_pct": f"{100 * (p25 / p20 - 1):.6f}",
            "porto_discount_compression_2020_2025_pp": f"{100 * (r25 / porto_2025 - r20 / porto_2020):.6f}",
            "own_nominal_rent_growth_2020_2025_pct": f"{100 * (r25 / r20 - 1):.6f}",
        }
        for end in WINDOW_ENDS:
            for mode in MODES:
                source_col = "completions" if mode == "completed_dwellings" else "permits"
                total = sum(int(records[name, year][source_col]) for year in range(WINDOW_START, end + 1))
                prefix = f"{mode}_2020_{end}"
                record[f"{prefix}_raw_count"] = str(total)
                record[f"{prefix}_per_1000_population_2020"] = f"{1000 * total / p20:.6f}"
                record[f"{prefix}_per_1000_population_2021"] = f"{1000 * total / p21:.6f}"
        result.append(record)
    return result


def _supply(rows: Sequence[dict[str, str]], mode: str, end: int, normalization: str) -> np.ndarray:
    if mode not in MODES or end not in WINDOW_ENDS or normalization not in NORMALIZATIONS:
        raise ValueError("Unregistered construction exposure convention")
    col = f"{mode}_2020_{end}_{normalization}"
    values = np.array([float(r[col]) for r in rows])
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.all(values == values[0]):
        raise ValueError("Invalid or constant construction exposure")
    return values


def _outcome(rows: Sequence[dict[str, str]], outcome: str) -> np.ndarray:
    if outcome not in OUTCOMES:
        raise ValueError("Unexpected construction/rent outcome")
    values = np.array([float(r[outcome]) for r in rows])
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite construction/rent outcome")
    return values


def diagnostics(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str | int]], list[dict[str, str]]]:
    """Evaluate every declared window, exposure, normalization and outcome."""
    if len(rows) != 16 or {r["municipality"] for r in rows} != MUNICIPALITIES - {"Porto"}:
        raise ValueError("Expected exactly the 16 other AMP municipalities")
    stats: list[dict[str, str | int]] = []
    deletions: list[dict[str, str]] = []
    for end in WINDOW_ENDS:
        for mode in MODES:
            for norm in NORMALIZATIONS:
                x = _supply(rows, mode, end, norm)
                for outcome in OUTCOMES:
                    y = _outcome(rows, outcome)
                    rho = spearman_rho(x, y)
                    leave = leave_one_out_rhos(x, y)
                    stats.append({
                        "construction_window_start": WINDOW_START,
                        "construction_window_end": end,
                        "construction_exposure": mode,
                        "normalization": norm,
                        "rent_outcome": outcome,
                        "municipalities": len(rows),
                        "spearman_rho": f"{rho:.6f}",
                        "leave_one_out_min_rho": f"{min(leave):.6f}",
                        "leave_one_out_max_rho": f"{max(leave):.6f}",
                        "includes_estimated_completions": str(mode == "completed_dwellings" and end >= 2024).lower(),
                    })
                    if end == 2023 and norm == "per_1000_population_2020" and outcome == OUTCOMES[0]:
                        for row, r in zip(rows, leave):
                            deletions.append({
                                "removed_municipality": row["municipality"],
                                "construction_exposure": mode,
                                "full_16_spearman_rho": f"{rho:.6f}",
                                "leave_one_out_15_spearman_rho": f"{r:.6f}",
                            })
    return stats, deletions


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty construction data")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_primary(rows: Sequence[dict[str, str]], stats: Sequence[dict[str, str | int]], path: Path) -> None:
    """Expose all primary municipal construction/rent pairings, no selection."""
    fig, axes = plt.subplots(1, 2, figsize=(13.1, 5.9), sharey=True)
    y = _outcome(rows, OUTCOMES[0])
    labels = {"Matosinhos", "Vila Nova de Gaia", "Espinho", "Arouca", "Valongo", "Póvoa de Varzim"}
    for ax, mode, title in zip(axes, MODES, ("Completed new dwellings", "Licensed new dwellings")):
        x = _supply(rows, mode, 2023, "per_1000_population_2020")
        rho = next(float(r["spearman_rho"]) for r in stats if r["construction_window_end"] == 2023
                   and r["construction_exposure"] == mode and r["normalization"] == "per_1000_population_2020"
                   and r["rent_outcome"] == OUTCOMES[0])
        ax.axhline(0, color="#788b92", ls="--", lw=1)
        ax.scatter(x, y, color="#246d9b" if mode == MODES[0] else "#c27330",
                   s=64, alpha=.88, edgecolor="white", linewidth=.6)
        for row, a, b in zip(rows, x, y):
            if row["municipality"] in labels:
                # Keep labels for high-exposure municipalities inside axes.
                far_right = a >= float(np.quantile(x, .85))
                ax.annotate(row["municipality"], (a, b),
                            xytext=(-4 if far_right else 4, 5), textcoords="offset points",
                            ha="right" if far_right else "left", fontsize=8)
        ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f}", fontsize=12)
        ax.set_xlabel("2020–2023 cumulative dwellings per 1,000 residents (2020)")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Porto-relative rent-discount compression, 2020–2025 (pp)")
    fig.suptitle("More construction does not identify a lower-rent causal effect", fontsize=14)
    fig.text(.015, .015, "Construction is an observed flow, not net available rental supply. 2021–2023 values are preliminary; windows overlap.",
             fontsize=8.7, color="#576771")
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.83, wspace=.16)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sensitivity(stats: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show three supply-window choices against two rent outcomes."""
    combos = [(mode, outcome) for outcome in OUTCOMES for mode in MODES]
    matrix = np.array([
        [float(next(r["spearman_rho"] for r in stats if r["construction_window_end"] == year
                    and r["construction_exposure"] == mode
                    and r["normalization"] == "per_1000_population_2020"
                    and r["rent_outcome"] == outcome)) for year in WINDOW_ENDS]
        for mode, outcome in combos
    ])
    fig, ax = plt.subplots(figsize=(9.1, 5))
    picture = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3), [f"2020–{year}" for year in WINDOW_ENDS])
    ax.set_yticks(range(4), ["Completions / discount compression", "Permits / discount compression",
                                  "Completions / own rent growth", "Permits / own rent growth"])
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=12, color="#1b2f41", weight="bold")
    ax.set_title("Construction-flow rank associations depend on window and outcome", fontsize=12.6, pad=12)
    fig.colorbar(picture, ax=ax, fraction=.04, pad=.03, label="Spearman rank correlation")
    fig.text(.015, .024, "Per 1,000 2020 residents; 2024–2025 completions estimated; all 2021+ construction provisional; not causal.",
             fontsize=8.4, color="#576771")
    fig.subplots_adjust(left=.33, right=.91, bottom=.16, top=.87)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate pinned noncausal supply/rent diagnostic artifacts offline."""
    rows = municipal_exposures()
    stats, deletions = diagnostics(rows)
    figdir, resultsdir = output / "figures", output / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resultsdir.mkdir(parents=True, exist_ok=True)
    plot_primary(rows, stats, figdir / FIGURES[0])
    plot_sensitivity(stats, figdir / FIGURES[1])
    for file, data in zip(RESULTS[:3], (rows, stats, deletions)):
        _write_csv(resultsdir / file, data)
    source = construction_panel()
    report: dict[str, object] = {
        "evidence_class": "exploratory_noncausal_municipal_construction_flow_rent_rank_screen",
        "source": "Frozen upstream INE-derived 17×7 municipal panel; indicators 0013248 (completed dwellings), 0013235 (licensed dwellings)",
        "source_sha256": SUBSET_SHA256,
        "municipalities": len(rows),
        "construction_windows": [[2020, end] for end in WINDOW_ENDS],
        "rent_window": [2020, END_RENT_YEAR],
        "exposures": list(MODES),
        "normalizations": list(NORMALIZATIONS),
        "outcomes": list(OUTCOMES),
        "rank_diagnostic_rows": len(stats),
        "influence_rows": len(deletions),
        "construction_preliminary_rows_2021_2025": sum("construction_preliminary" in str(r["flags"]) for (name, year), r in source.items() if year >= 2021),
        "estimated_completed_dwellings_rows_2024_2025": sum("completions_estimated" in str(r["flags"]) for (name, year), r in source.items() if year >= 2024),
        "primary_completion_vs_discount_rho": next(r["spearman_rho"] for r in stats if r["construction_window_end"] == 2023
             and r["construction_exposure"] == MODES[0] and r["normalization"] == NORMALIZATIONS[1]
             and r["rent_outcome"] == OUTCOMES[0]),
        "primary_permit_vs_discount_rho": next(r["spearman_rho"] for r in stats if r["construction_window_end"] == 2023
             and r["construction_exposure"] == MODES[1] and r["normalization"] == NORMALIZATIONS[1]
             and r["rent_outcome"] == OUTCOMES[0]),
        "limitations": [
            "New family-housing dwellings completed/licensed are gross flows, not vacant affordable or rental housing units",
            "Licensed dwellings do not necessarily become completed, occupied or available to rent",
            "Annual completion and permit series are preliminary from 2021 onward; 2024-2025 completions are estimated",
            "Supply counts and 2020-2025 rent outcomes overlap, allowing reverse causality and common demand shocks",
            "One municipality's new-contract rent median does not identify a constant-quality dwelling",
            "Fixed 2020/2021 population denominators are administrative counts, not renter households or new household formation",
            "The sixteen observations share Porto's reference rent, are spatially dependent and not an exchangeable sample",
            "Many correlated exposure and outcome conventions are exploratory diagnostics, not confirmatory tests",
            "No supply elasticity, household welfare, vacancy, residential displacement or causal metropolitan price transmission is identified",
        ],
    }
    (resultsdir / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Generated {report['rank_diagnostic_rows']} municipal supply-rent rank checks")


if __name__ == "__main__":
    main()
