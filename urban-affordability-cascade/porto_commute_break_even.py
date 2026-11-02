"""Break-even *incremental* commuting time for Porto-relative rent options.

Uses observed 2025 municipal new-contract rent medians and strictly declared
household cost assumptions. No actual journey, traveller, dwelling choice or
housing welfare is observed. All money amounts are nominal 2025 EUR/month.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import ROOT, SOURCE_COMMIT, SUBSET_SHA256, load_panel, require_float
from porto_rent_options import AREAS_M2, COMPARATORS, END_YEAR


EXTRA_ONE_WAY_MINUTES = (15, 30, 45, 60)
FIGURES = ("fig30_porto_rent_break_even_extra_commute_minutes.png",
           "fig31_rent_saving_exhaustion_by_commute_scenario.png")
RESULTS = ("porto_commute_break_even_municipality_scenarios.csv",
           "porto_commute_break_even_duration_thresholds.csv",
           "porto_commute_break_even_70m2_municipal.csv",
           "porto_commute_break_even_provenance.json")


def _positive(value: float, label: str) -> float:
    """Reject invalid/bool monetary quantities and physical durations."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def _nonnegative(value: float, label: str) -> float:
    """Reject invalid/bool negative costs or time durations."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return float(value)


def _finite_real(value: float, label: str) -> float:
    """Accept finite signed rent gaps, including negative comparisons."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


@dataclass(frozen=True, slots=True)
class CommuteScenario:
    """Incremental household journey-cost assumptions, not observed prices.

    All commuters are assigned the *same added one-way minutes per workday*.
    The daily return journey doubles this addition. `cash_extra_eur_month`
    is the *household's incremental* monthly cash cost, not a pass price.
    """

    name: str
    commuters: int
    commuting_days_month: int
    time_value_eur_hour: float
    cash_extra_eur_month: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Scenario name must be a nonempty string")
        for value, label in ((self.commuters, "commuters"), (self.commuting_days_month, "commuting days")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        _positive(self.time_value_eur_hour, "hourly value of time")
        _nonnegative(self.cash_extra_eur_month, "extra household cash travel cost")


SCENARIOS: tuple[CommuteScenario, ...] = (
    CommuteScenario("low_friction", commuters=1, commuting_days_month=16,
                    time_value_eur_hour=5, cash_extra_eur_month=0),
    CommuteScenario("reference", commuters=1, commuting_days_month=20,
                    time_value_eur_hour=10, cash_extra_eur_month=40),
    CommuteScenario("two_commuter_high_cost", commuters=2, commuting_days_month=22,
                    time_value_eur_hour=15, cash_extra_eur_month=80),
)

SCENARIO_DISPLAY = {
    "low_friction": "Low friction",
    "reference": "Reference",
    "two_commuter_high_cost": "Two commuters, high cost",
}


def scenario_legend_label(scenario: CommuteScenario) -> str:
    """Render assumptions directly from the validated scenario object."""
    if not isinstance(scenario, CommuteScenario):
        raise ValueError("Expected a declared CommuteScenario")
    title = SCENARIO_DISPLAY.get(scenario.name, scenario.name.replace("_", " ").title())
    noun = "commuter" if scenario.commuters == 1 else "commuters"
    return (f"{title}: {scenario.commuters} {noun} / {scenario.commuting_days_month}d "
            f"/ €{scenario.time_value_eur_hour:g}/h / €{scenario.cash_extra_eur_month:g} extra")


def additional_time_cost_eur_month(extra_one_way_minutes: float, scenario: CommuteScenario) -> float:
    """Value of two added one-way legs on each assumed commuting day."""
    minutes = _nonnegative(extra_one_way_minutes, "added one-way minutes")
    if not isinstance(scenario, CommuteScenario):
        raise ValueError("Expected a declared CommuteScenario")
    return (2 * minutes / 60 * scenario.commuting_days_month
            * scenario.commuters * scenario.time_value_eur_hour)


def net_monthly_saving_eur(
    rent_gap_eur_month: float, extra_one_way_minutes: float, scenario: CommuteScenario,
) -> float:
    """Difference between synthetic rent-only saving and extra travel costs."""
    gap = _finite_real(rent_gap_eur_month, "rent gap")
    if not isinstance(scenario, CommuteScenario):
        raise ValueError("Expected a declared CommuteScenario")
    return gap - scenario.cash_extra_eur_month - additional_time_cost_eur_month(extra_one_way_minutes, scenario)


def break_even_extra_one_way_minutes(rent_gap_eur_month: float, scenario: CommuteScenario) -> float:
    """Maximal *additional* one-way commute before net saving becomes ≤0.

    Return zero if assumed monthly cash travel costs already consume the
    rent gap. Zero does not imply that an actual commute takes zero minutes.
    """
    gap = _finite_real(rent_gap_eur_month, "rent gap")
    if not isinstance(scenario, CommuteScenario):
        raise ValueError("Expected a declared CommuteScenario")
    remaining = max(0, gap - scenario.cash_extra_eur_month)
    per_minute = additional_time_cost_eur_month(1, scenario)
    if per_minute <= 0:
        raise ValueError("An assumed positive time valuation is required")
    return remaining / per_minute


def rent_gaps_2025() -> dict[str, float]:
    """Exact source-pinned difference between independent municipal medians."""
    panel = load_panel()
    porto = require_float(panel["Porto", END_YEAR].rent_m2, "Porto 2025 new-contract median")
    if porto <= 0:
        raise ValueError("Invalid 2025 Porto median")
    result: dict[str, float] = {}
    for municipality in COMPARATORS:
        other = require_float(panel[municipality, END_YEAR].rent_m2, "comparator 2025 new-contract median")
        if other <= 0 or not math.isfinite(other):
            raise ValueError("Invalid 2025 comparator rent")
        result[municipality] = porto - other
    if len(result) != 16:
        raise ValueError("Expected 16 independent alternative municipal medians")
    return result


def scenario_table() -> list[dict[str, str | int]]:
    """Compute all 16×3×3 municipal / floor-area / scenario outcomes."""
    gaps = rent_gaps_2025()
    rows: list[dict[str, str | int]] = []
    for municipality in COMPARATORS:
        for area in AREAS_M2:
            rent_gap = area * gaps[municipality]
            for scenario in SCENARIOS:
                break_even = break_even_extra_one_way_minutes(rent_gap, scenario)
                rows.append({
                    "municipality": municipality,
                    "year": END_YEAR,
                    "assumed_dwelling_area_m2": area,
                    "scenario": scenario.name,
                    "commuters": scenario.commuters,
                    "commuting_days_month": scenario.commuting_days_month,
                    "time_value_eur_hour": f"{scenario.time_value_eur_hour:.2f}",
                    "incremental_cash_cost_eur_month": f"{scenario.cash_extra_eur_month:.2f}",
                    "municipal_new_rent_gap_eur_m2_month": f"{gaps[municipality]:.6f}",
                    "synthetic_rent_only_saving_eur_month": f"{rent_gap:.6f}",
                    "already_no_saving_at_zero_extra_time": str(rent_gap <= scenario.cash_extra_eur_month).lower(),
                    "break_even_extra_one_way_minutes_workday": f"{break_even:.6f}",
                })
    if len(rows) != 16 * len(AREAS_M2) * len(SCENARIOS):
        raise ValueError("Expected complete scenario grid")
    return rows


def duration_thresholds(rows: Sequence[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Count comparisons exhausted by different *assumed* added durations."""
    expected = {(name, area, s.name) for name in COMPARATORS for area in AREAS_M2 for s in SCENARIOS}
    found = {(str(row["municipality"]), int(row["assumed_dwelling_area_m2"]), str(row["scenario"])) for row in rows}
    if len(rows) != len(expected) or found != expected:
        raise ValueError("Incomplete or duplicate rent-commute comparison grid")
    summary: list[dict[str, str | int]] = []
    for scenario in SCENARIOS:
        for area in AREAS_M2:
            selected = [r for r in rows if r["scenario"] == scenario.name and r["assumed_dwelling_area_m2"] == area]
            threshold = [float(r["break_even_extra_one_way_minutes_workday"]) for r in selected]
            if len(threshold) != 16:
                raise ValueError("Expected 16 comparator thresholds")
            for duration in EXTRA_ONE_WAY_MINUTES:
                # Use the exact cost inequality, without rounding a break-even
                # minute value before classifying boundary comparisons.
                exhausted = sum(net_monthly_saving_eur(
                    float(r["synthetic_rent_only_saving_eur_month"]), duration, scenario,
                ) <= 0 for r in selected)
                summary.append({
                    "scenario": scenario.name,
                    "assumed_dwelling_area_m2": area,
                    "extra_one_way_minutes_workday": duration,
                    "comparator_municipalities": len(selected),
                    "comparisons_with_no_positive_synthetic_net_saving": exhausted,
                    "median_break_even_extra_one_way_minutes": f"{median(threshold):.6f}",
                    "min_break_even_extra_one_way_minutes": f"{min(threshold):.6f}",
                    "max_break_even_extra_one_way_minutes": f"{max(threshold):.6f}",
                })
    return summary


def primary_municipality_table(rows: Sequence[dict[str, str | int]]) -> list[dict[str, str]]:
    """One 70 m² row per municipality: compare all three stated assumptions."""
    lookup = {(str(r["municipality"]), int(r["assumed_dwelling_area_m2"]), str(r["scenario"])): r for r in rows}
    expected = {(name, area, s.name) for name in COMPARATORS for area in AREAS_M2 for s in SCENARIOS}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Duplicate or incomplete scenario records")
    output: list[dict[str, str]] = []
    for name in COMPARATORS:
        records = {s.name: lookup[name, 70, s.name] for s in SCENARIOS}
        output.append({
            "municipality": name,
            "synthetic_70m2_rent_only_saving_2025_eur_month": str(records["reference"]["synthetic_rent_only_saving_eur_month"]),
            **{f"{s.name}_break_even_added_one_way_minutes": str(records[s.name]["break_even_extra_one_way_minutes_workday"])
               for s in SCENARIOS},
        })
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty rent-commute table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_break_even(primary: Sequence[dict[str, str]], path: Path) -> None:
    """Display scenario thresholds per comparator without treating them as CIs."""
    sorted_rows = sorted(primary, key=lambda r: float(r["reference_break_even_added_one_way_minutes"]))
    places = [r["municipality"] for r in sorted_rows]
    y = np.arange(len(places))
    fig, ax = plt.subplots(figsize=(10.4, 8.2))
    for scenario, color in zip(SCENARIOS, ("#2f8b69", "#347db0", "#c4703d")):
        x = np.array([float(r[f"{scenario.name}_break_even_added_one_way_minutes"]) for r in sorted_rows])
        ax.scatter(x, y, s=55, color=color, label=scenario_legend_label(scenario), alpha=.9, zorder=3)
    ax.set_yticks(y, places, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Break-even added one-way commute minutes per working day")
    fig.suptitle("2025 synthetic 70 m² rent gaps: how much extra travel time?", y=.98, fontsize=12.5)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="upper center", bbox_to_anchor=(.59, .941), ncol=3, fontsize=7.0, frameon=False)
    fig.text(.02, .019, "Assumed time/cash costs, not actual journeys. Break-even dots are scenarios, not statistical uncertainty.",
             fontsize=8.5, color="#566570")
    fig.subplots_adjust(left=.21, right=.97, bottom=.12, top=.865)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_threshold_counts(summary: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show counts, not probabilities, of hypothetical exhausted rent gaps."""
    area = 70
    matrix = np.array([
        [int(next(r["comparisons_with_no_positive_synthetic_net_saving"] for r in summary
                  if r["scenario"] == scenario.name and r["assumed_dwelling_area_m2"] == area
                  and r["extra_one_way_minutes_workday"] == duration))
         for duration in EXTRA_ONE_WAY_MINUTES]
        for scenario in SCENARIOS
    ])
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    heat = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=16, aspect="auto")
    ax.set_yticks(range(len(SCENARIOS)),
                  [SCENARIO_DISPLAY.get(s.name, s.name.replace("_", " ").title()) for s in SCENARIOS])
    ax.set_xticks(range(4), [f"{d} min" for d in EXTRA_ONE_WAY_MINUTES])
    ax.set_xlabel("Assumed additional one-way minutes on each commuting day")
    ax.set_title("Synthetic rent-only advantage exhausted (out of 16 comparators)", fontsize=12.6, pad=11)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] >= 10 else "#273544", fontsize=13, weight="bold")
    fig.colorbar(heat, ax=ax, fraction=.04, pad=.03, label="Municipal comparisons, 0–16")
    fig.text(.02, .02, "70 m² hypothetical rent comparison, 2025 nominal EUR. Counts are not numbers of households or real alternatives.",
             fontsize=8.5, color="#566570")
    fig.subplots_adjust(left=.27, right=.90, bottom=.21, top=.86)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Regenerate the fixed-source synthetic commute thresholds and ledger."""
    table = scenario_table()
    summary = duration_thresholds(table)
    primary = primary_municipality_table(table)
    figures, results = output / "figures", output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plot_break_even(primary, figures / FIGURES[0])
    plot_threshold_counts(summary, figures / FIGURES[1])
    for name, rows in zip(RESULTS[:3], (table, summary, primary)):
        _write_csv(results / name, rows)
    report: dict[str, object] = {
        "evidence_class": "fixed_source_rent_gap_declared_synthetic_household_commute_cost_scenarios",
        "housing_upstream_commit": SOURCE_COMMIT,
        "housing_panel_sha256": SUBSET_SHA256,
        "rent_reference_year": END_YEAR,
        "money_units": "nominal 2025 EUR/month; no HICP deflation needed for within-2025 cash comparison",
        "comparator_municipalities": len(COMPARATORS),
        "assumed_dwelling_areas_m2": list(AREAS_M2),
        "assumed_added_one_way_minutes": list(EXTRA_ONE_WAY_MINUTES),
        "scenarios": [{"name": s.name, "commuters": s.commuters, "commuting_days_month": s.commuting_days_month,
                       "time_value_eur_hour": s.time_value_eur_hour,
                       "incremental_household_cash_eur_month": s.cash_extra_eur_month} for s in SCENARIOS],
        "municipal_scenario_rows": len(table), "duration_summary_rows": len(summary),
        "reference_70m2_median_break_even_added_one_way_minutes": f"{median(float(r['reference_break_even_added_one_way_minutes']) for r in primary):.6f}",
        "limits": [
            "2025 municipal medians for independent newly signed rental contracts do not compare available equivalent homes",
            "50/70/90 m² floor areas are hypothetical; the published rent median is per square metre",
            "Commuting days, time valuations, number of commuters and cash increments are assumptions, not measured Portugal costs",
            "Extra one-way time is the incremental time *versus the current commute*, not the entire trip from Porto",
            "Every assumed added minute is incurred on both trip legs on each commuting day by each assumed commuter",
            "Extra travel cash is an incremental monthly household amount, not an Andante price or transit fare",
            "Time value is a modeled opportunity cost, not observed expenditure or a statutory salary rate",
            "No dwelling availability, moving cost, household composition, employment access, care needs or amenity differences are measured",
            "Municipal options are administrative comparisons, not verified feasible relocation choices",
            "Break-even thresholds do not identify migration, regional price spillovers, displacement or a causal policy effect",
        ],
    }
    (results / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Reproduced {report['municipal_scenario_rows']} hypothetical household commute scenarios")


if __name__ == "__main__":
    main()
