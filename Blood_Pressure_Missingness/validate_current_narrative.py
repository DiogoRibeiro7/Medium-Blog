"""Check that the public blood-pressure narrative matches the current snapshot.

This is intentionally a snapshot-sensitive documentation gate, not a numerical
regression test. A private-data refresh is allowed to change the aggregate
results; the resulting review PR should then fail this check until the README
and notebook are updated to describe the new privacy-safe snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analysis as primary
import day_influence_sensitivity as influence
import episode_observation_sensitivity as episode_observation
import episode_time_form_sensitivity as time_form
import gap_aware_trend_decomposition as gap_aware


def _require(text: str, fragment: str, surface: str) -> None:
    if fragment not in text:
        raise RuntimeError(
            f"Public narrative drift: {surface} is missing expected fragment: {fragment!r}"
        )


def validate_current_narrative(root: Path) -> None:
    data_path = root / "data" / "analysis_snapshot.csv"
    audit_path = root / "data" / "source_audit.json"
    readme_path = root / "README.md"
    notebook_path = root / "blood-pressure-missingness.ipynb"

    records = primary.load_snapshot(data_path)
    observed = primary.observed_records(records)
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    readme = readme_path.read_text(encoding="utf-8")
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    notebook_markdown = "\n".join(
        str(cell.get("source", ""))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )

    n_calendar = len(records)
    n_observed = len(observed)
    n_missing = n_calendar - n_observed
    n_readings = int(audit["valid_measurements"])
    n_sessions = int(audit["sessionization"]["n_sessions"])
    coverage = 100.0 * n_observed / n_calendar

    global_fit = primary.linear_trend(records, "mean_systolic_mmHg")
    gap = gap_aware.analyze_gap_aware_trend(records)
    centered = gap["episode_centered_model"]
    decomposition = gap["exact_global_slope_decomposition"]
    influence_result = influence.summarize_influence(records)
    loo = influence_result["leave_one_day_out_summary"]
    episode_obs = episode_observation.fit_episode_observation_sensitivity(records)
    time_results = time_form.fit_episode_time_form_sensitivity(records)

    dominant_gap = gap["dominant_internal_gap"]
    longest_gap = int(dominant_gap["length_days"])

    readme_fragments = [
        f"| Valid measurements | {n_readings} |",
        f"| Measurement sessions | {n_sessions} |",
        f"| Observed calendar days | {n_observed} |",
        f"| Calendar days in analysis window | {n_calendar} |",
        f"| Missing calendar days | {n_missing} |",
        f"| Longest missing run | {longest_gap} days |",
        f"Calendar-day coverage is therefore **{coverage:.1f}%**",
        f"\\boxed{{{float(global_fit['slope_per_30_days']):.2f}\\ \\text{{mmHg per 30 days}}}}",
        (
            f"[{float(global_fit['ci95_low_per_30_days']):.2f},"
            f"{float(global_fit['ci95_high_per_30_days']):.2f}]."
        ),
        f"\\boxed{{{float(centered['post_minus_pre_mean_difference_mmHg']):.2f}\\ \\text{{mmHg}}}}",
        (
            f"[{float(centered['post_minus_pre_ci95_low_mmHg']):.2f},"
            f"{float(centered['post_minus_pre_ci95_high_mmHg']):.2f}]."
        ),
        (
            f"about **{100.0 * float(decomposition['between_fraction_of_time_pressure_covariance']):.1f}% "
            "of the negative global time-pressure covariance**"
        ),
        (
            f"**{float(loo['global_slope_min_per_30_days']):.2f} to "
            f"{float(loo['global_slope_max_per_30_days']):.2f} mmHg/30d**"
        ),
        (
            f"**{float(loo['episode_level_difference_min_mmHg']):.2f} to "
            f"{float(loo['episode_level_difference_max_mmHg']):.2f} mmHg**"
        ),
    ]
    for fragment in readme_fragments:
        _require(readme, fragment, "README")

    inverse_episode = next(
        item for item in episode_obs["estimates"] if item["name"] == "inverse_intensity_stress"
    )
    if float(inverse_episode["episode_ci95_low_mmHg"]) < 0.0 < float(
        inverse_episode["episode_ci95_high_mmHg"]
    ):
        _require(
            readme,
            "Only the deliberately aggressive inverse-intensity stress case is inconclusive.",
            "README",
        )

    time_estimates = time_results["estimates"]
    all_time_form_intervals_below_zero = all(
        float(item["episode_ci95_high_mmHg"]) < 0.0 for item in time_estimates
    )
    if all_time_form_intervals_below_zero:
        _require(readme, "All five intervals are now below zero.", "README")
        _require(
            notebook_markdown,
            "all five tested within-episode time forms",
            "notebook",
        )

    _require(
        notebook_markdown,
        f"**{n_observed} observed days across {n_calendar} calendar days**",
        "notebook",
    )
    _require(
        notebook_markdown,
        f"{n_readings} readings are treated as exchangeable i.i.d. observations",
        "notebook",
    )
    _require(
        notebook_markdown,
        f"smooth downward trajectory over {n_calendar} days",
        "notebook",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Blood_Pressure_Missingness project directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_current_narrative(args.root)
    print("Public narrative matches the current privacy-safe snapshot.")


if __name__ == "__main__":
    main()
