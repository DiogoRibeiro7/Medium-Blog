"""Validate current-snapshot influence findings during private-data refresh.

These checks are intentionally *not* ordinary code invariants. They encode the
scientific statements that should remain stable as the private source evolves.
The secret-backed refresh workflow runs this module immediately after rebuilding
all derived outputs. CI also runs it whenever the committed snapshot or influence
analysis changes, so scientific drift is detected on the same PR that introduces
new aggregate data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import analysis as primary
import day_influence_sensitivity as influence
import gap_aware_trend_decomposition as gap_aware


def validate_current_findings(data_path: Path) -> None:
    """Raise when the documented influence conclusions no longer hold."""

    records = primary.load_snapshot(data_path)
    results = influence.summarize_influence(records)
    summary = results["leave_one_day_out_summary"]
    if not isinstance(summary, dict):
        raise TypeError("leave_one_day_out_summary must be a dictionary.")

    if not bool(summary["global_interval_excludes_zero_for_all_deletions"]):
        raise RuntimeError(
            "Scientific finding changed: the global systolic HC3 interval no longer "
            "stays below zero under every single-day deletion."
        )
    if not bool(
        summary["episode_level_interval_excludes_zero_for_all_deletions"]
    ):
        raise RuntimeError(
            "Scientific finding changed: the episode-level contrast HC3 interval "
            "no longer stays below zero under every single-day deletion."
        )

    # The stable within-episode finding is about sign and deletion sensitivity,
    # not whether one particular full-data confidence interval happens to straddle
    # zero. As the live source grows, the baseline HC3 interval may legitimately
    # move from crossing zero to lying just below zero without changing the core
    # conclusion: all leave-one-day-out point estimates remain negative, while
    # deletion-specific significance is mixed.
    gap = gap_aware.dominant_internal_gap(records)
    before, after = gap_aware.split_observation_episodes(records, gap)
    baseline_episode = gap_aware._episode_centered_model(before, after)
    baseline_slope = float(
        baseline_episode["common_within_episode_slope_per_30_days"]
    )

    if baseline_slope >= 0.0:
        raise RuntimeError(
            "Scientific finding changed: the baseline within-episode slope point "
            "estimate is no longer negative."
        )

    deletion_min = float(summary["within_episode_slope_min_per_30_days"])
    deletion_max = float(summary["within_episode_slope_max_per_30_days"])
    if deletion_min >= 0.0 or deletion_max >= 0.0:
        raise RuntimeError(
            "Scientific finding changed: at least one leave-one-day-out "
            "within-episode slope point estimate is no longer negative."
        )

    significant = summary["within_episode_significant_negative_deletions"]
    if not isinstance(significant, list):
        raise TypeError(
            "within_episode_significant_negative_deletions must be a list."
        )
    n_observed = int(results["n_observed_days"])
    if not 0 < len(significant) < n_observed:
        raise RuntimeError(
            "Scientific finding changed: within-episode significance is no longer "
            "partially deletion-sensitive."
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/analysis_snapshot.csv"),
        help="Current privacy-safe day-indexed snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the current documented influence findings."""

    args = parse_args()
    validate_current_findings(args.data)
    print("Current influence findings remain valid for this snapshot.")


if __name__ == "__main__":
    main()
