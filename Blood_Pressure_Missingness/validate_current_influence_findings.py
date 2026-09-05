"""Validate current-snapshot influence findings during private-data refresh.

These checks are intentionally *not* ordinary code invariants. They encode the
scientific statements currently documented for the latest committed snapshot.
The secret-backed refresh workflow runs this module immediately after rebuilding
all derived outputs. If new measurements change one of these conclusions, the
refresh stops so the article and documentation can be reviewed before publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import analysis as primary
import day_influence_sensitivity as influence


def validate_current_findings(data_path: Path) -> None:
    """Raise when current documented influence conclusions no longer hold."""

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
