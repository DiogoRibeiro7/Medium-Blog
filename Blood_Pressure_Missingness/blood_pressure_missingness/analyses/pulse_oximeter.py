"""Privacy-safe diagnostics for optional pulse-oximeter measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from blood_pressure_missingness.data_sources import google_sheets as source
from blood_pressure_missingness.data_sources.google_sheets import Measurement

DAILY_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "day_index",
    "n_spo2_readings",
    "mean_spo2_percent",
    "n_bpm_spo2_readings",
    "mean_bpm_spo2",
    "n_paired_bpm_readings",
    "mean_bpm_difference",
)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return descriptive statistics without inventing values for empty samples."""

    count = len(values)
    return {
        "n": count,
        "mean": round(statistics.fmean(values), 8) if values else None,
        "median": round(statistics.median(values), 8) if values else None,
        "sample_sd": round(statistics.stdev(values), 8) if count >= 2 else None,
        "minimum": round(min(values), 8) if values else None,
        "maximum": round(max(values), 8) if values else None,
    }


def build_pulse_oximeter_diagnostics(
    measurements: Sequence[Measurement],
) -> dict[str, Any]:
    """Summarize coverage and cross-device BPM agreement.

    The output is aggregate-only. It contains no timestamps, dates, or row-level
    SpO2/BPM values.
    """

    total = len(measurements)
    if total == 0:
        raise ValueError("At least one measurement is required.")

    spo2_values = [item.spo2 for item in measurements if item.spo2 is not None]
    bpm_spo2_values = [
        item.bpm_spo2 for item in measurements if item.bpm_spo2 is not None
    ]

    both_observed = sum(
        item.spo2 is not None and item.bpm_spo2 is not None for item in measurements
    )
    spo2_only = sum(
        item.spo2 is not None and item.bpm_spo2 is None for item in measurements
    )
    bpm_spo2_only = sum(
        item.spo2 is None and item.bpm_spo2 is not None for item in measurements
    )
    both_missing = total - both_observed - spo2_only - bpm_spo2_only

    paired = [
        (item.bpm, item.bpm_spo2)
        for item in measurements
        if item.bpm_spo2 is not None
    ]
    differences = [bp_bpm - spo2_bpm for bp_bpm, spo2_bpm in paired]

    agreement: dict[str, float | int | str | None] = {
        "observed_pairs": len(paired),
        "difference_definition": "bpm_minus_bpm_spo2",
        "mean_bpm_bp_device": (
            round(statistics.fmean(bp for bp, _ in paired), 8) if paired else None
        ),
        "mean_bpm_spo2_device": (
            round(statistics.fmean(spo2_bpm for _, spo2_bpm in paired), 8)
            if paired
            else None
        ),
        "mean_difference": (
            round(statistics.fmean(differences), 8) if differences else None
        ),
        "median_difference": (
            round(statistics.median(differences), 8) if differences else None
        ),
        "sample_sd_difference": (
            round(statistics.stdev(differences), 8) if len(differences) >= 2 else None
        ),
        "mean_absolute_difference": (
            round(statistics.fmean(abs(value) for value in differences), 8)
            if differences
            else None
        ),
        "rmse": (
            round(math.sqrt(statistics.fmean(value**2 for value in differences)), 8)
            if differences
            else None
        ),
        "minimum_difference": round(min(differences), 8) if differences else None,
        "maximum_difference": round(max(differences), 8) if differences else None,
        "maximum_absolute_difference": (
            round(max(abs(value) for value in differences), 8)
            if differences
            else None
        ),
    }

    return {
        "n_measurements": total,
        "coverage": {
            "spo2_observed": len(spo2_values),
            "spo2_coverage_rate": round(len(spo2_values) / total, 8),
            "bpm_spo2_observed": len(bpm_spo2_values),
            "bpm_spo2_coverage_rate": round(len(bpm_spo2_values) / total, 8),
        },
        "joint_field_completeness": {
            "both_observed": both_observed,
            "spo2_without_bpm_spo2": spo2_only,
            "bpm_spo2_without_spo2": bpm_spo2_only,
            "both_missing": both_missing,
        },
        "spo2_percent": _summary(spo2_values),
        "bpm_spo2": _summary(bpm_spo2_values),
        "paired_bpm_device_agreement": agreement,
        "interpretation": {
            "scope": "descriptive_device_diagnostics",
            "clinical_thresholds_applied": False,
            "row_level_values_persisted": False,
        },
    }


def build_daily_snapshot(
    measurements: Sequence[Measurement],
) -> list[dict[str, float | int | None]]:
    """Build a date-free daily pulse-oximeter aggregate snapshot.

    Only days with at least one pulse-oximeter value are emitted. ``day_index`` is
    relative to the first blood-pressure measurement day, so it can be aligned to
    the established public snapshot without exposing calendar dates.
    """

    if not measurements:
        raise ValueError("At least one measurement is required.")

    first_day = min(item.day for item in measurements)
    by_day: defaultdict[date, list[Measurement]] = defaultdict(list)
    for item in measurements:
        if item.spo2 is not None or item.bpm_spo2 is not None:
            by_day[item.day].append(item)

    snapshot: list[dict[str, float | int | None]] = []
    for current_day in sorted(by_day):
        rows = by_day[current_day]
        spo2 = [item.spo2 for item in rows if item.spo2 is not None]
        bpm_spo2 = [item.bpm_spo2 for item in rows if item.bpm_spo2 is not None]
        bpm_differences = [
            item.bpm - item.bpm_spo2
            for item in rows
            if item.bpm_spo2 is not None
        ]
        snapshot.append(
            {
                "day_index": (current_day - first_day).days,
                "n_spo2_readings": len(spo2),
                "mean_spo2_percent": (
                    round(statistics.fmean(spo2), 8) if spo2 else None
                ),
                "n_bpm_spo2_readings": len(bpm_spo2),
                "mean_bpm_spo2": (
                    round(statistics.fmean(bpm_spo2), 8) if bpm_spo2 else None
                ),
                "n_paired_bpm_readings": len(bpm_differences),
                "mean_bpm_difference": (
                    round(statistics.fmean(bpm_differences), 8)
                    if bpm_differences
                    else None
                ),
            }
        )
    return snapshot


def write_daily_snapshot(
    path: Path,
    snapshot: Sequence[dict[str, float | int | None]],
) -> None:
    """Write the date-free pulse-oximeter daily snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_SNAPSHOT_COLUMNS)
        writer.writeheader()
        writer.writerows(snapshot)


def run_private_source_outputs() -> tuple[dict[str, Any], list[dict[str, float | int | None]]]:
    """Fetch the private Sheet and build aggregate pulse-oximeter outputs."""

    values = source.fetch_sheet_values()
    measurements, _ = source.parse_measurements(values)
    return (
        build_pulse_oximeter_diagnostics(measurements),
        build_daily_snapshot(measurements),
    )


def write_diagnostics(path: Path, diagnostics: dict[str, Any]) -> None:
    """Write one aggregate-only diagnostic JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pulse_oximeter_diagnostics.json"),
        help="Path receiving aggregate pulse-oximeter diagnostics.",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=Path("data/pulse_oximeter_daily_snapshot.csv"),
        help="Path receiving the date-free daily pulse-oximeter snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate aggregate pulse-oximeter artifacts from the private source."""

    args = parse_args()
    diagnostics, daily_snapshot = run_private_source_outputs()
    write_diagnostics(args.output, diagnostics)
    write_daily_snapshot(args.daily_output, daily_snapshot)
    print(
        "Wrote pulse-oximeter diagnostics: "
        f"{diagnostics['coverage']['spo2_observed']} SpO2 readings, "
        f"{diagnostics['paired_bpm_device_agreement']['observed_pairs']} BPM pairs, "
        f"{len(daily_snapshot)} observed pulse-oximeter days."
    )


if __name__ == "__main__":
    main()
