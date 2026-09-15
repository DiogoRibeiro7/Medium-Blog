"""Coverage of pulse-oximeter capture after pulse observations first appear."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from blood_pressure_missingness.analyses.pulse_oximeter_longitudinal import (
    PulseOximeterDay,
    load_pulse_oximeter_days,
)


@dataclass(frozen=True)
class BloodPressureObservationDay:
    """One privacy-safe blood-pressure calendar day with its reading count."""

    day_index: int
    observed: bool
    n_readings: int


def load_blood_pressure_observation_days(
    path: Path,
) -> list[BloodPressureObservationDay]:
    """Load the public BP calendar with the reading counts needed as denominators."""

    rows: list[BloodPressureObservationDay] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"day_index", "observed", "n_readings"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Blood-pressure snapshot is missing observation-window columns.")

        for raw in reader:
            day_index = int(raw["day_index"])
            observed_raw = int(raw["observed"])
            n_readings = int(raw["n_readings"])

            if day_index < 0:
                raise ValueError("Blood-pressure day_index must be non-negative.")
            if day_index in seen:
                raise ValueError("Blood-pressure day_index values must be unique.")
            seen.add(day_index)
            if observed_raw not in (0, 1):
                raise ValueError("Blood-pressure observed must be 0 or 1.")
            if n_readings < 0:
                raise ValueError("Blood-pressure n_readings must be non-negative.")
            if observed_raw == 1 and n_readings == 0:
                raise ValueError("Observed blood-pressure days must have positive n_readings.")
            if observed_raw == 0 and n_readings != 0:
                raise ValueError("Unobserved blood-pressure days must have zero n_readings.")

            rows.append(
                BloodPressureObservationDay(
                    day_index=day_index,
                    observed=bool(observed_raw),
                    n_readings=n_readings,
                )
            )

    if not rows:
        raise ValueError("Blood-pressure snapshot is empty.")

    rows.sort(key=lambda item: item.day_index)
    expected = list(range(rows[-1].day_index + 1))
    actual = [item.day_index for item in rows]
    if actual != expected:
        raise ValueError("Blood-pressure calendar grid must be contiguous from day zero.")
    return rows


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a rounded rate, or no estimand when the denominator is absent."""

    if denominator == 0:
        return None
    return round(numerator / denominator, 8)


def build_observation_window_coverage(
    pulse_days: Sequence[PulseOximeterDay],
    bp_days: Sequence[BloodPressureObservationDay],
) -> dict[str, Any]:
    """Describe pulse capture from the first observed pulse day onward.

    The first observed pulse day is an observable boundary, not an identified device
    introduction date. Historical days before this boundary are therefore excluded
    from the observation-window denominators rather than labelled as missed pulse
    measurements.
    """

    if not bp_days:
        raise ValueError("Blood-pressure observation days cannot be empty.")

    bp_by_day = {item.day_index: item for item in bp_days}
    if pulse_days:
        for pulse_day in pulse_days:
            bp_day = bp_by_day.get(pulse_day.day_index)
            if bp_day is None:
                raise ValueError("Pulse days must lie inside the blood-pressure calendar.")
            if not bp_day.observed:
                raise ValueError("Pulse days must correspond to observed blood-pressure days.")
            if pulse_day.n_spo2_readings > bp_day.n_readings:
                raise ValueError("Daily SpO2 count cannot exceed blood-pressure readings.")
            if pulse_day.n_bpm_spo2_readings > bp_day.n_readings:
                raise ValueError("Daily bpm_spo2 count cannot exceed blood-pressure readings.")

    if not pulse_days:
        return {
            "window": {
                "defined": False,
                "definition": "first_observed_pulse_day_through_bp_calendar_end",
                "reason": "no_observed_pulse_days",
                "start_day_index": None,
                "end_day_index": bp_days[-1].day_index,
                "calendar_days": 0,
                "blood_pressure_observed_days": 0,
                "blood_pressure_readings": 0,
            },
            "pulse_capture": {
                "pulse_observed_days": 0,
                "pulse_day_capture_of_bp_observed_days": None,
                "spo2_readings": 0,
                "spo2_reading_capture_of_bp_readings": None,
                "bpm_spo2_readings": 0,
                "bpm_spo2_reading_capture_of_bp_readings": None,
                "paired_bpm_readings": 0,
                "paired_bpm_reading_capture_of_bp_readings": None,
            },
            "interpretation": _interpretation(),
        }

    first_pulse_day = min(item.day_index for item in pulse_days)
    last_bp_day = bp_days[-1].day_index
    window_days = [item for item in bp_days if item.day_index >= first_pulse_day]
    observed_window_days = [item for item in window_days if item.observed]

    n_bp_observed_days = len(observed_window_days)
    n_bp_readings = sum(item.n_readings for item in observed_window_days)
    n_pulse_days = len(pulse_days)
    n_spo2 = sum(item.n_spo2_readings for item in pulse_days)
    n_bpm_spo2 = sum(item.n_bpm_spo2_readings for item in pulse_days)
    n_paired = sum(item.n_paired_bpm_readings for item in pulse_days)

    return {
        "window": {
            "defined": True,
            "definition": "first_observed_pulse_day_through_bp_calendar_end",
            "reason": None,
            "start_day_index": first_pulse_day,
            "end_day_index": last_bp_day,
            "calendar_days": len(window_days),
            "blood_pressure_observed_days": n_bp_observed_days,
            "blood_pressure_readings": n_bp_readings,
        },
        "pulse_capture": {
            "pulse_observed_days": n_pulse_days,
            "pulse_day_capture_of_bp_observed_days": _safe_rate(
                n_pulse_days,
                n_bp_observed_days,
            ),
            "spo2_readings": n_spo2,
            "spo2_reading_capture_of_bp_readings": _safe_rate(n_spo2, n_bp_readings),
            "bpm_spo2_readings": n_bpm_spo2,
            "bpm_spo2_reading_capture_of_bp_readings": _safe_rate(
                n_bpm_spo2,
                n_bp_readings,
            ),
            "paired_bpm_readings": n_paired,
            "paired_bpm_reading_capture_of_bp_readings": _safe_rate(
                n_paired,
                n_bp_readings,
            ),
        },
        "interpretation": _interpretation(),
    }


def _interpretation() -> dict[str, bool | str]:
    """Return the fixed interpretation guardrails for this diagnostic."""

    return {
        "scope": "observed_pulse_window_coverage",
        "window_start_is_first_observed_pulse_day": True,
        "device_introduction_date_identified": False,
        "pre_window_absence_treated_as_missing": False,
        "missingness_mechanism_identified": False,
        "clinical_thresholds_applied": False,
        "causal_interpretation": False,
    }


def write_results(path: Path, results: dict[str, Any]) -> None:
    """Write the privacy-safe observation-window coverage artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pulse-data",
        type=Path,
        default=Path("data/pulse_oximeter_daily_snapshot.csv"),
    )
    parser.add_argument(
        "--bp-data",
        type=Path,
        default=Path("data/analysis_snapshot.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pulse_oximeter_observation_window.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Build and write the observation-window coverage diagnostic."""

    args = parse_args()
    pulse_days = load_pulse_oximeter_days(args.pulse_data)
    bp_days = load_blood_pressure_observation_days(args.bp_data)
    results = build_observation_window_coverage(pulse_days, bp_days)
    write_results(args.output, results)
    window = results["window"]
    capture = results["pulse_capture"]
    print(
        "Wrote pulse observation-window coverage: "
        f"defined={window['defined']}, "
        f"{capture['pulse_observed_days']} pulse days, "
        f"{capture['spo2_readings']} SpO2 readings."
    )


if __name__ == "__main__":
    main()
