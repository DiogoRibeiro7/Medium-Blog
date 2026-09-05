"""Refresh privacy-safe blood-pressure aggregates from a private Google Sheet.

The source Sheet URL and Google service-account credentials are supplied through
environment variables (normally GitHub Actions secrets). Raw health measurements
are transformed in memory; only date-free aggregate artifacts are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Sequence

EXPECTED_COLUMNS: tuple[str, ...] = (
    "date",
    "hour",
    "systolic_mmHg",
    "diastolic_mmHg",
    "diff",
    "bpm",
    "Pill",
    "Home",
    "Sleep",
    "Meal",
    "Symptoms",
)
SESSION_GAP_MINUTES = 15
EXCEL_EPOCH = datetime(1899, 12, 30)
SECONDS_PER_DAY = 86_400
MICROSECONDS_PER_SECOND = 1_000_000
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * MICROSECONDS_PER_SECOND


@dataclass(frozen=True)
class Measurement:
    """One validated row-level blood-pressure measurement."""

    day: date
    timestamp: datetime
    systolic: float
    diastolic: float
    pulse_pressure: float
    bpm: float
    pill: str | None
    home: str | None
    sleep: str | None
    meal: str | None
    symptoms: str | None


def _required_env(name: str) -> str:
    """Return a required environment variable without printing its value."""

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set.")
    return value


def fetch_sheet_values() -> list[list[Any]]:
    """Read the private Google Sheet using secret-backed service-account auth."""

    sheet_url = _required_env("BLOOD_PRESSURE_SHEET_URL")
    raw_credentials = _required_env("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON")
    worksheet_name = os.environ.get("BLOOD_PRESSURE_WORKSHEET", "").strip()

    try:
        credentials = json.loads(raw_credentials)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON is not valid JSON."
        ) from exc
    if not isinstance(credentials, dict):
        raise RuntimeError("Google service-account credentials must be a JSON object.")

    import gspread

    client = gspread.service_account_from_dict(credentials)
    spreadsheet = client.open_by_url(sheet_url)
    worksheet = (
        spreadsheet.worksheet(worksheet_name)
        if worksheet_name
        else spreadsheet.get_worksheet(0)
    )
    if worksheet is None:
        raise RuntimeError("The Google Sheet does not contain a worksheet.")

    values = worksheet.get_all_values(value_render_option="UNFORMATTED_VALUE")
    if not values:
        raise ValueError("The Google Sheet is empty.")
    return values


def _is_blank(value: Any) -> bool:
    """Return whether a source cell should be treated as empty."""

    return value is None or (isinstance(value, str) and not value.strip())


def _optional_text(value: Any) -> str | None:
    """Normalize optional text without inventing a negative category."""

    return None if _is_blank(value) else str(value).strip()


def _as_float(value: Any, field: str) -> float:
    """Parse a required finite numeric source value."""

    if isinstance(value, bool) or _is_blank(value):
        raise ValueError(f"{field} is missing or non-numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    return parsed


def _excel_serial_to_date(value: float) -> date:
    """Convert a finite Excel/Google-Sheets serial day to a date."""

    if not math.isfinite(value):
        raise ValueError("date serial must be finite.")
    try:
        return (EXCEL_EPOCH + timedelta(days=value)).date()
    except OverflowError as exc:
        raise ValueError("date serial is outside the supported range.") from exc


def _parse_source_date(value: Any) -> tuple[date, str | None]:
    """Parse one source date and report the legacy repair category, if any.

    In the first legacy block, intended July dates such as 4 July were parsed by
    Excel as 7 April. Those affected 2026 values have parsed day 7 and month
    4..12, so swapping day and month restores 4 July through 12 July.

    Two later rows are stored as text in YYYY/DD/MM form (for example
    ``2026/14/08``), so they are explicitly interpreted as year/day/month.
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = _excel_serial_to_date(float(value))
        if parsed.year == 2026 and parsed.day == 7 and 4 <= parsed.month <= 12:
            return date(parsed.year, parsed.day, parsed.month), "excel"
        return parsed, None

    text = str(value).strip()
    ydm = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", text)
    if ydm:
        year, day_value, month_value = map(int, ydm.groups())
        return date(year, month_value, day_value), "text"

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date(), None
        except ValueError:
            continue
    raise ValueError("Unsupported date representation in source sheet.")


def _parse_source_time(value: Any) -> time:
    """Parse a Sheet serial time in ``[0, 1)`` or a conventional clock string."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        serial = _as_float(value, "hour")
        if not 0.0 <= serial < 1.0:
            raise ValueError("numeric hour serial must be in the interval [0, 1).")

        total_microseconds = int(round(serial * MICROSECONDS_PER_DAY))
        if not 0 <= total_microseconds < MICROSECONDS_PER_DAY:
            raise ValueError("numeric hour serial rounds outside a valid clock day.")

        hour_value, remainder = divmod(
            total_microseconds, 3_600 * MICROSECONDS_PER_SECOND
        )
        minute_value, remainder = divmod(
            remainder, 60 * MICROSECONDS_PER_SECOND
        )
        second_value, microsecond_value = divmod(
            remainder, MICROSECONDS_PER_SECOND
        )
        return time(
            hour_value,
            minute_value,
            second_value,
            microsecond_value,
        )

    text = str(value).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise ValueError("Unsupported time representation in source sheet.")


def _normalize_matrix(values: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Pad source rows and validate the expected leading schema."""

    if not values:
        raise ValueError("No source rows were supplied.")
    header = [str(value).strip() for value in values[0]]
    if tuple(header[: len(EXPECTED_COLUMNS)]) != EXPECTED_COLUMNS:
        raise ValueError(
            "Google Sheet columns do not match the expected blood-pressure schema."
        )

    width = len(EXPECTED_COLUMNS)
    normalized: list[list[Any]] = [list(header[:width])]
    for row in values[1:]:
        padded = list(row[:width]) + [None] * max(0, width - len(row))
        normalized.append(padded[:width])
    return normalized


def parse_measurements(
    values: Sequence[Sequence[Any]],
) -> tuple[list[Measurement], dict[str, Any]]:
    """Validate raw sheet rows and build aggregate source-audit metadata."""

    matrix = _normalize_matrix(values)
    measurements: list[Measurement] = []
    blank_placeholders = 0
    summary_rows = 0
    excel_repairs = 0
    text_repairs = 0
    pulse_mismatches = 0
    reported_pulse_mean: float | None = None

    context_missing: Counter[str] = Counter()
    context_fields = {
        "Pill": 6,
        "Home": 7,
        "Sleep": 8,
        "Meal": 9,
        "Symptoms": 10,
    }

    for row_number, row in enumerate(matrix[1:], start=2):
        date_cell = row[0]
        if isinstance(date_cell, str) and date_cell.strip().lower() == "average":
            summary_rows += 1
            if not _is_blank(row[4]):
                reported_pulse_mean = _as_float(row[4], "summary diff")
            continue

        systolic_cell = row[2]
        diastolic_cell = row[3]
        if _is_blank(systolic_cell) and _is_blank(diastolic_cell):
            if _is_blank(row[4]):
                continue
            placeholder_diff = _as_float(row[4], "placeholder diff")
            if placeholder_diff != 0.0:
                raise ValueError(
                    f"Row {row_number} has no pressure values but a non-zero diff."
                )
            blank_placeholders += 1
            continue
        if _is_blank(systolic_cell) != _is_blank(diastolic_cell):
            raise ValueError(
                f"Row {row_number} contains only one blood-pressure component."
            )

        parsed_day, repair_kind = _parse_source_date(date_cell)
        if repair_kind == "excel":
            excel_repairs += 1
        elif repair_kind == "text":
            text_repairs += 1

        parsed_time = _parse_source_time(row[1])
        systolic = _as_float(systolic_cell, "systolic_mmHg")
        diastolic = _as_float(diastolic_cell, "diastolic_mmHg")
        source_pulse_pressure = _as_float(row[4], "diff")
        derived_pulse_pressure = systolic - diastolic
        bpm = _as_float(row[5], "bpm")

        if derived_pulse_pressure <= 0.0:
            raise ValueError(
                f"Row {row_number}: systolic must exceed diastolic."
            )

        if not math.isclose(
            derived_pulse_pressure,
            source_pulse_pressure,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            pulse_mismatches += 1

        for field, index in context_fields.items():
            if _is_blank(row[index]):
                context_missing[field] += 1

        measurements.append(
            Measurement(
                day=parsed_day,
                timestamp=datetime.combine(parsed_day, parsed_time),
                systolic=systolic,
                diastolic=diastolic,
                pulse_pressure=derived_pulse_pressure,
                bpm=bpm,
                pill=_optional_text(row[6]),
                home=_optional_text(row[7]),
                sleep=_optional_text(row[8]),
                meal=_optional_text(row[9]),
                symptoms=_optional_text(row[10]),
            )
        )

    if not measurements:
        raise ValueError("No valid blood-pressure measurements were found.")
    measurements.sort(key=lambda item: item.timestamp)

    cleaned_pulse_mean = statistics.fmean(
        measurement.pulse_pressure for measurement in measurements
    )
    if cleaned_pulse_mean <= 0.0:
        raise ValueError("Cleaned pulse-pressure mean must be positive.")

    if reported_pulse_mean is None:
        denominator = len(measurements) + blank_placeholders
        reported_pulse_mean = (
            sum(item.pulse_pressure for item in measurements) / denominator
            if denominator
            else cleaned_pulse_mean
        )

    context_payload: dict[str, dict[str, float | int]] = {}
    for field in context_fields:
        missing = int(context_missing[field])
        context_payload[field] = {
            "missing": missing,
            "rate": round(missing / len(measurements), 8),
        }

    audit: dict[str, Any] = {
        "source_rows_excluding_header": len(matrix) - 1,
        "valid_measurements": len(measurements),
        "blank_placeholder_rows": blank_placeholders,
        "spreadsheet_summary_rows": summary_rows,
        "date_repairs": {
            "excel_day_month_inversion_measurements": excel_repairs,
            "text_day_month_inversion_measurements": text_repairs,
        },
        "derived_field_check": {
            "pulse_pressure_mismatches_on_valid_rows": pulse_mismatches,
            "pulse_pressure_output_policy": "derived_from_systolic_and_diastolic",
            "reported_spreadsheet_mean_pulse_pressure_mmHg": round(
                reported_pulse_mean, 8
            ),
            "cleaned_mean_pulse_pressure_mmHg": round(cleaned_pulse_mean, 8),
            "relative_bias_percent": round(
                100.0
                * (reported_pulse_mean - cleaned_pulse_mean)
                / cleaned_pulse_mean,
                8,
            ),
        },
        "context_missingness_on_valid_measurements": context_payload,
    }
    return measurements, audit


def sessionize(
    measurements: Sequence[Measurement],
    gap_minutes: int = SESSION_GAP_MINUTES,
) -> list[list[Measurement]]:
    """Group consecutive same-day readings no more than ``gap_minutes`` apart."""

    if gap_minutes <= 0:
        raise ValueError("gap_minutes must be positive.")

    sessions: list[list[Measurement]] = []
    current: list[Measurement] = []
    for measurement in sorted(measurements, key=lambda item: item.timestamp):
        if not current:
            current = [measurement]
            continue
        previous = current[-1]
        gap = (measurement.timestamp - previous.timestamp).total_seconds() / 60.0
        if measurement.day == previous.day and gap <= gap_minutes:
            current.append(measurement)
        else:
            sessions.append(current)
            current = [measurement]
    if current:
        sessions.append(current)
    return sessions


def build_snapshot(
    measurements: Sequence[Measurement],
    sessions: Sequence[Sequence[Measurement]],
) -> list[dict[str, Any]]:
    """Build one date-free aggregate row for every relative calendar day."""

    by_day: defaultdict[date, list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        by_day[measurement.day].append(measurement)
    session_count_by_day: Counter[date] = Counter(
        session[0].day for session in sessions if session
    )

    first_day = min(by_day)
    last_day = max(by_day)
    snapshot: list[dict[str, Any]] = []
    for day_index in range((last_day - first_day).days + 1):
        current_day = first_day + timedelta(days=day_index)
        rows = by_day.get(current_day, [])
        if not rows:
            snapshot.append(
                {
                    "day_index": day_index,
                    "observed": 0,
                    "n_readings": 0,
                    "n_sessions": 0,
                    "mean_systolic_mmHg": None,
                    "mean_diastolic_mmHg": None,
                    "mean_pulse_pressure_mmHg": None,
                    "mean_bpm": None,
                }
            )
            continue

        snapshot.append(
            {
                "day_index": day_index,
                "observed": 1,
                "n_readings": len(rows),
                "n_sessions": int(session_count_by_day[current_day]),
                "mean_systolic_mmHg": round(
                    statistics.fmean(row.systolic for row in rows), 8
                ),
                "mean_diastolic_mmHg": round(
                    statistics.fmean(row.diastolic for row in rows), 8
                ),
                "mean_pulse_pressure_mmHg": round(
                    statistics.fmean(row.pulse_pressure for row in rows), 8
                ),
                "mean_bpm": round(statistics.fmean(row.bpm for row in rows), 8),
            }
        )
    return snapshot


def _write_snapshot(path: Path, snapshot: Sequence[dict[str, Any]]) -> None:
    """Write only the date-free aggregate calendar grid to disk."""

    fieldnames = [
        "day_index",
        "observed",
        "n_readings",
        "n_sessions",
        "mean_systolic_mmHg",
        "mean_diastolic_mmHg",
        "mean_pulse_pressure_mmHg",
        "mean_bpm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(snapshot)


def refresh(values: Sequence[Sequence[Any]], data_dir: Path) -> dict[str, Any]:
    """Transform private source rows into the two public aggregate artifacts."""

    measurements, audit = parse_measurements(values)
    sessions = sessionize(measurements)
    size_counts = Counter(len(session) for session in sessions)
    audit["sessionization"] = {
        "gap_threshold_minutes": SESSION_GAP_MINUTES,
        "n_sessions": len(sessions),
        "session_size_counts": {
            str(key): value for key, value in sorted(size_counts.items())
        },
    }
    snapshot = build_snapshot(measurements, sessions)

    _write_snapshot(data_dir / "analysis_snapshot.csv", snapshot)
    with (data_dir / "source_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return audit


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory receiving privacy-safe aggregate outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch the private Google Sheet and refresh public aggregate artifacts."""

    args = parse_args()
    values = fetch_sheet_values()
    audit = refresh(values, args.data_dir)
    sessionization = audit["sessionization"]
    if not isinstance(sessionization, dict):
        raise TypeError("sessionization audit must be a dictionary.")
    print(
        "Refreshed privacy-safe snapshot: "
        f"{audit['valid_measurements']} readings, "
        f"{sessionization['n_sessions']} sessions."
    )


if __name__ == "__main__":
    main()
