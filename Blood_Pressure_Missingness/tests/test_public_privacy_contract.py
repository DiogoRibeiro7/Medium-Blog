"""Machine-checkable privacy contract for committed public artifacts."""

from __future__ import annotations

import csv
import json
import re
import unittest
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

ALLOWED_SNAPSHOT_COLUMNS = {
    "day_index",
    "observed",
    "n_readings",
    "n_sessions",
    "mean_systolic_mmHg",
    "mean_diastolic_mmHg",
    "mean_pulse_pressure_mmHg",
    "mean_bpm",
}

FORBIDDEN_DIRECT_FIELDS = {
    "date",
    "datetime",
    "timestamp",
    "time",
    "hour",
    "latitude",
    "longitude",
    "coordinates",
    "sheet_url",
    "spreadsheet_id",
}

FORBIDDEN_CONTEXT_FIELDS = {"pill", "home", "sleep", "meal", "symptoms"}
ALLOWED_CONTEXT_AGGREGATE_PARENT = "context_missingness_on_valid_measurements"

FORBIDDEN_TEMPERATURE_SERIES_FIELDS = {
    "temperature_series",
    "temperatures",
    "matched_temperatures",
    "interpolated_temperatures",
    "day_indexed_temperature_series",
    "daily_temperature_series",
}

ISO_CALENDAR_DATE = re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{2}-\d{2}(?!\d)")
SVG_RENDERER_DATE = re.compile(r"<dc:date>.*?</dc:date>", flags=re.DOTALL)


def _normalise_key(value: object) -> str:
    return str(value).strip().lower()


def _walk_json(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Return privacy-contract violations found in one parsed JSON value."""

    violations: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _normalise_key(raw_key)
            child_path = (*path, key)
            dotted = ".".join(child_path)

            if key in FORBIDDEN_DIRECT_FIELDS:
                violations.append(f"forbidden public field: {dotted}")

            if key in FORBIDDEN_TEMPERATURE_SERIES_FIELDS:
                violations.append(f"reconstructable temperature series field: {dotted}")

            if key in FORBIDDEN_CONTEXT_FIELDS:
                if ALLOWED_CONTEXT_AGGREGATE_PARENT not in path:
                    violations.append(f"raw health-context field: {dotted}")

            violations.extend(_walk_json(child, child_path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_walk_json(child, (*path, f"[{index}]")))

    elif isinstance(value, str) and ISO_CALENDAR_DATE.search(value):
        violations.append(
            f"calendar date value at {'.'.join(path) or '<root>'}: {value!r}"
        )

    return violations


class PublicPrivacyContractTests(unittest.TestCase):
    """Ensure committed artifacts preserve the documented privacy boundary."""

    def test_public_snapshot_schema_is_date_free_and_context_free(self) -> None:
        snapshot = DATA_DIR / "analysis_snapshot.csv"
        with snapshot.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(set(reader.fieldnames or ()), ALLOWED_SNAPSHOT_COLUMNS)
            for row_number, row in enumerate(reader, start=2):
                for value in row.values():
                    if value:
                        self.assertIsNone(
                            ISO_CALENDAR_DATE.search(value),
                            f"calendar date leaked in {snapshot.name}:{row_number}: {value!r}",
                        )

    def test_public_json_artifacts_do_not_expose_private_fields_or_dates(self) -> None:
        paths = sorted(DATA_DIR.glob("*.json")) + sorted(FIGURES_DIR.glob("*.json"))
        self.assertTrue(paths, "Expected at least one committed public JSON artifact.")

        for path in paths:
            with self.subTest(path=path.relative_to(PROJECT_DIR)):
                payload = json.loads(path.read_text(encoding="utf-8"))
                violations = _walk_json(payload)
                self.assertEqual(violations, [], "\n".join(violations))

    def test_public_svgs_do_not_embed_health_calendar_dates(self) -> None:
        paths = sorted(FIGURES_DIR.glob("*.svg"))
        self.assertTrue(paths, "Expected at least one committed SVG figure.")

        for path in paths:
            with self.subTest(path=path.relative_to(PROJECT_DIR)):
                text = path.read_text(encoding="utf-8")
                # Matplotlib's dc:date is render provenance, not a measurement date.
                text_without_renderer_metadata = SVG_RENDERER_DATE.sub("", text)
                match = ISO_CALENDAR_DATE.search(text_without_renderer_metadata)
                self.assertIsNone(
                    match,
                    f"calendar date leaked into public SVG {path.name}: {match.group(0) if match else ''}",
                )

    def test_privacy_policy_file_exists(self) -> None:
        policy = PROJECT_DIR / "DATA_PRIVACY.md"
        text = policy.read_text(encoding="utf-8")
        self.assertIn("row-level timestamps", text)
        self.assertIn("day-indexed temperature sequence", text)
        self.assertIn("never committed", text)


if __name__ == "__main__":
    unittest.main()
