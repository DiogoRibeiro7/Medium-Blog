"""Resolve the private Google-Sheet worksheet used by blood-pressure refreshes.

When ``BLOOD_PRESSURE_WORKSHEET`` is configured it remains authoritative. When it
is blank, the selector inspects worksheet headers only and requires exactly one
tab to contain the complete canonical blood-pressure schema.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

from blood_pressure_missingness.data_sources.google_sheets import EXPECTED_COLUMNS


class WorksheetLike(Protocol):
    """Minimal worksheet interface required for schema selection."""

    title: str

    def row_values(self, row: int) -> Sequence[Any]: ...


class SpreadsheetLike(Protocol):
    """Minimal spreadsheet interface required for schema selection."""

    def worksheet(self, title: str) -> WorksheetLike: ...

    def worksheets(self) -> list[WorksheetLike]: ...


def _required_env(name: str) -> str:
    """Return a required environment variable without printing its value."""

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set.")
    return value


def _header(worksheet: WorksheetLike) -> tuple[str, ...]:
    """Return one worksheet header with surrounding whitespace removed."""

    return tuple(str(value).strip() for value in worksheet.row_values(1))


def _missing_columns(worksheet: WorksheetLike) -> tuple[str, ...]:
    """Return canonical columns absent from a worksheet header."""

    header = _header(worksheet)
    return tuple(name for name in EXPECTED_COLUMNS if name not in header)


def select_worksheet(
    spreadsheet: SpreadsheetLike,
    configured_name: str = "",
) -> WorksheetLike:
    """Select the unique worksheet containing the canonical source schema.

    An explicit worksheet name is authoritative and is still schema-validated.
    Without one, every worksheet header is inspected and exactly one complete
    match is required. The function never chooses a partial match heuristically.
    """

    configured_name = configured_name.strip()
    if configured_name:
        worksheet = spreadsheet.worksheet(configured_name)
        missing = _missing_columns(worksheet)
        if missing:
            raise ValueError(
                f"Configured worksheet {configured_name!r} is missing required "
                f"columns: {', '.join(missing)}."
            )
        return worksheet

    worksheets = spreadsheet.worksheets()
    if not worksheets:
        raise RuntimeError("The Google Sheet does not contain a worksheet.")

    matches = [worksheet for worksheet in worksheets if not _missing_columns(worksheet)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        titles = ", ".join(repr(worksheet.title) for worksheet in matches)
        raise ValueError(
            "Multiple worksheets contain the complete blood-pressure schema: "
            f"{titles}. Set BLOOD_PRESSURE_WORKSHEET explicitly."
        )

    summaries = "; ".join(
        f"{worksheet.title!r} missing {', '.join(_missing_columns(worksheet))}"
        for worksheet in worksheets
    )
    raise ValueError(
        "No worksheet contains the complete expected blood-pressure schema. "
        f"Header-only scan: {summaries}."
    )


def resolve_private_worksheet_title() -> str:
    """Authenticate and return the validated worksheet title without reading rows."""

    sheet_url = _required_env("BLOOD_PRESSURE_SHEET_URL")
    raw_credentials = _required_env("GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON")
    configured_name = os.environ.get("BLOOD_PRESSURE_WORKSHEET", "").strip()

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
    return select_worksheet(spreadsheet, configured_name).title


def write_github_env(path: Path, worksheet_title: str) -> None:
    """Persist the resolved worksheet title for later GitHub Actions steps."""

    if "\n" in worksheet_title or "\r" in worksheet_title:
        raise ValueError("Worksheet titles containing newlines are unsupported.")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"BLOOD_PRESSURE_WORKSHEET={worksheet_title}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github-env",
        type=Path,
        required=True,
        help="GitHub Actions environment file receiving BLOOD_PRESSURE_WORKSHEET.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve the worksheet and export it without printing private source data."""

    args = parse_args()
    title = resolve_private_worksheet_title()
    write_github_env(args.github_env, title)
    print("Resolved a worksheet containing the complete blood-pressure schema.")


if __name__ == "__main__":
    main()
