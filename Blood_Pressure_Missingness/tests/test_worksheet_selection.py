"""Regression tests for private Google-Sheet worksheet selection."""

from __future__ import annotations

import unittest

from blood_pressure_missingness.data_sources.google_sheets import EXPECTED_COLUMNS
from blood_pressure_missingness.data_sources.worksheet_selection import select_worksheet


class FakeWorksheet:
    """Minimal worksheet double exposing only a title and first row."""

    def __init__(self, title: str, header: list[str]) -> None:
        self.title = title
        self._header = header

    def row_values(self, row: int) -> list[str]:
        if row != 1:
            raise AssertionError("Selector must inspect only the header row.")
        return list(self._header)


class FakeSpreadsheet:
    """Minimal spreadsheet double used by the selector."""

    def __init__(self, worksheets: list[FakeWorksheet]) -> None:
        self._worksheets = worksheets

    def worksheets(self) -> list[FakeWorksheet]:
        return list(self._worksheets)

    def worksheet(self, title: str) -> FakeWorksheet:
        for worksheet in self._worksheets:
            if worksheet.title == title:
                return worksheet
        raise KeyError(title)


class WorksheetSelectionTests(unittest.TestCase):
    """Ensure blank configuration never means 'take the first tab'."""

    def test_unique_complete_schema_is_selected_even_when_not_first(self) -> None:
        spreadsheet = FakeSpreadsheet(
            [
                FakeWorksheet("Old", list(EXPECTED_COLUMNS[:-2])),
                FakeWorksheet("Current", list(EXPECTED_COLUMNS)),
                FakeWorksheet("Notes", ["note", "value"]),
            ]
        )

        selected = select_worksheet(spreadsheet)

        self.assertEqual(selected.title, "Current")

    def test_explicit_worksheet_name_remains_authoritative(self) -> None:
        spreadsheet = FakeSpreadsheet(
            [
                FakeWorksheet("Current", list(EXPECTED_COLUMNS)),
                FakeWorksheet("Other", list(EXPECTED_COLUMNS)),
            ]
        )

        selected = select_worksheet(spreadsheet, "Other")

        self.assertEqual(selected.title, "Other")

    def test_explicit_worksheet_with_old_schema_is_rejected(self) -> None:
        spreadsheet = FakeSpreadsheet(
            [FakeWorksheet("Old", list(EXPECTED_COLUMNS[:-2]))]
        )

        with self.assertRaisesRegex(ValueError, "spo2, bpm_spo2"):
            select_worksheet(spreadsheet, "Old")

    def test_multiple_complete_matches_require_explicit_configuration(self) -> None:
        spreadsheet = FakeSpreadsheet(
            [
                FakeWorksheet("A", list(EXPECTED_COLUMNS)),
                FakeWorksheet("B", list(EXPECTED_COLUMNS)),
            ]
        )

        with self.assertRaisesRegex(ValueError, "Multiple worksheets"):
            select_worksheet(spreadsheet)

    def test_no_complete_match_reports_header_scan(self) -> None:
        spreadsheet = FakeSpreadsheet(
            [
                FakeWorksheet("Old", list(EXPECTED_COLUMNS[:-2])),
                FakeWorksheet("Notes", ["note"]),
            ]
        )

        with self.assertRaisesRegex(ValueError, "No worksheet contains"):
            select_worksheet(spreadsheet)


if __name__ == "__main__":
    unittest.main()
