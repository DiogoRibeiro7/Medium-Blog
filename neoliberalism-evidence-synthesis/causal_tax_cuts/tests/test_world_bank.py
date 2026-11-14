"""Offline schema, provenance and missing-data tests for official WDI ingestion."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from causal_tax_cuts.world_bank import (
    ALL_COLUMNS,
    CORE_COLUMNS,
    INDICATORS,
    PanelRow,
    SeriesAudit,
    coverage_audit,
    download_indicator,
    indicator_url,
    parse_wdi_response,
    read_outcome_panel,
    write_results,
)


def api_observation(
    *, country: str = "USA", year: int = 1970, value: float | None = 2.5
) -> dict[str, object]:
    """Generate a synthetic row matching the documented World Bank JSON schema."""
    return {
        "indicator": {"id": INDICATORS[0].code, "value": INDICATORS[0].name},
        "country": {"id": "US", "value": "United States"},
        "countryiso3code": country,
        "date": str(year),
        "value": value,
        "unit": "",
        "obs_status": "",
    }


def api_document(rows: list[dict[str, object]], **metadata: int) -> bytes:
    """Wrap observations in the official two-element paginated JSON envelope."""
    page = {"page": 1, "pages": 1, "per_page": 10000, "total": len(rows)}
    page.update(metadata)
    return json.dumps([page, rows]).encode("utf-8")


class FakeResponse(io.BytesIO):
    """A context-managed byte stream with an HTTP content-type header."""

    def __init__(self, content: bytes, content_type: str = "application/json") -> None:
        super().__init__(content)
        self.headers = {"Content-Type": content_type}


class WorldBankSourceTests(unittest.TestCase):
    """Reject malformed values and incorrect scope before using public data."""

    def test_url_identifies_wdi_source_and_full_paper_sample(self) -> None:
        url = indicator_url(INDICATORS[0])
        self.assertIn("/v2/country/", url)
        self.assertIn("/indicator/NY.GDP.PCAP.KD.ZG?", url)
        self.assertIn("source=2", url)
        self.assertIn("date=1965:2015", url)
        self.assertIn("USA", url)
        self.assertNotIn("PRT", url)
        with self.assertRaisesRegex(ValueError, "Invalid WDI"):
            indicator_url(INDICATORS[0], countries=frozenset({"PRT"}))

    def test_null_stays_missing_but_observed_zero_is_kept(self) -> None:
        rows = [
            api_observation(year=1970, value=None),
            api_observation(year=1971, value=0.0),
            api_observation(year=1972, value=-1.25),
        ]
        values, n_rows = parse_wdi_response(
            api_document(rows),
            indicator=INDICATORS[0],
            countries=frozenset({"USA"}),
            start_year=1970,
            end_year=1972,
        )
        self.assertEqual(n_rows, 3)
        self.assertNotIn(("USA", 1970), values)
        self.assertEqual(values[("USA", 1971)], 0.0)
        self.assertEqual(values[("USA", 1972)], -1.25)

    def test_duplicate_scope_and_indicator_errors_are_rejected(self) -> None:
        row = api_observation()
        cases = (
            ([row, row], "Duplicated"),
            ([api_observation(country="PRT")], "Unexpected"),
            ([api_observation(year=2020)], "Unexpected"),
            ([dict(row, indicator={"id": "OTHER"})], "Wrong WDI"),
            ([api_observation(value=float("inf"))], "Non-finite"),
        )
        for rows, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                parse_wdi_response(
                    api_document(rows),
                    indicator=INDICATORS[0],
                    countries=frozenset({"USA"}),
                    start_year=1965,
                    end_year=2015,
                )
        with self.assertRaisesRegex(TypeError, "Non-numeric"):
            parse_wdi_response(
                api_document([dict(row, value="2.5")]),
                indicator=INDICATORS[0],
                countries=frozenset({"USA"}),
            )

    def test_truncated_pagination_and_api_error_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "paginated"):
            parse_wdi_response(
                api_document([api_observation()], pages=2),
                indicator=INDICATORS[0],
                countries=frozenset({"USA"}),
            )
        with self.assertRaisesRegex(ValueError, "paginated"):
            parse_wdi_response(
                api_document([api_observation()], total=2),
                indicator=INDICATORS[0],
                countries=frozenset({"USA"}),
            )
        with self.assertRaisesRegex(ValueError, "envelope"):
            parse_wdi_response(
                b'{"message": "invalid request"}',
                indicator=INDICATORS[0],
                countries=frozenset({"USA"}),
            )

    def test_download_hash_and_missing_years_are_audited(self) -> None:
        raw = api_document(
            [
                api_observation(year=1970, value=3.0),
                api_observation(year=1971, value=None),
            ]
        )
        with patch(
            "causal_tax_cuts.world_bank.urlopen", return_value=FakeResponse(raw)
        ):
            values, audit = download_indicator(
                INDICATORS[0],
                countries=frozenset({"USA"}),
                start_year=1970,
                end_year=1972,
            )
        self.assertEqual(audit.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(audit.n_api_rows, 2)
        self.assertEqual(audit.n_observed_values, 1)
        self.assertEqual(audit.n_missing_values, 2)
        self.assertEqual(audit.missing_by_country, {"USA": 2})
        self.assertEqual(values, {("USA", 1970): 3.0})

    def test_download_never_accepts_html(self) -> None:
        with (
            patch(
                "causal_tax_cuts.world_bank.urlopen",
                return_value=FakeResponse(b"<html>login</html>", "text/html"),
            ),
            self.assertRaisesRegex(ValueError, "HTML"),
        ):
            download_indicator(INDICATORS[0], countries=frozenset({"USA"}))


class CoverageAuditTests(unittest.TestCase):
    """Check outcome-cohort retention and distinct lagged/event-year missingness."""

    def test_lag_coverage_does_not_require_event_year_macro_data(self) -> None:
        rows = [
            PanelRow("USA", year, int(year == 1970), 12.0) for year in range(1965, 1971)
        ]
        values = {
            column: {("USA", year): 1.0 for year in range(1965, 1970)}
            for column in ALL_COLUMNS
        }
        # One missing value in a pre-treatment lag prevents complete history.
        values["trade_open_pct_gdp"].pop(("USA", 1967))
        report = coverage_audit(rows, values)
        self.assertEqual(report["n_original_outcome_panel_rows"], 6)
        self.assertEqual(report["n_original_treated_event_rows"], 1)
        self.assertEqual(report["n_original_event_rows_missing_core_at_event_year"], 1)
        self.assertEqual(
            report["n_original_event_rows_with_complete_core_pre_treatment_lags"], 0
        )

        values["trade_open_pct_gdp"][("USA", 1967)] = 5.0
        report = coverage_audit(rows, values)
        self.assertEqual(
            report["n_original_event_rows_with_complete_core_pre_treatment_lags"], 1
        )
        self.assertEqual(report["n_original_event_rows_missing_core_at_event_year"], 1)
        self.assertEqual(report["n_joint_rows_complete_core"], 5)

    def test_outcome_panel_rejects_invalid_exposure_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcome.csv"
            path.write_text(
                "country,year,treated,top1_share_pct\nUSA,1970,1,12\nGBR,1971,0,15\n",
                encoding="utf-8",
            )
            rows = read_outcome_panel(path)
            self.assertEqual(
                [(r.country, r.year) for r in rows], [("GBR", 1971), ("USA", 1970)]
            )
            path.write_text(
                "country,year,treated,top1_share_pct\nUSA,1970,2,12\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid outcome-panel values"):
                read_outcome_panel(path)
            path.write_text(
                "country,year,treated,top1_share_pct\nUSA,1970,1,12\nUSA,1970,1,12\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicated outcome-panel"):
                read_outcome_panel(path)

    def test_output_retains_missing_rows_and_source_provenance(self) -> None:
        values = {column: {("USA", 1970): 0.0} for column in ALL_COLUMNS}
        values["unemployment_pct_labor"] = {}  # Optional series has a gap.
        audits = [
            SeriesAudit(
                indicator.column,
                indicator.code,
                "https://api.worldbank.org/example",
                "abc123",
                60,
                1,
                1,
                0,
                {"USA": 0},
            )
            for indicator in INDICATORS
        ]
        panel = [PanelRow("USA", 1970, 1, 12.5)]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            write_results(
                destination,
                values,
                audits,
                start_year=1970,
                end_year=1971,
                panel=panel,
            )
            with (destination / "world_bank_controls.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                grid = list(csv.DictReader(source))
            self.assertEqual(len(grid), 36)  # 18 countries × 2 years.
            observed = next(
                r for r in grid if r["country"] == "USA" and r["year"] == "1970"
            )
            self.assertEqual(observed[CORE_COLUMNS[0]], "0")
            self.assertEqual(observed["unemployment_pct_labor"], "")
            self.assertTrue((destination / "outcome_panel_with_macro.csv").is_file())
            manifest = json.loads(
                (destination / "world_bank_provenance.json").read_text()
            )
            self.assertEqual(manifest["series"][0]["source_sha256"], "abc123")
            audit = json.loads((destination / "macro_overlap_audit.json").read_text())
            self.assertEqual(audit["n_original_outcome_panel_rows"], 1)
            self.assertEqual(audit["n_joint_rows_complete_core"], 1)
            self.assertEqual(audit["n_joint_rows_complete_all_four"], 0)


if __name__ == "__main__":
    unittest.main()
