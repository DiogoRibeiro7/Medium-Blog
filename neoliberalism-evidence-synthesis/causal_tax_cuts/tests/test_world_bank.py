"""Offline parsing, checksum and coverage tests for official WDI ZIP archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from causal_tax_cuts.world_bank import (
    ALL_COLUMNS,
    CORE_COLUMNS,
    INDICATORS,
    PanelRow,
    SeriesAudit,
    archive_url,
    coverage_audit,
    download_indicator,
    parse_wdi_archive,
    read_outcome_panel,
    write_results,
)


def archive_bytes(
    *,
    indicator_index: int = 0,
    rows: list[tuple[str, dict[int, str]]] | None = None,
    years: tuple[int, ...] = (1970, 1971, 1972),
    source_name: str = "World Development Indicators",
    reported_indicator_code: str | None = None,
) -> bytes:
    """Create synthetic observations in the World Bank's real CSV ZIP layout."""
    indicator = INDICATORS[indicator_index]
    if rows is None:
        rows = [("USA", {1970: "2.5", 1971: "", 1972: "0"})]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Data Source", source_name, ""])
    writer.writerow([])
    writer.writerow(["Last Updated Date", "2026-11-11", ""])
    writer.writerow([])
    writer.writerow(
        ["Country Name", "Country Code", "Indicator Name", "Indicator Code", *years, ""]
    )
    for country, recorded in rows:
        writer.writerow(
            [
                country,
                country,
                indicator.name,
                reported_indicator_code or indicator.code,
                *(recorded.get(year, "") for year in years),
                "",
            ]
        )
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"API_{indicator.code}_DS2_en_csv_v2_1.csv", output.getvalue().encode()
        )
        archive.writestr("Metadata_Indicator_example.csv", b"metadata only")
    return target.getvalue()


class FakeResponse(io.BytesIO):
    """A context-managed HTTP response with an explicit content-type."""

    def __init__(self, content: bytes, content_type: str = "application/zip") -> None:
        super().__init__(content)
        self.headers = {"Content-Type": content_type}


class WorldBankArchiveTests(unittest.TestCase):
    """Reject wrong series, nonfinite measurements, duplicates and corrupt ZIPs."""

    def test_archive_url_references_exact_official_series(self) -> None:
        url = archive_url(INDICATORS[0])
        self.assertEqual(
            url,
            "https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.KD.ZG?downloadformat=csv",
        )

    def test_null_stays_missing_but_zero_is_retained(self) -> None:
        content = archive_bytes()
        values, member, revised, country_count = parse_wdi_archive(
            content,
            indicator=INDICATORS[0],
            countries=frozenset({"USA"}),
            start_year=1970,
            end_year=1972,
        )
        self.assertNotIn(("USA", 1971), values)
        self.assertEqual(values[("USA", 1970)], 2.5)
        self.assertEqual(values[("USA", 1972)], 0.0)
        self.assertIn("DS2_en_csv", member)
        self.assertEqual(revised, "2026-11-11")
        self.assertEqual(country_count, 1)

    def test_wider_official_archive_is_filtered_to_study_years(self) -> None:
        content = archive_bytes(
            years=(1969, 1970, 1971, 1972, 2025),
            rows=[("USA", {1969: "100", 1970: "2", 2025: "30"})],
        )
        values, _, _, _ = parse_wdi_archive(
            content,
            indicator=INDICATORS[0],
            countries=frozenset({"USA"}),
            start_year=1970,
            end_year=1972,
        )
        self.assertEqual(values, {("USA", 1970): 2.0})

    def test_missing_country_and_duplicate_country_are_rejected(self) -> None:
        for rows, error in (
            ([("PRT", {1970: "5"})], "lacks paper countries"),
            ([("USA", {1970: "5"}), ("USA", {1971: "3"})], "Duplicated"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                parse_wdi_archive(
                    archive_bytes(rows=rows),
                    indicator=INDICATORS[0],
                    countries=frozenset({"USA"}),
                    start_year=1970,
                    end_year=1972,
                )

    def test_wrong_indicator_invalid_value_and_bad_metadata_are_rejected(self) -> None:
        cases = (
            (archive_bytes(reported_indicator_code="WRONG"), "Wrong WDI"),
            (archive_bytes(rows=[("USA", {1970: "nan"})]), "Non-finite"),
            (archive_bytes(rows=[("USA", {1970: "something"})]), "Invalid numeric"),
            (archive_bytes(source_name="World Atlas"), "metadata"),
            (archive_bytes(years=(1970, 1971)), "requested annual window"),
        )
        for content, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                parse_wdi_archive(
                    content,
                    indicator=INDICATORS[0],
                    countries=frozenset({"USA"}),
                    start_year=1970,
                    end_year=1972,
                )

    def test_reject_invalid_zip_and_unsupported_country(self) -> None:
        with self.assertRaisesRegex(ValueError, "ZIP"):
            parse_wdi_archive(
                b"not a zip",
                indicator=INDICATORS[0],
                countries=frozenset({"USA"}),
            )
        with self.assertRaisesRegex(ValueError, "Invalid WDI"):
            parse_wdi_archive(
                archive_bytes(),
                indicator=INDICATORS[0],
                countries=frozenset({"PRT"}),
            )

    def test_download_retains_hash_and_complete_country_coverage(self) -> None:
        content = archive_bytes()
        with patch(
            "causal_tax_cuts.world_bank.urlopen", return_value=FakeResponse(content)
        ):
            values, audit = download_indicator(
                INDICATORS[0],
                countries=frozenset({"USA"}),
                start_year=1970,
                end_year=1972,
            )
        self.assertEqual(audit.source_sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(audit.source_last_updated, "2026-11-11")
        self.assertEqual(audit.n_paper_country_rows, 1)
        self.assertEqual(audit.n_observed_values, 2)
        self.assertEqual(audit.n_missing_values, 1)
        self.assertEqual(audit.missing_by_country, {"USA": 1})
        self.assertEqual(values[("USA", 1972)], 0.0)

    def test_download_retries_only_transient_http_errors(self) -> None:
        content = archive_bytes()
        error = HTTPError(
            "https://api.worldbank.org/example", 502, "Bad Gateway", {}, None
        )
        with (
            patch(
                "causal_tax_cuts.world_bank.urlopen",
                side_effect=[error, FakeResponse(content)],
            ) as mocked,
            patch("causal_tax_cuts.world_bank.time.sleep") as pause,
        ):
            values, _ = download_indicator(
                INDICATORS[0],
                countries=frozenset({"USA"}),
                start_year=1970,
                end_year=1972,
            )
        self.assertEqual(mocked.call_count, 2)
        pause.assert_called_once_with(1)
        self.assertEqual(values[("USA", 1970)], 2.5)

    def test_download_rejects_html(self) -> None:
        with (
            patch(
                "causal_tax_cuts.world_bank.urlopen",
                return_value=FakeResponse(b"<html>login</html>", "text/html"),
            ),
            self.assertRaisesRegex(ValueError, "HTML"),
        ):
            download_indicator(INDICATORS[0], countries=frozenset({"USA"}))


class CoverageAuditTests(unittest.TestCase):
    """Check cohort retention and distinct lagged versus event-year missingness."""

    def test_lag_history_does_not_require_event_year_macro_data(self) -> None:
        rows = [
            PanelRow("USA", year, int(year == 1970), 12.0) for year in range(1965, 1971)
        ]
        values = {
            column: {("USA", year): 1.0 for year in range(1965, 1970)}
            for column in ALL_COLUMNS
        }
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

    def test_output_retains_missing_rows_and_archive_provenance(self) -> None:
        values = {column: {("USA", 1970): 0.0} for column in ALL_COLUMNS}
        values["unemployment_pct_labor"] = {}
        audits = [
            SeriesAudit(
                indicator.column,
                indicator.code,
                "https://api.worldbank.org/example",
                "abc123",
                100,
                "API_example.csv",
                "2026-11-11",
                18,
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
            self.assertEqual(len(grid), 36)
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

    def test_frozen_live_coverage_tables_reconcile(self) -> None:
        """Assert arithmetic of the real source/overlap audit without re-fetching."""
        path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "wdi_macro_coverage_2026-11-14.json"
        )
        frozen = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(frozen["sample"]["countries"], 18)
        self.assertEqual(frozen["sample"]["possible_country_years"], 918)
        for source in frozen["source_archives"].values():
            self.assertEqual(
                source["available_country_years"] + source["missing_country_years"],
                918,
            )
            self.assertEqual(len(source["sha256"]), 64)
        overlap = frozen["outcome_panel_overlap"]
        self.assertEqual(sum(frozen["core_joint_rows_by_country"].values()), 810)
        self.assertEqual(overlap["joint_core_three_series_complete"], 810)
        self.assertEqual(overlap["source_panel_country_years"], 853)
        self.assertEqual(overlap["source_rows_missing_core_three_series"], 853 - 810)
        self.assertLessEqual(
            overlap["event_years_with_five_complete_pre_treatment_core_years"],
            overlap["source_panel_treated_event_years"],
        )


if __name__ == "__main__":
    unittest.main()
