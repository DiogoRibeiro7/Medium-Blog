"""Isolated synthetic-only tests for the official-data source pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from source_pipeline import build_from_paths, read_forbes_aggregate, read_owid_poverty, read_wdi_indicator


def wdi_response(indicator: str, rows: list[tuple[str, int, float]], pages: int = 1) -> list[object]:
    """Construct a synthetic two-part World Bank API response."""
    return [
        {"page": 1, "pages": pages, "per_page": 20_000, "total": len(rows)},
        [
            {
                "countryiso3code": country,
                "date": str(year),
                "value": value,
                "indicator": {"id": indicator},
            }
            for country, year, value in rows
        ],
    ]


class SourcePipelineTests(unittest.TestCase):
    """Verify parsing and validation without representing fixtures as observations."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.owid = self.root / "owid.csv"
        self.gdp = self.root / "gdp.json"
        self.pop = self.root / "pop.json"
        self.forbes = self.root / "forbes.csv"
        self.owid.write_text(
            "Entity,Code,Year,Share of population below $30 a day\n"
            "Example A,AAA,2021,30\nExample A,AAA,2024,22\n"
            "Example B,BBB,2023,76\nExample B,BBB,2025,64\n"
            "Example C,CCC,2025,99\nAggregated World,OWID_WRL,2025,55\n",
            encoding="utf-8",
        )
        self.gdp.write_text(
            json.dumps(
                wdi_response(
                    "NY.GDP.PCAP.PP.KD",
                    [("AAA", 2024, 45000), ("BBB", 2025, 12000), ("CCC", 2023, 3500)],
                )
            ),
            encoding="utf-8",
        )
        self.pop.write_text(
            json.dumps(
                wdi_response(
                    "SP.POP.TOTL",
                    [("AAA", 2025, 10_000_000), ("BBB", 2025, 30_000_000), ("CCC", 2025, 60_000_000)],
                )
            ),
            encoding="utf-8",
        )
        self.forbes.write_text(
            "iso3,country,billionaires,billionaire_method,forbes_year\n"
            "AAA,Example A,10,residence,2026\n"
            "BBB,Example B,2,residence,2026\n"
            "CCC,Example C,0,residence,2026\n",
            encoding="utf-8",
        )

    def build(self) -> dict[str, object]:
        """Execute offline assembly against synthetic country fixtures."""
        return build_from_paths(
            poverty_csv=self.owid,
            gdp_json=self.gdp,
            population_json=self.pop,
            forbes_csv=self.forbes,
            output=self.root / "countries.csv",
            manifest=self.root / "manifest.json",
        )

    def test_owid_observation_year_is_not_survey_date(self) -> None:
        frame = read_owid_poverty(self.owid)
        self.assertEqual(len(frame), 3)
        self.assertEqual(int(frame.loc[frame.iso3 == "AAA", "poverty_year"].iloc[0]), 2024)
        self.assertEqual(float(frame.loc[frame.iso3 == "BBB", "poverty_pct"].iloc[0]), 64)
        self.assertNotIn("survey_year", frame.columns)

    def test_reject_invalid_poverty(self) -> None:
        self.owid.write_text(
            "Entity,Code,Year,poverty\nExample A,AAA,2025,101\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            read_owid_poverty(self.owid)

    def test_reject_duplicate_observation(self) -> None:
        with self.owid.open("a", encoding="utf-8") as f:
            f.write("Example A,AAA,2024,23\n")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            read_owid_poverty(self.owid)

    def test_reject_multiple_indicator_columns(self) -> None:
        self.owid.write_text("Entity,Code,Year,a,b\nX,AAA,2025,1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "one OWID"):
            read_owid_poverty(self.owid)

    def test_wdi_latest_in_window(self) -> None:
        frame = read_wdi_indicator(
            self.gdp, indicator="NY.GDP.PCAP.PP.KD", field="gdp_ppp_per_capita"
        )
        self.assertEqual(len(frame), 3)
        self.assertEqual(int(frame.loc[frame.iso3 == "AAA", "gdp_ppp_per_capita_year"].iloc[0]), 2024)

    def test_reject_incomplete_wdi_pagination(self) -> None:
        self.gdp.write_text(
            json.dumps(wdi_response("NY.GDP.PCAP.PP.KD", [("AAA", 2025, 1000)], pages=3)),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "paginated"):
            read_wdi_indicator(
                self.gdp, indicator="NY.GDP.PCAP.PP.KD", field="gdp_ppp_per_capita"
            )

    def test_reject_wrong_indicator(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected WDI"):
            read_wdi_indicator(self.gdp, indicator="SP.POP.TOTL", field="population")

    def test_reject_mixed_country_attribution(self) -> None:
        with self.forbes.open("a", encoding="utf-8") as f:
            f.write("DDD,Example D,3,citizenship,2026\n")
        with self.assertRaisesRegex(ValueError, "one documented attribution"):
            read_forbes_aggregate(self.forbes)

    def test_reject_wrong_forbes_year(self) -> None:
        frame = pd.read_csv(self.forbes)
        frame.loc[0, "forbes_year"] = 2025
        frame.to_csv(self.forbes, index=False)
        with self.assertRaisesRegex(ValueError, "vintage"):
            read_forbes_aggregate(self.forbes)

    def test_unmatched_not_inferred_zero(self) -> None:
        with self.forbes.open("a", encoding="utf-8") as f:
            f.write("DDD,Example D,6,residence,2026\n")
        report = self.build()
        coverage = report["coverage"]
        self.assertEqual(coverage["n_forbes_country_rows"], 4)
        self.assertEqual(coverage["n_matched_rows"], 3)
        self.assertEqual(coverage["unmatched_iso3"], ["DDD"])

    def test_offline_panel_and_manifest(self) -> None:
        report = self.build()
        self.assertEqual(report["coverage"]["n_matched_rows"], 3)
        self.assertEqual(report["coverage"]["zero_counts_explicitly_listed"], 1)
        self.assertEqual(len(report["panel_sha256"]), 64)
        self.assertEqual(len(report["input_sha256"]), 4)
        panel = pd.read_csv(self.root / "countries.csv")
        self.assertEqual(set(panel.iso3), {"AAA", "BBB", "CCC"})
        self.assertTrue((panel.billionaire_method == "residence").all())
        self.assertEqual(int(panel.loc[panel.iso3 == "AAA", "poverty_year"].iloc[0]), 2024)
        self.assertEqual(int(panel.loc[panel.iso3 == "AAA", "gdp_year"].iloc[0]), 2024)
        self.assertEqual(int(panel.loc[panel.iso3 == "AAA", "population_year"].iloc[0]), 2025)

    def test_missing_input_fails(self) -> None:
        self.owid.unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()

    def test_output_hash_deterministic(self) -> None:
        a = self.build()
        b = self.build()
        self.assertEqual(a["panel_sha256"], b["panel_sha256"])


if __name__ == "__main__":
    unittest.main()
