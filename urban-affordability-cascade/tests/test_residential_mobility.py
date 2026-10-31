"""Independent data-accounting and source-schema tests for Porto Census moves."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_residential_mobility import (  # noqa: E402
    DEFAULT_CSV,
    DEFAULT_PROVENANCE,
    FIELD_MAP,
    REQUIRED_GEOGRAPHIES,
    ResidentialOrigin,
    parse_source_sheet,
    verify_porto_reconciliation,
)
from residential_mobility import (  # noqa: E402
    FIGURES,
    RESULTS,
    load_residential,
    parish_summary,
    run,
)


class FakeSheet:
    """Minimal typed test double for the key cells of an INE XLS worksheet."""

    name = "Quadro-parte1"
    ncols = 255
    nrows = 29

    def __init__(self, records: list[ResidentialOrigin]) -> None:
        self.cells: dict[tuple[int, int], object] = {(13, 0): "2021"}
        for field, (column, label) in FIELD_MAP.items():
            self.cells[9, column] = label
            self.cells[11, column] = "Total"
        for row, record in enumerate(records, start=13):
            self.cells[row, 1] = record.geography
            self.cells[row, 2] = record.geography_code
            for field, (col, _) in FIELD_MAP.items():
                self.cells[row, col] = float(getattr(record, field))

    def cell_value(self, rowx: int, colx: int) -> object:
        return self.cells.get((rowx, colx), "")


class SourceIntegrityTests(unittest.TestCase):
    """The official portal is frozen in data; live downloads are not needed by CI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_residential()
        cls.by_code = {item.geography_code: item for item in cls.rows}

    def test_11_geographies_include_exact_seven_porto_parishes(self) -> None:
        self.assertEqual(tuple(self.by_code), REQUIRED_GEOGRAPHIES)
        self.assertEqual(sum(item.level == "parish" for item in self.rows), 7)
        self.assertEqual(self.by_code["1312"].geography, "Porto")

    def test_official_archive_provenance_and_derived_sha(self) -> None:
        meta = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
        self.assertEqual(meta["source_archive_sha256"], "52f447ed1e1e159132ca94c9719d74ab7553f3eefaee4107f391ea0cd54fdf14")
        self.assertEqual(meta["source_member_sha256"], "0d82ac3a057202c3781ec73ebfbb32ee29d3c60f0c86e9b64a3e1b2b8d47def5")
        self.assertEqual(hashlib.sha256(DEFAULT_CSV.read_bytes()).hexdigest(), meta["derived_csv_sha256"])
        self.assertEqual(meta["reference_start"], "2019-12-31")
        self.assertEqual(meta["census_reference"], "2021-04-19")

    def test_2021_counts_and_exact_porto_parish_reconciliation(self) -> None:
        porto, amp = self.by_code["1312"], self.by_code["11A"]
        self.assertEqual(porto.resident_total_2021, 231800)
        self.assertEqual(porto.not_born_2019, 2073)
        self.assertEqual(porto.eligible_2019_population, 229727)
        self.assertEqual(porto.moved_from_other_municipality, 15724)
        self.assertEqual(amp.moved_from_other_municipality, 124618)
        self.assertEqual(porto.moved_from_abroad, 4345)
        self.assertEqual(sum(item.moved_from_other_municipality for item in self.rows if item.level == "parish"), 15724)
        verify_porto_reconciliation(self.rows)

    def test_decomposition_and_2019_eligible_denominator(self) -> None:
        porto = self.by_code["1312"]
        self.assertEqual(porto.domestic_cross_parish_movers, 7471 + 15724)
        self.assertEqual(porto.resident_total_2021, 2073 + 195182 + 7005 + 23195 + 4345)
        self.assertAlmostEqual(100 * 15724 / 229727, 6.844646, places=5)
        self.assertEqual(porto.all_movers, 34545)

    def test_central_union_and_campanha_are_distinct_observations(self) -> None:
        old_center = self.by_code["131217"]
        campanha = self.by_code["131203"]
        self.assertEqual(old_center.moved_from_other_municipality, 3122)
        self.assertAlmostEqual(100 * old_center.moved_from_other_municipality / old_center.eligible_2019_population, 8.404900, places=5)
        self.assertAlmostEqual(100 * campanha.moved_from_other_municipality / campanha.eligible_2019_population, 5.942686, places=5)
        self.assertGreater(old_center.all_movers / old_center.eligible_2019_population,
                           campanha.all_movers / campanha.eligible_2019_population)

    def test_missing_or_repeated_csv_geographies_fail(self) -> None:
        lines = DEFAULT_CSV.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected geographic scope"):
                load_residential(path)
            path.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected geographic scope"):
                load_residential(path)


class XLSParsingTests(unittest.TestCase):
    """The workbook reader validates labelled dimensions, counts and codes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_residential()

    def test_fake_export_roundtrip_matches_frozen_source(self) -> None:
        sheet = FakeSheet(self.records)
        parsed = parse_source_sheet(sheet)
        self.assertEqual(parsed, self.records)

    def test_reject_category_header_drift(self) -> None:
        sheet = FakeSheet(self.records)
        sheet.cells[9, 99] = "Noutro país"
        with self.assertRaisesRegex(ValueError, "Source category"):
            parse_source_sheet(sheet)

    def test_reject_suppression_and_noninteger_excel_values(self) -> None:
        sheet = FakeSheet(self.records)
        sheet.cells[16, 99] = "x"
        with self.assertRaisesRegex(ValueError, "suppressed"):
            parse_source_sheet(sheet)
        sheet.cells[16, 99] = 5.5
        with self.assertRaisesRegex(ValueError, "Invalid Census count"):
            parse_source_sheet(sheet)

    def test_reject_broken_domestic_subtotal(self) -> None:
        sheet = FakeSheet(self.records)
        sheet.cells[16, 67] = 1.0
        with self.assertRaisesRegex(ValueError, "Domestic origin categories"):
            parse_source_sheet(sheet)


class ResidentialDiagnosticTests(unittest.TestCase):
    """New charts and CSV are reproducible, with unambiguous denominators."""

    def test_parish_summary_sorted_and_city_contrasts(self) -> None:
        rows = parish_summary(load_residential())
        shares = [float(row["2019_other_municipality_pct_eligible"]) for row in rows]
        self.assertEqual(len(rows), 7)
        self.assertEqual(shares, sorted(shares))
        self.assertEqual(rows[-1]["geography_code"], "131217")
        self.assertEqual(rows[0]["geography_code"], "131216")

    def test_reproduced_figures_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run(root)
            self.assertEqual(result["porto_other_municipality_in_movers"], 15724)
            self.assertEqual(result["porto_parish_reconciliation"], "exact for all eight source count fields")
            self.assertTrue(any("cannot measure Porto-to-suburb exits" in limit for limit in result["limits"]))
            for name in FIGURES:
                self.assertGreater((root / "figures" / name).stat().st_size, 5000)
            for name in RESULTS:
                self.assertGreater((root / "results" / name).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
