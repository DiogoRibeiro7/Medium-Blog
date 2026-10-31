"""Reproduce Porto's 2021 residential-origin census extract from an INE XLS.

The official Porto open-data archive contains an INE Census 2021 workbook.
Its categories describe residents' location on 31 December 2019 relative to
their residence at the census on 19 April 2021. They are NOT annual flows,
origin-destination municipality pairs, or observations of outward movement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "porto_census2021_residential_mobility.csv"
DEFAULT_PROVENANCE = ROOT / "data" / "porto_census2021_residential_provenance.json"
SOURCE_URL = (
    "https://dadosabertos.cm-porto.pt/dataset/71f0710c-9e19-11f1-84ed-6abdb6d5cf34/"
    "resource/e00c8545-474a-4b6e-a87d-cf5159a1cb69/download/"
    "indicadores-diversos-relativos-a-populacao-residente-individuos.zip"
)
SOURCE_SHA256 = "52f447ed1e1e159132ca94c9719d74ab7553f3eefaee4107f391ea0cd54fdf14"
SOURCE_MEMBER_SHA256 = "0d82ac3a057202c3781ec73ebfbb32ee29d3c60f0c86e9b64a3e1b2b8d47def5"
SOURCE_QUESTIONNAIRE = "https://censos.ine.pt/scripts/censos_css_js/quest/PT_Q_Individual_Censos2021_INE.pdf"

FIELD_MAP = {
    "resident_total_2021": (3, "Total"),
    "not_born_2019": (19, "Ainda não tinha nascido"),
    "same_residence": (35, "População que não mudou de residência"),
    "moved_within_same_parish": (51, "População que mudou de residência na mesma freguesia"),
    "domestic_cross_parish_movers": (67, "Provenientes do país"),
    "moved_from_other_parish_same_municipality": (
        83, "População que mudou de freguesia de residência no mesmo município"
    ),
    "moved_from_other_municipality": (99, "População que mudou de município"),
    "moved_from_abroad": (115, "Provenientes do estrangeiro"),
}
FIELDS = (
    "geography_code", "geography", "level", *FIELD_MAP.keys(), "eligible_2019_population",
    "moved_from_other_municipality_pct_eligible",
    "moved_from_abroad_pct_eligible", "all_residence_movers_pct_eligible",
)
REQUIRED_GEOGRAPHIES = (
    "PT", "11", "11A", "1312", "131202", "131203", "131210", "131211",
    "131216", "131217", "131218",
)


class XLSWorksheet(Protocol):
    """Read-only sheet contract, shared with test doubles and xlrd worksheets."""

    name: str
    ncols: int
    nrows: int

    def cell_value(self, rowx: int, colx: int) -> object: ...


@dataclass(frozen=True, slots=True)
class ResidentialOrigin:
    """One complete Census 2021 residence-versus-2019 categorical margin."""

    geography_code: str
    geography: str
    level: str
    resident_total_2021: int
    not_born_2019: int
    same_residence: int
    moved_within_same_parish: int
    domestic_cross_parish_movers: int
    moved_from_other_parish_same_municipality: int
    moved_from_other_municipality: int
    moved_from_abroad: int

    @property
    def eligible_2019_population(self) -> int:
        """2021 residents old enough to have a residence on 31 Dec 2019."""
        return self.resident_total_2021 - self.not_born_2019

    @property
    def all_movers(self) -> int:
        return (
            self.moved_within_same_parish
            + self.moved_from_other_parish_same_municipality
            + self.moved_from_other_municipality
            + self.moved_from_abroad
        )

    def verify(self) -> None:
        """Assert both exact census category partitions and valid geography."""
        values = [getattr(self, field) for field in FIELD_MAP]
        if any(not isinstance(n, int) or n < 0 for n in values):
            raise ValueError(f"Negative or non-integer Census count for {self.geography}")
        if self.eligible_2019_population <= 0:
            raise ValueError(f"No eligible Census denominator for {self.geography}")
        if self.domestic_cross_parish_movers != (
            self.moved_from_other_parish_same_municipality + self.moved_from_other_municipality
        ):
            raise ValueError(f"Domestic origin categories do not reconcile for {self.geography}")
        if self.resident_total_2021 != (
            self.not_born_2019
            + self.same_residence
            + self.moved_within_same_parish
            + self.domestic_cross_parish_movers
            + self.moved_from_abroad
        ):
            raise ValueError(f"Residential origin categories do not sum to residents for {self.geography}")

    def as_row(self) -> dict[str, str | int]:
        """A stable typed CSV row with rates in percentage points, not fractions."""
        denominator = self.eligible_2019_population
        row: dict[str, str | int] = {
            "geography_code": self.geography_code,
            "geography": self.geography,
            "level": self.level,
        }
        row.update({field: getattr(self, field) for field in FIELD_MAP})
        row.update({
            "eligible_2019_population": denominator,
            "moved_from_other_municipality_pct_eligible": f"{100 * self.moved_from_other_municipality / denominator:.6f}",
            "moved_from_abroad_pct_eligible": f"{100 * self.moved_from_abroad / denominator:.6f}",
            "all_residence_movers_pct_eligible": f"{100 * self.all_movers / denominator:.6f}",
        })
        return row


def verify_porto_reconciliation(records: Sequence[ResidentialOrigin]) -> None:
    """Ensure seven parish counts add to Porto for every mutually defined field."""
    index = {item.geography_code: item for item in records}
    if len(records) != len(REQUIRED_GEOGRAPHIES) or tuple(index) != REQUIRED_GEOGRAPHIES:
        raise ValueError("Unexpected geographic scope or order in Census source")
    if len([item for item in records if item.level == "parish"]) != 7:
        raise ValueError("Porto must have precisely seven 2021 parish units")
    for field in FIELD_MAP:
        aggregated = sum(getattr(index[code], field) for code in REQUIRED_GEOGRAPHIES[4:])
        if aggregated != getattr(index["1312"], field):
            raise ValueError(f"Porto parish reconciliation failed for {field}")


def parse_source_sheet(sheet: XLSWorksheet) -> list[ResidentialOrigin]:
    """Parse audited sheet positions, rejecting a silently changed export."""
    # The XLS is a Portuguese INE spreadsheet export with two header levels.
    # Validate its labels so fixed positional parsing cannot silently drift.
    if sheet.name != "Quadro-parte1" or sheet.ncols < 131 or sheet.nrows < 24:
        raise ValueError("Unexpected INE source worksheet dimensions")
    if str(sheet.cell_value(13, 0)) != "2021":
        raise ValueError("The source does not refer to Census 2021")
    for _, (column, label) in FIELD_MAP.items():
        actual = str(sheet.cell_value(9, column)).strip()
        if actual != label:
            raise ValueError(f"Source category at column {column} changed: {actual!r}")
        if str(sheet.cell_value(11, column)).strip() != "Total":
            raise ValueError(f"Only all-birthplace totals can enter the extract: {column}")

    records: list[ResidentialOrigin] = []
    for row_index, expected_code in zip(range(13, 24), REQUIRED_GEOGRAPHIES):
        code = str(sheet.cell_value(row_index, 2)).strip()
        name = str(sheet.cell_value(row_index, 1)).strip()
        if code != expected_code or not name:
            raise ValueError(f"Geography mismatch in source row {row_index}")
        counts: dict[str, int] = {}
        for field, (column, _) in FIELD_MAP.items():
            raw = sheet.cell_value(row_index, column)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"Unpublished/suppressed count in {name}: {field}")
            if raw < 0 or not float(raw).is_integer():
                raise ValueError(f"Invalid Census count in {name}: {field}")
            counts[field] = int(raw)
        level = "parish" if len(code) == 6 else ("municipality" if code == "1312" else "aggregate")
        record = ResidentialOrigin(code, name, level, **counts)
        record.verify()
        records.append(record)
    verify_porto_reconciliation(records)
    return records


def extract(archive_path: Path, *, output: Path = DEFAULT_CSV,
            provenance: Path = DEFAULT_PROVENANCE) -> list[ResidentialOrigin]:
    """Verify the exact official-portal ZIP and export deterministic small CSV."""
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Official Porto Census archive SHA-256 mismatch")

    import xlrd  # Extraction-only dependency; tests use pure synthetic sheet objects.

    with zipfile.ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist() if "proveni" in name.lower() and name.lower().endswith(".xls")]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one provenance/nativity INE workbook")
        member = archive.read(candidates[0])
        if hashlib.sha256(member).hexdigest() != SOURCE_MEMBER_SHA256:
            raise ValueError("Source member XLS SHA-256 mismatch")
        book = xlrd.open_workbook(file_contents=member, on_demand=True)
        try:
            meta = book.sheet_by_name("MetaInfo")
            if "Local de proveniência (Migrações relativamente a 31/12/2019)" not in str(meta.cell_value(5, 1)):
                raise ValueError("Unexpected residential origin indicator definition")
            if str(meta.cell_value(7, 1)) != "INE, Recenseamento da população e habitação - Censos 2021":
                raise ValueError("Incorrect census source attribution")
            records = parse_source_sheet(book.sheet_by_name("Quadro-parte1"))
        finally:
            book.release_resources()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.as_row() for row in records)
    record = {
        "publisher": "INE Census 2021, redistributed by Município do Porto open-data portal",
        "source_url": SOURCE_URL,
        "source_archive_sha256": SOURCE_SHA256,
        "source_member_sha256": SOURCE_MEMBER_SHA256,
        "source_workbook_extracted": "2023-01-04",
        "source_workbook_last_updated": "2022-11-23",
        "portal_resource_published": "2026-09-04",
        "source_questionnaire": SOURCE_QUESTIONNAIRE,
        "source_question": 10,
        "reference_start": "2019-12-31",
        "census_reference": "2021-04-19",
        "geographies": list(REQUIRED_GEOGRAPHIES),
        "derived_csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "row_count": len(records),
        "semantic_limits": [
            "current 2021 residents grouped by their broad 2019 residence-location category",
            "in-movers from another municipality are NOT identified by origin municipality",
            "does not measure Porto resident outflows or municipality-to-municipality migration pairs",
            "not an annual rate: 31 Dec 2019 to 19 Apr 2021 is more than one year",
            "individuals, not households; no rents, reasons for moving or individual finances",
        ],
        "license": "CCZero as specified by the Porto open-data portal (credit INE)",
    }
    provenance.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args(argv)
    records = extract(args.source_zip, output=args.output, provenance=args.provenance)
    print(f"Extracted {len(records)} geographic residential-origin margins")


if __name__ == "__main__":
    main()
