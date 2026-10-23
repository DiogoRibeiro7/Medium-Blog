"""Eurostat's official NUTS change tables and the chains of pure recodes they record.

A NUTS-2 code joins a NUTS-2021 region when every step between its version and NUTS 2021
records either no change for it or a pure recode (METHODS.md, change log). The four
tables are frozen with SHA-256 hashes; their NUTS-2 rows are re-derived by the audit.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eu_wheat_weather.contract import TARGET_NUTS, DataError
from eu_wheat_weather.crops import sha256
from eu_wheat_weather.downloads import download

BASE = "https://ec.europa.eu/eurostat/documents/345175/629341/"
# (from, to) -> (file, sheet with the NUTS-2 changes)
TABLES: dict[tuple[int, int], tuple[str, str]] = {
    (2006, 2010): ("2006-2010.xls", "Correspondence NUTS-2"),
    (2010, 2013): ("NUTS+2010+-+NUTS+2013.xls", "Correspondence NUTS-2"),
    (2013, 2016): ("NUTS2013-NUTS2016.xlsx", "Correspondence NUTS-2"),
    (2016, 2021): ("NUTS2021.xlsx", "Changes NUTS-2"),
}
PURE_RECODES = frozenset({"code change", "code change, label change", "recoded",
                          "recoded and relabelled"})
CHANGES_NAME = "processed/changes_nuts2.csv"
COLUMNS = ("version_from", "version_to", "code_from", "code_to", "change", "pure_recode")
NUTS2 = re.compile(r"^[A-Z]{2}[0-9A-Z]{2}$")


@dataclass(frozen=True)
class Change:
    version_from: int
    version_to: int
    code_from: str
    code_to: str
    change: str

    @property
    def pure_recode(self) -> bool:
        return bool(self.code_from and self.code_to) and normalise(self.change) in PURE_RECODES


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _cell(value: object) -> str:
    return "" if value is None else str(value).strip()


def read_sheet(path: Path, sheet: str) -> list[list[str]]:
    """The first five columns of every row of one sheet, as text."""
    if path.suffix == ".xls":
        import xlrd

        book = xlrd.open_workbook(path)
        sh = book.sheet_by_name(sheet)
        return [[_cell(sh.cell_value(r, c)) for c in range(min(5, sh.ncols))]
                for r in range(sh.nrows)]
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return [[_cell(v) for v in row[:5]] for row in workbook[sheet].iter_rows(values_only=True)]
    finally:
        workbook.close()


def parse_changes(step: tuple[int, int], rows: Sequence[Sequence[str]]) -> list[Change]:
    """The NUTS-2 rows of one change table (header first: old code, new code, ...)."""
    if not rows or not rows[0][0].startswith(f"Code {step[0]}") or \
            not rows[0][1].startswith(f"Code {step[1]}"):
        raise DataError(f"Unexpected header in the {step} change table: {rows[:1]}.")
    out = []
    for row in rows[1:]:
        row = [*row, "", "", "", ""][:5]
        old, new, change = row[0], row[1], row[3]
        if not (NUTS2.match(old) or NUTS2.match(new)):
            continue
        out.append(Change(step[0], step[1], old if NUTS2.match(old) else "",
                          new if NUTS2.match(new) else "", change))
    return out


def changes_text(changes: Sequence[Change]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    for c in changes:
        writer.writerow([c.version_from, c.version_to, c.code_from, c.code_to, c.change,
                         c.pure_recode])
    return out.getvalue()


def derive(directory: Path) -> list[Change]:
    out: list[Change] = []
    for step, (name, sheet) in TABLES.items():
        out += parse_changes(step, read_sheet(directory / "raw" / name, sheet))
    return out


def fetch_nuts_changes(directory: Path) -> dict[str, Any]:
    files = {}
    for name, _ in TABLES.values():
        sha, _size = download(BASE + name, directory / "raw" / name)
        files[f"raw/{name}"] = sha
    changes = derive(directory)
    table = changes_text(changes).encode("utf-8")
    (directory / CHANGES_NAME).parent.mkdir(parents=True, exist_ok=True)
    (directory / CHANGES_NAME).write_bytes(table)
    files[CHANGES_NAME] = sha256(table)
    manifest = {"source": BASE, "tables": {f"{a}-{b}": f for (a, b), (f, _) in TABLES.items()},
                "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": files, "nuts2_rows": len(changes),
                "pure_recodes": sum(c.pure_recode for c in changes)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    return manifest


def audit_nuts_changes(directory: Path) -> list[Change]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256((directory / name).read_bytes()) != expected:
            raise DataError(f"{name} does not match its recorded SHA-256.")
    changes = derive(directory)
    if changes_text(changes).encode("utf-8") != (directory / CHANGES_NAME).read_bytes():
        raise DataError("The NUTS-2 changes do not re-derive from the raw tables.")
    return changes


def recode_target(code: str, version: int, changes: Sequence[Change]) -> tuple[str | None, str]:
    """Follow ``code`` from ``version`` to NUTS 2021 through pure recodes.

    Returns the NUTS-2021 code, or None and the step that stopped the chain.
    """
    by_step: dict[tuple[int, int], dict[str, Change]] = {}
    for c in changes:
        if c.code_from:
            by_step.setdefault((c.version_from, c.version_to), {})[c.code_from] = c
    current = code
    for step in sorted(TABLES):
        if step[0] < version:
            continue
        change = by_step.get(step, {}).get(current)
        if change is None:
            continue                 # unchanged in this step
        if not change.pure_recode:
            return None, f"{step[0]}-{step[1]}: {change.change}"
        current = change.code_to
    if version > TARGET_NUTS:
        return None, f"version {version} is later than NUTS {TARGET_NUTS}"
    return current, ""

