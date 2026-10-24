"""Tests of the official NUTS change tables and the recode chains."""

from __future__ import annotations

from pathlib import Path

import pytest

from eu_wheat_weather.contract import DataError
from eu_wheat_weather.nuts_changes import (
    Change,
    audit_nuts_changes,
    parse_changes,
    read_sheet,
    recode_target,
)

ROOT = Path(__file__).resolve().parents[1]

CHANGES = [
    Change(2006, 2010, "GR11", "EL11", "Code change"),
    Change(2010, 2013, "EL11", "EL51", "Code change"),
    Change(2006, 2010, "ITD5", "", "Boundary shift"),
    Change(2013, 2016, "PL33", "PL72", "recoded"),
    Change(2013, 2016, "FR24", "FRB0", "recoded and relabelled"),
    Change(2016, 2021, "XX11", "XX21", "recoded (boundary change)"),
]


def test_pure_recodes_are_classified_by_their_label() -> None:
    assert [c.pure_recode for c in CHANGES] == [True, True, False, True, True, False]
    assert not Change(2006, 2010, "AA11", "", "Code change").pure_recode   # no new code


def test_chains_follow_every_step_to_nuts_2021() -> None:
    assert recode_target("GR11", 2006, CHANGES) == ("EL51", "")
    assert recode_target("EL11", 2010, CHANGES) == ("EL51", "")
    assert recode_target("PL33", 2013, CHANGES) == ("PL72", "")
    assert recode_target("FR24", 2013, CHANGES) == ("FRB0", "")
    assert recode_target("ITD5", 2006, CHANGES) == (None, "2006-2010: Boundary shift")
    assert recode_target("XX11", 2016, CHANGES) == (None,
                                                    "2016-2021: recoded (boundary change)")
    assert recode_target("AT11", 2006, CHANGES) == ("AT11", "")       # never changed
    assert recode_target("PT19", 2024, CHANGES)[0] is None             # after NUTS 2021


def test_parser_checks_the_header_and_reads_xlsx(tmp_path: Path) -> None:
    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "Changes NUTS-2"
    for row in (["Code 2016", "Code 2021", "Label", "Change", "Explanation"],
                ["HR04", None, "Kontinentalna Hrvatska", "discontinued", None],
                [None, "HR05", "Grad Zagreb", "new", "HR05 = HR041"],
                ["HR041", "HR050", "Grad Zagreb", "recoded", None],          # NUTS-3: skipped
                ["PL33 ", "PL72", "Swietokrzyskie", " Recoded ", None]):
        sheet.append(row)
    path = tmp_path / "t.xlsx"
    book.save(path)
    rows = read_sheet(path, "Changes NUTS-2")
    changes = parse_changes((2016, 2021), rows)
    assert [(c.code_from, c.code_to, c.pure_recode) for c in changes] == [
        ("HR04", "", False), ("", "HR05", False), ("PL33", "PL72", True)]
    with pytest.raises(DataError, match="Unexpected header"):
        parse_changes((2013, 2016), rows)


def test_committed_snapshot_audits_and_contains_the_known_recodes() -> None:
    snapshots = sorted((ROOT / "snapshots").glob("nuts-changes-*"))
    if not snapshots:
        pytest.skip("No committed NUTS change tables.")
    changes = audit_nuts_changes(snapshots[-1])
    for code, version, target in (("ITD3", 2006, "ITH3"), ("EL11", 2010, "EL51"),
                                  ("PL33", 2013, "PL72")):
        assert recode_target(code, version, changes) == (target, "")
    for code, version in (("ITD5", 2006), ("ITE3", 2006), ("SI01", 2010), ("HU10", 2013),
                          ("PL12", 2013), ("HR04", 2016)):
        assert recode_target(code, version, changes)[0] is None
