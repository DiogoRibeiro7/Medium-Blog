"""Tests of the measurement checks: the revisions, the claims quoted in the text, and the
regeneration of the committed tables."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import pytest

from eu_okun.measurement import revisions, write_report

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "snapshots" / "eurostat-2026-10-15"
LATEST = ROOT / "snapshots" / "eurostat-2026-10-22"
COMMITTED = ROOT / "docs" / "measurement"


def test_revisions_of_a_release_against_itself_are_zero() -> None:
    rows = revisions(FROZEN, FROZEN)
    assert rows and all(r["changed"] == 0 and r["flag_changes"] == 0 for r in rows)


def test_a_revised_cell_is_counted(tmp_path: Path) -> None:
    copy = tmp_path / "release"
    shutil.copytree(FROZEN / "processed", copy / "processed")
    path = copy / "processed" / "eu27_quarterly.csv"
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    target = next(r for r in rows if r["geo"] == "PT" and r["quarter"] == "2015-Q1")
    target["gdp_growth_qoq_pct"] = str(float(target["gdp_growth_qoq_pct"]) + 0.5)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pt = next(r for r in revisions(FROZEN, copy) if r["series"] == "gdp" and r["geo"] == "PT")
    assert pt["changed"] == 1 and pt["max_abs_revision"] == pytest.approx(0.5)


def test_text_claims_match_the_committed_results() -> None:
    """The text says the latest release revised no labour-force cell and changed no count
    or country classification."""
    if not (COMMITTED / "summary.json").exists():
        pytest.skip("No committed results.")
    s = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    assert s["revisions"]["lfs_cells_changed"] == 0
    assert s["counts"]["vintage"] == s["counts"]["headline"]
    assert all(not v for v in s["changes"]["vintage"].values())


def test_committed_outputs_regenerate(tmp_path: Path) -> None:
    if not (COMMITTED / "summary.json").exists():
        pytest.skip("No committed results.")
    write_report(FROZEN, LATEST, ROOT / "docs" / "review" / "grid.csv", tmp_path)
    for name in ("breaks.csv", "vintage.csv", "revisions.csv"):
        with (COMMITTED / name).open(encoding="utf-8") as a, (tmp_path / name).open(
                encoding="utf-8") as b:
            old, new = list(csv.DictReader(a)), list(csv.DictReader(b))
        assert len(old) == len(new), name
        for x, y in zip(old, new, strict=True):
            for key, value in x.items():
                try:
                    va, vb = float(value), float(y[key])
                except ValueError:
                    assert value == y[key], (name, key)
                    continue
                if math.isnan(va):
                    assert math.isnan(vb), (name, key)
                else:
                    assert va == pytest.approx(vb, rel=1e-7, abs=1e-10), (name, key)
    committed = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    regenerated = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert committed["counts"] == regenerated["counts"]
    assert committed["changes"] == regenerated["changes"]
