"""The committed derived tables: provenance, invariants and regeneration."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from routing_roads.contract import DAYS, EXTRACT_MD5, EXTRACT_SHA256, PAIRS
from routing_roads.data import draw_days
from routing_roads.report import load, render

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
REPORT = ROOT / "docs" / "DATA.md"

pytestmark = pytest.mark.skipif(not (DERIVED / "manifest.json").exists(),
                                reason="No derived tables.")


def test_the_tables_come_from_the_pinned_extract() -> None:
    e = load(DERIVED).manifest["extract"]
    assert (e["md5"], e["sha256"]) == (EXTRACT_MD5, EXTRACT_SHA256)


def test_matrices_and_points_agree() -> None:
    d = load(DERIVED)
    for pair in PAIRS:
        rows = [p for p in d.points if p["pair"] == pair.key]
        assert [int(p["index"]) for p in rows] == list(range(len(rows)))
        assert rows[0]["role"] == "depot" and rows[0]["municipality"] == pair.city
        towns = [p["municipality"] for p in rows[1:]]
        assert towns == sorted(towns, key=lambda t: t != pair.city)   # city block first
        road, straight = d.road[pair.key], d.straight[pair.key]
        assert road.shape == straight.shape == (len(rows), len(rows))
        assert np.all(np.diag(road) == 0) and np.all(np.diag(straight) == 0)
        assert np.array_equal(straight, straight.T)
        assert np.all(road >= straight - 1)      # a road path is never shorter than the line


def test_days_regenerate_from_their_seed() -> None:
    d = load(DERIVED)
    sizes = {p.key: sum(1 for r in d.points if r["pair"] == p.key) - 1 for p in PAIRS}
    expected = [{k: str(v) for k, v in row.items()} for row in draw_days(sizes)]
    assert d.days == expected
    assert {int(r["day"]) for r in d.days} <= set(range(1, DAYS + 1))


def test_the_report_regenerates() -> None:
    text = REPORT.read_text(encoding="utf-8")
    command = re.search(r"```\n(routing-roads data [^\n]+)\n```", text)
    assert command is not None
    assert render(load(DERIVED), command.group(1)) == text
