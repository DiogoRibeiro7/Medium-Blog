"""The committed derived tables: the registered days, provenance, invariants, the report."""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pytest

from iberian_lp.contract import DAYS_PER_MONTH, FIRST_DAY, KINDS, LAST_DAY
from iberian_lp.days import draw_days
from iberian_lp.report import TOLERANCE_MW, demand, load, render, supply

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
REPORT = ROOT / "docs" / "DATA.md"

needs_data = pytest.mark.skipif(not (DERIVED / "periods.csv").exists(),
                                reason="No derived tables.")


def test_days_are_drawn_as_registered() -> None:
    days = draw_days()
    assert days == draw_days()
    assert all(FIRST_DAY <= d <= LAST_DAY for d in days) and len(set(days)) == len(days)
    per_month = Counter((d.year, d.month) for d in days)
    assert set(per_month.values()) == {DAYS_PER_MONTH} and len(per_month) == 12


@needs_data
def test_manifest_lists_every_file_of_every_drawn_day() -> None:
    d = load(DERIVED)
    assert d.days == [x.isoformat() for x in draw_days()]
    files = {(x["day"], x["kind"]): x for x in d.manifest["files"]}
    assert set(files) == {(day, kind) for day in d.days for kind in KINDS}
    for (day, kind), x in files.items():
        assert re.fullmatch(rf"{kind}_{day.replace('-', '')}\.\d+", x["file"])
        assert re.fullmatch(r"[0-9a-f]{64}", x["sha256"]) and x["bytes"] > 0


@needs_data
def test_periods_are_complete_and_balanced() -> None:
    d = load(DERIVED)
    zones: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in d.periods:
        zones[(r["date"], r["period"])].add(r["zone"])
        assert abs(supply(r) - demand(r)) <= TOLERANCE_MW + 1e-9
    for day in d.days:
        numbers = sorted(int(p) for (dd, p) in zones if dd == day)
        assert numbers == list(range(1, len(numbers) + 1))
        assert len(numbers) == 96 or date.fromisoformat(day).month in (3, 10)
    assert all(z in ({"MI"}, {"ES", "PT"}) for z in zones.values())


@needs_data
def test_step_tables_sum_to_the_period_totals() -> None:
    d = load(DERIVED)
    totals = {(r["date"], r["period"], r["zone"]): r for r in d.periods}
    for day in d.days:
        sums: dict[tuple[str, str, str], float] = defaultdict(float)
        count: Counter[tuple[str, str]] = Counter()
        with gzip.open(DERIVED / "steps" / f"{day.replace('-', '')}.csv.gz", "rt",
                       encoding="utf-8") as stream:
            for s in csv.DictReader(stream):
                column = "offered_buy" if s["side"] == "C" else f"offered_sell_{s['type']}"
                sums[(s["period"], s["zone"], column)] += float(s["mw"])
                count[(s["period"], s["zone"])] += 1
        for (period, zone), n in count.items():
            r = totals[(day, period, zone)]
            assert n == int(r["steps_aggregated"])
            for column in ("offered_buy", "offered_sell_S", "offered_sell_C01",
                           "offered_sell_C02", "offered_sell_C04"):
                assert abs(sums[(period, zone, column)] - float(r[column])) < 0.05


@needs_data
def test_data_report_regenerates() -> None:
    text = REPORT.read_text(encoding="utf-8")
    command = re.search(r"\n(iberian-lp data [^\n]+)\n```", text)
    assert command is not None
    assert render(load(DERIVED), command.group(1)) == text


@needs_data
def test_manifest_records_the_source() -> None:
    m = json.loads((DERIVED / "manifest.json").read_text(encoding="utf-8"))
    assert m["source"].startswith("OMIE") and "omie.es" in m["source"]
