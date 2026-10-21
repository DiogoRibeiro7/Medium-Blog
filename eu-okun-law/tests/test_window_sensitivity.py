"""Tests of the post-hoc pandemic-window sensitivity: the shift-share margin and the
regeneration of the committed table."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eu_okun.contract import quarter_sequence
from eu_okun.shift_share import origins
from eu_okun.window_sensitivity import run

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "window-sensitivity" / "window_sensitivity.csv"


def test_shift_share_margin_keeps_the_level_at_t_minus_5_clear() -> None:
    quarters = quarter_sequence("2010-Q1", "2026-Q2")
    registered = {quarters[t] for t in origins(quarters)}
    widened = {quarters[t] for t in origins(quarters, margin=1)}
    # 2023-Q1 has t-4 = 2022-Q1, clear of 2020-2021, but its t-5 is 2021-Q4.
    assert "2023-Q1" in registered and "2023-Q1" not in widened
    assert "2023-Q2" in widened and "2019-Q4" in widened
    assert widened < registered


def test_committed_table_regenerates(tmp_path: Path) -> None:
    if not COMMITTED.exists():
        pytest.skip("No committed results.")
    run(ROOT / "snapshots" / "eurostat-2026-10-15",
        ROOT / "snapshots" / "external-shocks-2026-10-16",
        ROOT / "snapshots" / "eurostat-monthly-2026-10-18",
        ROOT / "snapshots" / "eurostat-deflator-2026-10-18",
        ROOT / "docs" / "sign-restrictions", tmp_path,
        shift_share_dir=ROOT / "snapshots" / "shift-share-2026-10-18")
    with COMMITTED.open(encoding="utf-8") as a, (tmp_path / "window_sensitivity.csv").open(
            encoding="utf-8") as b:
        old, new = list(csv.DictReader(a)), list(csv.DictReader(b))
    assert len(old) == len(new)
    for x, y in zip(old, new, strict=True):
        assert {k: x[k] for k in ("design", "margin", "target", "statistic", "n")} == {
            k: y[k] for k in ("design", "margin", "target", "statistic", "n")}
        try:
            assert float(x["value"]) == pytest.approx(float(y["value"]), rel=1e-5, abs=1e-9)
        except ValueError:
            assert x["value"] == y["value"], x
