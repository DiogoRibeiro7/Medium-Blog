"""Tests of the hours study: the pandemic rows, the stratified resamples, the reproduction
gate, the margin identity, the p-value floor and the regeneration of the committed tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eu_okun.heterogeneity import FIRST, CountryData, Estimate, block_counts, quarters_for
from eu_okun.hours import (
    MEASURES,
    Run,
    Settings,
    check_reproduction,
    pandemic_rows,
    render,
    run,
    stratified_counts,
    write_report,
)
from eu_okun.publication import EXPERIMENTS

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "hours"


def test_pandemic_rows_follow_the_lag_window() -> None:
    quarters = quarters_for()
    first = quarters.index(FIRST)
    for lags, last in ((1, "2022-Q1"), (2, "2022-Q2"), (4, "2022-Q4")):
        mask = pandemic_rows(quarters, lags)
        rows = [quarters[first + i] for i in np.flatnonzero(mask)]
        assert rows[0] == "2020-Q1" and rows[-1] == last
        assert len(rows) == 8 + lags


def test_stratified_resamples_keep_the_pandemic_rows() -> None:
    mask = pandemic_rows(quarters_for(), 2)
    counts = stratified_counts(mask, 4, 200, np.random.default_rng(1))
    assert np.all(counts[:, mask].sum(axis=1) == mask.sum())
    assert np.all(counts[:, ~mask].sum(axis=1) == (~mask).sum())


@pytest.mark.reproduction
def test_reproduction_gate(tmp_path: Path) -> None:
    run = Run(Settings(), {("AT", "h", "excluded"): Estimate(0.5, np.zeros(3), 40),
                           ("AT", "h", "full"): Estimate(0.7, np.zeros(3), 48)},
              "registered", 0.0)
    path = tmp_path / "elasticities.csv"
    for value, ok in (("0.5", True), ("0.6", False)):
        path.write_text(f"geo,measure,elasticity,low,high,n\nAT,h,{value},0,1,40\n"
                        "AT,temp,0.1,0,1,40\n", encoding="utf-8")
        if ok:
            assert check_reproduction(run, path) == 1
        else:
            with pytest.raises(ValueError, match="does not reproduce"):
                check_reproduction(run, path)


def _rows(name: str) -> list[dict[str, str]]:
    with (COMMITTED / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_margins_sum_to_one_in_both_samples() -> None:
    if not (COMMITTED / "countries.csv").exists():
        pytest.skip("No committed results.")
    beta = {(r["geo"], r["measure"], r["sample"]): float(r["elasticity"])
            for r in _rows("countries.csv")}
    checked = 0
    for geo, _, sample in [k for k in beta if k[1] == "hpw"]:
        total = sum(beta[(geo, m, sample)] for m in ("prod", "hpw", "e_dom"))
        assert total == pytest.approx(1.0, abs=1e-9), (geo, sample)
        checked += 1
    assert checked == 40          # 20 countries with persons, two samples


def test_registered_p_values_respect_the_floor() -> None:
    """The flaw recorded in the change log: no registered difference test can reach a
    p-value below the floor, and the floor exceeds 0.05."""
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    summary = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    floor = summary["p_floor"]["primary"]["p_floor"]
    assert floor > 0.05
    for r in _rows("tests.csv"):
        if r["bootstrap"] == "registered" and r["key"] != "P3":
            assert float(r["p_value"]) >= floor - 1e-12, r["key"]


def test_differences_are_exactly_zero_without_a_pandemic_row() -> None:
    """In a resample without a pandemic row the full and excluded regressions use the same
    rows and weights, so their draws must be equal exactly, on every platform."""
    quarters = quarters_for()
    rng = np.random.default_rng(0)
    g = rng.normal(0.5, 1.0, len(quarters))
    y = 0.4 * g + rng.normal(0.0, 0.3, len(quarters))
    data = {"AT": CountryData(g, {m: y for m in MEASURES}, {}, {})}
    r = run(data, quarters, Settings(), replicates=400, seed=3)
    mask = pandemic_rows(quarters, 2)
    counts = block_counts(len(mask), 4, 400, np.random.default_rng(3))
    none = counts[:, mask].sum(axis=1) == 0
    assert none.any() and r.no_pandemic_share == pytest.approx(none.mean())
    full, excl = r.estimates[("AT", "h", "full")], r.estimates[("AT", "h", "excluded")]
    assert np.array_equal(full.draws[none], excl.draws[none])
    assert not np.array_equal(full.draws[~none], excl.draws[~none])


def test_verdicts_quoted_in_the_text() -> None:
    """The manuscript and article describe these verdicts in words."""
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    verdicts = {(r["bootstrap"], r["key"]): r["verdict"] for r in _rows("tests.csv")}
    assert {v for (b, _), v in verdicts.items() if b == "registered"} == {"not determined"}
    assert verdicts[("stratified", "P1")] == verdicts[("stratified", "P2a")] == "supported"
    assert verdicts[("stratified", "P2b")] == verdicts[("stratified", "P3")] == (
        "not determined")


@pytest.mark.reproduction
def test_committed_outputs_regenerate(tmp_path: Path) -> None:
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    study = write_report(ROOT / "snapshots" / "eurostat-2026-10-15",
                         ROOT / "snapshots" / "heterogeneity-2026-10-18",
                         ROOT / "docs" / "heterogeneity" / "elasticities.csv", tmp_path)
    for name in ("countries.csv", "tests.csv", "sensitivity.csv"):
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
                    assert va == pytest.approx(vb, rel=1e-9, abs=1e-12), (name, key)
    (command,) = next(e for e in EXPERIMENTS if e["key"] == "hours")["commands"]
    assert render(study, f"poetry run {command}") == (ROOT / "docs" / "HOURS.md").read_text(
        encoding="utf-8")
