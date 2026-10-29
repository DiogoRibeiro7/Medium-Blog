"""The committed results: re-solved offline from the committed steps, and regenerated."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iberian_lp.estimates import estimate, load_programmes, render
from iberian_lp.experiment import PROGRAMMES, SPANISH, solve_day
from iberian_lp.programme import Exact, Steps, exact, solve
from iberian_lp.report import load

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "RESULTS.md"

needs_results = pytest.mark.skipif(not (RESULTS / "programmes.csv").exists(),
                                   reason="No results.")


def _exact_only(s: Steps) -> tuple[Exact, float]:
    return exact(s), 0.0


def _by_day() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in load(DERIVED).periods:
        out[row["date"]].append(row)
    return out


@needs_results
def test_every_zone_period_has_its_programmes() -> None:
    rows = load_programmes(RESULTS / "programmes.csv")
    keys = [(r["date"], r["period"], r["zone"]) for r in rows]
    assert len(keys) == len(set(keys)) == len(load(DERIVED).periods)
    for r in rows:
        names = PROGRAMMES if r["zone"] in SPANISH else ("R",)
        assert all(r[f"{p}_lo"] and r[f"{p}_hi"] for p in names)
        assert all(r[f"{p}_lo"] == "" for p in PROGRAMMES if p not in names)


@needs_results
def test_every_interval_is_reproduced_from_the_committed_steps() -> None:
    committed = load_programmes(RESULTS / "programmes.csv")
    checks: dict[str, Any] = {"solves": 0, "objective_gap_max": 0.0}
    again = [row for day, rows in sorted(_by_day().items())
             for row in solve_day(DERIVED, day, rows, checks, _exact_only)]
    assert again == committed


@needs_results
def test_highs_agrees_on_a_day_with_a_split() -> None:
    committed = load_programmes(RESULTS / "programmes.csv")
    day = next(r["date"] for r in committed if r["zone"] == "ES")
    checks: dict[str, Any] = {"solves": 0, "objective_gap_max": 0.0}
    again = list(solve_day(DERIVED, day, _by_day()[day], checks, solve))
    assert again == [r for r in committed if r["date"] == day]


@needs_results
def test_more_supply_never_raises_the_interval() -> None:
    for r in load_programmes(RESULTS / "programmes.csv"):
        if r["zone"] in SPANISH:
            for end in ("lo", "hi"):
                ladder = [float(r[f"{p}_{end}"]) for p in ("R", "M500", "M1000", "M2000")]
                assert ladder == sorted(ladder, reverse=True)


@needs_results
def test_estimates_and_report_regenerate() -> None:
    committed = json.loads((RESULTS / "estimates.json").read_text(encoding="utf-8"))
    solver = json.loads((RESULTS / "solver.json").read_text(encoding="utf-8"))
    again = estimate(load_programmes(RESULTS / "programmes.csv"), load(DERIVED), solver)
    assert json.loads(json.dumps(again)) == committed
    text = REPORT.read_text(encoding="utf-8")
    command = re.search(r"\n(iberian-lp estimates [^\n]+)\n```", text)
    assert command is not None
    assert render(committed, command.group(1)) == text


@needs_results
def test_bootstrap_resamples_are_registered() -> None:
    r = json.loads((RESULTS / "estimates.json").read_text(encoding="utf-8"))
    assert r["bootstrap"] == {"draws": 9_999, "seed": 20261029, "level": 0.95,
                              "simultaneous_level": 0.975}
    assert r["days"] == 48
    assert np.isclose(r["E2"]["all"]["mean"]["level"], 0.975)
    assert np.isclose(r["E3"]["G"]["level"], 0.975)
