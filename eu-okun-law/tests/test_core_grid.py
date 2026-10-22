"""Tests of the participation-free core grid: the grid it reads, the reproduction gate on
the published verdicts, and the regeneration of the committed outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eu_okun.core_grid import core, read_grid, reclassify, render, write_report
from eu_okun.publication import EXPERIMENTS
from eu_okun.review import CORE_LAGS, CORE_SAMPLES

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "docs" / "review" / "grid.csv"
ROBUSTNESS = ROOT / "docs" / "review" / "robustness.csv"
COMMITTED = ROOT / "docs" / "core-grid"


def test_core_specifications_without_participation() -> None:
    lookup = read_grid(GRID)
    free = core(lookup, "PT", (False,))
    assert len(free) == len(CORE_SAMPLES) * len(CORE_LAGS) == 12
    assert not any(r.participation for r in free)
    assert {r.gdp_lags for r in free} == set(CORE_LAGS) and 0 not in CORE_LAGS
    assert len(core(lookup, "PT", (False, True))) == 24


@pytest.mark.reproduction
def test_published_verdicts_are_reproduced_before_reclassifying(tmp_path: Path) -> None:
    rows = reclassify(GRID, ROBUSTNESS)
    with ROBUSTNESS.open(encoding="utf-8") as stream:
        published = {r["geo"]: r for r in csv.DictReader(stream)}
    assert {r.geo: r.published for r in rows} == {g: r["verdict"]
                                                  for g, r in published.items()}
    # A published verdict the grid does not reproduce stops the run (protocol, section 2).
    with ROBUSTNESS.open(encoding="utf-8") as stream:
        altered = list(csv.DictReader(stream))
    altered[0]["verdict"] = "sign_unstable"
    path = tmp_path / "robustness.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(altered[0]))
        writer.writeheader()
        writer.writerows(altered)
    with pytest.raises(ValueError, match="not reproduced"):
        reclassify(GRID, path)


def test_a_changed_verdict_follows_from_the_participation_estimates() -> None:
    lookup = read_grid(GRID)
    for row in reclassify(GRID, ROBUSTNESS):
        if not row.changed:
            continue
        with_control = core(lookup, row.geo, (True,))
        # The 12 estimates without the control are unchanged, so only those with it can
        # have moved the published verdict.
        assert row.verdict == "robust_negative"
        assert any(r.ci_high >= 0 for r in with_control)


@pytest.mark.reproduction
def test_committed_outputs_regenerate(tmp_path: Path) -> None:
    if not (COMMITTED / "verdicts.csv").exists():
        pytest.skip("No committed results.")
    rows = write_report(GRID, ROBUSTNESS, tmp_path)
    for name in ("verdicts.csv", "summary.json"):
        assert (tmp_path / name).read_text(encoding="utf-8") == (
            COMMITTED / name).read_text(encoding="utf-8"), name
    summary = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    assert summary["robust"] == sum(r.verdict == "robust_negative" for r in rows)
    (command,) = next(e for e in EXPERIMENTS if e["key"] == "core_grid")["commands"]
    assert render(rows, command) == (ROOT / "docs" / "CORE_GRID.md").read_text(
        encoding="utf-8")
