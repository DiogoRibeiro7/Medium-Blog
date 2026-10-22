"""Tests of the small-sample inference for the classical headline sums: the contrast, the
check against the published estimates, and the regeneration of the committed tables from
the frozen snapshot."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eu_okun.headline_inference import (
    BANDWIDTH,
    check_published,
    headline_designs,
    sum_contrast,
    write_report,
)
from eu_okun.small_sample import hac_design, hac_t, null_sieve_test, segment_starts

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshots" / "eurostat-2026-10-15"
GRID = ROOT / "docs" / "review" / "grid.csv"
COMMITTED = ROOT / "docs" / "headline-inference"


def test_sum_contrast_picks_the_three_gdp_coefficients() -> None:
    assert sum_contrast(4).tolist() == [0.0, 1.0, 1.0, 1.0]


@pytest.mark.reproduction
def test_designs_reproduce_the_published_headline() -> None:
    heads, _ = headline_designs(SNAPSHOT)
    assert [h.geo for h in heads][:3] == ["AT", "BE", "BG"] and len(heads) == 27
    assert len({len(h.y) for h in heads}) == 1          # one common sample
    assert check_published(heads, GRID) < 1e-10


def test_a_mismatch_stops_the_run(tmp_path: Path) -> None:
    heads, _ = headline_designs(SNAPSHOT)
    with GRID.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        if r["geo"] == "AT" and r["sample"] == "pandemic_excluded":
            r["se"] = str(float(r["se"]) * 1.01)
    wrong = tmp_path / "grid.csv"
    with wrong.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="does not reproduce"):
        check_published(heads, wrong)


def test_null_bootstrap_keeps_the_observed_statistic() -> None:
    heads, _ = headline_designs(SNAPSHOT)
    head = heads[0]
    design = hac_design(head.x, head.periods, sum_contrast(head.x.shape[1]), BANDWIDTH)
    t_obs, t_star, sieve = null_sieve_test(design, head.y, segment_starts(head.periods), 50,
                                           np.random.default_rng(1))
    assert t_obs == pytest.approx(float(hac_t(design, head.y)[2][0]))
    assert t_star.shape == (50,) and len(sieve.period_rho) == 2


@pytest.mark.reproduction
def test_committed_tables_reproduce(tmp_path: Path) -> None:
    """The registered run regenerates from the frozen snapshot (validation gate)."""
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed headline-inference results.")
    write_report(SNAPSHOT, GRID, tmp_path)
    for name in ("tests.csv", "size.csv"):
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
    for key in ("bootstrap_negative", "bootstrap_negative_holm", "normal_negative",
                "normal_negative_holm", "lost", "scheme_dependent"):
        assert committed[key] == regenerated[key], key
