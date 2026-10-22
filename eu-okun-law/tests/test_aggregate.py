"""Tests of the EU aggregate: the labour-force weights, the data check, the snapshot audit,
the claims quoted in the text, and the regeneration of the committed tables."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import pytest

from eu_okun.aggregate import data_check, labour_force, weights, write_report
from eu_okun.aggregate_data import EU, audit_eu_aggregate
from eu_okun.contract import EU27, EurostatDataError
from eu_okun.heterogeneity import quarters_for

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshots" / "eu-aggregate-2026-10-21"
COMMITTED = ROOT / "docs" / "aggregate"


def synthetic(sizes: dict[str, float]) -> dict[tuple[str, str], dict[str, float | None]]:
    """Every country has a 10% rate; unemployed persons are 10% of its labour force. The EU
    rate is the labour-force-weighted rate."""
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    for t, q in enumerate(quarters_for()):
        total = 0.0
        for i, geo in enumerate(EU27):
            rate = 5.0 + 0.2 * i + math.sin(0.3 * t + i)
            out[(geo, q)] = {"unemployed_ths": sizes[geo] * rate / 100, "rate": rate,
                             "gdp": 0.4}
            total += sizes[geo] * rate
        out[(EU, q)] = {"unemployed_ths": None, "rate": total / sum(sizes.values()),
                        "gdp": 0.4}
    return out


def test_labour_force_and_weights() -> None:
    sizes = {geo: 100.0 + 10 * i for i, geo in enumerate(EU27)}
    data = synthetic(sizes)
    assert labour_force(data, "DE", "2015-Q1") == pytest.approx(sizes["DE"])
    w = weights(data, quarters_for())
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["SE"] == pytest.approx(sizes["SE"] / sum(sizes.values()))


def test_data_check_matches_a_consistent_aggregate() -> None:
    check = data_check(synthetic({geo: 50.0 + i for i, geo in enumerate(EU27)}),
                       quarters_for())
    assert check["max_abs_difference"] < 1e-12


def test_snapshot_audit_detects_tampering(tmp_path: Path) -> None:
    audit_eu_aggregate(SNAPSHOT)
    copy = tmp_path / "snapshot"
    shutil.copytree(SNAPSHOT, copy)
    raw = copy / "raw" / "rate.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(EurostatDataError, match="SHA-256"):
        audit_eu_aggregate(copy)


def test_text_claims_match_the_committed_results() -> None:
    """The manuscript and article say neither registered difference is decided."""
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    with (COMMITTED / "tests.csv").open(encoding="utf-8") as stream:
        assert {r["verdict"] for r in csv.DictReader(stream)} == {"not determined"}


@pytest.mark.reproduction
def test_committed_outputs_regenerate(tmp_path: Path) -> None:
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    write_report(ROOT / "snapshots" / "eurostat-2026-10-15", SNAPSHOT,
                 ROOT / "docs" / "heterogeneity" / "elasticities.csv",
                 ROOT / "docs" / "cce" / "countries.csv", tmp_path)
    for name in ("weights.csv", "tests.csv", "sensitivity.csv"):
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
    assert committed["verdicts"] == regenerated["verdicts"]
