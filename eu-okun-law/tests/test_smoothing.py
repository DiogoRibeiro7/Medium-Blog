"""Tests of the smoothing study: the month mapping, the exact permutation test, agreement
with the Bayesian persistence statistic, and the regeneration of the committed tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from eu_okun.smoothing import DIRECT, SMOOTHED, _months, permutation_test, write_report

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "smoothing"


def test_months_of_a_quarter() -> None:
    assert _months("2015-Q1") == ["2015-01", "2015-02", "2015-03"]
    assert _months("2015-Q4") == ["2015-10", "2015-11", "2015-12"]


def test_groups_are_disjoint_and_as_registered() -> None:
    assert set(SMOOTHED) == {"DE", "NL", "EE", "PT", "HR", "LU"}
    assert set(DIRECT) == {"AT", "CZ", "EL", "FI", "HU", "IT", "RO", "SE"}
    assert not set(SMOOTHED) & set(DIRECT)


def test_exact_permutation_test() -> None:
    diff, greater, less = permutation_test([3.0, 4.0], [1.0, 2.0])
    assert diff == pytest.approx(2.0)
    # 6 assignments; only the observed one reaches 2.0 or more.
    assert greater == pytest.approx(1 / 6) and less == pytest.approx(1.0)


@pytest.mark.reproduction
def test_committed_tables_reproduce_and_match_the_bayesian_check(tmp_path: Path) -> None:
    """The run regenerates from the frozen snapshots, and its mean autocorrelation is the
    observed statistic of the Bayesian predictive check (protocol, section 4)."""
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    study = write_report(ROOT / "snapshots" / "eurostat-2026-10-15",
                         ROOT / "snapshots" / "eurostat-monthly-2026-10-18",
                         ROOT / "docs" / "review", ROOT / "docs" / "bayes", tmp_path)
    diagnostics = json.loads((ROOT / "docs" / "bayes" / "diagnostics.json").read_text(
        encoding="utf-8"))
    observed = diagnostics["primary"]["predictive_check"]["observed"]["lag1_autocorrelation"]
    assert study.misfit["observed_all"] == pytest.approx(observed, abs=1e-12)
    for name in ("inheritance.csv", "countries.csv", "tests.csv"):
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
