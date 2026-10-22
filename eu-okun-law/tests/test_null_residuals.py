"""Tests of the post-hoc null-residuals check: the false-discovery-rate adjustments, the
null-imposed bootstrap with unrestricted residuals, and the regeneration of the committed
tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eu_okun.null_residuals import bh_adjust, by_adjust, sieve_test, write_report
from eu_okun.small_sample import hac_design, segment_starts

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "null-residuals"


def test_benjamini_hochberg_and_yekutieli() -> None:
    p = [0.01, 0.04, 0.03, 0.20]
    # Sorted 0.01, 0.03, 0.04, 0.20: BH = min over k >= rank of p_k * 4 / k.
    assert bh_adjust(p) == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20])
    factor = 1 + 1 / 2 + 1 / 3 + 1 / 4
    assert by_adjust(p) == pytest.approx([min(1.0, factor * x) for x in bh_adjust(p)])


def test_bootstrap_imposes_the_null_on_the_mean() -> None:
    """A clearly negative sum is rejected, while the bootstrap t* is centred on zero because
    the simulated outcomes keep the restricted (null) mean."""
    rng = np.random.default_rng(0)
    periods = [f"{y}-Q{q}" for y in range(2010, 2020) for q in range(1, 5)]
    g = rng.normal(0.5, 1.0, (len(periods), 3))
    x = np.column_stack([np.ones(len(periods)), g])
    y = x @ np.array([0.1, -0.3, -0.2, -0.1]) + rng.normal(0, 0.2, len(periods))
    contrast = np.zeros(4)
    contrast[1:4] = 1.0
    design = hac_design(x, periods, contrast, 4)
    t_obs, t_star = sieve_test(design, y, segment_starts(periods), 2000,
                               np.random.default_rng(1))
    assert t_obs < -3
    assert abs(float(np.median(t_star))) < 0.2       # centred on the null


@pytest.mark.reproduction
def test_committed_outputs_regenerate(tmp_path: Path) -> None:
    if not (COMMITTED / "summary.json").exists():
        pytest.skip("No committed results.")
    summary = write_report(ROOT / "snapshots" / "eurostat-2026-10-15",
                           ROOT / "docs" / "headline-inference", ROOT / "docs" / "small-sample",
                           tmp_path)
    committed = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    for study in ("headline", "changes"):
        for errors in ("registered", "unrestricted"):
            assert committed[study][errors] == summary[study][errors], (study, errors)
    for name in ("headline.csv", "changes.csv"):
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
