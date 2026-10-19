"""Tests of the measurement-noise and finance study: the weighted statistics, the
non-financial growth formula, the block resample, the comparison with the median, the
signature of transitory noise on synthetic data, and the regeneration of the committed
tables from the frozen snapshots."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from eu_okun.gdp_measurement import (
    FOCUS,
    block_counts,
    comparison,
    measures,
    two_lag_sum,
    weighted_corr,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshots" / "gdp-measurement-2026-10-19"
EUROSTAT = ROOT / "snapshots" / "eurostat-2026-10-15"
COMMITTED = ROOT / "docs" / "gdp-measurement"


def test_weighted_corr_matches_duplicated_rows() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20), rng.normal(size=20)
    w = rng.integers(0, 3, size=20).astype(float)
    duplicated = np.repeat(np.arange(20), w.astype(int))
    expected = np.corrcoef(a[duplicated], b[duplicated])[0, 1]
    assert weighted_corr(a, b, w) == pytest.approx(expected)
    assert weighted_corr(a, b, np.ones(20)) == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_two_lag_sum_recovers_the_sum() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3))
    y = 0.3 + x @ np.array([0.5, 0.2, -0.1])
    data = np.column_stack((y, x))
    assert two_lag_sum(data, np.ones(40)) == pytest.approx(0.6)
    assert math.isnan(two_lag_sum(data[:4], np.ones(4)))


def test_block_counts_cover_the_sample() -> None:
    counts = block_counts(68, 4, np.random.default_rng(2))
    assert counts.sum() == 68 and counts.shape == (68,)


def test_comparison_excludes_the_focus_countries() -> None:
    values = {"LU": 1.0, "IE": 100.0, "MT": -100.0, "AT": 0.0, "BE": 0.2, "DE": 0.4}
    assert comparison(values, "LU") == pytest.approx(0.8)
    assert set(FOCUS) == {"LU", "IE", "MT"}


def test_non_financial_growth_and_noise_signature() -> None:
    quarters = [f"20{10 + i // 4:02d}-Q{i % 4 + 1}" for i in range(40)]
    rng = np.random.default_rng(3)
    growth_n = 0.5 + rng.normal(0, 0.5, 40)
    growth_k = 1.0 + rng.normal(0, 3.0, 40)
    n = 100 * np.exp(np.cumsum(growth_n) / 100)
    k = 20 * np.exp(np.cumsum(growth_k) / 100)
    noise = np.exp(rng.normal(0, 0.02, 40))         # transitory level noise
    panel = {("LU", q): {"va_clv": float(n[i] + k[i]), "va_clv_k": float(k[i]),
                         "va_cp": float(n[i] + k[i]), "va_cp_k": float(k[i]),
                         "hours": float(n[i]), "hours_k": 1.0} for i, q in enumerate(quarters)}
    m = measures(panel, {}, "LU", quarters)
    exact = 100 * np.diff(np.log(n))
    assert np.allclose(m["g_n"][1:], exact, atol=0.05)
    noisy = {key: {**value, "va_clv": value["va_clv"] * float(noise[i])}
             for i, (key, value) in enumerate(panel.items())}
    m_noisy = measures(noisy, {}, "LU", quarters)
    rho = np.corrcoef(m_noisy["g"][2:], m_noisy["g"][1:-1])[0, 1]
    assert rho < -0.2                                # level noise: negative autocorrelation


def test_committed_tables_reproduce(tmp_path: Path) -> None:
    """The registered run regenerates from the frozen snapshots (validation gate)."""
    from eu_okun.gdp_measurement import write_report

    if not (COMMITTED / "predictions.csv").exists():
        pytest.skip("No committed results.")
    write_report(SNAPSHOT, EUROSTAT, tmp_path)
    for name in ("predictions.csv", "statistics.csv", "descriptive.csv"):
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
