"""Tests of the asymmetry study: the estimators on synthetic data with a known asymmetry,
the pooled estimator against a direct least-squares fit, and the regeneration of the
committed tables."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eu_okun.asymmetry import PRIMARY, Series, design, fit, write_report
from eu_okun.contract import EU27
from eu_okun.heterogeneity import quarters_for

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "asymmetry"


def synthetic(asymmetry: float, seed: int = 3, noise: float = 0.1) -> dict[str, Series]:
    """du = -0.3 sum_k 0.5^k g[t-k] + asymmetry/3 sum_k g_neg[t-k] + noise."""
    rng = np.random.default_rng(seed)
    quarters = quarters_for()
    out = {}
    for i, geo in enumerate(EU27):
        g = rng.normal(0.4, 1.0, len(quarters))
        neg = np.minimum(g, 0.0)
        du = np.full(len(quarters), np.nan)
        slope = -0.2 - 0.01 * i
        for t in range(2, len(quarters)):
            du[t] = (slope * (g[t] + 0.5 * g[t - 1] + 0.25 * g[t - 2])
                     + asymmetry / 3 * (neg[t] + neg[t - 1] + neg[t - 2])
                     + noise * rng.normal())
        out[geo] = Series(g, du)
    return out


def test_estimators_recover_a_known_asymmetry() -> None:
    quarters = quarters_for()
    for truth in (-0.2, 0.0):
        # With noise 0.1 the bootstrap sd of C is about 0.016; 0.02 makes it about 0.003.
        f = fit(synthetic(truth, noise=0.02), quarters, PRIMARY, replicates=50)
        assert float(f.pooled_c[0]) == pytest.approx(truth, abs=0.03)
        assert float(f.mean_group[0]) == pytest.approx(truth, abs=0.05)


def test_pooled_estimate_equals_a_direct_fit() -> None:
    quarters = quarters_for()
    series = synthetic(-0.2, seed=5)
    countries = EU27[:4]
    settings = replace(PRIMARY, exclude=tuple(g for g in EU27 if g not in countries))
    f = fit(series, quarters, settings, replicates=5)
    blocks, ys = [], []
    for i, geo in enumerate(countries):
        d = design(series[geo], quarters, settings)
        z, h, y = d.v[:, :4], d.v[:, 4:7], d.v[:, -1]
        own = np.zeros((len(y), 4 * len(countries)))
        own[:, 4 * i: 4 * i + 4] = z
        blocks.append(np.column_stack([own, h]))
        ys.append(y)
    x, y = np.vstack(blocks), np.concatenate(ys)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    assert float(f.pooled_c[0]) == pytest.approx(float(beta[-3:].sum()), abs=1e-10)
    slopes = np.mean([beta[4 * i + 1: 4 * i + 4].sum() for i in range(len(countries))])
    assert float(f.pooled_theta_plus[0]) == pytest.approx(float(slopes), abs=1e-10)


def test_text_claims_match_the_committed_results() -> None:
    """The manuscript and article say: every setting's pooled difference is negative, no
    setting decides either test, and neither registered test is decided."""
    if not (COMMITTED / "sensitivity.csv").exists():
        pytest.skip("No committed results.")
    with (COMMITTED / "sensitivity.csv").open(encoding="utf-8") as stream:
        sens = list(csv.DictReader(stream))
    assert all(float(r["estimate"]) < 0 for r in sens if r["key"] == "P1")
    assert not any(r["verdict"] in ("supported", "contradicted") for r in sens)
    with (COMMITTED / "tests.csv").open(encoding="utf-8") as stream:
        assert {r["verdict"] for r in csv.DictReader(stream)} == {"not determined"}


def test_committed_outputs_regenerate(tmp_path: Path) -> None:
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    write_report(ROOT / "snapshots" / "eurostat-2026-10-15",
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
                    assert va == pytest.approx(vb, rel=1e-7, abs=1e-10), (name, key)
    committed = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    regenerated = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert committed["verdicts"] == regenerated["verdicts"]
