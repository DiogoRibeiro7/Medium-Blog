"""Tests of the CCE mean-group panel: the estimator on synthetic data with a common
factor, the mean-group statistic, and the regeneration of the committed tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eu_okun.asymmetry import Series
from eu_okun.cce import PRIMARY, fit, mean_group_t, write_report
from eu_okun.contract import EU27
from eu_okun.heterogeneity import quarters_for

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "cce"


def synthetic(seed: int = 7) -> tuple[dict[str, Series], float]:
    """A common factor f moves GDP growth in every country and, separately, unemployment:
    g = 0.4 + gamma_i f + e, du = theta_i g + lambda_i f + u. OLS on g alone is biased;
    CCE removes f through the cross-country averages."""
    rng = np.random.default_rng(seed)
    quarters = quarters_for()
    f = rng.normal(0.0, 1.0, len(quarters))
    thetas = []
    out = {}
    for i, geo in enumerate(EU27):
        theta = -0.2 - 0.01 * i
        thetas.append(theta)
        g = 0.4 + (0.8 + 0.02 * i) * f + rng.normal(0.0, 0.7, len(quarters))
        du = theta * g + (0.3 + 0.01 * i) * f + rng.normal(0.0, 0.05, len(quarters))
        du[:1] = np.nan
        out[geo] = Series(g, du)
    return out, float(np.mean(thetas))


def test_cce_removes_a_common_factor_that_biases_ols() -> None:
    series, truth = synthetic()
    f = fit(series, quarters_for(), PRIMARY, replicates=20)
    assert float(f.mean("cce")[0]) == pytest.approx(truth, abs=0.02)
    assert abs(float(f.mean("ols")[0]) - truth) > 0.1


def test_mean_group_statistic() -> None:
    series, _ = synthetic(seed=8)
    f = fit(series, quarters_for(), PRIMARY, replicates=5)
    estimate, se, p = mean_group_t(f)
    theta = np.asarray([f.cce[g][0] for g in EU27])
    assert estimate == pytest.approx(theta.mean())
    assert se == pytest.approx(theta.std(ddof=1) / math.sqrt(len(theta)))
    assert 0 <= p < 1e-6


def test_text_claims_match_the_committed_results() -> None:
    """The manuscript says both registered tests are supported and that both results hold in
    every sensitivity setting."""
    if not (COMMITTED / "sensitivity.csv").exists():
        pytest.skip("No committed results.")
    with (COMMITTED / "tests.csv").open(encoding="utf-8") as stream:
        assert {r["verdict"] for r in csv.DictReader(stream)} == {"supported"}
    with (COMMITTED / "sensitivity.csv").open(encoding="utf-8") as stream:
        assert {r["verdict"] for r in csv.DictReader(stream)} == {"supported"}


def test_committed_outputs_regenerate(tmp_path: Path) -> None:
    if not (COMMITTED / "tests.csv").exists():
        pytest.skip("No committed results.")
    write_report(ROOT / "snapshots" / "eurostat-2026-10-15",
                 ROOT / "docs" / "heterogeneity" / "elasticities.csv",
                 ROOT / "docs" / "bayes" / "country_estimates.csv", tmp_path)
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
