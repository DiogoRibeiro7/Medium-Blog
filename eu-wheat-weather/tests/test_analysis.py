"""Tests of the estimands on synthetic data, and regeneration of the committed report."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from eu_wheat_weather.analysis import (
    BASE,
    Obs,
    decide,
    dry_season_effect,
    out_of_sample,
    render,
)
from eu_wheat_weather.estimation import Fit

ROOT = Path(__file__).resolve().parents[1]


def synthetic_obs(noise: float = 0.0, seed: int = 0) -> list[Obs]:
    """Log yields generated exactly by M2: region level, region trend and weather."""
    rng = np.random.default_rng(seed)
    beta = {"gdd": 0.02, "edd": -0.05, "p": 0.3, "p2": -0.06}
    out = []
    for c in range(3):
        for r in range(4):
            geo, level, slope = f"C{c}R{r}", rng.normal(1.8, 0.2), rng.normal(0.01, 0.005)
            for year in range(2000, 2025):
                p = rng.uniform(1.0, 4.0)
                x = {"gdd": rng.uniform(15, 25), "edd": rng.uniform(0, 3), "p": p, "p2": p * p}
                y = level + slope * (year - 2012) + sum(beta[k] * x[k] for k in BASE)
                out.append(Obs(geo, f"C{c}", year, y + rng.normal(0, noise),
                               rng.uniform(5, 50), x))
    return out


def test_out_of_sample_is_exact_without_noise() -> None:
    rows, regions = out_of_sample(synthetic_obs())
    assert all(r["predicted"] == pytest.approx(r["observed"]) for r in regions)
    assert [r["year"] for r in rows] == [2003, 2018, 2022]
    for r in rows:
        assert r["correlation"] == pytest.approx(1.0)
        assert r["variance_explained"] == pytest.approx(1.0)
        assert r["eu_yield_predicted_pct"] == pytest.approx(r["eu_yield_observed_pct"])


def test_dry_season_effect_matches_a_hand_computation() -> None:
    obs = synthetic_obs(noise=0.1)
    rng = np.random.default_rng(1)
    n = len(obs)
    x = np.column_stack([[o.x[k] for o in obs] for k in BASE])
    beta = np.array([0.0, 0.0, 0.3, -0.06])
    f = Fit(BASE, beta, x, rng.normal(size=n), rng.normal(size=n),
            np.array([o.country for o in obs]), np.array([str(o.year) for o in obs]))
    got = dry_season_effect(f, obs)
    d1, d2 = [], []
    for geo in sorted({o.geo for o in obs}):
        p = [o.x["p"] for o in obs if o.geo == geo]
        p10, p50 = np.percentile(p, 10), np.percentile(p, 50)
        d1.append(p10 - p50)
        d2.append(p10 ** 2 - p50 ** 2)
    assert got["effect"] == pytest.approx(0.3 * np.mean(d1) - 0.06 * np.mean(d2))
    assert got["low"] < got["effect"] < got["high"]


def _fit(names: tuple[str, ...], beta: list[float]) -> Fit:
    z = np.zeros((2, len(names)))
    return Fit(names, np.array(beta), z, np.zeros(2), np.zeros(2), np.array(["A", "B"]),
               np.array(["1", "2"]))


@pytest.mark.parametrize(("p_t", "p_b", "edd", "expected"), [
    (0.001, 0.002, -0.05, "established negative"),
    (0.001, 0.2, -0.05, "not determined"),          # only one method rejects
    (0.001, 0.002, 0.05, "contrary to expectation (positive)"),
])
def test_decisions_need_both_methods_and_the_sign(p_t: float, p_b: float, edd: float,
                                                  expected: str) -> None:
    m1 = _fit(BASE, [0.0, edd, 0.0, 0.0])
    m3 = _fit((*BASE, "edd_post"), [0.0, edd, 0.0, 0.0, 0.01])
    e1 = {"p_twoway": p_t, "p_bootstrap": p_b}
    e2 = {"p_twoway": 0.5, "p_bootstrap": 0.5, "dry_season": {"effect": -0.02}}
    e3 = {"p_twoway": 0.5, "p_bootstrap": 0.5}
    out = decide(m1, e1, e2, m3, e3)
    assert out["E1"]["verdict"] == expected
    assert out["E2"]["verdict"] == "not determined"


def assert_same(committed: object, regenerated: object, path: str = "") -> None:
    """Equal structure and non-float values; floats equal to a relative 1e-7.

    The last digits of least-squares fits differ between processors and BLAS builds;
    every number the reports show is far coarser than this tolerance.
    """
    if isinstance(committed, dict):
        assert isinstance(regenerated, dict) and set(committed) == set(regenerated), path
        for key in committed:
            assert_same(committed[key], regenerated[key], f"{path}.{key}")
    elif isinstance(committed, list):
        assert isinstance(regenerated, list) and len(committed) == len(regenerated), path
        for i, (a, b) in enumerate(zip(committed, regenerated, strict=True)):
            assert_same(a, b, f"{path}[{i}]")
    elif isinstance(committed, float) and isinstance(regenerated, float):
        assert math.isclose(committed, regenerated, rel_tol=1e-7, abs_tol=1e-12), \
            (path, committed, regenerated)
    else:
        assert committed == regenerated, (path, committed, regenerated)


def test_committed_results_regenerate() -> None:
    """The whole estimation, bootstraps included, reproduces results.json."""
    results = ROOT / "docs" / "results" / "results.json"
    if not results.exists():
        pytest.skip("No committed results.")
    from eu_wheat_weather.analysis import _round, run

    panel = ROOT / "docs" / "panel"
    regenerated = _round(run(panel / "panel.csv", panel / "weather.csv"))
    assert_same(json.loads(results.read_text(encoding="utf-8")),
                json.loads(json.dumps(regenerated)))


def test_report_regenerates_from_the_committed_results() -> None:
    results = ROOT / "docs" / "results" / "results.json"
    if not results.exists():
        pytest.skip("No committed results.")
    report = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    command = re.search(r"```\n(poetry run eu-wheat estimate [^\n]+)\n```", report)
    assert command is not None
    document = json.loads(results.read_text(encoding="utf-8"))
    assert render(document, command.group(1)) == report
