"""Unit tests for the pass-through, the debt dynamics and the co-movement measures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contagion import PANEL, shared  # noqa: E402
from fiscal import estimate_rho, simulate, stabilising_primary  # noqa: E402
from stress import project, project_many, step  # noqa: E402


def test_simulate_closes_gap_geometrically():
    rho, start, x = 0.2, 2.0, 5.0
    path = simulate(start, np.full(10, x), rho)
    for k, i in enumerate(path, start=1):
        assert abs((x - i) - (x - start) * (1 - rho) ** k) < 1e-12


def test_estimate_rho_recovers_known_value():
    rng = np.random.default_rng(1)
    rows = []
    for c, rho in (("AA", 0.15), ("BB", 0.15), ("CC", 0.15)):
        i = 5.0
        for y in range(1990, 2026):
            x = 3 + 2 * np.sin(y / 3) + rng.normal(0, 0.5)
            i = i + rho * (x - i) + rng.normal(0, 0.01)
            rows.append(dict(geo=c, year=y, implicit=i, **{"yield": x}))
    a = pd.DataFrame(rows)
    est, se, each = estimate_rho(a, ("AA", "BB", "CC"), (1991, 2025))
    assert abs(est - 0.15) < 0.02 and se < 0.02
    assert all(abs(v[0] - 0.15) < 0.04 for v in each.values())


def test_stabilising_primary_holds_debt_ratio():
    d, i, g = 115.0, 4.0, 2.5
    pb = stabilising_primary(d, i, g)
    d_new, _ = step(d, i, g, pb)
    # interest is charged on average debt within the year, so the ratio holds to second order
    assert abs(d_new - d) < 0.1


def test_step_interest_on_average_debt():
    d_new, ig = step(100.0, 4.0, 0.0, 0.0)
    assert abs(ig - 0.04 * (100.0 + d_new) / 2) < 1e-12
    assert abs(d_new - (100.0 + ig)) < 1e-9


def test_vectorised_projection_matches_loop():
    years = np.arange(2026, 2037)
    rng = np.random.default_rng(3)
    X = 4 + rng.normal(0, 0.5, (5, len(years)))
    G = 2.5 + rng.normal(0, 1.0, (5, len(years)))
    for primary in ("plan", "held"):
        IG, D = project_many(115.7, 1.94, X, G, 0.165, primary, years)
        for n in range(5):
            f = project(115.7, 1.94, X[n], G[n], 0.165, primary, years)
            assert np.allclose(f["int_gdp"].to_numpy(), IG[n]) and np.allclose(f["debt_gdp"].to_numpy(), D[n])


def test_shared_separates_common_and_specific_moves():
    rng = np.random.default_rng(7)
    n = 24
    common = rng.normal(0.10, 0.05, n)
    data = {c: common + rng.normal(0, 0.01, n) for c in PANEL}
    together = pd.DataFrame(data)
    r = shared(together, "FR")
    assert r["r2"] > 0.8
    assert abs(r["common"] - r["cum"]) < 0.25 * abs(r["cum"])
    alone = together.copy()
    alone["FR"] = rng.normal(0.02, 0.1, n)
    r2 = shared(alone, "FR")["r2"]
    assert r2 < 0.4
