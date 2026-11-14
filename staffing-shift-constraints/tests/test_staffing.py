from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from evaluate import KS, comparisons, cost, run, verdicts
from feed import erlang as E
from feed import forecast as F
from feed import schedule as S
from feed.calls import START, Calls, evaluation_weeks, grid


def erlang_c_direct(a: float, s: int) -> float:
    top = a**s / math.factorial(s) * s / (s - a)
    return top / (sum(a**k / math.factorial(k) for k in range(s)) + top)


def test_service_level_matches_direct_formula():
    for lam, aht, s in ((120.0, 300.0, 12), (60.0, 180.0, 5), (400.0, 480.0, 60)):
        a = lam * aht / 3600
        want = 1 - erlang_c_direct(a, s) * math.exp(-(s - a) * 30 / aht)
        got = E.service_level(np.array([lam]), np.array([s]), aht)[0]
        assert got == pytest.approx(want, rel=1e-9)


def test_requirement_is_the_smallest_meeting_the_target():
    lam = np.array([0.0, 1.0, 50.0, 120.0, 900.0, 2500.0])
    need = E.requirement(lam, 300.0)
    assert need[0] == 0 and need[1] >= 1
    for x, s in zip(lam[1:], need[1:], strict=True):
        assert E.service_level(np.array([x]), np.array([s]), 300.0)[0] >= 0.8
        assert E.service_level(np.array([x]), np.array([s - 1]), 300.0)[0] < 0.8


def test_regimes_cover_and_rank():
    rng = np.random.default_rng(1)
    hod = np.r_[np.full(7, 3), np.full(5, 40), np.full(6, 55), np.full(6, 20)]
    need = np.tile(hod, 7) + rng.integers(0, 5, 168)
    paid = {}
    for r in S.REGIMES:
        on, p = S.staffing(r, need)
        assert (on >= need - 1e-9).all()
        paid[r] = p
    assert paid["R0"] <= paid["R2"] <= paid["R1"] <= paid["R4"] + 1e-9
    assert paid["R1"] <= paid["R3"] + 1e-9
    on, p = S.staffing("R2", need)
    sp = S.shadow_prices("R4", need)
    assert sp.shape == (168,) and (sp >= -1e-9).all()


def test_tours_are_five_consecutive_days():
    p = S.programme("R4")
    col = p.cover[:, 0]  # first day 0, start 00:00
    assert col.sum() == 40
    assert col[: 24 * 5].reshape(5, 24)[:, :8].all() and not col[24 * 5 :].any()


def synthetic_calls(seed: int = 0) -> Calls:
    rng = np.random.default_rng(seed)
    g = grid()
    hod = np.r_[np.full(7, 0.15), np.linspace(0.5, 1.0, 3), np.full(7, 1.0), np.linspace(0.9, 0.3, 7)]
    dow = np.array([1.1, 1.05, 1.0, 1.0, 0.95, 0.5, 0.4])
    lam = 60 * hod[g.hour] * dow[g.dayofweek] * (1 + 0.1 * np.sin(2 * np.pi * g.dayofyear / 365))
    return Calls(g, rng.poisson(lam).astype(float))


@pytest.fixture(scope="module")
def result():
    return run(synthetic_calls(), ahts=(300.0,), settings={"rounds": 20}, boot=50)


def test_end_to_end(result):
    assert len(result["weeks"]) == 52
    assert cost(result, 300.0, "oracle", "R0", 1) == 0.0
    comp = comparisons(result, 300.0)
    assert {x["hypothesis"] for x in comp} == {"H1", "H2", "H3"}
    assert len([x for x in comp if x["hypothesis"] == "H1"]) == 4 * len(KS)
    v = verdicts(comp)
    assert set(v) == {"H1@300", "H2@300", "H3@300"}


def test_forecasts_use_only_data_two_weeks_old():
    c = synthetic_calls()
    w = evaluation_weeks()[10]
    base = F.simple(c, w, "mean4")
    c2 = Calls(c.hours, c.calls.copy())
    i = c2.index(w - pd.Timedelta(days=7))
    c2.calls[i:] = 1e6  # week w - 1 and later must not matter
    assert np.allclose(F.simple(c2, w, "mean4"), base)
    assert np.allclose(F.simple(c2, w, "two_weeks"), F.simple(c, w, "two_weeks"))
    hol = F.holidays()
    assert np.allclose(F.features(c2, w, hol).to_numpy(), F.features(c, w, hol).to_numpy())
    assert START == pd.Timestamp("2016-01-01")
