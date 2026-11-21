"""Unit tests on synthetic data (no EMS or 311 figure is read)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch import busy_by_hour  # noqa: E402
from staff import comb, data, schedule  # noqa: E402

H = np.arange(168) % 24


def synthetic_need(seed=0, base=60.0):
    rng = np.random.default_rng(seed)
    x = base + 0.6 * base * np.sin(2 * np.pi * (H - 8) / 24) + 0.2 * base * np.sin(4 * np.pi * (H - 3) / 24)
    x = x + rng.normal(0, 3, 168)
    x[np.arange(168) // 24 >= 5] *= 0.85
    return np.maximum(np.round(x), 0)


def brute_comb(need, L):
    """Exhaustive comb value for a short cycle."""
    n = len(need)
    best = 0.0
    for mask in range(1, 1 << n):
        pts = [i for i in range(n) if mask >> i & 1]
        gaps = [(pts[(k + 1) % len(pts)] - pts[k]) % n or n for k in range(len(pts))]
        if len(pts) == 1 or min(gaps) >= L:
            best = max(best, sum(need[i] for i in pts))
    return best


def test_comb_matches_brute_force_on_short_cycles():
    rng = np.random.default_rng(3)
    import staff.comb as cm

    for n, L in ((12, 3), (14, 4), (13, 5)):
        need = rng.integers(0, 10, n).astype(float)
        old = cm.WEEK
        try:
            cm.WEEK = n
            got = max(cm.comb_through(need, L).max(), 0.0)
        finally:
            cm.WEEK = old
        assert np.isclose(got, brute_comb(need, L))


def test_comb_bound_and_binding_definitions():
    need = synthetic_need(1)
    for L in (6, 8, 9, 10, 12):
        ip, x = schedule.roster(need, L)
        assert ip >= L * comb.comb_value(need, L) - 1e-6  # the comb bound is a lower bound
        assert np.isclose(schedule.relaxation(need, L), ip)
        assert (schedule.cover(L) @ x >= need).all()
    for L in (8, 10):
        by_ip = schedule.binding_by_resolve(need, L)
        by_lp = schedule.binding_by_relaxation(need, L)
        assert (by_ip == by_lp).all()
    # where the comb bound is exact (L = 10 here), every binding hour is a comb hour
    assert np.isclose(schedule.roster(need, 10)[0], 10 * comb.comb_value(need, 10))
    lp10 = schedule.binding_by_relaxation(need, 10)
    assert (lp10 <= comb.binding(need, 10)).all()


def test_best_comb_is_feasible_and_optimal():
    need = synthetic_need(2)
    for L in (6, 9, 12):
        teeth = comb.best_comb(need, L)
        gaps = np.diff(teeth + [teeth[0] + 168])
        assert gaps.min() >= L
        assert np.isclose(need[teeth].sum(), comb.comb_value(need, L))


def test_phase_by_comb_picks_the_heaviest_teeth():
    m = np.zeros(24)
    m[[2, 10, 18]] = 5
    assert comb.phase_by_comb(m, 8) == 2
    m2 = np.zeros(24)
    m2[[9, 21]] = 3
    assert comb.phase_by_comb(m2, 12) == 9


def test_busy_by_hour_is_exact():
    start = np.array([0.0, 1800.0, 3500.0, 7200.0 * 2])
    end = np.array([600.0, 5400.0, 3 * 3600.0 + 60.0, 7200.0 * 2 + 3600.0])
    b = busy_by_hour(start, end, 6)
    assert np.allclose(b, [600 + 1800 + 100, 1800 + 3600, 3600, 60, 3600, 0])
    assert np.isclose(b.sum(), (end - start).sum())


def test_erlang_requirement():
    need = data.ems_need(np.array([0.0, 5.0, 100.0]))
    assert need[0] == 0
    for s, a in zip(need[1:], (5.0, 100.0), strict=True):
        assert data.erlang_c(np.array([s]), np.array([a]))[0] <= data.EMS_ALPHA
        assert data.erlang_c(np.array([s - 1]), np.array([a]))[0] > data.EMS_ALPHA or s - 1 <= a
    assert list(data.ems_need(np.array([4.2, 7.0]), alpha=None)) == [5, 7]


def test_loaders_on_the_grid(tmp_path):
    g = data.grid()
    assert len(g) == 52 * 168 and g[0] == pd.Timestamp("2018-12-31") and g[-1] == pd.Timestamp("2019-12-29 23:00")
    (tmp_path / "ems_hourly.csv").write_text(
        "hour,borough,busy_minutes,assigned\n2018-12-31 00,BRONX,120,2\n2018-12-31 00,QUEENS,60,1\n"
        "2018-12-30 23,BRONX,60,1\n"
    )
    ems = data.load_ems(tmp_path / "ems_hourly.csv")
    assert ems["ems_city"][0] == 3.0 and ems["ems_bronx"][0] == 2.0 and ems["ems_queens"][0] == 1.0
    (tmp_path / "c.csv").write_text("hour,calls\n2018-12-31 01,10\n2016-01-01 00,5\n")
    c = data.load_311(tmp_path / "c.csv")
    assert c[1] == 10 and c.sum() == 10


def test_fast_resolve_equals_full_resolve():
    need = synthetic_need(4)
    for L in (8, 9):
        base, x = schedule.roster(need, L)
        full = np.zeros(168, dtype=bool)
        for h in range(168):
            b = need.copy()
            b[h] += 1
            full[h] = schedule.roster(b, L)[0] > base + 1e-6
        assert (schedule.binding_by_resolve(need, L, base, x) == full).all()
