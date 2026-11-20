"""Unit tests on synthetic data with the M5 layout (no real sales are read)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import success_core as sc  # noqa: E402


def synthetic_panel(n=6, seed=0):
    rng = np.random.default_rng(seed)
    days = np.arange(1, sc.N_DAYS + 1)
    wday = ((days - 1) % 7 + 1).astype(np.int8)  # day 1 is a Saturday, wday 1
    sales = rng.poisson(2 + (wday == 1)[None, :] * 3, size=(n, sc.N_DAYS)).astype(np.float32)
    price = np.full((n, sc.N_DAYS), 2.5, dtype=np.float32)
    return sc.Panel(
        sales=sales, price=price,
        item=np.arange(n) // 2, store=np.arange(n) % 2, dept=np.zeros(n, int), state=np.zeros(n, int),
        ids=np.array([f"s{i}" for i in range(n)]), wday=wday,
        month=np.ones(sc.N_DAYS, dtype=np.int8), event=np.zeros(sc.N_DAYS, dtype=np.int8),
        snap=np.zeros((3, sc.N_DAYS), dtype=np.int8),
        dates=np.arange(sc.N_DAYS).astype("datetime64[D]"),
    )


def test_origins_are_fridays():
    for o in sc.ALL_ORIGINS + sc.S0_ORIGINS + sc.REFITS:
        assert o % 7 == 0  # day 7 is the first Friday
    assert len(sc.TEST_ORIGINS) == 52 and len(sc.WARMUP_ORIGINS) == 14
    assert sc.TEST_ORIGINS[-1] + 14 <= 1939
    assert sc.S0_ORIGINS[-1] + 28 <= 1939 and len(sc.S0_ORIGINS) == 13


def test_snaive_and_dow4_use_same_weekday():
    p = synthetic_panel()
    o = 1568
    f = sc.snaive(p.sales, o)
    for h in range(1, 29):
        src = o - 6 + (h - 1) % 7
        assert p.wday[src - 1] == p.wday[o + h - 1]
        assert np.allclose(f[:, h - 1], p.sales[:, src - 1])
    d = sc.dow4(p.sales, o)
    assert np.allclose(d[:, 0], p.sales[:, [o - 6 - 1 - 7 * j for j in range(4)]].mean(axis=1))


def test_week_totals_and_window_mean():
    p = synthetic_panel()
    o = 1568
    a, b = sc.week_totals(p.sales, o)
    assert np.allclose(a, p.sales[:, o:o + 7].sum(axis=1))
    assert np.allclose(b, p.sales[:, o + 7:o + 14].sum(axis=1))
    cs = sc.cumsum0(p.sales)
    assert np.allclose(sc.window_mean(cs, o, 28), p.sales[:, o - 28:o].mean(axis=1))


def test_features_do_not_look_past_origin():
    p = synthetic_panel()
    o = 1568
    cs = sc.cumsum0(p.sales)
    X1, t = sc.daily_features(p, cs, o)
    q = synthetic_panel()
    q.sales[:, o:] = 999.0  # change everything after the origin
    X2, _ = sc.daily_features(q, sc.cumsum0(q.sales), o)
    assert np.allclose(X1, X2, equal_nan=True)
    W1 = sc.weekly_features(p, cs, o)
    W2 = sc.weekly_features(q, sc.cumsum0(q.sales), o)
    assert np.allclose(W1, W2, equal_nan=True)
    assert t.min() == o + 1 and t.max() == o + 28


def test_ses_matches_recursion():
    p = synthetic_panel(n=3)
    levels, sse = sc.ses_run(p.sales, [1568], [1568])
    y = p.sales[0].astype(float)
    lev = y[700 - 28 - 1:700 - 1].mean()
    for d in range(700, 1569):
        lev = lev + 0.1 * (y[d - 1] - lev)
    assert np.isclose(levels[1568][2, 0], lev, rtol=1e-5)
    assert sse[1568].shape == (5, 3)


def test_residual_quantile_orders_use_only_finished_weeks():
    n = 4
    origins = sc.ALL_ORIGINS
    rng = np.random.default_rng(1)
    fc = rng.random((n, len(origins)))
    act = rng.random((n, len(origins)))
    orders = sc.residual_quantile_orders(fc, act, origins, 0.8)
    assert min(orders) == 1568
    o = 1568
    i = origins.index(o)
    act2 = act.copy()
    act2[:, i - 1] += 100  # origin o-7: its week B ends at o+7, not known at o
    orders2 = sc.residual_quantile_orders(fc, act2, origins, 0.8)
    assert np.allclose(orders[o], orders2[o])


def test_pinball_and_bad_weeks():
    assert np.isclose(sc.pinball(np.array([3.0]), np.array([5.0]), 0.8)[0], 1.6)
    assert np.isclose(sc.pinball(np.array([5.0]), np.array([3.0]), 0.8)[0], 0.4)
    bw = sc.bad_weeks(np.array([0.0, 10.0, 30.0]), np.array([4.0, 20.0, 18.0]))
    assert list(bw) == [False, False, True]


def test_kendall():
    assert sc.kendall_tau(np.array([1, 2, 3]), np.array([1, 2, 3])) == 1
    assert sc.kendall_tau(np.array([1, 2, 3]), np.array([3, 2, 1])) == -1
