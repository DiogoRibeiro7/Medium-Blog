"""Checks of the queue simulator against cases worked by hand and theory."""

from __future__ import annotations

import math

import numpy as np

from gentradeoffs.queue import (ABANDONED, ANSWERED, OFFERED, TOTAL_WAIT,
                                WITHIN, Callers, run)


def callers(arrivals, service, patience, retry_u=None, pool=0):
    n = len(arrivals)
    retry_u = np.ones(n) if retry_u is None else np.asarray(retry_u, float)
    return Callers(np.asarray(arrivals, float), np.asarray(service, float),
                   np.asarray(patience, float), retry_u, np.full(n, 10.0),
                   np.full(pool, 5.0), np.full(pool, np.inf),
                   np.ones(pool), np.zeros(pool))


def test_single_agent_by_hand():
    # One agent. Calls at 0, 1, 2 needing 10 s each; the third is impatient.
    c = callers([0.0, 1.0, 2.0], [10.0, 10.0, 10.0], [np.inf, np.inf, 5.0])
    out, _ = run(c, staff=np.ones(24, np.int64))
    tot = out.sum(axis=0)
    assert tot[OFFERED] == 3
    assert tot[ANSWERED] == 2
    assert tot[ABANDONED] == 1
    # waits: 0, 9 (answered at 10), and 5 s before hanging up
    assert math.isclose(tot[TOTAL_WAIT], 0 + 9 + 5)
    assert tot[WITHIN] == 2


def test_second_agent_logs_on_at_interval_start():
    # One agent all day, a second from 1800 s. The first caller holds agent
    # 0 for an hour, so the second waits for agent 1 to log on.
    staff = np.ones(24, np.int64)
    staff[1:] = 2
    c = callers([0.0, 10.0], [3600.0, 10.0], [np.inf, np.inf])
    out, _ = run(c, staff=staff)
    assert math.isclose(out[:, TOTAL_WAIT].sum(), 1790.0)


def test_redial_joins_queue():
    # One agent busy for 100 s; the second caller gives up after 5 s and
    # always redials 10 s later; the redial has infinite patience.
    c = callers([0.0, 1.0], [100.0, 10.0], [np.inf, 5.0], retry_u=[1.0, 0.0],
                pool=4)
    out, _ = run(c, staff=np.ones(24, np.int64), p_retry=1.0)
    tot = out.sum(axis=0)
    assert tot[OFFERED] == 3          # two fresh calls plus one redial
    assert tot[ABANDONED] == 1
    assert tot[ANSWERED] == 2


def erlang_c_wait_probability(c: int, a: float) -> float:
    """P(wait > 0) in M/M/c with offered load a (Erlang C formula)."""
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) * c / (c - a)
    return top / (s + top)


def test_matches_erlang_c():
    # M/M/5 with load 3.5 Erlangs over a long day: share of calls that wait.
    rng = np.random.default_rng(0)
    lam, mean_s, c = 3.5 / 100.0, 100.0, 5
    horizon = 24 * 1800.0
    waited = offered = 0
    for _ in range(20):
        m = rng.poisson(lam * horizon)
        arr = np.sort(rng.uniform(0, horizon, m))
        cl = callers(arr, rng.exponential(mean_s, m), np.full(m, np.inf))
        out, _ = run(cl, staff=np.full(24, c, np.int64))
        tot = out.sum(axis=0)
        offered += tot[OFFERED]
        waited += tot[TOTAL_WAIT]
    pw = erlang_c_wait_probability(c, 3.5)
    expected_wait = pw * mean_s / (c - 3.5)
    assert abs(waited / offered - expected_wait) / expected_wait < 0.08
