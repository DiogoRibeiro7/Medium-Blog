"""Erlang C service level and staffing requirement (PLAN.md, "Requirement per hour"), vectorized over hours."""

from __future__ import annotations

import numpy as np

TARGET = 0.80
WAIT = 30.0  # seconds


def _erlang_b_table(a: np.ndarray, smax: int) -> np.ndarray:
    """B[k, i] = Erlang B blocking with k servers and offered load a[i], k = 0..smax."""
    B = np.empty((smax + 1, a.size))
    B[0] = 1.0
    for k in range(1, smax + 1):
        B[k] = a * B[k - 1] / (k + a * B[k - 1])
    return B


def service_level(calls: np.ndarray, agents: np.ndarray, aht: float, wait: float = WAIT) -> np.ndarray:
    """Share of calls answered within `wait` seconds, M/M/s, `calls` per hour, `aht` seconds."""
    calls = np.asarray(calls, dtype=np.float64).ravel()
    s = np.asarray(agents, dtype=np.int64).ravel()
    a = calls * aht / 3600.0
    out = np.zeros_like(a)
    out[calls == 0] = 1.0
    ok = (calls > 0) & (s > a)
    if ok.any():
        smax = int(s[ok].max())
        B = _erlang_b_table(a[ok], smax)
        b = B[s[ok], np.arange(ok.sum())]
        sk, ak = s[ok], a[ok]
        C = sk * b / (sk - ak * (1 - b))
        out[ok] = 1 - C * np.exp(-(sk - ak) * wait / aht)
    return out


def requirement(calls: np.ndarray, aht: float, target: float = TARGET, wait: float = WAIT) -> np.ndarray:
    """Smallest number of agents meeting `target` within `wait` seconds; 0 for an hour with no calls."""
    calls = np.asarray(calls, dtype=np.float64)
    shape = calls.shape
    lam = calls.ravel()
    a = lam * aht / 3600.0
    need = np.zeros(lam.size, dtype=np.int64)
    pos = lam > 0
    if not pos.any():
        return need.reshape(shape)
    ap = a[pos]
    smax = int(np.ceil(ap.max() + 10 * np.sqrt(ap.max()) + 20))
    B = _erlang_b_table(ap, smax)
    found = np.zeros(ap.size, dtype=np.int64)
    for k in range(1, smax + 1):
        todo = (found == 0) & (k > ap)
        if not todo.any():
            if (found > 0).all():
                break
            continue
        b = B[k, todo]
        C = k * b / (k - ap[todo] * (1 - b))
        sl = 1 - C * np.exp(-(k - ap[todo]) * wait / aht)
        idx = np.flatnonzero(todo)[sl >= target]
        found[idx] = k
    if (found == 0).any():
        raise ValueError("requirement search did not converge")
    need[pos] = found
    return need.reshape(shape)
