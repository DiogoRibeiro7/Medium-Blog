"""Combs: sets of hours of the cyclic week pairwise at least L apart (PLAN.md, "Mechanism" and "Measures").

The comb value V(L) is the largest sum of need(h) over such a set. L x V(L) is a lower bound on the paid hours of any
roster of L-hour tours (a feasible point of the dual of the covering programme).
"""

from __future__ import annotations

import numpy as np

WEEK = 168


def comb_through(need: np.ndarray, L: int) -> np.ndarray:
    """For each hour h, the largest comb value of a comb that contains h.

    With h fixed, the other teeth lie on the path h+L .. h+168-L (indices modulo 168), where a comb is a set with
    gaps of at least L: dp[i] = max(dp[i-1], dp[i-L] + w[i]). Vectorised over the 168 anchors.
    """
    need = np.asarray(need, dtype=np.float64)
    W = need.size
    n = W - 2 * L + 1  # path length h+L .. h+W-L inclusive
    if n <= 0:
        return need.copy()
    idx = (np.arange(W)[:, None] + L + np.arange(n)[None, :]) % W
    w = need[idx]  # (W anchors, n positions)
    dp = np.zeros((W, n))
    for i in range(n):
        take = w[:, i] + (dp[:, i - L] if i - L >= 0 else 0.0)
        dp[:, i] = np.maximum(dp[:, i - 1] if i >= 1 else 0.0, take)
    return need + dp[:, -1]


def comb_value(need: np.ndarray, L: int) -> float:
    return float(max(comb_through(need, L).max(), 0.0))


def binding(need: np.ndarray, L: int, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of the hours that belong to at least one optimal comb."""
    t = comb_through(need, L)
    v = t.max()
    return (t >= v - tol) & (np.asarray(need) > 0)


def best_comb(need: np.ndarray, L: int) -> list[int]:
    """One optimal comb (the teeth, sorted), reconstructed from the best anchor."""
    need = np.asarray(need, dtype=np.float64)
    W = need.size
    t = comb_through(need, L)
    h = int(np.argmax(t))
    n = W - 2 * L + 1
    if n <= 0:
        return [h]
    pos = (h + L + np.arange(n)) % W
    w = need[pos]
    dp = np.zeros(n)
    take = np.zeros(n, dtype=bool)
    for i in range(n):
        tk = w[i] + (dp[i - L] if i - L >= 0 else 0.0)
        sk = dp[i - 1] if i >= 1 else 0.0
        take[i] = tk > sk
        dp[i] = max(tk, sk)
    teeth = [h]
    i = n - 1
    while i >= 0:
        if take[i]:
            teeth.append(int(pos[i]))
            i -= L
        else:
            i -= 1
    return sorted(teeth)


def phase_by_comb(profile: np.ndarray, L: int) -> int:
    """For L dividing 24: the phase phi in 0..L-1 maximising the sum of profile[phi + jL]."""
    profile = np.asarray(profile, dtype=np.float64)
    if 24 % L:
        raise ValueError("L must divide 24")
    sums = [profile[phi::L].sum() for phi in range(L)]
    return int(np.argmax(sums))


def phase_of_share(f: np.ndarray, L: int) -> int:
    """Observed phase: phi in 0..L-1 maximising the sum of the binding shares f[phi + jL]."""
    return phase_by_comb(f, L)
