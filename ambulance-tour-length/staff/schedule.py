"""Rosters of L-hour tours over a cyclic week of 168 hours, as integer programmes (PLAN.md, "Rosters")."""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

WEEK = 168
SHORT = 4
SHORT_SHARE = 0.25


@cache
def cover(L: int) -> np.ndarray:
    """(168, 168) matrix: column j is a tour starting at hour j, covering j .. j+L-1 modulo 168."""
    A = np.zeros((WEEK, WEEK))
    for j in range(WEEK):
        A[(j + np.arange(L)) % WEEK, j] = 1.0
    return A


def roster(need: np.ndarray, L: int) -> tuple[float, np.ndarray]:
    """Paid hours of the cheapest integer roster, and the number of tours starting at each hour."""
    A = cover(L)
    res = milp(
        c=np.full(WEEK, float(L)),
        constraints=[LinearConstraint(A, lb=np.asarray(need, dtype=np.float64), ub=np.inf)],
        integrality=np.ones(WEEK),
        bounds=Bounds(0, np.inf),
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if res.status != 0:
        raise RuntimeError(res.message)
    x = np.round(res.x)
    if (A @ x < np.asarray(need) - 1e-9).any():
        raise RuntimeError("rounded roster does not cover the requirement")
    return float(L * x.sum()), x


def relaxation(need: np.ndarray, L: int) -> float:
    res = linprog(
        np.full(WEEK, float(L)),
        A_ub=-cover(L),
        b_ub=-np.asarray(need, dtype=np.float64),
        bounds=(0, None),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(res.fun)


def binding_by_relaxation(need: np.ndarray, L: int, base: float | None = None) -> np.ndarray:
    """Hours where one more required tour raises the optimum of the linear relaxation (PLAN.md, binding hours)."""
    need = np.asarray(need, dtype=np.float64)
    if base is None:
        base = relaxation(need, L)
    out = np.zeros(WEEK, dtype=bool)
    for h in range(WEEK):
        bumped = need.copy()
        bumped[h] += 1
        out[h] = relaxation(bumped, L) > base + 1e-6
    return out


def binding_by_resolve(need: np.ndarray, L: int, base: float | None = None, x: np.ndarray | None = None) -> np.ndarray:
    """Hours where one more required tour raises the integer optimum.

    Only hours covered exactly by an optimal roster x can bind: where coverage exceeds the requirement, x stays
    feasible after adding one, so the optimum cannot rise. The other hours are re-solved one at a time.
    """
    need = np.asarray(need, dtype=np.float64)
    if base is None or x is None:
        base, x = roster(need, L)
    tight = cover(L) @ x <= need + 1e-9
    out = np.zeros(WEEK, dtype=bool)
    for h in np.flatnonzero(tight):
        bumped = need.copy()
        bumped[h] += 1
        out[h] = roster(bumped, L)[0] > base + 1e-6
    return out


def roster_with_short(need: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Cheapest roster with L-hour tours and 4-hour tours, 4-hour paid hours at most 25% of all paid hours.

    Returns the tours of each kind starting at each hour.
    """
    A = np.hstack([cover(L), cover(SHORT)])
    c = np.r_[np.full(WEEK, float(L)), np.full(WEEK, float(SHORT))]
    share = np.r_[np.full(WEEK, -SHORT_SHARE * L), np.full(WEEK, (1 - SHORT_SHARE) * SHORT)][None, :]
    res = milp(
        c=c,
        constraints=[
            LinearConstraint(A, lb=np.asarray(need, dtype=np.float64), ub=np.inf),
            LinearConstraint(share, lb=-np.inf, ub=0.0),
        ],
        integrality=np.ones(2 * WEEK),
        bounds=Bounds(0, np.inf),
        options={"disp": False, "mip_rel_gap": 1e-6},
    )
    if res.status != 0:
        raise RuntimeError(res.message)
    x = np.round(res.x)
    return x[:WEEK], x[WEEK:]
