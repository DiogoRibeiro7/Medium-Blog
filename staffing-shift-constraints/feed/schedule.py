"""Shift regimes as integer programmes over a cyclic week of 168 hours (PLAN.md, "Shift regimes")."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from feed.calls import WEEK

REGIMES = ("R0", "R1", "R2", "R3", "R4")
LABELS = {
    "R0": "hourly",
    "R1": "8h any start",
    "R2": "8h + 4h",
    "R3": "three fixed starts",
    "R4": "weekly tours",
}
FIXED_STARTS = (7, 15, 23)
PART_TIME_SHARE = 0.25


def block(start: int, length: int) -> np.ndarray:
    col = np.zeros(WEEK)
    col[(start + np.arange(length)) % WEEK] = 1.0
    return col


@dataclass
class Programme:
    cover: np.ndarray  # (168, n) hours covered by each shift type
    hours: np.ndarray  # (n,) paid hours of each shift type
    extra_A: np.ndarray | None = None  # rows of additional constraints A x <= 0
    names: list[str] | None = None


def programme(regime: str) -> Programme:
    if regime == "R1":
        cols = [block(j, 8) for j in range(WEEK)]
        return Programme(np.array(cols).T, np.full(WEEK, 8.0))
    if regime == "R2":
        cols = [block(j, 8) for j in range(WEEK)] + [block(j, 4) for j in range(WEEK)]
        hours = np.r_[np.full(WEEK, 8.0), np.full(WEEK, 4.0)]
        # 4-hour paid hours <= share x all paid hours  <=>  (1 - share) 4 y - share 8 x <= 0
        extra = np.r_[np.full(WEEK, -PART_TIME_SHARE * 8.0), np.full(WEEK, (1 - PART_TIME_SHARE) * 4.0)][None, :]
        return Programme(np.array(cols).T, hours, extra)
    if regime == "R3":
        starts = [d * 24 + h for d in range(7) for h in FIXED_STARTS]
        return Programme(np.array([block(j, 8) for j in starts]).T, np.full(len(starts), 8.0))
    if regime == "R4":
        cols, names = [], []
        for first in range(7):
            for h in range(24):
                col = np.zeros(WEEK)
                for d in range(5):
                    col += block(((first + d) % 7) * 24 + h, 8)
                cols.append(np.minimum(col, 1.0))
                names.append(f"day{first}_h{h}")
        return Programme(np.array(cols).T, np.full(len(cols), 40.0), None, names)
    raise ValueError(regime)


_CACHE: dict[str, Programme] = {}


def staffing(regime: str, need: np.ndarray) -> tuple[np.ndarray, float]:
    """Agents on duty in each of the 168 hours, and paid agent-hours, for the cheapest roster covering `need`."""
    need = np.asarray(need, dtype=np.float64)
    if regime == "R0":
        return need.copy(), float(need.sum())
    p = _CACHE.setdefault(regime, programme(regime))
    cons = [LinearConstraint(p.cover, lb=need, ub=np.inf)]
    if p.extra_A is not None:
        cons.append(LinearConstraint(p.extra_A, lb=-np.inf, ub=0.0))
    res = milp(
        c=p.hours,
        constraints=cons,
        integrality=np.ones(p.hours.size),
        bounds=Bounds(0, np.inf),
        options={"disp": False, "mip_rel_gap": 1e-6},
    )
    if res.status != 0:
        raise RuntimeError(f"{regime}: {res.message}")
    x = np.round(res.x)
    on = p.cover @ x
    if (on < need - 1e-9).any():
        raise RuntimeError(f"{regime}: rounded roster does not cover the requirement")
    return on, float(p.hours @ x)


def shadow_prices(regime: str, need: np.ndarray) -> np.ndarray:
    """Dual values of the hourly coverage constraints in the linear relaxation (paid hours per extra agent)."""
    p = _CACHE.setdefault(regime, programme(regime))
    A_ub = -p.cover
    b_ub = -np.asarray(need, dtype=np.float64)
    if p.extra_A is not None:
        A_ub = np.vstack([A_ub, p.extra_A])
        b_ub = np.r_[b_ub, 0.0]
    res = linprog(p.hours, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return -res.ineqlin.marginals[:WEEK]
