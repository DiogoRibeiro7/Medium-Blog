"""The clearing programme of one zone-period (METHODS.md, section 4), solved two ways.

The variables are the divisible sell and buy steps, each between 0 and its quantity; the
programme maximises the buyers' bid value minus the sellers' offer value, subject to one
balance constraint: sell volume − buy volume = R, where R is the fixed demand (exports)
minus the fixed supply (accepted non-convex volume and imports).

- ``exact`` computes the optimal dual interval Λ = [lo, hi] and the optimal value from the
  integer steps (tenths of a MW, cents of a euro), by the fact in section 1.
- ``solve`` solves the same programme with HiGHS through SciPy and stops the run unless
  HiGHS's dual value lies in Λ and its objective matches the exact one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

IntArray = NDArray[np.int64]

DUAL_TOLERANCE = 1e-6                      # EUR/MWh, HiGHS's dual against Λ
OBJECTIVE_TOLERANCE = 1e-9                 # relative to the largest possible objective


@dataclass(frozen=True)
class Steps:
    """A programme's divisible steps: prices in cents, quantities in tenths of a MW; and R,
    in tenths, the sell volume that must exceed the buy volume."""
    sell_price: IntArray
    sell_qty: IntArray
    buy_price: IntArray
    buy_qty: IntArray
    r: int


@dataclass(frozen=True)
class Exact:
    lo: float              # cents; -inf or +inf when Λ is unbounded
    hi: float
    welfare: int           # cents × tenths
    buy_min: int           # the optimal accepted buy volume, in tenths: [buy_min, buy_max]
    buy_max: int


def _sums(prices: IntArray, qty: IntArray, at: IntArray) -> tuple[IntArray, IntArray]:
    """For each price in ``at``: the quantity priced below it, and at or below it."""
    order = np.argsort(prices, kind="stable")
    p, cum = prices[order], np.concatenate(([0], np.cumsum(qty[order])))
    return cum[np.searchsorted(p, at, "left")], cum[np.searchsorted(p, at, "right")]


def exact(s: Steps) -> Exact:
    """Λ is the set of λ at which the excess supply correspondence
    Z(λ) = [S(<λ) − D(≥λ), S(≤λ) − D(>λ)] − R contains zero; Z is monotone, so Λ is an
    interval whose ends are step prices, or infinite."""
    c = np.unique(np.concatenate((s.sell_price, s.buy_price)))
    if c.size == 0:
        raise ValueError("A programme with no steps.")
    s_lt, s_le = _sums(s.sell_price, s.sell_qty, c)
    total_buy = int(s.buy_qty.sum())
    b_lt, b_le = _sums(s.buy_price, s.buy_qty, c)
    d_ge, d_gt = total_buy - b_lt, total_buy - b_le
    low, up = s_lt - d_ge - s.r, s_le - d_gt - s.r
    if low[0] > 0 or up[-1] < 0:
        raise ValueError("The programme is infeasible.")
    k_lo = int(np.argmax(up >= 0))
    k_hi = int(np.flatnonzero(low <= 0)[-1])
    lo = -np.inf if low[0] == 0 else float(c[k_lo])
    hi = np.inf if up[-1] == 0 else float(c[k_hi])
    if not np.isfinite(lo) and not np.isfinite(hi):
        raise ValueError("Every price is optimal.")
    # An optimal primal at a finite end λ: steps priced inside the money in full, and the
    # steps at λ sharing r = R − S(<λ) + D(>λ). Its value does not depend on how they share.
    k = k_lo if np.isfinite(lo) else k_hi
    lam = int(c[k])
    r = s.r - int(s_lt[k]) + int(d_gt[k])
    inside_buy = s.buy_price > lam
    inside_sell = s.sell_price < lam
    welfare = (int(s.buy_price[inside_buy] @ s.buy_qty[inside_buy])
               - int(s.sell_price[inside_sell] @ s.sell_qty[inside_sell]) - lam * r)
    if lo == hi:
        q_at, b_at = int(s_le[k] - s_lt[k]), int(b_le[k] - b_lt[k])
        buy_min, buy_max = int(d_gt[k]) + max(0, -r), int(d_gt[k]) + min(b_at, q_at - r)
    else:
        # With Λ wider than a point, no step is priced inside it and the primal is unique:
        # the buy steps priced at or above hi are accepted, those at or below lo are not.
        buy_min = buy_max = int(d_gt[k_lo]) if np.isfinite(lo) else int(d_ge[k_hi])
    return Exact(lo, hi, welfare, buy_min, buy_max)


def solve(s: Steps) -> tuple[Exact, float]:
    """``exact``, checked against HiGHS (section 4): HiGHS's dual must lie in Λ and its
    objective must match. Returns the exact result and the objective's relative gap."""
    e = exact(s)
    cost = np.concatenate((s.sell_price, -s.buy_price)) / 100
    upper = np.concatenate((s.sell_qty, s.buy_qty)) / 10
    a_eq = np.concatenate((np.ones(s.sell_qty.size), -np.ones(s.buy_qty.size)))[None, :]
    res = linprog(cost, A_eq=a_eq, b_eq=[s.r / 10],
                  bounds=np.column_stack((np.zeros_like(upper), upper)), method="highs")
    if res.status != 0:
        raise RuntimeError(f"HiGHS: {res.message}")
    dual = float(res.eqlin.marginals[0]) * 100
    scale = float(np.abs(cost) @ upper) + 1.0
    gap = abs(res.fun + e.welfare / 1000) / scale
    if gap > OBJECTIVE_TOLERANCE:
        raise RuntimeError(f"HiGHS's objective {res.fun} differs from the exact "
                           f"{-e.welfare / 1000}.")
    if not e.lo - 100 * DUAL_TOLERANCE <= dual <= e.hi + 100 * DUAL_TOLERANCE:
        raise RuntimeError(f"HiGHS's dual {dual / 100} is outside [{e.lo / 100}, "
                           f"{e.hi / 100}].")
    return e, gap
