"""A market for an essential good with slow capacity, borrowed supply and income-limited demand.

Everything is per household. Each household needs one unit of the good per year
(a basket of food, a dwelling's worth of housing services). Incomes are lognormal
with median 1, and a household spends at most a share ``s`` of its income on the
good: below that it buys less than it needs. That is all demand is.

Supply has two parts:

- *anchored* capacity ``K``: domestic firms, the people who run them, the skills,
  the land in use and the suppliers around them. It is a stock. It grows only
  when production pays, at a rate limited by how many people can be trained by
  the ones already there, and it shrinks quickly when production loses money;
- *borrowed* supply: imports, or output that depends on workers who come from
  elsewhere. It is available at a unit cost ``p_w`` up to a cap ``M``.

The price clears the market every year. Capacity follows the margin
``p - c(K)``, with a unit cost ``c(K)`` that is high when the sector is small
(nobody left to buy from, sell to or learn from), falls as it grows, and rises
again when it gets large (worse land, congestion). ``phi`` is the premium that
must be paid to get people into physical work rather than an office job.

Two set-ups are used:

``FoodModel``    state ``K``; the good is a flow, borrowed supply can fill gaps.
``HousingModel`` states ``H`` (dwellings per household) and ``K`` (dwellings
                 built per household per year); the good is a stock, nothing can
                 be imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

Schedule = Callable[[float], float]


def const(x: float) -> Schedule:
    return lambda t: x


def step(before: float, after: float, t0: float, t1: float = np.inf) -> Schedule:
    """``after`` on [t0, t1), ``before`` elsewhere."""
    return lambda t: after if t0 <= t < t1 else before


def ramp(x0: float, x1: float, t0: float, t1: float) -> Schedule:
    """Linear from ``x0`` at ``t0`` to ``x1`` at ``t1``, flat outside."""

    def f(t: float) -> float:
        if t <= t0:
            return x0
        if t >= t1:
            return x1
        return x0 + (x1 - x0) * (t - t0) / (t1 - t0)

    return f


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Households:
    """Lognormal incomes with median 1 and a budget ceiling ``s``."""

    sigma: float
    s: float

    @classmethod
    def from_gini(cls, gini: float, s: float) -> "Households":
        # Gini of a lognormal is 2 Phi(sigma / sqrt 2) - 1
        return cls(float(np.sqrt(2.0) * ndtri((1.0 + gini) / 2.0)), s)

    def demand(self, p):
        """Units bought per household at price ``p``: E[min(1, s y / p)]."""
        p = np.asarray(p, dtype=float)
        a = np.log(p / self.s) / self.sigma
        return 1.0 - ndtr(a) + (self.s / p) * np.exp(self.sigma**2 / 2.0) * ndtr(a - self.sigma)

    def priced_out(self, p):
        """Share of households who cannot buy their full need: P(s y < p)."""
        return ndtr(np.log(np.asarray(p, dtype=float) / self.s) / self.sigma)

    def unmet_need(self, p):
        """Average shortfall from need, per household."""
        return 1.0 - self.demand(p)

    def price_with_priced_out(self, share: float) -> float:
        """Price at which ``share`` of households cannot afford their need."""
        return float(self.s * np.exp(self.sigma * ndtri(share)))

    def inverse_demand(self, q: float) -> float:
        """Price at which households buy ``q`` units in total (0 < q < 1)."""
        q = float(np.clip(q, 1e-9, 1.0 - 1e-12))
        f = lambda lp: float(self.demand(np.exp(lp))) - q
        return float(np.exp(brentq(f, -40.0, 40.0, xtol=1e-13)))


@dataclass
class InverseDemandTable:
    """Fast inverse demand by interpolation on a log-price grid."""

    hh: Households
    lo: float = -12.0
    hi: float = 8.0
    n: int = 20001
    _q: np.ndarray = field(init=False, repr=False)
    _lp: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lp = np.linspace(self.lo, self.hi, self.n)
        q = self.hh.demand(np.exp(lp))
        # demand is decreasing in price; np.interp needs increasing x
        self._q = q[::-1]
        self._lp = lp[::-1]

    def __call__(self, q: float) -> float:
        q = min(max(q, self._q[0]), self._q[-1])
        return float(np.exp(np.interp(q, self._q, self._lp)))


@dataclass
class Demand:
    """Need-limited demand plus a discretionary layer, in units of consumption at ``p0``.

    ``total(p)`` with ``p`` in units of ``p0``: need bought by households,
    E[min(1, s y / (p p0))], plus discretionary use ``x p^-eta`` (waste, premium
    products, meals out: whatever is cut first when the price rises), divided by
    their sum at ``p = 1`` so that consumption at the reference price is 1.
    """

    hh: Households
    p0: float
    x: float = 0.25
    eta: float = 1.0
    lo: float = -8.0
    hi: float = 6.0
    n: int = 20001
    _q: np.ndarray = field(init=False, repr=False)
    _lp: np.ndarray = field(init=False, repr=False)
    scale: float = field(init=False)

    def __post_init__(self) -> None:
        self.scale = float(self.hh.demand(self.p0)) + self.x
        lp = np.linspace(self.lo, self.hi, self.n)
        q = self.total(np.exp(lp))
        self._q = q[::-1]
        self._lp = lp[::-1]

    def total(self, p):
        p = np.asarray(p, dtype=float)
        return (self.hh.demand(p * self.p0) + self.x * p ** (-self.eta)) / self.scale

    def inverse(self, q):
        q = np.clip(q, self._q[0], self._q[-1])
        return np.exp(np.interp(q, self._q, self._lp))

    def priced_out(self, p):
        return self.hh.priced_out(np.asarray(p, dtype=float) * self.p0)

    def unmet_need(self, p):
        return self.hh.unmet_need(np.asarray(p, dtype=float) * self.p0)

    def elasticity(self, p: float = 1.0, h: float = 1e-4) -> float:
        return float((np.log(self.total(p * (1 + h))) - np.log(self.total(p * (1 - h)))) / (np.log(1 + h) - np.log(1 - h)))


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capacity:
    """Unit cost and the response of anchored capacity to the margin.

    Prices and costs are in units of the reference price ``p0`` (see ``calibrate``).
    """

    c_inf: float = 0.55  # unit cost of a full-sized sector, before the effort premium
    dc: float = 0.60  # extra unit cost when the sector is tiny (ecosystem penalty)
    K_h: float = 0.25  # capacity at which half of that penalty is gone
    n: float = 4.0  # steepness of the penalty
    beta: float = 0.35  # rising cost at large scale (land, congestion), times K^2
    g_max: float = 0.07  # fastest per-capita growth, per year (construction rebuilt at 3.7%/yr, 2014-2025)
    l_max: float = 0.09  # fastest per-capita exit, per year (construction shed 11.2%/yr, 2008-2013)
    pi_s: float = 0.10  # margin (share of p0) at which growth is ~3/4 of its maximum
    K_g: float = 0.10  # newcomers learn from incumbents: growth scales with K/(K+K_g)
    eps: float = 2e-4  # entry from nothing (no incumbents), per year at full margin

    def cost(self, K: float, phi: float) -> float:
        K = max(K, 0.0)
        return self.c_inf + phi + self.dc / (1.0 + (K / self.K_h) ** self.n) + self.beta * K * K

    def growth(self, K: float, margin: float, delta: float) -> float:
        """dK/dt given the margin (share of p0) and attrition ``delta``."""
        x = np.tanh(margin / self.pi_s)
        if margin >= 0.0:
            return K * (self.g_max * x * K / (K + self.K_g) - delta) + self.eps * x
        return K * (self.l_max * x - delta)


def rk4(f, y0: np.ndarray, t0: float, t1: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(round((t1 - t0) / dt))
    ts = t0 + dt * np.arange(n + 1)
    ys = np.empty((n + 1, len(y0)))
    ys[0] = y0
    y = np.array(y0, dtype=float)
    for i in range(n):
        t = ts[i]
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt / 2 * k1)
        k3 = f(t + dt / 2, y + dt / 2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y = np.maximum(y, 0.0)
        ys[i + 1] = y
    return ts, ys


# ---------------------------------------------------------------------------
# Food: a flow good that can be borrowed
# ---------------------------------------------------------------------------


@dataclass
class FoodModel:
    """Anchored capacity ``K``, in units of consumption at the reference price."""

    hh: Households
    cap: Capacity = field(default_factory=Capacity)
    p0: float = 1.0  # reference price, absolute units
    p_w: float = 1.0  # unit cost of borrowed supply, in units of p0
    x: float = 0.25  # discretionary layer, per household, at p0
    eta: float = 1.0
    phi: Schedule = field(default_factory=lambda: const(0.10))
    delta: Schedule = field(default_factory=lambda: const(0.025))
    M: Schedule = field(default_factory=lambda: const(1.0))
    dem: Demand | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.dem is None or self.dem.hh != self.hh or self.dem.p0 != self.p0 or self.dem.x != self.x:
            self.dem = Demand(self.hh, self.p0, self.x, self.eta)
        self._q_w = float(self.dem.total(self.p_w))

    def price(self, K: float, M: float) -> tuple[float, float]:
        """Clearing price (units of p0) and borrowed quantity, given capacity and the cap."""
        q_w = self._q_w
        if K >= q_w:
            return float(self.dem.inverse(K)), 0.0
        if K + M >= q_w:
            return self.p_w, q_w - K
        return float(self.dem.inverse(K + M)), M

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        K = y[0]
        p, _ = self.price(K, self.M(t))
        margin = p - self.cap.cost(K, self.phi(t))
        return np.array([self.cap.growth(K, margin, self.delta(t))])

    def dKdt_frozen(self, K: float, phi: float, delta: float, M: float) -> float:
        p, _ = self.price(K, M)
        return self.cap.growth(K, p - self.cap.cost(K, phi), delta)

    def equilibria(self, phi: float, delta: float, M: float, grid: np.ndarray | None = None):
        """Roots of dK/dt for frozen parameters, with their stability."""
        if grid is None:
            grid = np.concatenate([np.linspace(1e-6, 0.02, 200), np.linspace(0.02, 1.2, 1400)[1:]])
        f = np.array([self.dKdt_frozen(k, phi, delta, M) for k in grid])
        out = []
        if f[0] < 0:
            out.append((0.0, True))  # capacity dies out from any small level
        for i in range(len(grid) - 1):
            if f[i] == 0.0 or f[i] * f[i + 1] < 0:
                r = brentq(lambda k: self.dKdt_frozen(k, phi, delta, M), grid[i], grid[i + 1], xtol=1e-12)
                stable = f[i] > 0 > f[i + 1]
                out.append((r, stable))
        return out

    def simulate(self, K0: float, t0: float, t1: float, dt: float = 0.05) -> dict[str, np.ndarray]:
        ts, ys = rk4(self.rhs, np.array([K0]), t0, t1, dt)
        K = ys[:, 0]
        p = np.empty_like(K)
        b = np.empty_like(K)
        for i, (t, k) in enumerate(zip(ts, K)):
            p[i], b[i] = self.price(k, self.M(t))
        return {
            "t": ts,
            "K": K,
            "price": p,
            "borrowed": b,
            "priced_out": self.dem.priced_out(p),
            "unmet": self.dem.unmet_need(p),
        }


# ---------------------------------------------------------------------------
# Housing: a stock good that cannot be imported
# ---------------------------------------------------------------------------


@dataclass
class HousingModel:
    """Dwellings per household ``H`` and building capacity ``K`` (dwellings per household per year).

    Demand for the stock has two parts: primary need, one dwelling per household,
    limited by income as in ``Households``; and non-primary use (second homes,
    holiday and short-term lets, dwellings held empty), ``x(t) p^-eta``, which is
    elastic and can grow from outside the resident population.
    """

    hh: Households
    cap: Capacity
    p0: float = 1.0
    phi: Schedule = field(default_factory=lambda: const(0.10))
    delta: Schedule = field(default_factory=lambda: const(0.025))  # builders leaving
    growth_need: Schedule = field(default_factory=lambda: const(0.010))  # household growth
    nonprimary: Schedule = field(default_factory=lambda: const(0.44))  # x(t), dwellings per household at p0
    eta: float = 1.5
    cost_shock: Schedule = field(default_factory=lambda: const(0.0))  # credit, materials (units of p0)
    obsolescence: float = 0.005  # dwellings lost per year, share of stock
    K_floor: Schedule = field(default_factory=lambda: const(0.0))  # capacity kept alive by public orders
    K_unit: float = 0.0216  # build rate that the capacity curve calls 1

    def price(self, H: float, t: float) -> float:
        x = self.nonprimary(t)
        f = lambda lp: float(self.hh.demand(np.exp(lp) * self.p0)) + x * np.exp(-self.eta * lp) - H
        return float(np.exp(brentq(f, -20.0, 20.0, xtol=1e-12)))

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        H, K = y
        dH = K - (self.obsolescence + self.growth_need(t)) * H
        k = K / self.K_unit
        margin = self.price(H, t) - self.cap.cost(k, self.phi(t)) - self.cost_shock(t)
        dk = self.cap.growth(k, margin, self.delta(t))
        if k <= self.K_floor(t):
            dk = max(dk, 0.0)
        return np.array([dH, dk * self.K_unit])

    def steady_state(self, t: float = 0.0) -> tuple[float, float]:
        """High steady state for the parameters frozen at time ``t``."""
        g = self.obsolescence + self.growth_need(t)

        def gap(k: float) -> float:
            H = k * self.K_unit / g
            margin = self.price(H, t) - self.cap.cost(k, self.phi(t)) - self.cost_shock(t)
            return self.cap.growth(k, margin, self.delta(t))

        ks = np.linspace(0.05, 4.0, 4000)
        vals = np.array([gap(k) for k in ks])
        for i in range(len(ks) - 1, 0, -1):
            if vals[i - 1] > 0 >= vals[i]:
                k = brentq(gap, ks[i - 1], ks[i])
                return k * self.K_unit / g, k * self.K_unit
        raise ValueError("no positive steady state")

    def simulate(self, y0: tuple[float, float], t0: float, t1: float, dt: float = 0.02) -> dict[str, np.ndarray]:
        ts, ys = rk4(self.rhs, np.array(y0), t0, t1, dt)
        H, K = ys[:, 0], ys[:, 1]
        p = np.array([self.price(h, t) for h, t in zip(H, ts)])
        return {
            "t": ts,
            "H": H,
            "K": K,
            "k": K / self.K_unit,
            "price": p,
            "priced_out": self.hh.priced_out(p * self.p0),
        }


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def reference_price(hh: Households, baseline_priced_out: float) -> float:
    """Absolute price at which ``baseline_priced_out`` of households fall short of need."""
    return hh.price_with_priced_out(baseline_priced_out)


def with_(obj, **kw):
    return replace(obj, **kw)
