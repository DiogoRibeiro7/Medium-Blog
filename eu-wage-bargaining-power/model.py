"""A labour market in which wages are set by sequential auctions with bargaining.

The model follows the structure of Postel-Vinay and Robin (2002) and Cahuc, Postel-Vinay
and Robin (2006) in a simplified, simulated form. Workers differ in ability e, firms in
productivity p, and a match produces y = e * p per month. Time is monthly.

* An unemployed worker meets a random firm with probability lambda0. The match forms if
  y is at least the minimum wage m and at least the worker's outside option b = b0 * e.
  The wage is  w = max(m, b + beta * (y - b)):  the threat point is unemployment and the
  worker keeps a share beta of the surplus above it.
* An employed worker meets a rival firm with probability lambda1. The two firms bid for
  the worker. If the rival is more productive (y' > y) the worker moves and the old
  employer becomes the threat point:  w = max(m, y + beta * (y' - y)).  If the rival is
  less productive but values the worker above the current threat point q, the worker stays
  and the threat point rises to y':  w = max(m, y' + beta * (y - y')).  Wages are never cut.
* Matches end with probability delta per month.

The wage is therefore the second-highest valuation (the auction part) plus a share beta of
what is left (the bargaining part). Option values of future offers are ignored, which moves
wage levels but not the comparative statics studied here.

Five labour-law levers map onto five parameters:

* minimum wage                       -> reserve price m
* collective bargaining strength     -> bargaining weight beta
* non-compete / no-poach rules        -> rate of rival offers lambda1 (number of bidders)
* unemployment insurance             -> outside option b0
* pay transparency (equal pay rule)  -> commitment: within a firm, workers in the same
  ability band ("equal work") must earn the same wage, so a raise to one is a raise to all.

Under transparency a firm counters an outside offer only if keeping the worker at the
higher wage is worth more than the raise it must then give every peer (Cullen and
Pakzad-Hurson 2023), and a poacher whose peers earn less must also raise them. Wages become
cell wages that rise only when that trade-off is favourable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Params:
    n_workers: int = 30_000
    n_firms: int = 300
    sigma_e: float = 0.30          # sd of log ability
    sigma_p: float = 0.35          # sd of log firm productivity
    b0: float = 0.45               # outside option per unit of ability
    beta: float = 0.10             # worker bargaining weight
    lambda0: float = 0.20          # monthly offer rate when unemployed
    lambda1: float = 0.05          # monthly rival-offer rate when employed
    delta: float = 0.015           # monthly separation rate
    m: float = 0.0                 # minimum wage (reserve price)
    transparency: bool = False     # equal pay within firm x ability band
    n_bands: int = 5               # ability bands that define "equal work"
    burn_in: int = 480             # months before measurement starts
    measure_every: int = 12        # months between cross-sections
    n_measure: int = 8             # number of cross-sections averaged
    seed: int = 0


@dataclass
class State:
    firm: np.ndarray               # -1 if unemployed
    wage: np.ndarray
    threat: np.ndarray
    moves: int = 0
    raises: int = 0
    counters_refused: int = 0
    cell_wage: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cell_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def draw_population(prm: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ability, firm productivity and ability band. The same seed gives the same population."""
    rng = np.random.default_rng(10_000 + prm.seed)
    # Generator.lognormal exponentiates with the C library, not numpy's SIMD exp, so the draws are
    # bit-identical across CPUs; the equal-pay engine is path-dependent and amplifies last-digit differences
    e = rng.lognormal(0.0, prm.sigma_e, prm.n_workers)
    p = rng.lognormal(0.0, prm.sigma_p, prm.n_firms)
    edges = np.quantile(e, np.linspace(0, 1, prm.n_bands + 1)[1:-1])
    band = np.searchsorted(edges, e)
    return e, p, band


def _step_bargaining(prm: Params, st: State, e, p, b, rng) -> None:
    n = prm.n_workers
    emp = st.firm >= 0
    # separations
    sep = emp & (rng.random(n) < prm.delta)
    st.firm[sep] = -1
    st.wage[sep] = 0.0
    st.threat[sep] = b[sep]
    emp &= ~sep
    # rival offers to the employed
    off = emp & (rng.random(n) < prm.lambda1)
    idx = np.flatnonzero(off)
    if idx.size:
        rival = rng.integers(0, prm.n_firms, idx.size)
        y = e[idx] * p[st.firm[idx]]
        y2 = e[idx] * p[rival]
        move = y2 > y
        stay = (~move) & (y2 > st.threat[idx])
        i_mv = idx[move]
        st.wage[i_mv] = np.maximum(prm.m, y[move] + prm.beta * (y2[move] - y[move]))
        st.threat[i_mv] = y[move]
        st.firm[i_mv] = rival[move]
        st.moves += int(move.sum())
        i_st = idx[stay]
        new_w = np.maximum(prm.m, y2[stay] + prm.beta * (y[stay] - y2[stay]))
        st.raises += int((new_w > st.wage[i_st]).sum())
        st.wage[i_st] = np.maximum(st.wage[i_st], new_w)
        st.threat[i_st] = y2[stay]
    # offers to the unemployed
    une = (st.firm < 0) & (rng.random(n) < prm.lambda0)
    idx = np.flatnonzero(une)
    if idx.size:
        f = rng.integers(0, prm.n_firms, idx.size)
        y = e[idx] * p[f]
        ok = y >= np.maximum(prm.m, b[idx])
        i_ok = idx[ok]
        st.firm[i_ok] = f[ok]
        st.threat[i_ok] = b[i_ok]
        st.wage[i_ok] = np.maximum(prm.m, b[i_ok] + prm.beta * (y[ok] - b[i_ok]))


def _step_transparent(prm: Params, st: State, e, p, b, band, rng) -> None:
    """Same events, but wages are cell wages (firm x ability band) under an equal-pay rule."""
    n = prm.n_workers
    B = prm.n_bands
    cw, cc = st.cell_wage, st.cell_count

    def leave(i: int) -> None:
        c = st.firm[i] * B + band[i]
        cc[c] -= 1
        if cc[c] == 0:
            cw[c] = np.nan

    emp = st.firm >= 0
    sep = np.flatnonzero(emp & (rng.random(n) < prm.delta))
    for i in sep:
        leave(i)
        st.firm[i] = -1
    # rival offers to the employed
    off = np.flatnonzero((st.firm >= 0) & (rng.random(n) < prm.lambda1))
    rivals = rng.integers(0, prm.n_firms, off.size)
    for i, r in zip(off, rivals):
        j = st.firm[i]
        if r == j:
            continue
        c, c2 = j * B + band[i], r * B + band[i]
        w = cw[c]
        y, y2 = e[i] * p[j], e[i] * p[r]
        if y2 <= w:
            continue                                   # the rival cannot beat the current wage
        # the rival's offer: its posted cell wage if that beats w, else w + beta * (y2 - w)
        if cc[c2] > 0 and cw[c2] > w:
            w_off, raise_cost2 = cw[c2], 0.0
        else:
            w_off = max(prm.m, w + prm.beta * (y2 - w))
            raise_cost2 = cc[c2] * max(0.0, w_off - (cw[c2] if cc[c2] > 0 else w_off))
        if y2 - w_off - raise_cost2 <= 0:
            continue                                   # hiring would cost the rival too much
        if y2 <= y:
            # the incumbent can match; it does so only if the raise to every peer pays
            gain = (y - w_off) - (cc[c] - 1) * (w_off - w)
            if gain > 0:
                cw[c] = w_off
                st.raises += 1
                continue
            st.counters_refused += 1
        # the worker moves to the rival at w_off; the rival's cell is raised to w_off if needed
        leave(i)
        st.firm[i] = r
        if cc[c2] == 0 or np.isnan(cw[c2]):
            cw[c2] = w_off
        else:
            cw[c2] = max(cw[c2], w_off)
        cc[c2] += 1
        st.moves += 1
    # offers to the unemployed
    une = np.flatnonzero((st.firm < 0) & (rng.random(n) < prm.lambda0))
    firms = rng.integers(0, prm.n_firms, une.size)
    for i, f in zip(une, firms):
        y = e[i] * p[f]
        if y < max(prm.m, b[i]):
            continue
        c = f * B + band[i]
        if cc[c] > 0:
            w = cw[c]
            if w >= b[i] and y < w:
                continue                               # the posted wage exceeds this worker's output
            if w < b[i]:
                w_new = max(prm.m, b[i] + prm.beta * (y - b[i]))
                if (y - w_new) - cc[c] * (w_new - w) <= 0:
                    continue
                cw[c] = w_new
        else:
            cw[c] = max(prm.m, b[i] + prm.beta * (y - b[i]))
        cc[c] += 1
        st.firm[i] = f
    emp = st.firm >= 0
    st.wage[:] = 0.0
    st.wage[emp] = cw[st.firm[emp] * B + band[emp]]


def measure(prm: Params, st: State, e, p) -> dict[str, float]:
    emp = st.firm >= 0
    w = st.wage[emp]
    y = e[emp] * p[st.firm[emp]]
    q = np.quantile(w, [0.10, 0.25, 0.50, 0.75, 0.90])
    X = np.column_stack([np.ones(w.size), np.log(e[emp]), np.log(p[st.firm[emp]])])
    coef, *_ = np.linalg.lstsq(X, np.log(w), rcond=None)
    out = {
        "employment_rate": emp.mean(),
        "mean_wage": w.mean(),
        "p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4],
        "p50_p10": q[2] / q[0], "p90_p50": q[4] / q[2],
        "labour_share": w.sum() / y.sum(),
        "mean_output": y.mean(),
        "passthrough_firm": coef[2],
        "share_at_floor": float(np.mean(w <= prm.m * 1.001)) if prm.m > 0 else 0.0,
        "kaitz": prm.m / q[2] if prm.m > 0 else 0.0,
    }
    return out


def simulate(prm: Params) -> dict[str, float]:
    e, p, band = draw_population(prm)
    b = prm.b0 * e
    rng = np.random.default_rng(prm.seed)
    n = prm.n_workers
    st = State(firm=-np.ones(n, dtype=int), wage=np.zeros(n), threat=b.copy())
    if prm.transparency:
        st.cell_wage = np.full(prm.n_firms * prm.n_bands, np.nan)
        st.cell_count = np.zeros(prm.n_firms * prm.n_bands, dtype=int)
    rows = []
    total = prm.burn_in + prm.measure_every * prm.n_measure
    moves0 = raises0 = refused0 = 0
    for t in range(1, total + 1):
        if t == prm.burn_in:
            moves0, raises0, refused0 = st.moves, st.raises, st.counters_refused
        if prm.transparency:
            _step_transparent(prm, st, e, p, b, band, rng)
        else:
            _step_bargaining(prm, st, e, p, b, rng)
        if t > prm.burn_in and (t - prm.burn_in) % prm.measure_every == 0:
            rows.append(measure(prm, st, e, p))
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    years = (total - prm.burn_in) / 12
    emp_avg = out["employment_rate"] * n
    out["job_to_job_per_year"] = (st.moves - moves0) / years / emp_avg
    out["counter_raises_per_year"] = (st.raises - raises0) / years / emp_avg
    out["counters_refused_per_year"] = (st.counters_refused - refused0) / years / emp_avg
    return out


def with_(prm: Params, **kw) -> Params:
    return replace(prm, **kw)
