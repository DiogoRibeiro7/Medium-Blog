"""Sequential auctions with bargaining, a minimum wage, and an equal-pay rule in part of the firms.

Workers differ in ability e, firms in productivity p, and a match produces y = e * p per month.
Time is monthly. The structure follows Postel-Vinay and Robin (2002) and Cahuc, Postel-Vinay and
Robin (2006) in a simplified, simulated form, as in the companion of the earlier article
(../eu-wage-bargaining-power). Two things are new here:

1. Firms come in two kinds. Uncovered firms pay each worker an individual wage. Covered firms are
   bound by an equal-pay rule: inside a firm, workers in the same ability band ("equal work")
   earn one cell wage, so a raise to one is a raise to all of them, and a new hire is paid at
   least the cell wage. Coverage is assigned to the largest firms until a target share of
   employment is covered, which mimics size thresholds in pay-transparency law.

2. Every bid is a valuation. An uncovered firm values a worker at the output y. A covered firm
   that already employs k workers in the worker's cell at wage w values keeping (or hiring) the
   worker at the highest wage it would pay once the raise to the peers is counted:
       keep:  v = w + (y - w) / k           (k includes the worker)
       hire:  v = w + (y - w) / (k + 1)     (and never below the posted cell wage w)
   The two firms bid, the higher valuation wins, and the wage is the losing valuation plus a
   share beta of the gap. An unemployed worker's threat point is b = b0 * e.

* An unemployed worker meets a random firm with probability lambda0. The match forms if y is
  at least the minimum wage m and the outside option b.
* An employed worker meets a rival with probability lambda1 and the two firms bid as above.
  Wages are never cut. When a covered incumbent loses a worker it could have kept under
  individual pay (y >= y'), the event is counted as a refused counter-offer.
* Matches end with probability delta per month.

With no covered firm the engine reproduces the bargaining model of the earlier companion; with
every firm covered it is the equal-pay economy. The minimum wage destroys matches with y < m,
which is the only employment margin: there is no labour-supply response, no monopsony gain from
a higher floor and no price pass-through, so the employment loss it implies is an upper bound
within the model's own logic.

The simulation core is compiled with numba. It draws its random numbers with numba's own
generator, seeded per run, and uses only additions, multiplications and comparisons on the path
of the state, so a run is reproducible across machines.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numba import njit


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
    cover_share: float = 0.0       # employment share of the largest firms bound by equal pay
    n_bands: int = 10              # ability bands that define "equal work"
    visibility: float = 1.0        # probability that a raise in a covered firm must be extended to peers
    burn_in: int = 480             # months before measurement starts
    measure_every: int = 12        # months between cross-sections
    n_measure: int = 8             # number of cross-sections averaged
    seed: int = 0


def with_(prm: Params, **kw) -> Params:
    return replace(prm, **kw)


def draw_population(prm: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ability, firm productivity and ability band. The same seed gives the same population."""
    rng = np.random.default_rng(10_000 + prm.seed)
    # Generator.lognormal exponentiates with the C library, not numpy's SIMD exp: bit-identical across CPUs
    e = rng.lognormal(0.0, prm.sigma_e, prm.n_workers)
    p = rng.lognormal(0.0, prm.sigma_p, prm.n_firms)
    edges = np.quantile(e, np.linspace(0, 1, prm.n_bands + 1)[1:-1])
    band = np.searchsorted(edges, e).astype(np.int64)
    return e, p, band


@njit(cache=True)
def _unlink(i, c, head, nxt, prv):
    if prv[i] >= 0:
        nxt[prv[i]] = nxt[i]
    else:
        head[c] = nxt[i]
    if nxt[i] >= 0:
        prv[nxt[i]] = prv[i]
    nxt[i] = -1
    prv[i] = -1


@njit(cache=True)
def _link(i, c, head, nxt, prv):
    nxt[i] = head[c]
    prv[i] = -1
    if head[c] >= 0:
        prv[head[c]] = i
    head[c] = i


@njit(cache=True)
def _valuation(y, c, skip, head, nxt, wage, buf, pi):
    """Highest wage w at which a covered firm still gains from employing a worker of output y in
    cell c, when with probability pi every peer paid less than w must be raised to w:
        y - w - pi * sum_p max(0, w - w_p) = 0."""
    k = 0
    i = head[c]
    while i >= 0:
        if i != skip:
            buf[k] = wage[i]
            k += 1
        i = nxt[i]
    if k == 0:
        return y
    a = np.sort(buf[:k])
    prefix = 0.0
    for j in range(k + 1):
        w = (y + pi * prefix) / (1 + pi * j)
        upper = a[j] if j < k else np.inf
        if w <= upper:
            return w
        prefix += a[j]
    return y


@njit(cache=True)
def _raise_cell(c, w, head, nxt, wage):
    """Equal pay: every member of cell c paid less than w is raised to w."""
    i = head[c]
    while i >= 0:
        if wage[i] < w:
            wage[i] = w
        i = nxt[i]


@njit(cache=True)
def _run(e, p, band, b, m_path, beta, lambda0, lambda1, delta, B, pi, cover_share, reform_month,
         total, measure_start, measure_every, seed):
    """Simulate `total` months with the minimum wage m_path[t] in month t. In month reform_month the
    largest firms are put under the equal-pay rule until cover_share of employment is covered.
    Returns the wage and firm of every worker at each measurement month (every measure_every months
    after measure_start), the covered-firm flags, and per measurement window the counts of
    job-to-job moves, counter raises, refused counters and the summed log wage gain at moves."""
    np.random.seed(seed)
    n = e.size
    F = p.size
    firm = -np.ones(n, dtype=np.int64)
    wage = np.zeros(n)
    threat = b.copy()
    covered = np.zeros(F, dtype=np.bool_)
    head = -np.ones(F * B, dtype=np.int64)       # members of each covered cell, as a linked list
    nxt = -np.ones(n, dtype=np.int64)
    prv = -np.ones(n, dtype=np.int64)
    buf = np.zeros(n)
    n_meas = (total - measure_start) // measure_every
    snap_w = np.zeros((n_meas, n))
    snap_f = -np.ones((n_meas, n), dtype=np.int64)
    moves = np.zeros(n_meas, dtype=np.int64)
    raises = np.zeros(n_meas, dtype=np.int64)
    refused = np.zeros(n_meas, dtype=np.int64)
    gain = np.zeros(n_meas)
    for t in range(1, total + 1):
        m = m_path[t]
        k = (t - measure_start - 1) // measure_every if t > measure_start else -1
        if t == reform_month and cover_share > 0:
            size = np.zeros(F)
            for i in range(n):
                if firm[i] >= 0:
                    size[firm[i]] += 1.0
            order = np.argsort(-size, kind="mergesort")
            tot = size.sum()
            cum = 0.0
            for f in order:
                if cum >= cover_share * tot - 1e-9:
                    break
                covered[f] = True
                cum += size[f]
            for i in range(n):
                if firm[i] >= 0 and covered[firm[i]]:
                    _link(i, firm[i] * B + band[i], head, nxt, prv)
        # separations
        for i in range(n):
            f = firm[i]
            if f >= 0 and np.random.random() < delta:
                if covered[f]:
                    _unlink(i, f * B + band[i], head, nxt, prv)
                firm[i] = -1
                wage[i] = 0.0
                threat[i] = b[i]
        # rival offers to the employed
        for i in range(n):
            j = firm[i]
            if j < 0:
                continue
            if np.random.random() >= lambda1:
                continue
            r = np.random.randint(0, F)
            if r == j:
                continue
            y = e[i] * p[j]
            y2 = e[i] * p[r]
            w = wage[i]
            c = j * B + band[i]
            c2 = r * B + band[i]
            # the incumbent can always keep the worker at the current wage, so its bid is at least w
            v_inc = max(w, _valuation(y, c, i, head, nxt, wage, buf, pi)) if covered[j] else y
            v_riv = _valuation(y2, c2, -1, head, nxt, wage, buf, pi) if covered[r] else y2
            if v_riv <= w:
                continue                          # the rival cannot beat the current wage
            if v_riv > v_inc:
                new = max(m, v_inc + beta * (v_riv - v_inc))
                if k >= 0:
                    moves[k] += 1
                    gain[k] += np.log(new / w)
                    if y >= y2:
                        refused[k] += 1
                if covered[j]:
                    _unlink(i, c, head, nxt, prv)
                firm[i] = r
                wage[i] = new
                threat[i] = v_inc
                if covered[r]:
                    if np.random.random() < pi:
                        _raise_cell(c2, new, head, nxt, wage)
                    _link(i, c2, head, nxt, prv)
            else:
                new = max(m, v_riv + beta * (v_inc - v_riv))
                if new > w:
                    wage[i] = new
                    if k >= 0:
                        raises[k] += 1
                    if covered[j] and np.random.random() < pi:
                        _raise_cell(c, new, head, nxt, wage)
                if v_riv > threat[i]:
                    threat[i] = v_riv
        # a higher minimum wage lifts everyone below it; matches that produce less than it end
        for i in range(n):
            f = firm[i]
            if f >= 0 and wage[i] < m:
                if e[i] * p[f] < m:
                    if covered[f]:
                        _unlink(i, f * B + band[i], head, nxt, prv)
                    firm[i] = -1
                    wage[i] = 0.0
                    threat[i] = b[i]
                else:
                    wage[i] = m
        # offers to the unemployed
        for i in range(n):
            if firm[i] >= 0:
                continue
            if np.random.random() >= lambda0:
                continue
            f = np.random.randint(0, F)
            y = e[i] * p[f]
            if y < max(m, b[i]):
                continue
            w = max(m, b[i] + beta * (y - b[i]))
            if covered[f]:
                c = f * B + band[i]
                if _valuation(y, c, -1, head, nxt, wage, buf, pi) < w:
                    continue                      # raising the peers would cost more than the hire is worth
                if np.random.random() < pi:
                    _raise_cell(c, w, head, nxt, wage)
                _link(i, c, head, nxt, prv)
            firm[i] = f
            wage[i] = w
            threat[i] = b[i]
        if t > measure_start and (t - measure_start) % measure_every == 0 and k < n_meas:
            snap_w[k] = wage
            snap_f[k] = firm
    return snap_w, snap_f, covered, moves, raises, refused, gain


def _common(prm: Params):
    e, p, band = draw_population(prm)
    return e, p, band, prm.b0 * e


def _summarise(prm: Params, m_meas: np.ndarray, snap_w, snap_f, covered, moves, raises, refused, gain,
               e, p) -> list[dict[str, float]]:
    """One dict of indicators per measurement window."""
    out = []
    years = prm.measure_every / 12
    for k in range(snap_w.shape[0]):
        r = measure(m_meas[k], snap_w[k], snap_f[k], e, p, covered)
        emp = r["employment_rate"] * prm.n_workers
        r["job_to_job_per_year"] = moves[k] / years / emp
        r["counter_raises_per_year"] = raises[k] / years / emp
        r["counters_refused_per_year"] = refused[k] / years / emp
        r["log_gain_at_move"] = gain[k] / moves[k] if moves[k] else 0.0
        out.append(r)
    return out


def simulate(prm: Params) -> dict[str, float]:
    """Steady state at a constant minimum wage, with the equal-pay rule (if any) from the start."""
    e, p, band, b = _common(prm)
    total = prm.burn_in + prm.measure_every * prm.n_measure
    m_path = np.full(total + 1, prm.m)
    # the rule (if any) starts half-way through the burn-in, long enough before measurement to settle
    res = _run(e, p, band, b, m_path, prm.beta, prm.lambda0, prm.lambda1, prm.delta, prm.n_bands,
               prm.visibility, prm.cover_share, prm.burn_in // 2, total, prm.burn_in, prm.measure_every, prm.seed)
    rows = _summarise(prm, np.full(prm.n_measure, prm.m), *res, e, p)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def simulate_reform(prm: Params, years: int) -> pd.DataFrame:
    """Steady state without the rule, then the equal-pay rule switched on for the largest firms;
    indicators each year after the switch, for the reform run and for the same economy without it
    (identical random numbers until the switch)."""
    e, p, band, b = _common(prm)
    total = prm.burn_in + 12 * years
    m_path = np.full(total + 1, prm.m)
    rows = []
    for label, share in (("reform", prm.cover_share), ("baseline", 0.0)):
        res = _run(e, p, band, b, m_path, prm.beta, prm.lambda0, prm.lambda1, prm.delta, prm.n_bands,
                   prm.visibility, share, prm.burn_in, total, prm.burn_in, 12, prm.seed)
        for k, r in enumerate(_summarise(with_(prm, measure_every=12), np.full(years, prm.m), *res, e, p)):
            rows.append({"run": label, "year": k + 1, **r})
    return pd.DataFrame(rows)


def simulate_path(prm: Params, m_by_year: list[float]) -> pd.DataFrame:
    """Steady state at prm.m, then the minimum wage set to m_by_year[k] in year k + 1 (from its first
    month). Indicators at the end of each year."""
    e, p, band, b = _common(prm)
    years = len(m_by_year)
    total = prm.burn_in + 12 * years
    m_path = np.full(total + 1, prm.m)
    for k, mk in enumerate(m_by_year):
        m_path[prm.burn_in + 12 * k + 1: prm.burn_in + 12 * (k + 1) + 1] = mk
    res = _run(e, p, band, b, m_path, prm.beta, prm.lambda0, prm.lambda1, prm.delta, prm.n_bands,
               prm.visibility, 0.0, total + 1, total, prm.burn_in, 12, prm.seed)
    rows = _summarise(with_(prm, measure_every=12), np.asarray(m_by_year, dtype=float), *res, e, p)
    return pd.DataFrame([{"year": k + 1, "m": m_by_year[k], **r} for k, r in enumerate(rows)])


def measure(m: float, wage: np.ndarray, firm: np.ndarray, e: np.ndarray, p: np.ndarray,
            covered: np.ndarray) -> dict[str, float]:
    emp = firm >= 0
    w = wage[emp]
    f = firm[emp]
    y = e[emp] * p[f]
    q = np.quantile(w, [0.10, 0.25, 0.50, 0.75, 0.90])
    ws = np.sort(w)
    cov = covered[f]
    out = {
        "employment_rate": float(emp.mean()),
        "mean_wage": float(w.mean()),
        "p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4],
        "p10_p50": q[0] / q[2], "p25_p50": q[1] / q[2], "p90_p50": q[4] / q[2],
        "mean_bottom25": float(ws[: max(1, ws.size // 4)].mean()),
        "labour_share": float(w.sum() / y.sum()),
        "mean_output": float(y.mean()),
        "share_at_floor": float(np.mean(w <= m * 1.001)) if m > 0 else 0.0,
        "kaitz": m / q[2] if m > 0 else 0.0,
        "covered_emp_share": float(cov.mean()),
        "mean_wage_covered": float(w[cov].mean()) if cov.any() else np.nan,
        "mean_wage_uncovered": float(w[~cov].mean()) if (~cov).any() else np.nan,
    }
    return out
