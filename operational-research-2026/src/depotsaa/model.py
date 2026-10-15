"""The two-stage depot model (linopy + HiGHS) and the weekly recourse LP (highspy).

`solve_two_stage` builds the extensive form for a matrix of equally likely
weeks. With a single row (a point forecast) it is the deterministic plan.

`Recourse` fixes the open depots and re-solves the weekly routing LP for one
week after another, changing only the demand bounds, so HiGHS starts every
solve from the previous optimal basis.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

import highspy
import linopy
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from scipy import stats

from .instance import Instance


@contextlib.contextmanager
def quiet():
    """Silence the solver banner, which HiGHS writes straight to file descriptor 1."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


def solve_two_stage(inst: Instance, D: np.ndarray, link: bool = False,
                    time_limit: float = 600.0):
    """Extensive form of the sample average approximation.

    D has shape (n_weeks, n_zones); every row is one equally likely week.
    Returns (open vector, objective, solve seconds, n variables, n constraints).
    """
    S = D.shape[0]
    week = pd.RangeIndex(S, name="week")
    site = pd.RangeIndex(inst.n_sites, name="site")
    zone = pd.RangeIndex(inst.n_zones, name="zone")

    demand = xr.DataArray(D, coords=[week, zone])
    c = xr.DataArray(inst.cost, coords=[site, zone])
    u = xr.DataArray(inst.cap, coords=[site])
    f = xr.DataArray(inst.fixed, coords=[site])

    m = linopy.Model()
    y = m.add_variables(binary=True, coords=[site], name="open")
    x = m.add_variables(lower=0, coords=[week, site, zone], name="ship")
    z = m.add_variables(lower=0, coords=[week, zone], name="outsource")

    m.add_constraints(x.sum("site") + z == demand, name="serve")
    m.add_constraints(x.sum("zone") <= u * y, name="capacity")
    if link:
        # textbook tightening: flow may only leave an open depot, zone by zone
        m.add_constraints(x <= demand * y, name="link")

    m.add_objective((f * y).sum() + ((c * x).sum() + inst.penalty * z.sum()) / S)

    t0 = time.perf_counter()
    with quiet():
        status, cond = m.solve(solver_name="highs", io_api="direct",
                               output_flag=False, mip_rel_gap=1e-6,
                               time_limit=time_limit)
    secs = time.perf_counter() - t0
    if status != "ok":
        raise RuntimeError(f"solve failed: {status} {cond}")
    y_val = np.round(m.solution["open"].values).astype(int)
    return y_val, float(m.objective.value), secs, m.nvars, m.ncons


class Recourse:
    """Weekly routing LP for a fixed set of open depots."""

    def __init__(self, inst: Instance, y):
        self.inst = inst
        self.open = np.flatnonzero(y)
        k, n = len(self.open), inst.n_zones
        nx = k * n
        # columns: x[k, i] (depot-major), then z[i]; rows: zone balances, capacities
        rows, cols = [], []
        for a in range(k):
            for i in range(n):
                rows += [i, n + a]
                cols += [a * n + i] * 2
        rows += list(range(n))
        cols += list(range(nx, nx + n))
        A = sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + k, nx + n))

        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = nx + n, n + k
        lp.col_cost_ = np.concatenate([inst.cost[self.open].ravel(),
                                       np.full(n, inst.penalty)])
        lp.col_lower_ = np.zeros(nx + n)
        lp.col_upper_ = np.full(nx + n, highspy.kHighsInf)
        lp.row_lower_ = np.concatenate([inst.mu, np.full(k, -highspy.kHighsInf)])
        lp.row_upper_ = np.concatenate([inst.mu, inst.cap[self.open]])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        with quiet():
            self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.passModel(lp)
        self.idx = np.arange(n, dtype=np.int32)
        self.n_x = nx
        self.fixed = float(inst.fixed[self.open].sum())

    def week(self, d: np.ndarray) -> tuple[float, float]:
        """Optimal (transport cost, outsourced pallets) for one week of demand."""
        self.h.changeRowsBounds(self.inst.n_zones, self.idx, d, d)
        self.h.run()
        out = float(np.sum(self.h.getSolution().col_value[self.n_x:]))
        transport = self.h.getInfo().objective_function_value - self.inst.penalty * out
        return transport, out


def evaluate(inst: Instance, y, weeks: np.ndarray, cache: dict | None = None) -> dict:
    """Weekly total cost of design y on every row of `weeks`.

    Pass the same `cache` dict for the same set of weeks to avoid re-scoring a
    design that has already been scored.
    """
    key = tuple(int(v) for v in y)
    if cache is not None and key in cache:
        return cache[key]
    r = Recourse(inst, y)
    tr, out = np.empty(len(weeks)), np.empty(len(weeks))
    for s, d in enumerate(weeks):
        tr[s], out[s] = r.week(d)
    res = dict(total=r.fixed + tr + inst.penalty * out, fixed=r.fixed,
               transport=tr, outsourced=out,
               capacity=float(inst.cap[np.flatnonzero(y)].sum()))
    if cache is not None:
        cache[key] = res
    return res


def ci95(a) -> tuple[float, float]:
    """Mean and 95% t half-width."""
    a = np.asarray(a, float)
    return a.mean(), stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))


def names(y) -> str:
    return "{" + ", ".join(f"D{j + 1}" for j in np.flatnonzero(y)) + "}"
