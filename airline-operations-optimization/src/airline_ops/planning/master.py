"""Column generation for crew pairing, alone or integrated with aircraft routing.

Restricted master problem (RMP), for pairings ``p`` (columns generated on demand),
aircraft arcs ``a = (i, j)`` and flights ``f``::

    min  sum_p c_p y_p + sum_a c_a x_a + c_dh sum_f s_f + M sum_f u_f
    s.t. sum_p A_fp y_p - s_f + u_f = 1                 for every flight f   (cover)
         sum_p B_kp y_p - sum_s x_{k,s} <= 0            for every short connection k (link)
         routing rows of :mod:`airline_ops.planning.routing` on x_{a,s}     (routing, maintenance)
         sum_{a,s} n_a x_{a,s} <= F - (flights airborne at the count line)  (fleet)
         y, x in {0, 1}, s, u >= 0

``A_fp`` is one when pairing ``p`` operates flight ``f``; over-coverage ``s_f`` means an
extra crew rides as a deadhead. ``B_kp`` is one when pairing ``p`` uses short
connection ``k``, a connection too short for the crew to change aircraft, which is then
allowed only if the aircraft makes the same connection (``x_k = 1``). ``u_f`` are
artificial variables with a large cost; a positive ``u_f`` in the final solution means
that flight ``f`` cannot be covered, and the plan is reported as infeasible. ``x_{a,s}`` is arc
``a`` taken by an aircraft in maintenance state ``s`` (nights since its last check).

The *sequential* approach fixes ``x`` to a routing found first; the *integrated*
approach optimises ``x`` and ``y`` together. Pricing computes the reduced cost of every
legal pairing from the duals ``pi_f`` (cover) and ``sigma_k <= 0`` (link)::

    rc_p = c_p - sum_f A_fp pi_f - sum_k B_kp sigma_k

which is additive over duties, so a pairing's reduced cost is the sum of its duties'
reduced costs plus hotel cost: a shortest path in the duty network.

Integer solutions come from solving the final RMP as a mixed-integer programme
(price-and-branch). Its gap to the column-generation LP bound is an upper bound on
the true optimality gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import highspy
import numpy as np
from numpy.typing import NDArray

from airline_ops.config import PlanningConfig
from airline_ops.errors import InfeasibleModelError, SolverError
from airline_ops.logging_setup import get_logger
from airline_ops.network.model import Network
from airline_ops.planning.arcs import AircraftArcs, maintenance_violations
from airline_ops.planning.duties import DutySet, PairingSpace
from airline_ops.planning.plan import Pairing, PlannedDuty
from airline_ops.planning.routing import (
    RoutingRows,
    build_routing_variables,
    routing_column_entries,
    successor_from_values,
)
from airline_ops.planning.rules import PlanningRules

LOG = get_logger("planning.master")
INF = highspy.kHighsInf


@dataclass
class MasterInputs:
    """Everything the master problem needs."""

    network: Network
    rules: PlanningRules
    duty_set: DutySet
    space: PairingSpace
    arcs: AircraftArcs
    arc_cost: NDArray[np.float64]
    duty_cost: NDArray[np.float64]
    duty_planned_cost: NDArray[np.float64]
    hotel_night_cost: float
    deadhead_cost: float
    fixed_successor: dict[int, int] | None
    fleet_limit: int
    artificial_cost: float = 1.0e6


@dataclass
class MasterResult:
    """Solution of the master problem."""

    successor: dict[int, int]
    pairings: list[Pairing]
    lp_bound: float
    mip_objective: float
    mip_dual_bound: float
    uncovered: list[int]
    history: list[dict[str, float]] = field(default_factory=list)
    columns: int = 0
    maintenance_cuts: int = 0
    seconds: float = 0.0

    @property
    def gap(self) -> float:
        """Relative gap of the integer solution to the column-generation LP bound."""
        if self.mip_objective <= 0:
            return 0.0
        return max(0.0, (self.mip_objective - self.lp_bound) / self.mip_objective)


def _status_ok(h: highspy.Highs, *, mip: bool) -> None:
    status = h.getModelStatus()
    ok = {highspy.HighsModelStatus.kOptimal}
    if mip:
        ok |= {highspy.HighsModelStatus.kTimeLimit, highspy.HighsModelStatus.kSolutionLimit}
    if status == highspy.HighsModelStatus.kInfeasible:
        raise InfeasibleModelError("master", "the restricted master problem is infeasible")
    if status not in ok:
        raise SolverError(f"HiGHS returned status {status}")


class ColumnGenerationMaster:
    """Restricted master problem with pricing over the implicit pairing space."""

    def __init__(self, inputs: MasterInputs, planning: PlanningConfig) -> None:
        self.inp = inputs
        self.cfg = planning
        net = inputs.network
        self.n_f = net.n_flights
        self.shorts = inputs.duty_set.short_connections
        self.n_s = len(self.shorts)
        self.short_index = {sc: k for k, sc in enumerate(self.shorts)}
        arc_index = inputs.arcs.index()
        missing = [sc for sc in self.shorts if sc not in arc_index]
        if missing:
            raise InfeasibleModelError("master", f"{len(missing)} short connection(s) have no aircraft arc")
        self.h = highspy.Highs()  # type: ignore[no-untyped-call]
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("threads", planning.threads)
        self.h.setOptionValue("random_seed", 0)
        self._build_rows()
        self._build_static_columns(arc_index)
        self.added: set[int] = set()
        self.column_ids: list[int] = []
        self.n_single = int(inputs.space.single.size)

    # Row layout -----------------------------------------------------------------
    def _build_rows(self) -> None:
        n_f, n_s = self.n_f, self.n_s
        net = self.inp.network
        g = self.inp.rules.maintenance_max_nights
        self.row_cover = 0
        self.row_link = n_f
        self.routing_rows = RoutingRows(enter=n_f + n_s, conserve=2 * n_f + n_s, fleet=2 * n_f + n_s + n_f * g)
        n_rows = self.routing_rows.fleet + 1
        airborne = float(net.airborne_at_count_line())
        lower = np.concatenate([np.ones(n_f), np.full(n_s, -INF), np.ones(n_f), np.zeros(n_f * g), [-INF]])
        upper = np.concatenate(
            [np.ones(n_f), np.zeros(n_s), np.ones(n_f), np.zeros(n_f * g), [float(self.inp.fleet_limit) - airborne]]
        )
        self.h.addRows(
            n_rows, lower, upper, 0, np.zeros(n_rows, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0)
        )

    def _build_static_columns(self, arc_index: dict[tuple[int, int], int]) -> None:
        n_f = self.n_f
        inp = self.inp
        # Artificial (uncovered) and surplus (deadhead) columns for every flight.
        idx = np.arange(n_f, dtype=np.int32)
        self.h.addCols(
            n_f, np.full(n_f, inp.artificial_cost), np.zeros(n_f), np.full(n_f, INF), n_f,
            np.arange(n_f, dtype=np.int32), idx, np.ones(n_f),
        )  # fmt: skip
        self.h.addCols(
            n_f, np.full(n_f, inp.deadhead_cost), np.zeros(n_f), np.full(n_f, INF), n_f,
            np.arange(n_f, dtype=np.int32), idx, -np.ones(n_f),
        )  # fmt: skip
        self.col_u = 0
        self.col_s = n_f
        self.col_x = 2 * n_f
        arcs = inp.arcs
        self.rv = build_routing_variables(inp.network, inp.rules, arcs)
        rv = self.rv
        short_of_arc = {arc_index[sc]: k for k, sc in enumerate(self.shorts)}
        starts: list[int] = []
        rows: list[int] = []
        vals: list[float] = []
        lb = np.zeros(rv.size)
        ub = np.ones(rv.size)
        for v in range(rv.size):
            a = int(rv.arc[v])
            link = self.row_link + short_of_arc[a] if a in short_of_arc else None
            starts.append(len(rows))
            r, x = routing_column_entries(v, rv, arcs, self.routing_rows, link)
            rows += r
            vals += x
            if inp.fixed_successor is not None and inp.fixed_successor.get(int(arcs.tail[a])) != int(arcs.head[a]):
                ub[v] = 0.0
        self.h.addCols(
            rv.size, inp.arc_cost[rv.arc].astype(float), lb, ub, len(rows),
            np.array(starts, dtype=np.int32), np.array(rows, dtype=np.int32), np.array(vals),
        )  # fmt: skip
        self.col_y = self.col_x + rv.size

    # Pricing ----------------------------------------------------------------------
    def _pairing_columns(self, ids: NDArray[np.int64]) -> tuple[list[float], list[int], list[int], list[float]]:
        """Costs and sparse coefficients of pairing columns."""
        ds, sp = self.inp.duty_set, self.inp.space
        costs: list[float] = []
        starts: list[int] = []
        rows: list[int] = []
        vals: list[float] = []
        for pid in ids.tolist():
            duty_ids = [int(sp.single[pid])] if pid < self.n_single else [
                int(sp.pair_first[pid - self.n_single]), int(sp.pair_second[pid - self.n_single])
            ]  # fmt: skip
            cost = float(sum(self.inp.duty_cost[d] for d in duty_ids))
            if pid >= self.n_single:
                cost += self.inp.hotel_night_cost * float(sp.pair_nights[pid - self.n_single])
            starts.append(len(rows))
            for d in duty_ids:
                duty = ds.duties[d]
                rows.extend(self.row_cover + f for f in duty.flights)
                vals.extend(1.0 for _ in duty.flights)
                for sc in duty.short_connections:
                    rows.append(self.row_link + self.short_index[sc])
                    vals.append(1.0)
            costs.append(cost)
        return costs, starts, rows, vals

    def _reduced_costs(self, duals: NDArray[np.float64]) -> NDArray[np.float64]:
        ds, sp = self.inp.duty_set, self.inp.space
        pi = np.append(duals[self.row_cover : self.row_cover + self.n_f], 0.0)
        sigma = np.append(duals[self.row_link : self.row_link + self.n_s], 0.0)
        fl = np.where(ds.flight_incidence < 0, self.n_f, ds.flight_incidence)
        sh = np.where(ds.short_incidence < 0, self.n_s, ds.short_incidence)
        rc_duty = self.inp.duty_cost - pi[fl].sum(axis=1) - sigma[sh].sum(axis=1)
        rc_single = rc_duty[sp.single]
        rc_pair = rc_duty[sp.pair_first] + rc_duty[sp.pair_second] + self.inp.hotel_night_cost * sp.pair_nights
        rc: NDArray[np.float64] = np.concatenate([rc_single, rc_pair])
        if self.added:
            rc[np.fromiter(self.added, dtype=np.int64)] = np.inf
        return rc

    def solve_lp(self) -> list[dict[str, float]]:
        """Column generation to LP optimality (or the iteration limit)."""
        history: list[dict[str, float]] = []
        tol = self.cfg.cg_tolerance
        t0 = time.perf_counter()
        for it in range(self.cfg.cg_max_iterations):
            self.h.run()
            _status_ok(self.h, mip=False)
            z = float(self.h.getInfo().objective_function_value)
            duals = np.asarray(self.h.getSolution().row_dual, dtype=float)
            rc = self._reduced_costs(duals)
            min_rc = float(rc.min()) if rc.size else 0.0
            lagrangian = z + min(0.0, min_rc) * self.n_f
            history.append(
                {
                    "iteration": float(it),
                    "lp_objective": z,
                    "lagrangian_bound": lagrangian,
                    "min_reduced_cost": min_rc,
                    "columns": float(len(self.column_ids)),
                    "seconds": time.perf_counter() - t0,
                }
            )
            if min_rc >= -tol:
                break
            k = min(self.cfg.cg_columns_per_iteration, int((rc < -tol).sum()))
            chosen = np.argpartition(rc, k - 1)[:k] if k < rc.size else np.arange(rc.size)
            chosen = chosen[rc[chosen] < -tol]
            costs, starts, rows, vals = self._pairing_columns(chosen.astype(np.int64))
            self.h.addCols(
                len(costs), np.array(costs), np.zeros(len(costs)), np.full(len(costs), INF), len(rows),
                np.array(starts, dtype=np.int32), np.array(rows, dtype=np.int32), np.array(vals),
            )  # fmt: skip
            self.added.update(int(c) for c in chosen)
            self.column_ids.extend(int(c) for c in chosen)
        else:
            LOG.warning("column generation hit the iteration limit", extra={"fields": {"iterations": len(history)}})
        return history

    # Integer stage ----------------------------------------------------------------
    def solve_mip(self, history: list[dict[str, float]]) -> MasterResult:
        """Solve the final restricted master as a MIP (price-and-branch)."""
        t0 = time.perf_counter()
        lp_bound = history[-1]["lp_objective"] if history else 0.0
        n_cols = self.h.getNumCol()
        integer_cols = np.arange(self.col_x, n_cols, dtype=np.int32)
        self.h.changeColsIntegrality(
            integer_cols.size, integer_cols, np.full(integer_cols.size, highspy.HighsVarType.kInteger)
        )
        u_cols = np.arange(self.col_u, self.col_u + self.n_f, dtype=np.int32)
        self.h.changeColsIntegrality(u_cols.size, u_cols, np.full(u_cols.size, highspy.HighsVarType.kInteger))
        self.h.setOptionValue("time_limit", float(self.cfg.time_limit_s))
        self.h.setOptionValue("mip_rel_gap", float(self.cfg.mip_gap))
        self.h.run()
        _status_ok(self.h, mip=True)
        values = np.asarray(self.h.getSolution().col_value, dtype=float)
        successor = self._successor(values)
        leftover = maintenance_violations(successor, self.inp.network, self.inp.rules)
        if leftover and self.inp.fixed_successor is None:
            raise SolverError(f"state-expanded routing returned {len(leftover)} maintenance violation(s)")
        info = self.h.getInfo()
        values = np.asarray(self.h.getSolution().col_value, dtype=float)
        uncovered = [f for f in range(self.n_f) if values[self.col_u + f] > 0.5]
        result = MasterResult(
            successor=successor,
            pairings=self._pairings(values),
            lp_bound=lp_bound,
            mip_objective=float(info.objective_function_value),
            mip_dual_bound=float(info.mip_dual_bound),
            uncovered=uncovered,
            history=history,
            columns=len(self.column_ids),
            maintenance_cuts=0,
            seconds=time.perf_counter() - t0,
        )
        return result

    def _successor(self, values: NDArray[np.float64]) -> dict[int, int]:
        succ = successor_from_values(self.rv, self.inp.arcs, values[self.col_x : self.col_x + self.rv.size])
        if len(succ) != self.n_f:
            raise SolverError("integer solution does not route every flight")
        return succ

    def _pairings(self, values: NDArray[np.float64]) -> list[Pairing]:
        ds, sp = self.inp.duty_set, self.inp.space
        out: list[Pairing] = []
        for k, pid in enumerate(self.column_ids):
            if values[self.col_y + k] < 0.5:
                continue
            if pid < self.n_single:
                duty_ids, rest, nights = [int(sp.single[pid])], 0, 0
            else:
                m = pid - self.n_single
                duty_ids = [int(sp.pair_first[m]), int(sp.pair_second[m])]
                rest, nights = int(sp.pair_rest[m]), int(sp.pair_nights[m])
            planned: list[PlannedDuty] = []
            start = 0
            cost = self.inp.hotel_night_cost * nights
            for n_d, d_id in enumerate(duty_ids):
                d = ds.duties[d_id]
                if n_d == 1:
                    start = planned[0].length + rest
                planned.append(
                    PlannedDuty(
                        flights=d.flights,
                        start=start,
                        length=d.length,
                        fdp=d.fdp,
                        report_utc=d.report_utc,
                        deadhead=tuple(False for _ in d.flights),
                    )
                )
                cost += float(self.inp.duty_planned_cost[d_id])
            out.append(Pairing(duties=tuple(planned), cost=cost, hotel_nights=nights))
        return out
