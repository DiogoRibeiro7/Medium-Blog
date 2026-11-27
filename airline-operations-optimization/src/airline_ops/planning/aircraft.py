"""Aircraft routing alone: the first stage of the sequential approach.

The daily routing problem, with arcs ``a`` copied by maintenance state (see
:mod:`airline_ops.planning.routing`)::

    min  sum_{a,k} c_a x_{a,k}
    s.t. sum_{(a,k) into f} x_{a,k} = 1                    for every flight f
         sum_{(a,k) into (f,k')} x - sum_{(a,k') out of (f,k')} x = 0   for every flight and state
         sum_{a,k} n_a x_{a,k} <= F - (flights airborne at the count line)
         x in {0, 1}

``n_a`` is the number of times connection ``a`` crosses the daily count line, so the
fleet row counts aircraft. Without states and the fleet row this is an assignment
problem with a totally unimodular matrix; with them it is a small MIP.
"""

from __future__ import annotations

import time

import highspy
import numpy as np
from numpy.typing import NDArray

from airline_ops.config import PlanningConfig
from airline_ops.errors import InfeasibleModelError, SolverError
from airline_ops.network.model import Network
from airline_ops.planning.arcs import AircraftArcs, maintenance_violations
from airline_ops.planning.routing import (
    RoutingRows,
    build_routing_variables,
    routing_column_entries,
    successor_from_values,
)
from airline_ops.planning.rules import PlanningRules

INF = highspy.kHighsInf


def solve_aircraft_routing(
    network: Network,
    rules: PlanningRules,
    arcs: AircraftArcs,
    arc_cost: NDArray[np.float64],
    planning: PlanningConfig,
) -> tuple[dict[int, int], dict[str, float]]:
    """Minimum-cost daily routing that respects the fleet size and maintenance checks.

    Returns:
        The successor map and solver statistics.

    Raises:
        InfeasibleModelError: When no routing satisfies the constraints.

    """
    t0 = time.perf_counter()
    n = network.n_flights
    rv = build_routing_variables(network, rules, arcs)
    g = rv.gamma
    rows = RoutingRows(enter=0, conserve=n, fleet=n + n * g)
    n_rows = rows.fleet + 1
    fleet_rhs = float(network.fleet_size - network.airborne_at_count_line())
    lower = np.concatenate([np.ones(n), np.zeros(n * g), [-INF]])
    upper = np.concatenate([np.ones(n), np.zeros(n * g), [fleet_rhs]])
    h = highspy.Highs()  # type: ignore[no-untyped-call]
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", planning.threads)
    h.setOptionValue("random_seed", 0)
    h.setOptionValue("time_limit", float(planning.time_limit_s))
    h.setOptionValue("mip_rel_gap", float(planning.mip_gap))
    h.addRows(n_rows, lower, upper, 0, np.zeros(n_rows, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0))
    starts: list[int] = []
    idx: list[int] = []
    val: list[float] = []
    for v in range(rv.size):
        starts.append(len(idx))
        i, x = routing_column_entries(v, rv, arcs, rows, None)
        idx += i
        val += x
    costs = arc_cost[rv.arc].astype(float)
    h.addCols(
        rv.size, costs, np.zeros(rv.size), np.ones(rv.size), len(idx),
        np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32), np.array(val),
    )  # fmt: skip
    h.changeColsIntegrality(
        rv.size, np.arange(rv.size, dtype=np.int32), np.full(rv.size, highspy.HighsVarType.kInteger)
    )
    h.run()
    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kInfeasible:
        raise InfeasibleModelError("aircraft routing", "no routing satisfies the fleet size and maintenance checks")
    if status not in (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kTimeLimit):
        raise SolverError(f"HiGHS returned status {status}")
    values = np.asarray(h.getSolution().col_value, dtype=float)
    successor = successor_from_values(rv, arcs, values)
    if len(successor) != n:
        raise SolverError("routing solution does not cover every flight")
    leftover = maintenance_violations(successor, network, rules)
    if leftover:
        raise SolverError(f"state-expanded routing returned {len(leftover)} maintenance violation(s)")
    info = h.getInfo()
    return successor, {
        "objective": float(info.objective_function_value),
        "mip_dual_bound": float(info.mip_dual_bound),
        "variables": float(rv.size),
        "seconds": time.perf_counter() - t0,
    }
