"""State-expanded aircraft routing: maintenance checks enforced exactly.

Every flight is copied once per maintenance state ``k = 0, ..., gamma - 1``, the number
of consecutive nights the aircraft has spent away from a check. A connection ``i -> j``
becomes one variable per state of ``i`` in which it is allowed:

* a night spent on the ground at the base with enough time for a check resets the
  state to 0;
* any other night increases it by one, and the variable does not exist if the state
  would reach ``gamma``;
* a connection within the day keeps the state.

Flow is conserved at every (flight, state) node and each flight is entered exactly once
over all its states, so every rotation is a cycle that visits each flight once with a
consistent state sequence. That is the exact per-aircraft rule "a check at least every
``gamma`` nights" written as linear constraints, without the cut loop that a
path-based formulation needs (the four-day version of maintenance routing is
NP-complete; Talluri, 1998).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from airline_ops.network.model import Network, crosses_count_line
from airline_ops.planning.arcs import AircraftArcs
from airline_ops.planning.rules import PlanningRules


@dataclass(frozen=True)
class RoutingVariables:
    """Variables of the state-expanded routing model."""

    arc: NDArray[np.int64]
    state_from: NDArray[np.int64]
    state_to: NDArray[np.int64]
    gamma: int

    @property
    def size(self) -> int:
        """Number of variables."""
        return int(self.arc.size)


def build_routing_variables(network: Network, rules: PlanningRules, arcs: AircraftArcs) -> RoutingVariables:
    """One variable per (arc, state) pair that keeps the maintenance state below ``gamma``."""
    gamma = rules.maintenance_max_nights
    air_nights = np.array([crosses_count_line(f.dep, f.block) for f in network.flights], dtype=np.int64)
    arc_ids: list[int] = []
    s_from: list[int] = []
    s_to: list[int] = []
    for a in range(arcs.size):
        head = int(arcs.head[a])
        for k in range(gamma):
            nxt = int(air_nights[head]) if arcs.maintenance[a] else k + int(arcs.cross[a]) + int(air_nights[head])
            if nxt >= gamma:
                continue
            arc_ids.append(a)
            s_from.append(k)
            s_to.append(nxt)
    return RoutingVariables(
        arc=np.array(arc_ids, dtype=np.int64),
        state_from=np.array(s_from, dtype=np.int64),
        state_to=np.array(s_to, dtype=np.int64),
        gamma=gamma,
    )


@dataclass(frozen=True, slots=True)
class RoutingRows:
    """Row offsets of the routing block inside a larger model."""

    enter: int  # one row per flight: entered exactly once
    conserve: int  # one row per (flight, state): in - out = 0
    fleet: int  # one row: aircraft counted at the count line


def routing_column_entries(
    var: int,
    rv: RoutingVariables,
    arcs: AircraftArcs,
    rows: RoutingRows,
    link_row: int | None,
) -> tuple[list[int], list[float]]:
    """Sparse coefficients of one routing variable."""
    a = int(rv.arc[var])
    tail, head = int(arcs.tail[a]), int(arcs.head[a])
    g = rv.gamma
    idx = [
        rows.enter + head,
        rows.conserve + head * g + int(rv.state_to[var]),
        rows.conserve + tail * g + int(rv.state_from[var]),
    ]
    val = [1.0, 1.0, -1.0]
    if arcs.cross[a]:
        idx.append(rows.fleet)
        val.append(float(arcs.cross[a]))
    if link_row is not None:
        idx.append(link_row)
        val.append(-1.0)
    return idx, val


def successor_from_values(rv: RoutingVariables, arcs: AircraftArcs, values: NDArray[np.float64]) -> dict[int, int]:
    """Successor map from routing variable values (an arc is on if any of its state copies is)."""
    on_arcs = np.unique(rv.arc[values > 0.5])
    return {int(arcs.tail[a]): int(arcs.head[a]) for a in on_arcs}
