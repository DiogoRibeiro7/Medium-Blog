"""Aircraft connection arcs of the daily routing problem and the maintenance rule.

In the daily problem each flight is followed by exactly one flight of the same aircraft
(possibly on the next day). The successor relation is a permutation of the flights;
its cycles are the aircraft rotations, and an aircraft is needed for every crossing of
the daily count line. A connection ``i -> j`` is an arc when the ground time from the
arrival of ``i`` to the next departure of ``j`` is at least the minimum turn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from airline_ops.network.model import DAY, Network, crosses_count_line
from airline_ops.planning.rules import PlanningRules


@dataclass(frozen=True)
class AircraftArcs:
    """All feasible aircraft connections."""

    tail: NDArray[np.int64]
    head: NDArray[np.int64]
    gap: NDArray[np.int64]
    cross: NDArray[np.int64]
    maintenance: NDArray[np.bool_]

    @property
    def size(self) -> int:
        """Number of arcs."""
        return int(self.tail.size)

    def index(self) -> dict[tuple[int, int], int]:
        """Map ``(tail, head)`` to the arc index."""
        return {(int(t), int(h)): k for k, (t, h) in enumerate(zip(self.tail, self.head, strict=True))}


def build_aircraft_arcs(network: Network, rules: PlanningRules) -> AircraftArcs:
    """List every connection whose ground time is at least the (buffered) minimum turn."""
    tails: list[int] = []
    heads: list[int] = []
    gaps: list[int] = []
    cross: list[int] = []
    maint: list[bool] = []
    for i in network.flights:
        station = i.destination
        turn = rules.aircraft_turn(station)
        for j in network.flights:
            if j.origin != station or j.idx == i.idx:
                continue
            gap = (j.dep - i.arr) % DAY
            if gap < turn:
                continue
            c = crosses_count_line(i.arr, gap)
            tails.append(i.idx)
            heads.append(j.idx)
            gaps.append(gap)
            cross.append(c)
            maint.append(bool(c and station == rules.base and gap >= rules.maintenance_min_ground))
    return AircraftArcs(
        tail=np.array(tails, dtype=np.int64),
        head=np.array(heads, dtype=np.int64),
        gap=np.array(gaps, dtype=np.int64),
        cross=np.array(cross, dtype=np.int64),
        maintenance=np.array(maint, dtype=bool),
    )


def rotations(successor: dict[int, int]) -> list[list[int]]:
    """Cycles of a successor permutation, each starting at its smallest flight index."""
    seen: set[int] = set()
    cycles: list[list[int]] = []
    for start in sorted(successor):
        if start in seen:
            continue
        cycle = [start]
        seen.add(start)
        nxt = successor[start]
        while nxt != start:
            if nxt in seen:
                raise ValueError("successor map is not a permutation")
            cycle.append(nxt)
            seen.add(nxt)
            nxt = successor[nxt]
        cycles.append(cycle)
    return cycles


@dataclass(frozen=True, slots=True)
class MaintenanceViolation:
    """A run of consecutive nights without a maintenance check, and the arcs on it."""

    cycle: tuple[int, ...]
    arcs: tuple[tuple[int, int], ...]
    nights_without_check: int
    never_checked: bool = False


def night_sequence(cycle: list[int], network: Network, rules: PlanningRules) -> list[tuple[int, bool]]:
    """For each night of a rotation: the position in the cycle where it falls and whether it is a check.

    A night falls on a connection (on the ground) or on a flight (in the air). Position
    ``k`` refers to the connection from ``cycle[k]`` to its successor, or to the flight
    ``cycle[k]`` itself; the boolean says whether the night is a maintenance check.
    """
    nights: list[tuple[int, bool]] = []
    flights = network.flights
    for k, f_idx in enumerate(cycle):
        f = flights[f_idx]
        for _ in range(crosses_count_line(f.dep, f.block)):
            nights.append((k, False))
        nxt = flights[cycle[(k + 1) % len(cycle)]]
        gap = (nxt.dep - f.arr) % DAY
        for _ in range(crosses_count_line(f.arr, gap)):
            is_check = f.destination == rules.base and gap >= rules.maintenance_min_ground
            nights.append((k, is_check))
    return nights


def maintenance_violations(
    successor: dict[int, int], network: Network, rules: PlanningRules
) -> list[MaintenanceViolation]:
    """Rotations that go more than ``maintenance_max_nights`` nights without a check at the base."""
    limit = rules.maintenance_max_nights
    out: list[MaintenanceViolation] = []
    for cycle in rotations(successor):
        nights = night_sequence(cycle, network, rules)
        n = len(nights)
        if n == 0:
            continue
        arcs_all = tuple((cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle)))
        if not any(check for _, check in nights):
            out.append(
                MaintenanceViolation(cycle=tuple(cycle), arcs=arcs_all, nights_without_check=n, never_checked=True)
            )
            continue
        # Rotate so the sequence starts right after a check, then find runs of non-check nights.
        first_check = next(i for i, (_, c) in enumerate(nights) if c)
        order = nights[first_check + 1 :] + nights[: first_check + 1]
        run: list[int] = []
        for pos, check in order:
            if check:
                if len(run) >= limit:
                    out.append(_run_violation(cycle, run[:limit], len(run)))
                run = []
            else:
                run.append(pos)
    return out


def _run_violation(cycle: list[int], positions: list[int], run_length: int) -> MaintenanceViolation:
    """Arcs of the path that spans the first ``limit`` nights of a run without checks."""
    n = len(cycle)
    start, end = positions[0], positions[-1]
    span = (end - start) % n
    arcs = tuple((cycle[(start + s) % n], cycle[(start + s + 1) % n]) for s in range(span + 1))
    return MaintenanceViolation(cycle=tuple(cycle), arcs=arcs, nights_without_check=run_length)


def aircraft_count(successor: dict[int, int], network: Network) -> int:
    """Aircraft needed by a daily routing: count-line crossings in the air and on the ground."""
    total = network.airborne_at_count_line()
    for i, j in successor.items():
        fi, fj = network.flights[i], network.flights[j]
        total += crosses_count_line(fi.arr, (fj.dep - fi.arr) % DAY)
    return total
