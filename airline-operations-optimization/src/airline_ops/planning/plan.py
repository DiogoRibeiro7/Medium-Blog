"""Plans, their planned cost, and an independent validator.

The validator re-derives every rule from the flight times instead of trusting the
planner that produced the plan: aircraft turns, the fleet size, maintenance checks,
crew connections, the EASA FDP limit, rest away from base, and the rule that a crew
connection too short to change aircraft must stay with its aircraft. Every planner's
output goes through it, and a plan with violations is reported as infeasible.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from airline_ops.config import CostConfig
from airline_ops.network.model import DAY, Network
from airline_ops.planning.arcs import aircraft_count, maintenance_violations, rotations
from airline_ops.planning.rules import PlanningRules


@dataclass(frozen=True, slots=True)
class PlannedDuty:
    """A duty inside a pairing; times are minutes after the pairing's first report."""

    flights: tuple[int, ...]
    start: int
    length: int
    fdp: int
    report_utc: int
    deadhead: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class Pairing:
    """A crew pairing: one or two duties from the base back to the base."""

    duties: tuple[PlannedDuty, ...]
    cost: float
    hotel_nights: int

    @property
    def flights(self) -> tuple[int, ...]:
        """All flights of the pairing, operated or deadheaded, in order."""
        return tuple(f for d in self.duties for f in d.flights)

    @property
    def operated(self) -> tuple[int, ...]:
        """Flights the crew operates (not deadheads)."""
        return tuple(f for d in self.duties for f, dh in zip(d.flights, d.deadhead, strict=True) if not dh)

    @property
    def span(self) -> int:
        """Minutes from first report to last release (time away from base)."""
        last = self.duties[-1]
        return last.start + last.length


@dataclass
class Plan:
    """A daily plan: an aircraft routing and a set of crew pairings."""

    name: str
    successor: dict[int, int]
    pairings: list[Pairing]
    solve_seconds: float = 0.0
    lower_bound: float | None = None
    objective: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def operating_crew(self) -> dict[int, int]:
        """Pairing index that operates each flight (the first one listed operates; others deadhead)."""
        out: dict[int, int] = {}
        for p_idx, p in enumerate(self.pairings):
            for f in p.operated:
                out.setdefault(f, p_idx)
        return out


def planned_cost(plan: Plan, network: Network, costs: CostConfig) -> dict[str, float]:
    """Planned daily cost by component (euros)."""
    aircraft = aircraft_count(plan.successor, network)
    crew_pay = 0.0
    hotel = 0.0
    deadhead = 0.0
    for p in plan.pairings:
        for d in p.duties:
            crew_pay += costs.crew_hour_eur * max(d.length / 60.0, costs.min_duty_paid_h)
            deadhead += costs.deadhead_eur * sum(d.deadhead)
        hotel += costs.hotel_layover_eur * p.hotel_nights
    # Over-coverage: a flight with more than one operating pairing carries the extra crews as deadheads.
    counts: dict[int, int] = defaultdict(int)
    for p in plan.pairings:
        for f in p.operated:
            counts[f] += 1
    deadhead += costs.deadhead_eur * sum(c - 1 for c in counts.values() if c > 1)
    ownership = costs.aircraft_day_eur * aircraft
    return {
        "aircraft": float(aircraft),
        "ownership": ownership,
        "crew_pay": crew_pay,
        "hotel": hotel,
        "deadhead": deadhead,
        "total": ownership + crew_pay + hotel + deadhead,
    }


def validate_plan(plan: Plan, network: Network, rules: PlanningRules) -> list[str]:
    """Return every hard-constraint violation of a plan (an empty list means feasible).

    The rules passed here should be the nominal ones (no robustness buffer): a robust
    plan is checked against the same legal minimums as every other plan.
    """
    v: list[str] = []
    flights = network.flights
    n = network.n_flights
    succ = plan.successor
    # Aircraft routing.
    if sorted(succ) != list(range(n)):
        v.append("aircraft: not every flight has exactly one successor")
    if sorted(succ.values()) != list(range(n)):
        v.append("aircraft: not every flight has exactly one predecessor")
    for i, j in succ.items():
        fi, fj = flights[i], flights[j]
        if fi.destination != fj.origin:
            v.append(f"aircraft: {fi.number}->{fj.number} changes station")
            continue
        gap = (fj.dep - fi.arr) % DAY
        if gap < rules.turn_min[fi.destination]:
            v.append(f"aircraft: {fi.number}->{fj.number} turn {gap} min below minimum")
    if not v:
        used = aircraft_count(succ, network)
        if used > network.fleet_size:
            v.append(f"aircraft: routing needs {used} aircraft, fleet is {network.fleet_size}")
        for mv in maintenance_violations(succ, network, rules):
            first = flights[mv.cycle[0]].number
            if mv.never_checked:
                nights = mv.nights_without_check
                v.append(f"maintenance: rotation from {first} ({nights} nights) never has a check night")
            else:
                v.append(f"maintenance: rotation from {first} goes {mv.nights_without_check} nights without a check")
    # Crew pairings.
    operated: dict[int, int] = defaultdict(int)
    for p_idx, p in enumerate(plan.pairings):
        tag = f"pairing {p_idx}"
        if not p.duties:
            v.append(f"{tag}: no duties")
            continue
        if len(p.duties) > rules.max_duties:
            v.append(f"{tag}: {len(p.duties)} duties exceeds {rules.max_duties}")
        all_flights = p.flights
        if len(set(all_flights)) != len(all_flights):
            v.append(f"{tag}: a daily flight appears twice")
        if flights[all_flights[0]].origin != rules.base or flights[all_flights[-1]].destination != rules.base:
            v.append(f"{tag}: does not start and end at the base")
        for d_idx, d in enumerate(p.duties):
            v.extend(_check_duty(d, plan, network, rules, f"{tag} duty {d_idx}"))
            if d_idx > 0:
                prev = p.duties[d_idx - 1]
                rest = d.start - (prev.start + prev.length)
                if rest < rules.min_rest_away(prev.length):
                    v.append(f"{tag}: rest {rest} min below the minimum after a {prev.length}-min duty")
                if flights[prev.flights[-1]].destination != flights[d.flights[0]].origin:
                    v.append(f"{tag}: duties do not connect in space")
        for f in p.operated:
            operated[f] += 1
    missing = [flights[f].number for f in range(n) if operated[f] == 0]
    if missing:
        v.append(f"crew: {len(missing)} flight(s) without an operating crew, e.g. {missing[:3]}")
    return v


def _check_duty(d: PlannedDuty, plan: Plan, network: Network, rules: PlanningRules, tag: str) -> list[str]:
    v: list[str] = []
    flights = network.flights
    op = rules.operator
    first = flights[d.flights[0]]
    if (first.dep - op.report_before_min) % DAY != d.report_utc % DAY:
        v.append(f"{tag}: report time inconsistent with first departure")
    t = op.report_before_min
    sectors = 0
    for k, f_idx in enumerate(d.flights):
        f = flights[f_idx]
        if k > 0:
            prev = flights[d.flights[k - 1]]
            if prev.destination != f.origin:
                v.append(f"{tag}: {prev.number}->{f.number} changes station")
            gap = (f.dep - prev.arr) % DAY
            same_aircraft = plan.successor.get(prev.idx) == f.idx
            need = rules.turn_min[f.origin] if same_aircraft else rules.crew_change_min
            if gap < need:
                v.append(f"{tag}: crew connection {prev.number}->{f.number} of {gap} min below {need}")
            if gap > rules.max_sit_min:
                v.append(f"{tag}: sit of {gap} min exceeds {rules.max_sit_min}")
            t += gap
        t += f.block
        if not d.deadhead[k]:
            sectors += 1
    if sectors > rules.max_sectors:
        v.append(f"{tag}: {sectors} sectors exceeds {rules.max_sectors}")
    if t != d.fdp:
        v.append(f"{tag}: FDP {d.fdp} inconsistent with flight times ({t})")
    report_local = network.airports[first.origin].local_minute(d.report_utc)
    limit = rules.max_fdp(report_local, max(1, min(sectors, 10)))
    if d.fdp > limit:
        v.append(f"{tag}: FDP {d.fdp} min exceeds the Table 2 limit of {limit}")
    if d.length != d.fdp + op.post_flight_min:
        v.append(f"{tag}: duty length inconsistent with FDP plus post-flight time")
    return v


def rotation_summary(plan: Plan, network: Network) -> dict[str, float]:
    """Descriptive statistics of the aircraft routing."""
    cycles = rotations(plan.successor)
    block = sum(f.block for f in network.flights)
    used = aircraft_count(plan.successor, network)
    return {
        "rotations": float(len(cycles)),
        "aircraft": float(used),
        "block_hours_per_aircraft_day": block / 60.0 / used if used else 0.0,
    }
