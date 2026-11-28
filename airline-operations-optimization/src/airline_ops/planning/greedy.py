"""Greedy planning: first-in-first-out aircraft, then a greedy cover of the flights by pairings.

Aircraft: at every station, each departure takes the aircraft that has been ready the
longest (first in, first out). For a single fleet this attains the minimum fleet, but it
ignores maintenance and crews; the validator reports any maintenance violation.

Crews: the classical greedy heuristic for set covering (Chvátal, 1979). Among all legal
pairings under the FIFO routing, repeatedly take the one with the lowest cost per flight
it newly covers, counting a deadhead charge for every flight it would cover twice,
until every flight is covered. No linear programme is solved.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from airline_ops.config import Config
from airline_ops.network.model import COUNT_LINE_UTC, DAY, Network
from airline_ops.planning.duties import build_pairing_space, enumerate_duties
from airline_ops.planning.plan import Pairing, PlannedDuty
from airline_ops.planning.rules import PlanningRules


def fifo_routing(network: Network, rules: PlanningRules) -> dict[int, int]:
    """First-in-first-out daily routing (minimum fleet, no maintenance or crew logic)."""
    succ: dict[int, int] = {}
    for code in network.airports:
        turn = rules.aircraft_turn(code)
        events: list[tuple[int, int, int]] = []  # (time after the count line, 0 = ready / 1 = departure, flight)
        for f in network.flights:
            if f.destination == code:
                events.append(((f.arr + turn - COUNT_LINE_UTC) % DAY, 0, f.idx))
            if f.origin == code:
                events.append(((f.dep - COUNT_LINE_UTC) % DAY, 1, f.idx))
        events.sort()
        ready: deque[int] = deque()
        waiting: deque[int] = deque()
        for _, kind, f_idx in events:
            if kind == 0:
                ready.append(f_idx)
            elif ready:
                succ[ready.popleft()] = f_idx
            else:
                waiting.append(f_idx)
        if len(ready) != len(waiting):
            raise ValueError(f"unbalanced station {code}")
        # Overnight: aircraft still on the ground at the end of the day take the departures that found
        # none. A pair whose arrival falls less than one turn before the departure would read as a
        # too-short same-day turn in a daily routing, so each departure takes the first leftover
        # aircraft that avoids it.
        arrivals = {f.idx: f.arr for f in network.flights}
        departures = {f.idx: f.dep for f in network.flights}
        left = list(ready)
        for d_idx in waiting:
            pick = next((a for a in left if (departures[d_idx] - arrivals[a]) % DAY >= turn), left[0])
            left.remove(pick)
            succ[pick] = d_idx
    return succ


def greedy_pairings(network: Network, rules: PlanningRules, succ: dict[int, int], config: Config) -> list[Pairing]:
    """Greedy set cover of the flights by legal pairings under a fixed routing."""
    c = config.costs
    duty_set = enumerate_duties(network, rules, allowed_short=set(succ.items()))
    space = build_pairing_space(duty_set, rules)
    duties = duty_set.duties
    n_single = int(space.single.size)
    members: list[tuple[int, ...]] = []
    costs: list[float] = []
    parts: list[tuple[tuple[int, ...], int, int]] = []  # duty ids, rest, nights
    for k in range(space.size):
        if k < n_single:
            ids: tuple[int, ...] = (int(space.single[k]),)
            rest, nights = 0, 0
        else:
            m = k - n_single
            ids = (int(space.pair_first[m]), int(space.pair_second[m]))
            rest, nights = int(space.pair_rest[m]), int(space.pair_nights[m])
        members.append(tuple(f for d in ids for f in duties[d].flights))
        cost = sum(c.crew_hour_eur * max(duties[d].length / 60.0, c.min_duty_paid_h) for d in ids)
        costs.append(cost + c.hotel_layover_eur * nights)
        parts.append((ids, rest, nights))
    covered = np.zeros(network.n_flights, dtype=bool)
    chosen: list[int] = []
    while not covered.all():
        best_k, best_ratio = -1, np.inf
        for k, flights in enumerate(members):
            new = sum(1 for f in flights if not covered[f])
            if new == 0:
                continue
            ratio = (costs[k] + c.deadhead_eur * (len(flights) - new)) / new
            if ratio < best_ratio:
                best_k, best_ratio = k, ratio
        if best_k < 0:
            break  # remaining flights have no legal pairing; the validator reports them
        chosen.append(best_k)
        covered[list(members[best_k])] = True
    pairings: list[Pairing] = []
    for k in chosen:
        ids, rest, nights = parts[k]
        planned: list[PlannedDuty] = []
        for n_d, d_id in enumerate(ids):
            d = duties[d_id]
            start = 0 if n_d == 0 else planned[0].length + rest
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
        pairings.append(Pairing(duties=tuple(planned), cost=costs[k], hotel_nights=nights))
    return pairings
