"""Discrete-event simulation of a daily plan over several days.

The plan is unrolled over the horizon: each aircraft rotation of ``L`` days becomes
``L`` tails, each pairing of up to two days starts every day, and unused aircraft wait
at the hub as spares. Flights are processed in order of scheduled departure. A flight
leaves at the latest of its schedule, its aircraft being ready, its crew being ready
and legal, and its arrival slot, plus its own primary delay.

The recovery policy is deliberately simple and identical for every plan:

1. wait for the aircraft and the crew (delay);
2. at the hub, swap to a spare aircraft when the assigned one is late by more than the
   swap threshold or grounded;
3. at the hub, call a standby crew when the planned crew is sick or would exceed the
   EASA FDP limit (with commander's discretion of up to two hours, if enabled);
4. cancel a flight that would leave more than ``max_delay_min`` late, land or leave
   inside the hub curfew, or has no aircraft or crew; cancelling a hub departure also
   cancels the aircraft's return from the outstation (out-and-back pair);
5. ferry an aircraft and reposition a crew left at the wrong airport by a cancellation.

Real operations control centres re-optimise; this policy is a fixed yardstick, so the
differences between plans come from the plans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from airline_ops.config import Config
from airline_ops.network.model import DAY, Network
from airline_ops.planning.arcs import aircraft_count, rotations
from airline_ops.planning.plan import Plan
from airline_ops.planning.rules import PlanningRules
from airline_ops.regulation.ftl import COMMANDER_DISCRETION_MAX_MIN, MIN_REST_AWAY_MIN
from airline_ops.simulation.delays import DelayModel
from airline_ops.simulation.scenarios import ScenarioEvents

CAUSES = ("none", "airline", "technical", "crew", "atc")
EXTRAORDINARY = {"atc"}
BIND_NONE, BIND_AIRCRAFT, BIND_CREW, BIND_SLOT = 0, 1, 2, 3
STANDBY_LOCAL = (5 * 60, 17 * 60)


@dataclass
class Unrolled:
    """A daily plan unrolled over the horizon."""

    n_flights: int
    horizon: int
    tail_seq: list[list[int]]
    tail_init_ready: list[float]
    tail_init_loc: list[str]
    n_planned_tails: int
    crew_seq: list[list[int]]
    crew_op: list[list[bool]]
    crew_duty_start: list[list[bool]]
    crew_planned_report: list[list[int]]
    crew_planned_len: list[list[int]]
    crew_pairing: list[int]

    def instance(self, day: int, flight: int) -> int:
        """Instance id of a flight on a day."""
        return day * self.n_flights + flight


def unroll(network: Network, plan: Plan, horizon: int, rules: PlanningRules) -> Unrolled:
    """Unroll a daily plan into tails and crews over ``horizon`` days."""
    flights = network.flights
    n = network.n_flights
    tail_seq: list[list[int]] = []
    tail_ready: list[float] = []
    tail_loc: list[str] = []
    for cycle in rotations(plan.successor):
        m = len(cycle)
        t = flights[cycle[0]].dep
        offsets: list[int] = []
        for k, f_idx in enumerate(cycle):
            offsets.append((t - flights[f_idx].dep) // DAY)
            nxt = cycle[(k + 1) % m]
            t += flights[f_idx].block + (flights[nxt].dep - flights[f_idx].arr) % DAY
        length = (t - flights[cycle[0]].dep) // DAY
        for r in range(length):
            seq: list[tuple[int, int]] = []
            for day in range(horizon):
                for k, f_idx in enumerate(cycle):
                    if (day - offsets[k]) % length == r:
                        seq.append((day * DAY + flights[f_idx].dep, day * n + f_idx))
            seq.sort()
            insts = [i for _, i in seq]
            if not insts:
                continue
            first = insts[0] % n
            pos = cycle.index(first)
            prev = flights[cycle[(pos - 1) % m]]
            gap = (flights[first].dep - prev.arr) % DAY
            first_dep = (insts[0] // n) * DAY + flights[first].dep
            tail_seq.append(insts)
            tail_ready.append(float(first_dep - gap + rules.turn_min[prev.destination]))
            tail_loc.append(flights[first].origin)
    n_planned = len(tail_seq)
    for _ in range(max(0, network.fleet_size - aircraft_count(plan.successor, network))):
        tail_seq.append([])
        tail_ready.append(-math.inf)
        tail_loc.append(network.hub)

    crew_seq: list[list[int]] = []
    crew_op: list[list[bool]] = []
    crew_start: list[list[bool]] = []
    crew_report: list[list[int]] = []
    crew_len: list[list[int]] = []
    crew_pairing: list[int] = []
    before = rules.operator.report_before_min
    for p_idx, p in enumerate(plan.pairings):
        rel: list[
            tuple[int, int, bool, bool, int, int]
        ] = []  # (dep rel to report, flight, op, duty start, rel report, len)
        for d in p.duties:
            t = d.start + before
            for k, f_idx in enumerate(d.flights):
                if k > 0:
                    t += (flights[f_idx].dep - flights[d.flights[k - 1]].arr) % DAY
                rel.append((t, f_idx, not d.deadhead[k], k == 0, d.start, d.length))
                t += flights[f_idx].block
        first_dep_rel = rel[0][0]
        span_days = (rel[-1][0] // DAY) + 2
        for s in range(-span_days, horizon):
            report_abs = s * DAY + flights[rel[0][1]].dep - first_dep_rel
            seq_i: list[int] = []
            ops: list[bool] = []
            starts: list[bool] = []
            reports: list[int] = []
            lens: list[int] = []
            for t_rel, f_idx, op, start, d_start, d_len in rel:
                abs_dep = report_abs + t_rel
                day, rem = divmod(abs_dep - flights[f_idx].dep, DAY)
                if rem != 0:
                    raise ValueError("pairing times are inconsistent with the schedule")
                if 0 <= day < horizon:
                    seq_i.append(day * n + f_idx)
                    ops.append(op)
                    starts.append(start)
                    reports.append(report_abs + d_start)
                    lens.append(d_len)
            if seq_i:
                crew_seq.append(seq_i)
                crew_op.append(ops)
                crew_start.append(starts)
                crew_report.append(reports)
                crew_len.append(lens)
                crew_pairing.append(p_idx)
    return Unrolled(
        n_flights=n,
        horizon=horizon,
        tail_seq=tail_seq,
        tail_init_ready=tail_ready,
        tail_init_loc=tail_loc,
        n_planned_tails=n_planned,
        crew_seq=crew_seq,
        crew_op=crew_op,
        crew_duty_start=crew_start,
        crew_planned_report=crew_report,
        crew_planned_len=crew_len,
        crew_pairing=crew_pairing,
    )


@dataclass
class Outcome:
    """Result of one simulated replication of one plan in one scenario."""

    operated: NDArray[np.bool_]
    dep_delay: NDArray[np.float64]
    arr_delay: NDArray[np.float64]
    reactionary: NDArray[np.float64]
    cause: NDArray[np.int64]
    bind_type: NDArray[np.int64]
    bind_pred: NDArray[np.int64]
    sched_dep: NDArray[np.float64]
    counters: dict[str, float] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)


def _hub_arrival_slots(network: Network, horizon: int, events: ScenarioEvents) -> dict[int, float]:
    """Ration-by-schedule arrival slots at the hub during a capacity restriction."""
    if events.capacity is None:
        return {}
    start, end, factor = events.capacity
    n = network.n_flights
    arrivals = sorted((f.arr, f.idx) for f in network.flights if f.destination == network.hub and start <= f.arr < end)
    if not arrivals:
        return {}
    hours = max(1.0, (end - start) / 60.0)
    rate = max(1.0, math.floor(factor * len(arrivals) / hours))
    spacing = 60.0 / rate
    slots: dict[int, float] = {}
    last = -math.inf
    for arr, f_idx in arrivals:
        slot = max(float(arr), last + spacing)
        slots[0 * n + f_idx] = slot
        last = slot
    del horizon
    return slots


def simulate(
    network: Network,
    plan: Plan,
    unrolled: Unrolled,
    config: Config,
    delay_model: DelayModel,
    events: ScenarioEvents,
    base_seed: int,
    rep: int,
) -> Outcome:
    """Simulate one replication. The primary delays depend only on ``(base_seed, rep)``."""
    rules = PlanningRules.from_config(config, network)
    sim = config.simulation
    flights = network.flights
    airports = network.airports
    n = network.n_flights
    horizon = unrolled.horizon
    n_inst = n * horizon
    hub = network.hub
    before, post = rules.operator.report_before_min, rules.operator.post_flight_min
    hub_off = airports[hub].utc_offset_min

    rng = np.random.default_rng([base_seed, rep])
    primary = delay_model.departure_delays(rng, n_inst)
    factor = delay_model.block_factors(rng, n_inst)
    sched_dep = np.array([(i // n) * DAY + flights[i % n].dep for i in range(n_inst)], dtype=float)
    sched_arr = sched_dep + np.array([flights[i % n].block for i in range(n_inst)], dtype=float)
    primary_cause = np.full(n_inst, 1, dtype=np.int64)  # airline
    if events.scale_window is not None:
        lo, hi = events.scale_window
        in_win = (sched_dep >= lo) & (sched_dep < hi)
        primary = np.where(in_win, primary * events.primary_scale, primary)
    for f_idx, extra in events.inbound_delays.items():
        primary[f_idx] += extra
        primary_cause[f_idx] = 2  # technical
    slots = _hub_arrival_slots(network, horizon, events)

    tail_seq = [list(s) for s in unrolled.tail_seq]
    tail_of = np.full(n_inst, -1, dtype=np.int64)
    for t, seq in enumerate(tail_seq):
        for i in seq:
            tail_of[i] = t
    tail_ready = list(unrolled.tail_init_ready)
    tail_loc = list(unrolled.tail_init_loc)
    tail_cause = [0] * len(tail_seq)
    tail_last = [-1] * len(tail_seq)
    n_tails = len(tail_seq)
    is_spare = [t >= unrolled.n_planned_tails for t in range(n_tails)]
    aog_tail: dict[int, tuple[float, float]] = {}
    for f_idx in events.aog_flights:
        t = int(tail_of[f_idx])
        if t >= 0:
            start = sched_dep[f_idx] - 60.0
            aog_tail[t] = (start, start + events.aog_minutes)

    # Crews: planned crew instances, then standby crews appended on call.
    n_crews = len(unrolled.crew_seq)
    crew_seq = [list(s) for s in unrolled.crew_seq]
    crew_op = [list(s) for s in unrolled.crew_op]
    crew_start = [list(s) for s in unrolled.crew_duty_start]
    crew_report_plan = [list(s) for s in unrolled.crew_planned_report]
    crew_len_plan = [list(s) for s in unrolled.crew_planned_len]
    pos_in_crew: dict[tuple[int, int], int] = {}
    op_crew = np.full(n_inst, -1, dtype=np.int64)
    riders: dict[int, list[int]] = {}
    for c in range(n_crews):
        for k, i in enumerate(crew_seq[c]):
            pos_in_crew[(c, i)] = k
            if crew_op[c][k] and op_crew[i] < 0:
                op_crew[i] = c
            else:
                riders.setdefault(i, []).append(c)
    crew_loc: list[str] = [flights[s[0] % n].origin for s in crew_seq]
    crew_last_arr: list[float] = [-math.inf] * n_crews
    crew_last_tail: list[int] = [-1] * n_crews
    crew_last_inst: list[int] = [-1] * n_crews
    crew_cause: list[int] = [0] * n_crews
    crew_report: list[float] = [math.nan] * n_crews
    crew_report_station: list[str] = [hub] * n_crews
    crew_sectors: list[int] = [0] * n_crews
    crew_duty_len_plan: list[int] = [0] * n_crews
    duty_log: list[tuple[float, float, int]] = []  # (actual report, last on-blocks, planned length)
    sick: set[int] = {int(op_crew[f_idx]) for f_idx in events.sick_flights if op_crew[f_idx] >= 0}
    standby_used = [0] * horizon

    operated = np.zeros(n_inst, dtype=bool)
    cancelled = np.zeros(n_inst, dtype=bool)
    dep_delay = np.full(n_inst, np.nan)
    arr_delay = np.full(n_inst, np.nan)
    actual_dep = np.full(n_inst, np.nan)
    actual_arr = np.full(n_inst, np.nan)
    reactionary = np.zeros(n_inst)
    cause = np.zeros(n_inst, dtype=np.int64)
    bind_type = np.zeros(n_inst, dtype=np.int64)
    bind_pred = np.full(n_inst, -1, dtype=np.int64)
    counters: dict[str, float] = {
        k: 0.0
        for k in (
            "swaps", "ferries", "ferry_block_min", "standby_calls", "discretion_uses", "repositionings",
            "cancel_curfew", "cancel_delay", "cancel_aircraft", "cancel_crew", "cancel_pair", "fdp_replacements",
        )
    }  # fmt: skip

    def after_curfew(code: str, t: float) -> float:
        curfew = airports[code].curfew_utc
        if curfew is None or not airports[code].in_curfew(int(t)):
            return t
        end = curfew[1]
        return t + ((end - int(t)) % DAY)

    def block_between(a: str, b: str) -> float:
        for f in flights:
            if f.origin == a and f.destination == b:
                return float(f.block)
        return float(block_between(a, hub) + block_between(hub, b) + 60.0)

    def cancel(i: int, reason: str) -> None:
        cancelled[i] = True
        counters[reason] += 1
        # Out-and-back rule: the aircraft stays at the hub, so its return leg is cancelled too.
        t = int(tail_of[i])
        if flights[i % n].origin == hub and t >= 0:
            seq = tail_seq[t]
            k = seq.index(i)
            if k + 1 < len(seq):
                nxt = seq[k + 1]
                if flights[nxt % n].origin == flights[i % n].destination and not cancelled[nxt]:
                    cancelled[nxt] = True
                    counters["cancel_pair"] += 1
                    cause[nxt] = cause[i]

    def call_standby(c_old: int, i: int, t_detect: float) -> int | None:
        day = int(sched_dep[i] // DAY)
        if flights[i % n].origin != hub or standby_used[day] >= config.crew.reserve_crews_per_day:
            return None
        lo = day * DAY + STANDBY_LOCAL[0] - hub_off
        hi = day * DAY + STANDBY_LOCAL[1] - hub_off
        report = max(t_detect + config.crew.reserve_callout_min, sched_dep[i] - before, lo)
        if report > hi:
            return None
        standby_used[day] += 1
        counters["standby_calls"] += 1
        c_new = len(crew_seq)
        k0 = pos_in_crew[(c_old, i)]
        crew_seq.append(crew_seq[c_old][k0:])
        crew_op.append(crew_op[c_old][k0:])
        crew_start.append([True, *crew_start[c_old][k0 + 1 :]])
        crew_report_plan.append([int(report), *crew_report_plan[c_old][k0 + 1 :]])
        crew_len_plan.append(crew_len_plan[c_old][k0:])
        for k, inst in enumerate(crew_seq[c_new]):
            pos_in_crew[(c_new, inst)] = k
            if op_crew[inst] == c_old:
                op_crew[inst] = c_new
            elif c_old in riders.get(inst, []):
                riders[inst] = [c_new if r == c_old else r for r in riders[inst]]
        crew_loc.append(hub)
        crew_last_arr.append(-math.inf)
        crew_last_tail.append(-1)
        crew_last_inst.append(-1)
        crew_cause.append(0)
        crew_report.append(math.nan)
        crew_report_station.append(hub)
        crew_sectors.append(0)
        crew_duty_len_plan.append(0)
        return c_new

    def crew_ready_time(c: int, i: int, tail: int) -> tuple[float, int]:
        """Earliest departure the crew allows, and the cause it carries (starts a duty if planned)."""
        k = pos_in_crew[(c, i)]
        origin = flights[i % n].origin
        ready = crew_last_arr[c]
        if crew_loc[c] != origin:
            counters["repositionings"] += 1
            leg = block_between(crew_loc[c], origin) + 60.0
            ready = max(ready, sched_dep[i] - before - leg) + leg
            crew_loc[c] = origin
        if crew_start[c][k] or math.isnan(crew_report[c]):
            if crew_report[c] == crew_report[c] and crew_last_arr[c] > -math.inf:  # close the previous duty
                duty_log.append((crew_report[c], crew_last_arr[c], crew_duty_len_plan[c]))
            prev_len = crew_last_arr[c] + post - crew_report[c] if crew_last_arr[c] > -math.inf else 0.0
            rest_ok = (
                crew_last_arr[c] + post + max(prev_len, MIN_REST_AWAY_MIN)
                if crew_last_arr[c] > -math.inf
                else -math.inf
            )
            report = max(float(crew_report_plan[c][k]), rest_ok, ready - before)
            crew_report[c] = report
            crew_report_station[c] = origin
            crew_sectors[c] = 0
            crew_duty_len_plan[c] = crew_len_plan[c][k]
            return report + before, crew_cause[c]
        same = crew_last_tail[c] == tail
        conn = rules.turn_min[origin] if same else rules.crew_change_min
        return ready + conn, crew_cause[c]

    order = np.argsort(sched_dep, kind="stable")
    for i in (int(x) for x in order):
        if cancelled[i]:
            cause[i] = cause[i] or 1
            continue
        f = flights[i % n]
        t = int(tail_of[i])
        if t < 0:
            cancel(i, "cancel_aircraft")
            continue
        # A grounded aircraft is unavailable until the end of its AOG window.
        if t in aog_tail and tail_ready[t] < aog_tail[t][1] and max(tail_ready[t], sched_dep[i]) >= aog_tail[t][0]:
            tail_ready[t], tail_cause[t] = aog_tail[t][1], 2
        # Aircraft at the wrong airport: ferry it in.
        if tail_loc[t] != f.origin:
            blk = block_between(tail_loc[t], f.origin)
            turn = rules.turn_min[f.origin]
            ferry_arr = after_curfew(f.origin, max(tail_ready[t], sched_dep[i] - blk - turn) + blk)
            counters["ferries"] += 1
            counters["ferry_block_min"] += blk
            tail_loc[t] = f.origin
            tail_ready[t] = ferry_arr + rules.turn_min[f.origin]
        ac_ready, ac_cause = tail_ready[t], tail_cause[t]
        # Swap to a spare at the hub when the aircraft is badly late or grounded.
        if f.origin == hub and ac_ready - sched_dep[i] > sim.swap_threshold_min:
            spares = [s for s in range(n_tails) if is_spare[s] and tail_loc[s] == hub and s not in aog_tail]
            if spares:
                s = min(spares, key=lambda x: tail_ready[x])
                if tail_ready[s] < ac_ready - 30.0:
                    k = tail_seq[t].index(i)
                    rest = tail_seq[t][k:]
                    tail_seq[t] = tail_seq[t][:k]
                    tail_seq[s] = rest
                    for inst in rest:
                        tail_of[inst] = s
                    is_spare[s], is_spare[t] = False, True
                    counters["swaps"] += 1
                    t = s
                    ac_ready, ac_cause = tail_ready[s], 0
        # Crew.
        c = int(op_crew[i])
        if c < 0:
            cancel(i, "cancel_crew")
            continue
        if c in sick and crew_start[c][pos_in_crew[(c, i)]]:
            sick.discard(c)
            replacement = call_standby(c, i, float(crew_report_plan[c][pos_in_crew[(c, i)]]))
            if replacement is None:
                cancel(i, "cancel_crew")
                cause[i] = 3
                continue
            c = replacement
            crew_cause[c] = 3
        cr_ready, cr_cause = crew_ready_time(c, i, t)
        slot_min = slots[i] - f.block if i in slots else -math.inf
        candidates = [(sched_dep[i], BIND_NONE, 0, -1), (ac_ready, BIND_AIRCRAFT, ac_cause, tail_last[t]),
                      (cr_ready, BIND_CREW, cr_cause, crew_last_inst[c]), (slot_min, BIND_SLOT, 4, -1)]  # fmt: skip
        ready, btype, bcause, bpred = max(candidates, key=lambda x: x[0])
        dep = ready + primary[i]
        # FDP check for the operating crew; standby or cancel if illegal.
        sectors = crew_sectors[c] + 1
        report_local = airports[crew_report_station[c]].local_minute(int(crew_report[c]))
        limit = rules.max_fdp(report_local, min(sectors, 10))
        fdp = dep + f.block - crew_report[c]
        if fdp > limit:
            allowance = COMMANDER_DISCRETION_MAX_MIN if sim.allow_commander_discretion else 0
            if fdp <= limit + allowance:
                counters["discretion_uses"] += 1
            else:
                replacement = call_standby(c, i, dep - 120.0)
                if replacement is None:
                    cancel(i, "cancel_crew")
                    cause[i] = bcause if bcause else 3
                    continue
                counters["fdp_replacements"] += 1
                c = replacement
                cr_ready, _ = crew_ready_time(c, i, t)
                ready = max(sched_dep[i], ac_ready, cr_ready, slot_min)
                dep = ready + primary[i]
        if dep - sched_dep[i] > sim.max_delay_min:
            cancel(i, "cancel_delay")
            cause[i] = bcause if bcause else 1
            continue
        arr = dep + f.block * factor[i]
        if airports[f.origin].in_curfew(dep) or airports[f.destination].in_curfew(arr):
            cancel(i, "cancel_curfew")
            cause[i] = bcause if bcause else 1
            continue
        # Commit.
        operated[i] = True
        actual_dep[i], actual_arr[i] = dep, arr
        dep_delay[i] = dep - sched_dep[i]
        arr_delay[i] = arr - sched_arr[i]
        reactionary[i] = max(0.0, ready - sched_dep[i])
        if ready > sched_dep[i] and reactionary[i] >= primary[i]:
            cause[i], bind_type[i], bind_pred[i] = (bcause or 1), btype, bpred
        else:
            cause[i] = int(primary_cause[i]) if primary[i] > 0 else 0
        tail_loc[t] = f.destination
        tail_ready[t] = arr + rules.turn_min[f.destination]
        tail_cause[t] = int(cause[i])
        tail_last[t] = i
        for cc in [c, *riders.get(i, [])]:
            crew_loc[cc] = f.destination
            crew_last_arr[cc] = arr
            crew_last_tail[cc] = t
            crew_last_inst[cc] = i
            crew_cause[cc] = int(cause[i])
            if math.isnan(crew_report[cc]):
                crew_report[cc] = dep - before
        crew_sectors[c] += 1
    for c in range(len(crew_seq)):
        if crew_last_arr[c] > -math.inf and crew_report[c] == crew_report[c]:
            duty_log.append((crew_report[c], crew_last_arr[c], crew_duty_len_plan[c]))

    out = Outcome(
        operated=operated,
        dep_delay=dep_delay,
        arr_delay=arr_delay,
        reactionary=reactionary,
        cause=cause,
        bind_type=bind_type,
        bind_pred=bind_pred,
        sched_dep=sched_dep,
        counters=counters,
    )
    _passenger_and_cost_accounting(network, plan, config, out, actual_dep, actual_arr, sched_arr, duty_log, horizon)
    return out


def _passenger_and_cost_accounting(
    network: Network,
    plan: Plan,
    config: Config,
    out: Outcome,
    actual_dep: NDArray[np.float64],
    actual_arr: NDArray[np.float64],
    sched_arr: NDArray[np.float64],
    duty_log: list[tuple[float, float, int]],
    horizon: int,
) -> None:
    flights = network.flights
    n = network.n_flights
    hub = network.hub
    costs = config.costs
    mct = float(config.network.min_connection_pax_min)
    beyond = float(config.simulation.rebooking_horizon_min)
    # Operated departures by route, sorted by actual departure, for rebooking.
    by_route: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for i in np.nonzero(out.operated)[0]:
        f = flights[int(i) % n]
        by_route.setdefault((f.origin, f.destination), []).append((float(actual_dep[i]), float(actual_arr[i])))
    for v in by_route.values():
        v.sort()

    def earliest(origin: str, dest: str, t: float) -> float:
        if (origin, dest) in by_route:
            for d, a in by_route[(origin, dest)]:
                if d >= t:
                    return a
            return math.inf
        if hub in (origin, dest) or origin == dest:
            return math.inf  # the route itself has no operated departure left
        mid = earliest(origin, hub, t)
        return math.inf if math.isinf(mid) else earliest(hub, dest, mid + mct)

    pax = delay_sum = missed = cancelled_pax = stranded = 0.0
    eu261 = care = hotel = rebook = 0.0
    for g in network.passengers:
        legs = g.legs
        for day in range(horizon):
            first = day * n + legs[0]
            if len(legs) == 2:
                gap = (flights[legs[1]].dep - flights[legs[0]].arr) % DAY
                d2 = (day * DAY + flights[legs[0]].arr + gap) // DAY
                if d2 >= horizon:
                    continue
                last = d2 * n + legs[1]
            else:
                last = first
            pax += g.count
            sched_final = float(sched_arr[last])
            disrupted = False
            leg_cause = 0
            if len(legs) == 1:
                if out.operated[first]:
                    final = float(actual_arr[first])
                else:
                    disrupted = True
                    cancelled_pax += g.count
                    leg_cause = int(out.cause[first])
                    final = earliest(g.origin, g.destination, float(out.sched_dep[first]))
            else:
                ok1, ok2 = out.operated[first], out.operated[last]
                if ok1 and ok2 and actual_dep[last] >= actual_arr[first] + mct:
                    final = float(actual_arr[last])
                else:
                    disrupted = True
                    if not ok1:
                        cancelled_pax += g.count
                        leg_cause = int(out.cause[first])
                        final = earliest(g.origin, g.destination, float(out.sched_dep[first]))
                    else:
                        if ok2:
                            missed += g.count
                            leg_cause = int(out.cause[first])
                        else:
                            cancelled_pax += g.count
                            leg_cause = int(out.cause[last])
                        final = earliest(hub, g.destination, float(actual_arr[first]) + mct)
            if math.isinf(final):
                final = sched_final + beyond
                stranded += g.count
                hotel += costs.hotel_pax_eur * g.count
            delay = max(0.0, final - sched_final)
            delay_sum += delay * g.count
            if disrupted:
                rebook += costs.rebooking_eur * g.count
            if delay >= 180.0:
                why = CAUSES[leg_cause] if leg_cause < len(CAUSES) else "airline"
                if why not in EXTRAORDINARY:
                    amount = costs.eu261_medium_eur if g.distance_km > 1500 else costs.eu261_short_eur
                    eu261 += amount * g.count
            threshold = 120.0 if g.distance_km <= 1500 else 180.0
            if delay >= threshold:
                care += costs.care_meal_eur * g.count
    c = out.counters
    c.update(
        {
            "passengers": pax,
            "pax_delay_min": delay_sum,
            "missed_connection_pax": missed,
            "cancelled_pax": cancelled_pax,
            "stranded_pax": stranded,
        }
    )
    post = 30.0  # post-flight duty, as in OperatorTimes
    overtime = sum(max(0.0, (last + post - rep) - plan_len) for rep, last, plan_len in duty_log)
    paid = sum(max(last + post - rep, 60.0 * costs.min_duty_paid_h) for rep, last, _ in duty_log)
    block_flown = float(sum(flights[int(i) % n].block for i in np.nonzero(out.operated)[0]))
    c["overtime_min"] = overtime
    c["crew_paid_min"] = paid
    c["block_min"] = block_flown
    c["operated"] = float(out.operated.sum())
    c["cancelled"] = float((~out.operated).sum())
    arr_d = np.where(out.operated, out.arr_delay, 0.0)
    dep_d = np.where(out.operated, out.dep_delay, 0.0)
    out.costs = {
        "delay": costs.delay_minute_eur * float(np.clip(arr_d, 0, None).sum()),
        "overtime": costs.crew_hour_eur * costs.overtime_factor * overtime / 60.0,
        "standby": costs.reserve_callout_eur * c["standby_calls"],
        "ferry": costs.ferry_block_hour_eur * c["ferry_block_min"] / 60.0,
        "repositioning": costs.deadhead_eur * c["repositionings"],
        "eu261": eu261,
        "care": care,
        "pax_hotel": hotel,
        "rebooking": rebook,
    }
    out.costs["operational_total"] = float(sum(out.costs.values()))
    out.costs["eurocontrol_reference"] = costs.eurocontrol_minute_eur * float(np.clip(dep_d, 0, None).sum())
    del plan
