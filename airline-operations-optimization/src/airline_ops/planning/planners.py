"""The four planners compared in the article.

* ``greedy``: first-in-first-out aircraft, crews that follow their aircraft.
* ``sequential``: aircraft routing first (fewest aircraft, tightest turns), then crew
  pairing by column generation with the routing fixed.
* ``integrated``: aircraft routing and crew pairing in one model.
* ``robust``: the integrated model with a buffer on every aircraft turn and crew
  connection, plus a penalty on the delay each connection is expected to pass on.

Every plan is checked by :func:`airline_ops.planning.plan.validate_plan` against the
nominal rules, so a robust plan is held to the same legal minimums as the others.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from airline_ops.config import Config
from airline_ops.errors import InfeasibleModelError
from airline_ops.logging_setup import get_logger
from airline_ops.network.model import Network
from airline_ops.planning.aircraft import solve_aircraft_routing
from airline_ops.planning.arcs import AircraftArcs, build_aircraft_arcs
from airline_ops.planning.duties import DutySet, build_pairing_space, enumerate_duties
from airline_ops.planning.greedy import fifo_routing, greedy_pairings
from airline_ops.planning.master import ColumnGenerationMaster, MasterInputs
from airline_ops.planning.plan import Plan, validate_plan
from airline_ops.planning.rules import PlanningRules

LOG = get_logger("planning.planners")

# Expected delay passed on by a connection, as a function of (incoming flight, slack in minutes).
ExcessFunction = Callable[[int, float], float]


@dataclass(frozen=True, slots=True)
class RobustSettings:
    """Buffer and penalty of the robust planner."""

    buffer_min: int
    penalty_weight: float


def _duty_planned_cost(duty_set: DutySet, config: Config) -> NDArray[np.float64]:
    c = config.costs
    lengths = np.array([d.length for d in duty_set.duties], dtype=float)
    out: NDArray[np.float64] = c.crew_hour_eur * np.maximum(lengths / 60.0, c.min_duty_paid_h)
    return out


def _arc_penalty(
    network: Network, nominal: PlanningRules, arcs: AircraftArcs, excess: ExcessFunction
) -> NDArray[np.float64]:
    out = np.zeros(arcs.size)
    for a in range(arcs.size):
        i = int(arcs.tail[a])
        slack = float(arcs.gap[a] - nominal.turn_min[network.flights[i].destination])
        out[a] = excess(i, slack)
    return out


def _duty_penalty(duty_set: DutySet, nominal: PlanningRules, excess: ExcessFunction) -> NDArray[np.float64]:
    out = np.zeros(duty_set.size)
    for d in duty_set.duties:
        out[d.idx] = sum(excess(i, float(gap - nominal.crew_change_min)) for i, _, gap in d.change_connections)
    return out


def _solve_master(
    name: str,
    network: Network,
    config: Config,
    rules: PlanningRules,
    arcs: AircraftArcs,
    arc_cost: NDArray[np.float64],
    fixed_successor: dict[int, int] | None,
    duty_penalty_fn: Callable[[DutySet], NDArray[np.float64]] | None,
) -> Plan:
    t0 = time.perf_counter()
    allowed = None
    if fixed_successor is not None:
        allowed = {(i, j) for i, j in fixed_successor.items()}
    duty_set = enumerate_duties(network, rules, allowed_short=allowed)
    space = build_pairing_space(duty_set, rules)
    planned = _duty_planned_cost(duty_set, config)
    penalty = duty_penalty_fn(duty_set) if duty_penalty_fn is not None else np.zeros(duty_set.size)
    master = ColumnGenerationMaster(
        MasterInputs(
            network=network,
            rules=rules,
            duty_set=duty_set,
            space=space,
            arcs=arcs,
            arc_cost=arc_cost,
            duty_cost=planned + penalty,
            duty_planned_cost=planned,
            hotel_night_cost=config.costs.hotel_layover_eur,
            deadhead_cost=config.costs.deadhead_eur,
            fixed_successor=fixed_successor,
            fleet_limit=network.fleet_size,
        ),
        config.planning,
    )
    history = master.solve_lp()
    result = master.solve_mip(history)
    if result.uncovered:
        numbers = [network.flights[f].number for f in result.uncovered]
        raise InfeasibleModelError(name, f"no legal pairing covers {len(numbers)} flight(s): {numbers[:5]}")
    plan = Plan(
        name=name,
        successor=result.successor,
        pairings=result.pairings,
        solve_seconds=time.perf_counter() - t0,
        lower_bound=result.lp_bound,
        objective=result.mip_objective,
        history=result.history,
        notes={
            "duties": duty_set.size,
            "pairings_in_space": space.size,
            "columns_generated": result.columns,
            "cg_iterations": len(result.history),
            "maintenance_cuts": result.maintenance_cuts,
            "gap_to_lp_bound": result.gap,
            "mip_dual_bound": result.mip_dual_bound,
            "short_connections": len(duty_set.short_connections),
            "aircraft_arcs": arcs.size,
        },
    )
    return plan


def plan_greedy(network: Network, config: Config) -> Plan:
    """Greedy plan (no optimisation)."""
    t0 = time.perf_counter()
    rules = PlanningRules.from_config(config, network)
    succ = fifo_routing(network, rules)
    pairings = greedy_pairings(network, rules, succ, config)
    return Plan(name="greedy", successor=succ, pairings=pairings, solve_seconds=time.perf_counter() - t0)


def plan_sequential(network: Network, config: Config) -> Plan:
    """Aircraft routing first, then crew pairing with the routing fixed."""
    t0 = time.perf_counter()
    rules = PlanningRules.from_config(config, network)
    arcs = build_aircraft_arcs(network, rules)
    c = config.costs
    arc_cost = c.aircraft_day_eur * arcs.cross + c.utilization_tiebreak_eur_per_min * arcs.gap
    succ, stats = solve_aircraft_routing(network, rules, arcs, arc_cost.astype(float), config.planning)
    plan = _solve_master(
        "sequential", network, config, rules, arcs, c.aircraft_day_eur * arcs.cross.astype(float), succ, None
    )
    plan.solve_seconds = time.perf_counter() - t0
    plan.notes["routing_seconds"] = stats["seconds"]
    plan.notes["routing_variables"] = stats["variables"]
    return plan


def plan_integrated(network: Network, config: Config) -> Plan:
    """Aircraft routing and crew pairing optimised together."""
    rules = PlanningRules.from_config(config, network)
    arcs = build_aircraft_arcs(network, rules)
    arc_cost = config.costs.aircraft_day_eur * arcs.cross.astype(float)
    return _solve_master("integrated", network, config, rules, arcs, arc_cost, None, None)


def plan_robust(
    network: Network,
    config: Config,
    settings: RobustSettings,
    excess: ExcessFunction,
    *,
    name: str = "robust",
) -> Plan:
    """Integrated model with buffers and an expected-propagation penalty.

    Args:
        network: The schedule.
        config: Experiment configuration.
        settings: Buffer (minutes added to every minimum connection) and penalty weight.
        excess: ``excess(i, slack)`` is the expected delay, in minutes, that a connection
            out of flight ``i`` with ``slack`` minutes above its legal minimum passes on.
            The penalty prices each such minute at the EUROCONTROL reference cost of
            tactical delay, times the weight.
        name: Name given to the plan.

    """
    nominal = PlanningRules.from_config(config, network)
    rules = PlanningRules.from_config(config, network, buffer_min=settings.buffer_min)
    arcs = build_aircraft_arcs(network, rules)
    price = settings.penalty_weight * config.costs.eurocontrol_minute_eur
    arc_cost = config.costs.aircraft_day_eur * arcs.cross.astype(float)
    if price > 0:
        arc_cost = arc_cost + price * _arc_penalty(network, nominal, arcs, excess)

    def duty_penalty(ds: DutySet) -> NDArray[np.float64]:
        return price * _duty_penalty(ds, nominal, excess) if price > 0 else np.zeros(ds.size)

    plan = _solve_master(name, network, config, rules, arcs, arc_cost, None, duty_penalty)
    plan.notes["buffer_min"] = settings.buffer_min
    plan.notes["penalty_weight"] = settings.penalty_weight
    return plan


def check_plan(plan: Plan, network: Network, config: Config) -> list[str]:
    """Validate a plan against the nominal rules and log the outcome."""
    nominal = PlanningRules.from_config(config, network)
    violations = validate_plan(plan, network, nominal)
    plan.notes["violations"] = violations
    LOG.info(
        "plan validated",
        extra={"fields": {"plan": plan.name, "violations": len(violations), "seed": network.seed}},
    )
    return violations
