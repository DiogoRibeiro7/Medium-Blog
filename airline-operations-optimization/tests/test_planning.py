from __future__ import annotations

from dataclasses import replace

import highspy
import numpy as np
import pytest

from airline_ops.config import Config
from airline_ops.network.model import DAY, Network
from airline_ops.planning.arcs import aircraft_count, build_aircraft_arcs, maintenance_violations
from airline_ops.planning.duties import build_pairing_space, enumerate_duties
from airline_ops.planning.plan import Pairing, Plan, PlannedDuty, planned_cost, validate_plan
from airline_ops.planning.planners import (
    RobustSettings,
    plan_greedy,
    plan_integrated,
    plan_robust,
    plan_sequential,
)
from airline_ops.planning.rules import PlanningRules


@pytest.fixture(scope="module")
def rules(network: Network, smoke_config: Config) -> PlanningRules:
    return PlanningRules.from_config(smoke_config, network)


@pytest.fixture(scope="module")
def plans(network: Network, smoke_config: Config) -> dict[str, Plan]:
    zero = lambda i, s: max(0.0, 20.0 - s)  # noqa: E731 - a simple decreasing penalty for the test
    return {
        "greedy": plan_greedy(network, smoke_config),
        "sequential": plan_sequential(network, smoke_config),
        "integrated": plan_integrated(network, smoke_config),
        "robust": plan_robust(network, smoke_config, RobustSettings(10, 1.0), zero),
    }


def test_every_enumerated_duty_is_legal(network: Network, rules: PlanningRules) -> None:
    ds = enumerate_duties(network, rules)
    assert ds.size > network.n_flights
    for d in ds.duties:
        local = network.airports[d.start_station].local_minute(d.report_utc)
        assert d.fdp <= rules.max_fdp(local, d.sectors)
        assert d.sectors <= rules.max_sectors
        for a, b in d.short_connections:
            gap = (network.flights[b].dep - network.flights[a].arr) % DAY
            assert rules.turn_min[network.flights[a].destination] <= gap < rules.crew_change_min


def test_pairing_space_rests_are_legal(network: Network, rules: PlanningRules) -> None:
    ds = enumerate_duties(network, rules)
    sp = build_pairing_space(ds, rules)
    for a, b, rest in zip(sp.pair_first, sp.pair_second, sp.pair_rest, strict=True):
        da, db = ds.duties[int(a)], ds.duties[int(b)]
        assert rest >= max(da.length, 600)
        assert rest <= rules.max_rest_min
        assert da.end_station == db.start_station != rules.base
        assert not set(da.flights) & set(db.flights)
        assert (da.report_utc + da.length + rest - db.report_utc) % DAY == 0


def test_optimised_plans_pass_the_validator(plans: dict[str, Plan], network: Network, rules: PlanningRules) -> None:
    for name in ("sequential", "integrated", "robust"):
        assert validate_plan(plans[name], network, rules) == [], name


def test_greedy_violations_are_reported_not_hidden(
    plans: dict[str, Plan], network: Network, rules: PlanningRules
) -> None:
    violations = validate_plan(plans["greedy"], network, rules)
    assert all(v.startswith("maintenance") for v in violations)


def test_integrated_is_never_costlier_than_sequential(
    plans: dict[str, Plan], network: Network, smoke_config: Config
) -> None:
    seq = planned_cost(plans["sequential"], network, smoke_config.costs)["total"]
    integ = planned_cost(plans["integrated"], network, smoke_config.costs)["total"]
    assert integ <= seq + 1e-6


def test_lp_bound_below_integer_solution(plans: dict[str, Plan]) -> None:
    for name in ("sequential", "integrated", "robust"):
        p = plans[name]
        assert p.lower_bound is not None
        assert p.objective is not None
        assert p.lower_bound <= p.objective + 1e-6


def test_column_generation_matches_full_enumeration(
    network: Network, smoke_config: Config, rules: PlanningRules
) -> None:
    """With the routing fixed, CG's LP value equals the LP over every legal pairing."""
    seq = plan_sequential(network, smoke_config)
    ds = enumerate_duties(network, rules, allowed_short=set(seq.successor.items()))
    sp = build_pairing_space(ds, rules)
    c = smoke_config.costs
    pay = c.crew_hour_eur * np.maximum(np.array([d.length for d in ds.duties]) / 60.0, c.min_duty_paid_h)
    n = network.n_flights
    columns: list[tuple[float, list[int]]] = [(float(pay[k]), list(ds.duties[k].flights)) for k in sp.single]
    for a, b, nights in zip(sp.pair_first, sp.pair_second, sp.pair_nights, strict=True):
        flights = list(ds.duties[int(a)].flights) + list(ds.duties[int(b)].flights)
        columns.append((float(pay[a] + pay[b] + c.hotel_layover_eur * nights), flights))
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.addRows(n, np.ones(n), np.ones(n), 0, np.zeros(n, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0))
    idx = np.arange(n, dtype=np.int32)
    h.addCols(n, np.full(n, 1e6), np.zeros(n), np.full(n, 1e30), n, idx, idx, np.ones(n))
    h.addCols(n, np.full(n, c.deadhead_eur), np.zeros(n), np.full(n, 1e30), n, idx, idx, -np.ones(n))
    for cost, flights in columns:
        h.addCol(cost, 0.0, 1e30, len(flights), np.array(flights, dtype=np.int32), np.ones(len(flights)))
    h.run()
    full_lp = h.getInfo().objective_function_value
    ownership = c.aircraft_day_eur * aircraft_count(seq.successor, network)
    assert seq.lower_bound is not None
    assert seq.lower_bound - ownership == pytest.approx(full_lp, rel=1e-6)


def test_routings_respect_maintenance(plans: dict[str, Plan], network: Network, rules: PlanningRules) -> None:
    for name in ("sequential", "integrated", "robust"):
        assert maintenance_violations(plans[name].successor, network, rules) == []


def test_robust_buffer_is_respected_at_the_hub(plans: dict[str, Plan], network: Network, rules: PlanningRules) -> None:
    p = plans["robust"]
    for i, j in p.successor.items():
        fi, fj = network.flights[i], network.flights[j]
        if fi.destination == network.hub:
            assert (fj.dep - fi.arr) % DAY >= rules.turn_min[network.hub] + 10


def test_validator_catches_injected_violations(plans: dict[str, Plan], network: Network, rules: PlanningRules) -> None:
    good = plans["sequential"]
    # A crew flying too long: double the FDP of the first duty.
    p0 = good.pairings[0]
    d0 = p0.duties[0]
    bad_duty = PlannedDuty(d0.flights, d0.start, d0.length + 900, d0.fdp + 900, d0.report_utc, d0.deadhead)
    bad = replace(good, pairings=[Pairing((bad_duty, *p0.duties[1:]), p0.cost, p0.hotel_nights), *good.pairings[1:]])
    assert any("FDP" in v for v in validate_plan(bad, network, rules))
    # A flight left without crew.
    missing = replace(good, pairings=good.pairings[1:])
    assert any("without an operating crew" in v for v in validate_plan(missing, network, rules))
    # Two flights swapping successors so that a turn falls below the minimum or changes station.
    succ = dict(good.successor)
    keys = list(succ)
    succ[keys[0]], succ[keys[1]] = succ[keys[1]], succ[keys[0]]
    assert validate_plan(replace(good, successor=succ), network, rules)


def test_arcs_respect_turns(network: Network, rules: PlanningRules) -> None:
    arcs = build_aircraft_arcs(network, rules)
    for t, h, g in zip(arcs.tail, arcs.head, arcs.gap, strict=True):
        assert network.flights[int(t)].destination == network.flights[int(h)].origin
        assert g >= rules.turn_min[network.flights[int(t)].destination]
