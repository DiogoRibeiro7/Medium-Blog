from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from airline_ops.config import Config
from airline_ops.network.model import Network
from airline_ops.planning.plan import Plan
from airline_ops.planning.planners import plan_sequential
from airline_ops.planning.rules import PlanningRules
from airline_ops.simulation.delays import DelayModel, EmpiricalPenalty, PropagationPenalty
from airline_ops.simulation.scenarios import SCENARIOS, ScenarioEvents, draw_events
from airline_ops.simulation.simulator import simulate, unroll

QUIET = DelayModel(mean_primary_min=1e-9, p_delayed=1e-9, sigma=0.5, block_noise_sd=0.0, block_bias=0.0)


@pytest.fixture(scope="module")
def plan(network: Network, smoke_config: Config) -> Plan:
    return plan_sequential(network, smoke_config)


def _run(network: Network, plan: Plan, config: Config, model: DelayModel, events: ScenarioEvents, rep: int = 0):
    rules = PlanningRules.from_config(config, network)
    u = unroll(network, plan, config.simulation.horizon_days, rules)
    return simulate(network, plan, u, config, model, events, 42, rep)


def test_unroll_assigns_every_instance_once(network: Network, plan: Plan, smoke_config: Config) -> None:
    rules = PlanningRules.from_config(smoke_config, network)
    u = unroll(network, plan, smoke_config.simulation.horizon_days, rules)
    n_inst = network.n_flights * smoke_config.simulation.horizon_days
    tails = sorted(i for seq in u.tail_seq for i in seq)
    assert tails == list(range(n_inst))
    operated = sorted(i for c, seq in enumerate(u.crew_seq) for k, i in enumerate(seq) if u.crew_op[c][k])
    assert set(operated) == set(range(n_inst))


def test_no_delays_no_disruption(network: Network, plan: Plan, quiet_config: Config) -> None:
    out = _run(network, plan, quiet_config, QUIET, ScenarioEvents(onset=0))
    assert out.operated.all()
    assert np.allclose(out.dep_delay, 0.0)
    assert out.counters["missed_connection_pax"] == 0
    assert out.counters["pax_delay_min"] == pytest.approx(0.0, abs=1e-6)
    assert out.costs["eu261"] == 0


def test_inbound_delay_propagates_and_never_shortens(network: Network, plan: Plan, quiet_config: Config) -> None:
    flight = next(f.idx for f in network.flights if f.destination == network.hub)
    base = _run(network, plan, quiet_config, QUIET, ScenarioEvents(onset=0))
    hit = _run(network, plan, quiet_config, QUIET, ScenarioEvents(onset=0, inbound_delays={flight: 120.0}))
    assert hit.dep_delay[flight] == pytest.approx(120.0)
    both = base.operated & hit.operated
    assert (np.nan_to_num(hit.dep_delay[both]) >= np.nan_to_num(base.dep_delay[both]) - 1e-9).all()
    knock_on = (hit.bind_type > 0) & hit.operated
    assert knock_on.sum() >= 1


def test_large_delay_cancels_out_and_back_pair(network: Network, plan: Plan, quiet_config: Config) -> None:
    out_leg = next(f.idx for f in network.flights if f.origin == network.hub)
    cfg = replace(quiet_config, simulation=replace(quiet_config.simulation, swap_threshold_min=100000))
    out = _run(network, plan, cfg, QUIET, ScenarioEvents(onset=0, inbound_delays={out_leg: 1000.0}))
    assert not out.operated[out_leg]
    assert out.counters["cancel_pair"] >= 1
    assert out.counters["cancelled_pax"] > 0


def test_sick_crew_uses_standby(network: Network, plan: Plan, quiet_config: Config) -> None:
    hub_dep = sorted((f.dep, f.idx) for f in network.flights if f.origin == network.hub)[1][1]
    out = _run(network, plan, quiet_config, QUIET, ScenarioEvents(onset=0, sick_flights=(hub_dep,)))
    assert out.counters["standby_calls"] == 1
    assert out.operated[hub_dep]


def test_common_random_numbers(network: Network, plan: Plan, smoke_config: Config) -> None:
    model = DelayModel.from_config(smoke_config.delays)
    ev = draw_events(SCENARIOS[0], network, np.random.default_rng([1, 2, 0]))
    a = _run(network, plan, smoke_config, model, ev, rep=5)
    b = _run(network, plan, smoke_config, model, ev, rep=5)
    assert np.array_equal(np.nan_to_num(a.dep_delay), np.nan_to_num(b.dep_delay))


def test_scenarios_are_plan_independent(network: Network) -> None:
    for sc in SCENARIOS:
        e1 = draw_events(sc, network, np.random.default_rng([3, 4, 5]))
        e2 = draw_events(sc, network, np.random.default_rng([3, 4, 5]))
        assert e1 == e2


def test_delay_model_hits_its_mean() -> None:
    m = DelayModel(mean_primary_min=9.5, p_delayed=0.45, sigma=0.8, block_noise_sd=0.04, block_bias=-0.03)
    draws = m.departure_delays(np.random.default_rng(0), 400_000)
    assert draws.mean() == pytest.approx(9.5, rel=0.02)
    assert (draws > 0).mean() == pytest.approx(0.45, abs=0.005)


def test_expected_excess_is_decreasing_and_matches_monte_carlo() -> None:
    m = DelayModel(mean_primary_min=9.5, p_delayed=0.45, sigma=0.8, block_noise_sd=0.04, block_bias=-0.03)
    pen = PropagationPenalty(m, 50_000, 7)
    values = [pen.expected_excess(120, s) for s in (0, 10, 20, 40, 80)]
    assert all(a >= b for a, b in itertools.pairwise(values))
    rng = np.random.default_rng(8)
    a = m.departure_delays(rng, 200_000) + 120 * (m.block_factors(rng, 200_000) - 1.0)
    assert pen.expected_excess(120, 20) == pytest.approx(float(np.clip(a - 20, 0, None).mean()), rel=0.05)


def test_point_forecast_penalty_ignores_the_tail() -> None:
    samples = [np.array([0.0] * 90 + [100.0] * 10)]
    point, dist = EmpiricalPenalty(samples, point=True), EmpiricalPenalty(samples)
    assert point.expected_excess_flight(0, 20.0) == 0.0  # mean 10 < slack 20
    assert dist.expected_excess_flight(0, 20.0) == pytest.approx(8.0)
