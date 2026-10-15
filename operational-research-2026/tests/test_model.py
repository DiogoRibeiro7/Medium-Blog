"""Checks on the instance, the two-stage model and the recourse LP.

The key consistency test solves the SAA model on a few weeks and re-scores the
chosen design on those same weeks with the independent highspy recourse code:
the two numbers must agree, since they are two implementations of one quantity.
"""

import numpy as np
import pytest

from depotsaa import (Recourse, build_instance, draw_weeks, evaluate,
                      solve_two_stage)


@pytest.fixture(scope="module")
def inst():
    return build_instance()[0]


def test_instance_is_seeded(inst):
    again = build_instance()[0]
    assert np.array_equal(inst.cost, again.cost)
    assert np.array_equal(inst.fixed, again.fixed)
    assert inst.n_sites == 10 and inst.n_zones == 30


def test_demand_has_the_stated_mean(inst):
    weeks = draw_weeks(inst, 100_000, np.random.default_rng(0))
    assert np.all(weeks > 0)
    np.testing.assert_allclose(weeks.mean(0) / inst.mu, 1.0, atol=0.01)


def test_deterministic_plan_equals_lease_plus_recourse(inst):
    y, obj, *_ = solve_two_stage(inst, inst.mu[None, :])
    r = Recourse(inst, y)
    transport, out = r.week(inst.mu)
    assert obj == pytest.approx(r.fixed + transport + inst.penalty * out, rel=1e-6)


def test_saa_objective_matches_rescoring_on_the_same_weeks(inst):
    weeks = draw_weeks(inst, 12, np.random.default_rng(1))
    y, obj, *_ = solve_two_stage(inst, weeks)
    assert evaluate(inst, y, weeks)["total"].mean() == pytest.approx(obj, rel=1e-6)


def test_cost_at_mean_demand_is_optimistic(inst):
    # the recourse value is convex in demand, so Jensen's inequality applies
    weeks = draw_weeks(inst, 2_000, np.random.default_rng(2))
    y, *_ = solve_two_stage(inst, weeks.mean(0, keepdims=True))
    r = Recourse(inst, y)
    t_mean, o_mean = r.week(weeks.mean(0))
    at_mean = t_mean + inst.penalty * o_mean
    scored = evaluate(inst, y, weeks)
    assert at_mean <= (scored["total"] - scored["fixed"]).mean() + 1e-6


def test_linking_rows_do_not_change_the_answer(inst):
    weeks = draw_weeks(inst, 8, np.random.default_rng(3))
    y0, obj0, *_ = solve_two_stage(inst, weeks)
    y1, obj1, *_ = solve_two_stage(inst, weeks, link=True)
    assert obj0 == pytest.approx(obj1, rel=1e-6)


def test_no_outsourcing_when_every_depot_is_open_and_demand_is_low(inst):
    r = Recourse(inst, np.ones(inst.n_sites, dtype=int))
    _, out = r.week(0.5 * inst.mu)
    assert out == pytest.approx(0.0, abs=1e-9)
