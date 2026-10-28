"""The plans on toy instances, and the estimators on arrays with known answers."""

from __future__ import annotations

import numpy as np

from routing_roads.estimates import interval, noise_by_day, regret_by_day
from routing_roads.routing import cost, decode, encode, solve


def toy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 5_000, (9, 2))
    d = np.rint(np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))).astype(np.int64)
    d[2, 5] += 1_500                                 # asymmetric, like a one-way street
    return d, xy


def test_every_client_is_served_once_within_capacity() -> None:
    d, xy = toy()
    points = [0, 1, 2, 3, 5, 6, 8]                   # depot 0 and a subset of clients
    routes = solve(d, xy, points, capacity=2, iterations=500, seed=1)
    visited = sorted(i for r in routes for i in r)
    assert visited == points[1:]
    assert all(len(r) <= 2 for r in routes)
    assert routes == solve(d, xy, points, capacity=2, iterations=500, seed=1)


def test_cost_counts_depot_legs_and_direction() -> None:
    d, _ = toy()
    assert cost([[2, 5]], d, 0) == d[0, 2] + d[2, 5] + d[5, 0]
    assert cost([[5, 2]], d, 0) == d[0, 5] + d[5, 2] + d[2, 0]
    assert cost([], d, 0) == 0
    assert decode(encode([[3, 1], [7]])) == [[3, 1], [7]] and decode("") == []


def test_regret_and_noise() -> None:
    road_s = np.array([[110.0, 120.0], [100.0, 100.0]])
    road_r = np.array([[100.0, 100.0], [100.0, 125.0]])
    assert np.allclose(regret_by_day(road_s, road_r), [0.15, (0.0 + (100 / 125 - 1)) / 2])
    assert np.allclose(noise_by_day(road_r), [0.0, (0.25 + 0.2) / 2])


def test_a_constant_has_a_zero_width_interval() -> None:
    idx = np.random.default_rng(0).integers(0, 5, size=(99, 5))
    x = interval(np.full(5, 0.07), idx, 0.95)
    assert x["estimate"] == x["low"] == x["high"] == 0.07
