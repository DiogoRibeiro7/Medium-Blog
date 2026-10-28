"""Distances on toy graphs, checked against networkx."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from routing_roads.contract import EARTH_RADIUS_M
from routing_roads.network import (
    Network,
    crossing_names,
    crossings,
    haversine,
    matrices,
    path,
)


def toy() -> nx.MultiDiGraph:
    """Four nodes on a line of longitude 0, 1 km apart; 1 -> 2 is one-way, with a long
    parallel edge that must lose to the short one."""
    g = nx.MultiDiGraph()
    for n in range(4):
        g.add_node(n + 10, x=0.0, y=n * 1000 / (math.pi / 180 * EARTH_RADIUS_M))
    for u, v in ((10, 11), (11, 10), (11, 12), (12, 13), (13, 12)):
        g.add_edge(u, v, length=1000.0)
    g.add_edge(11, 12, length=5000.0)
    g.add_edge(12, 10, length=3000.0)            # the way back from 12 is a detour
    return g


def test_haversine_one_degree_of_latitude() -> None:
    assert math.isclose(haversine(0.0, 0.0, 0.0, 1.0), math.pi / 180 * EARTH_RADIUS_M)


def test_matrices_agree_with_networkx() -> None:
    g = toy()
    net = Network.from_graph(g)
    nodes = [0, 1, 2, 3]
    road, straight, pred = matrices(net, nodes)
    for i, u in enumerate(net.ids.tolist()):
        for j, v in enumerate(net.ids.tolist()):
            assert road[i, j] == round(nx.shortest_path_length(g, u, v, weight="length"))
    assert road[1, 2] == 1000 and road[2, 1] == 4000          # one-way: not symmetric
    assert np.array_equal(straight, straight.T) and np.all(np.diag(straight) == 0)
    assert np.all(road >= straight - 1)                       # never shorter than the line
    assert path(pred, 2, 1) == [2, 0, 1]


def test_a_crossing_is_an_edge_between_the_two_banks() -> None:
    g = toy()
    g[11][12][0]["name"] = "The bridge"        # the short parallel edge
    g[11][12][1]["name"] = "The long way"
    net = Network.from_graph(g)
    _, _, pred = matrices(net, [0, 1, 2, 3])
    bank = ["north", "north", "south", ""]     # node 13 lies outside both
    assert crossing_names(net, path(pred, 0, 3), bank) == ["The bridge"]
    report = crossings(net, pred, [0, 1, 2, 3], ["north", "north", "south", "south"], bank)
    assert report["paths"] == 8 and report["bridges"] == {"The bridge": 0.5, "(unnamed)": 0.5}


def test_snap_picks_the_nearest_node() -> None:
    net = Network.from_graph(toy())
    index, metres = net.snap(0.0, 1900 / (math.pi / 180 * EARTH_RADIUS_M))
    assert index == 2 and math.isclose(metres, 100, rel_tol=1e-6)
