"""Checks of the cut measures and the two fault models on small synthetic networks.

Coordinates are near the equator, where one degree is about 111 km. The continent lies
about 330 km east of the island in every case.
"""
import numpy as np
import pytest

from cableresilience.cuts import build_graph, edge_cut_lambda, external_cables, min_incidents
from cableresilience.data import Edge, Network, haversine_km
from cableresilience.risk import IslandModel, Params, incident_risk, independent_risk

ZONE = 100.0


def make_net(island_landings, cables):
    """island_landings: {id: (lat, lng)}; cables: list of (cable id, island landing id, route).

    Each route is a list of (lat, lng) points from the island landing to the continent.
    Every cable gets its own continental landing at the end of its route.
    """
    units = [
        {"mainland": True, "landings": [], "countries": {"Continent": 1}, "point": False, "area_km2": 1e7},
        {"mainland": False, "landings": [], "countries": {"Island": 1}, "point": False, "area_km2": 500.0},
    ]
    nodes, kind, node_unit, node_lp, lps, node_of = [], [], [], [], {}, {}

    def add_landing(lid, latlng, unit):
        node_of[lid] = len(nodes)
        nodes.append(latlng)
        kind.append("landing")
        node_unit.append(unit)
        node_lp.append(lid)
        lps[lid] = {"name": f"{lid}, Test", "lat": latlng[0], "lng": latlng[1], "unit": unit}
        units[unit]["landings"].append(lid)

    for lid, latlng in island_landings.items():
        add_landing(lid, latlng, 1)
    edges, cables_meta = [], {}
    for cid, start, route in cables:
        end = f"{cid}-continent"
        add_landing(end, tuple(route[-1]), 0)
        coords = np.array([island_landings[start]] + [tuple(p) for p in route])
        length = float(sum(haversine_km(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1)))
        edges.append(Edge(cid, node_of[start], node_of[end], coords, length))
        cables_meta[cid] = {"name": cid, "rfs": 2020, "owners": "", "length_km": length}
    return Network(cables=cables_meta, nodes=nodes, node_kind=kind, node_unit=node_unit, node_lp=node_lp,
                   edges=edges, units=units, landing_points=lps)


@pytest.fixture(scope="module")
def shared():
    """Two cables landing at the same point and leaving in slightly different directions."""
    return make_net({"A": (0.0, 0.0)},
                    [("c1", "A", [(0.3, 1.5), (0.3, 3.0)]), ("c2", "A", [(-0.3, 1.5), (-0.3, 3.0)])])


@pytest.fixture(scope="module")
def diverse():
    """Two cables landing 44 km apart and running in parallel to the continent."""
    return make_net({"A": (0.0, 0.0), "B": (0.4, 0.0)},
                    [("c1", "A", [(0.0, 3.0)]), ("c2", "B", [(0.4, 3.0)])])


@pytest.fixture(scope="module")
def single():
    return make_net({"A": (0.0, 0.0)}, [("c1", "A", [(0.0, 3.0)])])


def test_counts_and_edge_connectivity(shared, diverse, single):
    for net, k, lam in ((shared, 2, 2), (diverse, 2, 2), (single, 1, 1)):
        assert len(external_cables(net, 1)) == k
        assert edge_cut_lambda(net, 1, build_graph(net)) == lam


def test_shared_landing_falls_to_one_small_incident(shared):
    G = build_graph(shared)
    assert min_incidents(shared, 1, 1.0, ZONE, G)[0] == 1


def test_separated_landings_need_one_incident_per_cable_until_the_radius_spans_them(diverse):
    G = build_graph(diverse)
    assert min_incidents(diverse, 1, 1.0, ZONE, G)[0] == 2
    assert min_incidents(diverse, 1, 10.0, ZONE, G)[0] == 2      # 44 km apart: a 20 km disk cannot reach both
    assert min_incidents(diverse, 1, 50.0, ZONE, G)[0] == 1


def test_incident_rate_is_calibrated_to_the_fault_count(shared):
    p = Params(samples=50_000)
    M = IslandModel(shared, 1, p.zone_km, build_graph(shared))
    out = incident_risk(M, p)
    expected = p.faults_per_km_year * M.length_in_zone.sum()
    assert out["faults_per_year"] == pytest.approx(expected)
    assert out["incidents_per_year"] * out["mean_segments_per_incident"] == pytest.approx(expected)


def test_models_agree_when_one_segment_is_a_cut(single):
    """With a single cable both models count the same faults and the same outages."""
    p = Params()
    M = IslandModel(single, 1, p.zone_km, build_graph(single))
    a = independent_risk(M, p)
    b = incident_risk(M, p)
    assert a["min_cut_size"] == 1
    assert b["hours_per_year"] == pytest.approx(a["hours_per_year"], rel=0.10)


def test_shared_landing_is_far_riskier_under_incidents(shared):
    p = Params(samples=100_000)
    M = IslandModel(shared, 1, p.zone_km, build_graph(shared))
    a = independent_risk(M, p)
    b = incident_risk(M, p)
    assert a["min_cut_size"] == 2
    assert b["hours_per_year"] > 100 * a["hours_per_year"]


def test_spread_landings_reduce_incident_risk(shared, diverse):
    p = Params(samples=100_000)
    h = {}
    for name, net in (("shared", shared), ("diverse", diverse)):
        M = IslandModel(net, 1, p.zone_km, build_graph(net))
        h[name] = incident_risk(M, p)["hours_per_year"]
    assert h["diverse"] < h["shared"] / 2
