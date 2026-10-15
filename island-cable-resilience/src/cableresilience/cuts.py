"""How many separate incidents does it take to cut an island off?

The network is a graph whose nodes are land units (continents and islands), landing
stations and branching units at sea, and whose edges are drawn cable segments. An island
is isolated when no path of intact segments joins it to any continent.

Two numbers per island:
  lambda      the minimum number of single-point cuts, anywhere, that isolates it
              (an edge cut in the classical sense, every cut affecting one segment);
  N(r)        the minimum number of incidents of radius r km, centred within Z km of the
              island's landing stations, that isolates it. One incident severs every
              segment that passes within r km of its centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from ortools.sat.python import cp_model
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from .data import Network, haversine_km

PLATFORM_WORDS = ("platform", "totalenergies", "bp ", "glen lyon", "south arne", "tyra", "halfdan")


def is_platform(net: Network, u: int) -> bool:
    names = " ".join(net.landing_points[l]["name"].lower() for l in net.units[u]["landings"])
    return any(w in names for w in PLATFORM_WORDS)


def unit_node(u: int) -> str:
    return f"U{u}"


def build_graph(net: Network, alive: np.ndarray | None = None) -> nx.MultiGraph:
    """Nodes: 'U<k>' land units and integer cable nodes. Landing nodes join their unit."""
    G = nx.MultiGraph()
    for n, u in enumerate(net.node_unit):
        if u >= 0:
            G.add_edge(n, unit_node(u), kind="land")
    for i, e in enumerate(net.edges):
        if alive is None or alive[i]:
            G.add_edge(e.u, e.v, kind="cable", idx=i)
    return G


def mainland_units(net: Network) -> list[int]:
    return [i for i, u in enumerate(net.units) if u["mainland"]]


def external_cables(net: Network, u: int) -> set[str]:
    """Cable systems that land on unit u and also land somewhere else."""
    land_units: dict[str, set] = {}
    for e in net.edges:
        for n in (e.u, e.v):
            if net.node_unit[n] >= 0:
                land_units.setdefault(e.cable, set()).add(net.node_unit[n])
    return {c for c, us in land_units.items() if u in us and len(us) > 1}


def edge_cut_lambda(net: Network, u: int, G: nx.MultiGraph | None = None) -> int:
    """Minimum number of cable segments whose loss isolates unit u from every continent."""
    G = build_graph(net) if G is None else G
    H = nx.DiGraph()
    big = 10 ** 6
    for a, b, d in G.edges(data=True):
        cap = 1 if d["kind"] == "cable" else big
        for x, y in ((a, b), (b, a)):
            if H.has_edge(x, y):
                H[x][y]["capacity"] += cap
            else:
                H.add_edge(x, y, capacity=cap)
    for m in mainland_units(net):
        if unit_node(m) in H:
            H.add_edge(unit_node(m), "SINK", capacity=big)
    src = unit_node(u)
    if src not in H or "SINK" not in H:
        return 0
    val, _ = nx.minimum_cut(H, src, "SINK")
    return int(val) if val < big else -1


@dataclass
class Zone:
    unit: int
    proj: Transformer
    edge_ids: list            # edges with some part inside the zone
    lines: list               # projected LineStrings for those edges (km)
    landings_xy: np.ndarray   # projected island landing points (km)


def make_zone(net: Network, u: int, zone_km: float) -> Zone:
    lats = [net.landing_points[l]["lat"] for l in net.units[u]["landings"]]
    lngs = [net.landing_points[l]["lng"] for l in net.units[u]["landings"]]
    lat0, lng0 = float(np.mean(lats)), float(np.mean(lngs))
    proj = Transformer.from_crs("EPSG:4326",
                                f"+proj=aeqd +lat_0={lat0} +lon_0={lng0} +units=km +datum=WGS84",
                                always_xy=True)
    lx, ly = proj.transform(lngs, lats)
    land = np.column_stack([lx, ly])
    reach = max(float(np.max(haversine_km(lat0, lng0, np.array(lats), np.array(lngs)))), 0.0) + zone_km
    edge_ids, lines = [], []
    for i, e in enumerate(net.edges):
        d = haversine_km(lat0, lng0, e.coords[:, 0], e.coords[:, 1])
        if d.min() > reach + 2000:   # far away even allowing long straight drawn segments
            continue
        x, y = proj.transform(e.coords[:, 1], e.coords[:, 0])
        line = LineString(np.column_stack([x, y]))
        if line.distance(Point(0, 0)) > reach + 1e-9 and min(Point(p).distance(line) for p in land) > zone_km:
            continue
        if min(Point(p).distance(line) for p in land) <= zone_km:
            edge_ids.append(i)
            lines.append(line)
    return Zone(u, proj, edge_ids, lines, land)


def candidate_disks(zone: Zone, r_km: float, zone_km: float, spacing_km: float | None = None):
    """Distinct maximal sets of zone edges that one incident of radius r can sever.

    Candidate centres: points along every segment, the midpoint of the closest approach of
    every pair of segments that come within 2r of each other, and a fine grid within 2r of
    each landing station, where segments converge.
    """
    spacing = spacing_km or max(0.5, r_km / 2.0)
    tree = STRtree(zone.lines)
    land_pts = [Point(p) for p in zone.landings_xy]
    centres = []
    for line in zone.lines:
        centres.extend(line.segmentize(spacing).coords)
    if r_km > 0:
        for i in range(len(zone.lines)):
            for j in tree.query(zone.lines[i].buffer(2 * r_km)):
                j = int(j)
                if j <= i:
                    continue
                a, b = nearest_points(zone.lines[i], zone.lines[j])
                if a.distance(b) <= 2 * r_km:
                    centres.append(((a.x + b.x) / 2, (a.y + b.y) / 2))
        step = max(0.25, r_km / 4.0)
        g = np.arange(-2 * r_km, 2 * r_km + step / 2, step)
        for lp in zone.landings_xy:
            for dx in g:
                for dy in g:
                    if dx * dx + dy * dy <= (2 * r_km) ** 2:
                        centres.append((lp[0] + dx, lp[1] + dy))
    sets = set()
    for x, y in centres:
        c = Point(x, y)
        if min(c.distance(p) for p in land_pts) > zone_km:
            continue
        hit = tree.query(c.buffer(r_km if r_km > 0 else 1e-6))
        cov = frozenset(int(zone.edge_ids[k]) for k in hit
                        if zone.lines[int(k)].distance(c) <= max(r_km, 1e-6))
        if cov:
            sets.add(cov)
    # keep only maximal sets
    ordered = sorted(sets, key=len, reverse=True)
    maximal = []
    for s in ordered:
        if not any(s <= m for m in maximal):
            maximal.append(s)
    return maximal


def min_incidents(net: Network, u: int, r_km: float, zone_km: float, G: nx.MultiGraph | None = None,
                  time_limit: float = 30.0) -> tuple[int, list]:
    """Exact minimum number of radius-r incidents (centres within zone_km of the island's
    landing stations) that leave unit u with no path to a continent."""
    G = build_graph(net) if G is None else G
    zone = make_zone(net, u, zone_km)
    disks = candidate_disks(zone, r_km, zone_km)
    coverable = set().union(*disks) if disks else set()
    # Contract everything that no incident in the zone can touch.
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    for a, b, d in G.edges(data=True):
        if d["kind"] == "land" or d["idx"] not in coverable:
            H.add_edge(a, b)
    comp = {}
    for k, cc in enumerate(nx.connected_components(H)):
        for n in cc:
            comp[n] = k
    src = comp[unit_node(u)]
    sinks = {comp[unit_node(m)] for m in mainland_units(net) if unit_node(m) in comp}
    if src in sinks:
        return -1, []
    m = cp_model.CpModel()
    side = {}

    def s(cid):
        if cid not in side:
            side[cid] = m.new_bool_var(f"s{cid}")
        return side[cid]

    m.add(s(src) == 1)
    for k in sinks:
        m.add(s(k) == 0)
    y = [m.new_bool_var(f"y{j}") for j in range(len(disks))]
    by_edge: dict[int, list] = {}
    for j, cov in enumerate(disks):
        for e in cov:
            by_edge.setdefault(e, []).append(y[j])
    for e in coverable:
        a, b = comp[net.edges[e].u], comp[net.edges[e].v]
        if a == b:
            continue
        cut = m.new_bool_var(f"c{e}")
        m.add(cut >= s(a) - s(b))
        m.add(cut >= s(b) - s(a))
        m.add(cut <= sum(by_edge[e]))
    m.minimize(sum(y))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 2
    st = solver.solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return -2, []
    chosen = [disks[j] for j in range(len(disks)) if solver.value(y[j])]
    return int(round(solver.objective_value)), chosen
