"""Load the TeleGeography-derived cable snapshot and the Natural Earth land polygons,
and join them into one network.

Inputs are read from a data directory passed in by the caller; nothing is fetched here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import nearest_points
from shapely.geometry import LineString, Point, shape
from shapely.strtree import STRtree

R_EARTH = 6371.0088
JOIN_KM = 2.0          # a path endpoint within this distance of a landing point lands there
INTERIOR_JOIN_KM = 0.5 # an interior vertex this close to a landing point is a landing on the way
BRANCH_SNAP_KM = 2.0   # a branch end this close to another piece of the same cable joins it
SEA_MERGE_KM = 2.0     # sea nodes of one cable closer than this are the same branching unit
LAND_SNAP_KM = 25.0    # a landing point is assigned to the nearest land polygon within this distance
MAINLAND_KM2 = 5.0e6   # land units larger than this are continents
FIXED_LINK_KM = 1.5    # polygons closer than this are treated as joined by land (bridge, causeway)
# Longer fixed links (tunnels and bridges) that carry terrestrial fibre, as pairs of
# (lat, lng) points on either side.
MANUAL_LINKS = {
    "Channel Tunnel": ((51.09, 1.15), (50.92, 1.80)),
    "Great Belt": ((55.34, 11.12), (55.32, 10.82)),
    "Oresund": ((55.62, 12.66), (55.57, 12.93)),
    "King Fahd Causeway": ((26.16, 50.47), (26.21, 50.17)),
    "Confederation Bridge": ((46.25, -63.67), (46.16, -63.77)),
    "Seikan Tunnel": ((41.45, 140.10), (41.17, 140.40)),
    "Seto-Ohashi": ((34.48, 133.80), (34.32, 133.83)),
    "Penang Bridge": ((5.36, 100.32), (5.36, 100.40)),
    "Storstrom/Faro": ((54.95, 11.93), (55.06, 12.00)),
    "Johor-Singapore Causeway": ((1.40, 103.80), (1.55, 103.75)),
    "Kanmon": ((33.88, 130.88), (34.02, 130.98)),
    "Shanghai Yangtze Tunnel-Bridge (Pudong-Changxing)": ((31.25, 121.60), (31.37, 121.72)),
    "Chongqi/Changxing-Chongming": ((31.37, 121.72), (31.62, 121.55)),
    "Deception Pass": ((48.37, -122.62), (48.45, -122.62)),
}


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.minimum(1.0, h)))


@dataclass
class Edge:
    """One drawn segment of a cable system, between two graph nodes."""
    cable: str
    u: int
    v: int
    coords: np.ndarray          # (n, 2) as (lat, lng)
    length_km: float


@dataclass
class Network:
    cables: dict                 # id -> {"name", "rfs", "owners", "length_km"}
    nodes: list                  # node -> (lat, lng)
    node_kind: list              # "landing" | "sea"
    node_unit: list              # land unit index for landing nodes, -1 for sea nodes
    node_lp: list                # landing point id for landing nodes, None otherwise
    edges: list                  # list[Edge]
    units: list                  # land unit -> {"area_km2", "mainland", "centroid", "countries", "landings"}
    landing_points: dict         # id -> {"name", "lat", "lng", "unit"}
    meta: dict = field(default_factory=dict)


def _area_km2(geom) -> float:
    """Spherical-excess free approximation: equal-area via a per-polygon cylindrical scale."""
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    a, _ = g.geometry_area_perimeter(geom)
    return abs(a) / 1e6


def load(data_dir: str | Path, include_planned: bool = False, extra: tuple = (),
         exclude: tuple = ()) -> Network:
    """extra: planned cables to include anyway; exclude: cables to leave out."""
    data_dir = Path(data_dir)
    snap = data_dir / "ixpavi_subsea-siting" / "public" / "data"
    raw_cables = json.loads((snap / "cables.json").read_text())
    details = json.loads((snap / "cable-details.json").read_text())
    lps = json.loads((snap / "landing-points.json").read_text())

    # --- cables in service, features merged by id ---------------------------------
    paths_by_id: dict[str, list] = {}
    names = {}
    for c in raw_cables:
        paths_by_id.setdefault(c["id"], []).extend(c["paths"])
        names[c["id"]] = c["name"]
    cables = {}
    for cid, paths in paths_by_id.items():
        d = details["cables"].get(cid)
        if d is None:
            continue
        if d["planned"] and not include_planned and cid not in extra:
            continue
        if cid in exclude:
            continue
        cables[cid] = {"name": names[cid], "rfs": d["rfsYear"], "owners": d["owners"],
                       "length_km": d["lengthKm"], "planned": d["planned"]}

    # --- land units ----------------------------------------------------------------
    polys = []
    for fn in ("ne_10m_land.geojson", "ne_10m_minor_islands.geojson"):
        gj = json.loads((data_dir / "ne" / "geojson" / fn).read_text())
        for f in gj["features"]:
            g = shape(f["geometry"])
            parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            polys.extend(p for p in parts if p.is_valid or p.buffer(0).is_valid)
    tree = STRtree(polys)

    lp_xy = np.array([[p["lat"], p["lng"]] for p in lps])

    def nearest_lp(lat, lng):
        d = haversine_km(lat, lng, lp_xy[:, 0], lp_xy[:, 1])
        j = int(np.argmin(d))
        return j, float(d[j])

    # --- graph ---------------------------------------------------------------------
    nodes, node_kind, node_lp = [], [], []
    key_to_node = {}
    lp_node = {}

    def node_for(lat, lng):
        j, dist = nearest_lp(lat, lng)
        if dist <= JOIN_KM:
            lid = lps[j]["id"]
            if lid not in lp_node:
                lp_node[lid] = len(nodes)
                nodes.append((lps[j]["lat"], lps[j]["lng"]))
                node_kind.append("landing")
                node_lp.append(lid)
            return lp_node[lid]
        key = (round(lat, 4), round(lng, 4))
        if key not in key_to_node:
            key_to_node[key] = len(nodes)
            nodes.append((lat, lng))
            node_kind.append("sea")
            node_lp.append(None)
        return key_to_node[key]

    def lands_at(lat, lng, tol):
        j, dist = nearest_lp(lat, lng)
        return dist <= tol

    def split_at(arr, idx):
        idx = sorted(set(i for i in idx if 0 < i < len(arr) - 1))
        cuts = [0] + idx + [len(arr) - 1]
        return [arr[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

    pieces_by_cable = {}
    for cid in cables:
        polys_c = [np.asarray(p, dtype=float) for p in paths_by_id[cid] if len(p) >= 2]
        # 1. Festoon systems are drawn as one polyline through their landing points:
        #    an interior vertex sitting on a landing point splits the path.
        stage = []
        for arr in polys_c:
            idx = [i for i in range(1, len(arr) - 1) if lands_at(arr[i, 0], arr[i, 1], INTERIOR_JOIN_KM)]
            stage.extend(split_at(arr, idx))
        # 2. Branches leave the trunk at a branching unit drawn as an interior vertex of
        #    the trunk, or as a point between two of its vertices. Split the trunk there.
        changed = True
        while changed:
            changed = False
            ends = [(k, e) for k, arr in enumerate(stage) for e in (arr[0], arr[-1])]
            for k, arr in enumerate(stage):
                if len(arr) < 3 and not changed:
                    pass
                for k2, e in ends:
                    if k2 == k:
                        continue
                    d = haversine_km(e[0], e[1], arr[:, 0], arr[:, 1])
                    i = int(np.argmin(d))
                    if 0 < i < len(arr) - 1 and d[i] <= BRANCH_SNAP_KM:
                        stage[k:k + 1] = split_at(arr, [i])
                        changed = True
                        break
                    if d[0] > BRANCH_SNAP_KM and d[-1] > BRANCH_SNAP_KM:
                        # point between vertices: insert the projection if close enough
                        line = LineString(arr[:, ::-1])
                        pt = Point(e[1], e[0])
                        proj = line.interpolate(line.project(pt))
                        dd = float(haversine_km(e[0], e[1], proj.y, proj.x))
                        if dd <= BRANCH_SNAP_KM:
                            pos = line.project(pt)
                            acc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(arr[:, ::-1], axis=0).T))])
                            j = int(np.searchsorted(acc, pos))
                            if 0 < j < len(arr):
                                new_arr = np.vstack([arr[:j], [[proj.y, proj.x]], arr[j:]])
                                stage[k:k + 1] = split_at(new_arr, [j])
                                changed = True
                                break
                if changed:
                    break
        pieces_by_cable[cid] = stage

    # 3. Nodes: landing points, otherwise sea points merged within SEA_MERGE_KM per cable.
    edges = []
    for cid, stage in pieces_by_cable.items():
        sea_nodes = []
        def sea_or_land(lat, lng):
            j, dist = nearest_lp(lat, lng)
            if dist <= JOIN_KM:
                return node_for(lps[j]["lat"], lps[j]["lng"])
            for n in sea_nodes:
                if haversine_km(lat, lng, nodes[n][0], nodes[n][1]) <= SEA_MERGE_KM:
                    return n
            n = node_for(lat, lng)
            sea_nodes.append(n)
            return n
        for piece in stage:
            u = sea_or_land(*piece[0])
            v = sea_or_land(*piece[-1])
            if u == v and len(piece) < 3:
                continue
            seg = haversine_km(piece[:-1, 0], piece[:-1, 1], piece[1:, 0], piece[1:, 1]).sum()
            edges.append(Edge(cid, u, v, piece, float(seg)))

    # --- land polygons -> landmass groups joined by fixed links -----------------------
    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    deg = FIXED_LINK_KM / 111.0
    for i, poly in enumerate(polys):
        lat_c = abs(poly.representative_point().y)
        pad = deg / max(0.2, math.cos(math.radians(min(lat_c, 85))))
        for j in tree.query(poly.buffer(pad)):
            j = int(j)
            if j <= i:
                continue
            a_pt, b_pt = nearest_points(poly, polys[j])
            if float(haversine_km(a_pt.y, a_pt.x, b_pt.y, b_pt.x)) <= FIXED_LINK_KM:
                union(i, j)

    def poly_at(lat, lng, max_km=10.0):
        pt = Point(lng, lat)
        best, best_d = None, math.inf
        for k in tree.query(pt.buffer(max_km / 111.0 * 2)):
            k = int(k)
            if polys[k].contains(pt):
                return k
            q = nearest_points(polys[k], pt)[0]
            dk = float(haversine_km(lat, lng, q.y, q.x))
            if dk < best_d:
                best, best_d = k, dk
        return best if best_d <= max_km else None

    for name, (p1, p2) in MANUAL_LINKS.items():
        k1, k2 = poly_at(*p1), poly_at(*p2)
        if k1 is None or k2 is None:
            raise RuntimeError(f"fixed link {name}: endpoint not on land")
        union(k1, k2)

    group_area: dict[int, float] = {}
    for i, poly in enumerate(polys):
        g = find(i)
        group_area[g] = group_area.get(g, 0.0) + 0.0  # filled lazily below

    # --- landing points -> land units ---------------------------------------------
    unit_of_group: dict[int, int] = {}
    units = []
    landing_points = {}
    node_unit = [-1] * len(nodes)
    group_members: dict[int, list] = {}
    for i in range(len(polys)):
        group_members.setdefault(find(i), []).append(i)
    for lid, n in lp_node.items():
        lat, lng = nodes[n]
        k = poly_at(lat, lng, LAND_SNAP_KM)
        if k is None:
            # Not on any Natural Earth polygon: a platform or an islet too small for the
            # 1:10m layers. It becomes a unit of its own.
            g = -(len(units) + 1)
        else:
            g = find(k)
        if g not in unit_of_group:
            unit_of_group[g] = len(units)
            if g >= 0:
                members = group_members[g]
                area = sum(_area_km2(polys[m]) for m in members) if len(members) < 400 else math.inf
                biggest = max(members, key=lambda m: polys[m].area)
                c = polys[biggest].representative_point()
                centroid = (c.y, c.x)
            else:
                members, area, centroid = [], 0.0, (lat, lng)
            units.append({"group": g, "polys": members, "area_km2": area,
                          "mainland": area > MAINLAND_KM2, "centroid": centroid,
                          "landings": [], "countries": {}, "point": g < 0})
        u = unit_of_group[g]
        node_unit[n] = u
        name = next(p["name"] for p in lps if p["id"] == lid)
        country = name.split(",")[-1].strip() if "," in name else name
        units[u]["landings"].append(lid)
        units[u]["countries"][country] = units[u]["countries"].get(country, 0) + 1
        landing_points[lid] = {"name": name, "lat": lat, "lng": lng, "unit": u}

    net = Network(cables=cables, nodes=nodes, node_kind=node_kind, node_unit=node_unit,
                  node_lp=node_lp, edges=edges, units=units, landing_points=landing_points,
                  meta={"fetched": details.get("fetchedAt"), "source": details.get("source"),
                        "licence": details.get("licence"), "polygons": polys})
    return net


def unit_label(net: Network, u: int) -> str:
    c = net.units[u]["countries"]
    top = max(c, key=c.get)
    return top
