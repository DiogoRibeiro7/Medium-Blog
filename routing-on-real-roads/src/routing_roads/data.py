"""The data step (METHODS.md, sections 3-4): areas, network, points, matrices and days.

Writes, under ``out``: ``points.csv`` (the depot and each pharmacy of every pair, with the
node it snaps to), ``<pair>_road.csv`` and ``<pair>_straight.csv`` (whole metres, rows and
columns in the order of ``points.csv``), ``days.csv`` and ``manifest.json``. No route is
solved here.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import Point
from shapely.ops import transform, unary_union

from routing_roads.contract import (
    BUFFER_M,
    DAYS,
    DAYS_SEED,
    EXTRACT,
    EXTRACT_BYTES,
    EXTRACT_MD5,
    EXTRACT_SHA256,
    MUNICIPALITIES,
    PAIRS,
    PROJECTED_CRS,
    Pair,
)
from routing_roads.network import Network, build_graph, crossings, matrices
from routing_roads.osm import Pharmacy, file_hashes, read_places, write_drivable

POINT_FIELDS = ("pair", "index", "role", "municipality", "osm_type", "osm_id", "lon", "lat",
                "node", "node_lon", "node_lat", "snap_m")
PACKAGES = ("osmium", "osmnx", "pyvrp", "networkx", "numpy", "scipy", "shapely", "pyproj")


def verify_extract(pbf: Path) -> dict[str, Any]:
    found = file_hashes(pbf)
    expected = {"bytes": EXTRACT_BYTES, "md5": EXTRACT_MD5, "sha256": EXTRACT_SHA256}
    if found != expected:
        raise ValueError(f"{pbf} is not the pinned extract: {found} against {expected}.")
    return {"file": EXTRACT, **found}


def buffered(polygon: Any) -> Any:
    """The polygon buffered by BUFFER_M metres in the Portuguese projection."""
    forward = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform
    back = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True).transform
    return transform(back, transform(forward, polygon).buffer(BUFFER_M))


def _pharmacies_of(pair: Pair, municipalities: dict[str, Any],
                   pharmacies: list[Pharmacy]) -> list[tuple[str, Pharmacy]]:
    """The pair's pharmacies, city first, each block sorted by OSM type and id."""
    out = []
    for name in (pair.city, pair.other):
        inside = [p for p in pharmacies if municipalities[name].contains(Point(p.lon, p.lat))]
        out += [(name, p) for p in sorted(inside, key=lambda p: (p.osm_type, p.osm_id))]
    return out


def draw_days(sizes: dict[str, int]) -> list[dict[str, Any]]:
    """Section 4: one seeded generator, pairs in PAIRS order; on each day every pharmacy
    of the pair orders when its uniform draw is below the pair's probability."""
    rng = np.random.default_rng(DAYS_SEED)
    rows = []
    for pair in PAIRS:
        draws = rng.random((DAYS, sizes[pair.key]))
        for day in range(DAYS):
            rows += [{"pair": pair.key, "day": day + 1, "index": int(i) + 1}
                     for i in np.flatnonzero(draws[day] < pair.order_probability)]
    return rows


def build(pbf: Path, out: Path, work: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {"extract": verify_extract(pbf), "buffer_m": BUFFER_M,
                                "projected_crs": PROJECTED_CRS, "days": DAYS,
                                "days_seed": DAYS_SEED,
                                "versions": {p: importlib.metadata.version(p) for p in PACKAGES},
                                "pairs": {}}
    municipalities, pharmacies = read_places(pbf, MUNICIPALITIES)
    out.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}
    for pair in PAIRS:
        area = buffered(unary_union([municipalities[pair.city], municipalities[pair.other]]))
        xml = work / f"{pair.key}_drive.osm"
        ways = write_drivable(pbf, area.bounds, xml)
        graph, counts = build_graph(xml, area)
        net = Network.from_graph(graph)
        members = _pharmacies_of(pair, municipalities, pharmacies)
        city = [p for name, p in members if name == pair.city]
        depot_at = (float(np.mean([p.lon for p in city])), float(np.mean([p.lat for p in city])))
        # Section 7's depot variant, at the mean of all the pair's pharmacies: the last point.
        variant_at = (float(np.mean([p.lon for _, p in members])),
                      float(np.mean([p.lat for _, p in members])))
        variant_in = next((name for name in (pair.city, pair.other)
                           if municipalities[name].contains(Point(*variant_at))), "")
        rows = [("depot", pair.city, "", 0, *depot_at)] + [
            ("pharmacy", name, p.osm_type, p.osm_id, p.lon, p.lat) for name, p in members] + [
            ("depot_variant", variant_in, "", 0, *variant_at)]
        nodes = []
        for index, (role, name, osm_type, osm_id, lon, lat) in enumerate(rows):
            node, snap_m = net.snap(lon, lat)
            nodes.append(node)
            points.append({"pair": pair.key, "index": index, "role": role, "municipality": name,
                           "osm_type": osm_type, "osm_id": osm_id, "lon": lon, "lat": lat,
                           "node": int(net.ids[node]), "node_lon": float(net.lon[node]),
                           "node_lat": float(net.lat[node]), "snap_m": round(snap_m, 1)})
        road, straight, pred = matrices(net, nodes)
        for name, matrix in (("road", road), ("straight", straight)):
            np.savetxt(out / f"{pair.key}_{name}.csv", matrix, fmt="%d", delimiter=",")
        bank = [name for _, name, *_ in rows[:-1]]        # the depot variant left out
        node_bank = np.full(net.ids.size, "", dtype=object)
        for name in (pair.city, pair.other):
            node_bank[contains_xy(municipalities[name], net.lon, net.lat)] = name
        manifest["pairs"][pair.key] = {
            "city": pair.city, "other": pair.other, "order_probability": pair.order_probability,
            "drivable_ways": ways, **counts,
            "crossings": crossings(net, pred, nodes[:-1], bank, list(node_bank))}
        sizes[pair.key] = len(members)
    with (out / "points.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=POINT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)
    with (out / "days.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("pair", "day", "index"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(draw_days(sizes))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                                       newline="\n")
    return manifest
