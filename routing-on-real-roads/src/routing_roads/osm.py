"""Reading the pinned extract: municipal boundaries, pharmacies and the drivable ways."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import osmium
from shapely import wkb
from shapely.geometry.base import BaseGeometry

from routing_roads.contract import DRIVE_EXCLUDE

_EXCLUDE = {key: re.compile(pattern) for key, pattern in DRIVE_EXCLUDE.items()}
# The way tags OSMnx 2.1.1 keeps (settings.useful_tags_way); oneway and junction drive its
# one-way logic.
WAY_TAGS = ("access", "area", "bridge", "est_width", "highway", "junction", "landuse", "lanes",
            "maxspeed", "name", "oneway", "ref", "service", "tunnel", "width")


def drivable(tags: Mapping[str, str]) -> bool:
    """OSMnx's "drive" filter, in Overpass semantics (contract.DRIVE_EXCLUDE)."""
    return "highway" in tags and not any(
        key in tags and pattern.search(tags[key]) for key, pattern in _EXCLUDE.items())


def file_hashes(path: Path) -> dict[str, str | int]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            md5.update(block)
            sha.update(block)
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


@dataclass(frozen=True)
class Pharmacy:
    osm_type: str            # "node" or "way"
    osm_id: int
    lon: float
    lat: float


def read_places(pbf: Path, names: Iterable[str],
                ) -> tuple[dict[str, BaseGeometry], list[Pharmacy]]:
    """The named municipalities' boundaries (relations with admin_level 7) and every
    pharmacy in the extract. A way's location is the mean of its distinct nodes."""
    wanted = set(names)
    factory = osmium.geom.WKBFactory()
    areas: dict[str, BaseGeometry] = {}
    pharmacies: list[Pharmacy] = []
    processor = (osmium.FileProcessor(str(pbf)).with_locations()
                 .with_areas(osmium.filter.TagFilter(("admin_level", "7")))
                 .with_filter(osmium.filter.KeyFilter("amenity", "admin_level")))
    for o in processor:
        tags = o.tags
        if isinstance(o, osmium.osm.Area):
            name = tags.get("name")
            if (name in wanted and not o.from_way() and tags.get("admin_level") == "7"
                    and tags.get("boundary") == "administrative"):
                if name in areas:
                    raise ValueError(f"Two municipal boundaries are named {name}.")
                areas[name] = wkb.loads(factory.create_multipolygon(o), hex=True)
        elif tags.get("amenity") == "pharmacy":
            if isinstance(o, osmium.osm.Node):
                pharmacies.append(Pharmacy("node", o.id, o.location.lon, o.location.lat))
            elif isinstance(o, osmium.osm.Way):
                nodes = {n.ref: (n.lon, n.lat) for n in o.nodes if n.location.valid()}
                if nodes:
                    lon = sum(x for x, _ in nodes.values()) / len(nodes)
                    lat = sum(y for _, y in nodes.values()) / len(nodes)
                    pharmacies.append(Pharmacy("way", o.id, lon, lat))
    missing = wanted - set(areas)
    if missing:
        raise ValueError(f"No municipal boundary for {', '.join(sorted(missing))}.")
    return areas, pharmacies


def write_drivable(pbf: Path, bounds: tuple[float, float, float, float], out: Path) -> int:
    """Write the drivable ways with a node inside ``bounds`` (min lon, min lat, max lon,
    max lat), and all their nodes, as an OSM XML file OSMnx can read. Returns the ways."""
    west, south, east, north = bounds
    ways: dict[int, tuple[list[int], dict[str, str]]] = {}
    nodes: dict[int, tuple[float, float]] = {}
    processor = (osmium.FileProcessor(str(pbf)).with_locations()
                 .with_filter(osmium.filter.KeyFilter("highway")))
    for way in processor:
        if not isinstance(way, osmium.osm.Way):
            continue
        tags = {t.k: t.v for t in way.tags}
        if not drivable(tags):
            continue
        refs = [(n.ref, n.lon, n.lat) for n in way.nodes if n.location.valid()]
        if any(west <= x <= east and south <= y <= north for _, x, y in refs):
            ways[way.id] = ([r for r, _, _ in refs], {k: v for k, v in tags.items()
                                                       if k in WAY_TAGS})
            nodes.update({r: (x, y) for r, x, y in refs})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    writer = osmium.SimpleWriter(str(out))
    try:
        for ref in sorted(nodes):
            writer.add_node(osmium.osm.mutable.Node(id=ref, location=nodes[ref], version=1))
        for way_id in sorted(ways):
            members, kept = ways[way_id]
            writer.add_way(osmium.osm.mutable.Way(id=way_id, nodes=members, tags=kept,
                                                  version=1))
    finally:
        writer.close()
    return len(ways)
