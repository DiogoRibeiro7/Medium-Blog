"""The directed road network, snapping, and the two distance matrices (METHODS.md, 3-4)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry.base import BaseGeometry

from routing_roads.contract import EARTH_RADIUS_M

FloatArray = NDArray[np.float64]


def build_graph(xml: Path, polygon: BaseGeometry) -> tuple[nx.MultiDiGraph, dict[str, int]]:
    """OSMnx's graph of the drivable ways, clipped to ``polygon``, simplified, and reduced
    to its largest strongly connected component. One-way streets are respected."""
    g = ox.graph_from_xml(xml, bidirectional=False, simplify=False, retain_all=True)
    sizes = {"nodes_read": g.number_of_nodes()}
    g = ox.truncate.truncate_graph_polygon(g, polygon)
    g = ox.simplify_graph(g)
    sizes |= {"nodes_clipped": g.number_of_nodes(), "edges_clipped": g.number_of_edges()}
    g = ox.truncate.largest_component(g, strongly=True)
    sizes |= {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
    return g, sizes


def haversine(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> Any:
    """Great-circle distance in metres (section 4); broadcasts over arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class Network:
    """The graph as arrays: node ids, coordinates and a CSR matrix of edge lengths (the
    shortest of any parallel edges)."""

    ids: NDArray[np.int64]
    lon: FloatArray
    lat: FloatArray
    lengths: csr_matrix
    graph: nx.MultiDiGraph

    @classmethod
    def from_graph(cls, g: nx.MultiDiGraph) -> Network:
        ids = np.array(sorted(g.nodes), dtype=np.int64)
        position = {n: i for i, n in enumerate(ids.tolist())}
        best: dict[tuple[int, int], float] = {}
        for u, v, data in g.edges(data=True):
            key = (position[u], position[v])
            best[key] = min(best.get(key, np.inf), float(data["length"]))
        rows, cols = zip(*best, strict=True) if best else ((), ())
        lengths = csr_matrix((list(best.values()), (rows, cols)), shape=(ids.size, ids.size))
        lon = np.array([g.nodes[n]["x"] for n in ids.tolist()], dtype=float)
        lat = np.array([g.nodes[n]["y"] for n in ids.tolist()], dtype=float)
        return cls(ids, lon, lat, lengths, g)

    def snap(self, lon: float, lat: float) -> tuple[int, float]:
        """The index of the nearest node by great-circle distance, and that distance."""
        d = haversine(lon, lat, self.lon, self.lat)
        i = int(np.argmin(d))
        return i, float(d[i])


def matrices(net: Network, nodes: Sequence[int],
             ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int32]]:
    """Road and straight-line distances between the given node indices, in whole metres,
    and the shortest-path predecessors (for the crossing report)."""
    dist, pred = dijkstra(net.lengths, directed=True, indices=list(nodes),
                          return_predecessors=True)
    road = dist[:, list(nodes)]
    if not np.all(np.isfinite(road)):
        raise ValueError("A point cannot reach another: the component is not strongly "
                         "connected.")
    lon, lat = net.lon[list(nodes)], net.lat[list(nodes)]
    straight = haversine(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    return (np.rint(road).astype(np.int64), np.rint(straight).astype(np.int64),
            pred.astype(np.int32))


def path(pred: NDArray[np.int32], source_row: int, target: int) -> list[int]:
    """Node indices of the shortest path to ``target`` from the source of ``source_row``."""
    out = [target]
    while (step := int(pred[source_row, out[-1]])) >= 0:
        out.append(step)
    return out[::-1]


def _name(data: dict[str, Any]) -> str:
    """An edge's name. Simplification merges the names of the ways it joins into a list
    whose order follows Python's per-process string hashing, so the list is sorted."""
    value = data.get("name", data.get("ref", "(unnamed)"))
    return " / ".join(sorted(map(str, value))) if isinstance(value, list) else str(value)


def crossing_names(net: Network, nodes: Sequence[int], node_bank: Sequence[str]) -> list[str]:
    """The edges along a path of node indices whose two ends lie in different
    municipalities of the pair. The two meet only in the river, so these are the bridges."""
    out: list[str] = []
    for a, b in zip(nodes, nodes[1:], strict=False):
        if node_bank[a] and node_bank[b] and node_bank[a] != node_bank[b]:
            u, v = int(net.ids[a]), int(net.ids[b])
            out.append(_name(min(net.graph.get_edge_data(u, v).values(),
                                 key=lambda d: d["length"])))
    return out


def crossings(net: Network, pred: NDArray[np.int32], nodes: Sequence[int],
              bank: Sequence[str], node_bank: Sequence[str]) -> dict[str, Any]:
    """For every ordered pair of points on opposite banks (``bank``: each point's
    municipality), where the shortest road path between them crosses (``node_bank``: each
    network node's municipality of the pair, or "" outside both): each crossing's share of
    those paths."""
    counts: Counter[str] = Counter()
    paths = 0
    for i, _ in enumerate(nodes):
        for j, target in enumerate(nodes):
            if bank[i] != bank[j]:
                paths += 1
                counts.update(set(crossing_names(net, path(pred, i, target), node_bank)))
    return {"paths": paths, "bridges": {name: n / paths for name, n in counts.most_common()}}
