"""Section 5: the plans, solved by PyVRP, and their costs on both matrices.

A plan is a list of routes; a route is a list of point indices of the pair (the depot is
not listed; every route starts and ends there).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer
from pyvrp import Model
from pyvrp.stop import MaxIterations

from routing_roads.contract import PROJECTED_CRS

IntArray = NDArray[np.int64]
Routes = list[list[int]]


def projected(lon: Sequence[float], lat: Sequence[float]) -> NDArray[np.float64]:
    """Points in EPSG:3763 metres. PyVRP's SWAP* operator compares routes by the angles of
    their clients around the depot, so it is given real coordinates; the distances it
    optimises come from the matrix alone."""
    to = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = to.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.column_stack([x, y])


def solve(planning: IntArray, xy: NDArray[np.float64], points: Sequence[int], capacity: int,
          iterations: int, seed: int) -> Routes:
    """Minimise total distance on ``planning`` over ``points`` (the depot first), with
    unit orders, identical vehicles of ``capacity`` and as many vehicles as orders."""
    m = Model()
    locations = [m.add_location(x=float(xy[p, 0]), y=float(xy[p, 1])) for p in points]
    m.add_depot(locations[0])
    for loc in locations[1:]:
        m.add_client(loc, delivery=1)
    m.add_vehicle_type(num_available=len(points) - 1, capacity=capacity)
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i != j:
                m.add_edge(locations[i], locations[j], distance=int(planning[a, b]))
    result = m.solve(stop=MaxIterations(iterations), seed=seed, display=False,
                     collect_stats=False)
    if not result.is_feasible():
        raise ValueError("PyVRP returned an infeasible plan.")
    routes = [[points[act.idx + 1] for act in route if act.is_client()]
              for route in result.best.routes()]
    if cost(routes, planning, points[0]) != round(result.cost()):
        raise ValueError("The plan's cost on its own matrix differs from PyVRP's.")
    return routes


def cost(routes: Routes, matrix: IntArray, depot: int) -> int:
    """Total distance of the routes on ``matrix``, depot legs included."""
    total = 0
    for route in routes:
        stops = [depot, *route, depot]
        total += int(sum(matrix[a, b] for a, b in zip(stops, stops[1:], strict=False)))
    return total


def encode(routes: Routes) -> str:
    return "|".join("-".join(map(str, r)) for r in routes)


def decode(text: str) -> Routes:
    return [[int(p) for p in r.split("-")] for r in text.split("|")] if text else []
