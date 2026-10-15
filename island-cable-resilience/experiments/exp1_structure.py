"""Experiment 1: for every island, cable count against the number of independent cuts
and of separate incidents needed to cut it off from every continent."""
import time

import networkx as nx

from common import *  # noqa: F401,F403
from cableresilience.cuts import (build_graph, edge_cut_lambda, external_cables, is_platform, mainland_units,
                      min_incidents, unit_node)
from cableresilience.data import unit_label

net = network()
G = build_graph(net)
main = {unit_node(m) for m in mainland_units(net)}
reach = set().union(*[c for c in nx.connected_components(G) if c & main])
rows = []
t0 = time.time()
islands = [u for u in range(len(net.units))
           if not net.units[u]["mainland"] and not is_platform(net, u)]
skipped = {"not connected to a continent in the data": 0}
for u in islands:
    if unit_node(u) not in reach:
        skipped["not connected to a continent in the data"] += 1
        continue
    cables = sorted(external_cables(net, u))
    lam = edge_cut_lambda(net, u, G)
    row = {"unit": u, "label": unit_label(net, u), "area_km2": net.units[u]["area_km2"],
           "landing_stations": len(net.units[u]["landings"]),
           "landings": [net.landing_points[l]["name"] for l in net.units[u]["landings"]],
           "cables": cables, "k": len(cables), "lambda": lam}
    for r in RADII_KM:
        n, chosen = min_incidents(net, u, r, ZONE_KM, G)
        row[f"N_{int(r)}km"] = n
    rows.append(row)
    if len(rows) % 25 == 0:
        print(len(rows), "islands", round(time.time() - t0), "s", flush=True)
save_json("exp1_structure.json", {"zone_km": ZONE_KM, "radii_km": RADII_KM, "rows": rows,
                                  "skipped": skipped, "n_islands_considered": len(islands),
                                  "data": {k: v for k, v in net.meta.items() if k != "polygons"}})
print("done", len(rows), skipped)
