"""Experiment 3: the Azores and Madeira today, with the cables planned for 2026, and after
the old CAM ring retires."""
import dataclasses
import pickle

from common import *  # noqa: F401,F403
from cableresilience.cuts import build_graph, edge_cut_lambda, external_cables, min_incidents
from cableresilience.data import load
from cableresilience.risk import IslandModel, Params, incident_risk, independent_risk

PLANNED = ("new-cam-ring", "nuvem")
OLD_RING = ("columbus-iii-azores-portugal", "cam-ring", "continente-madeira")
SCENARIOS = {
    "in service (Sept 2026 snapshot)": {"extra": (), "exclude": ()},
    "plus New CAM Ring and Nuvem": {"extra": PLANNED, "exclude": ()},
    "after the old ring retires": {"extra": PLANNED, "exclude": OLD_RING},
}
ISLANDS = [
    ("São Miguel", "Ponta Delgada, Portugal"),
    ("Terceira", "Angra do Heroismo, Portugal"),
    ("Faial", "Faial, Portugal"),
    ("Madeira", "Funchal, Portugal"),
    ("Porto Santo", "Porto Santo, Portugal"),
]

if __name__ == "__main__":
    out = {"scenarios": {}, "params": dataclasses.asdict(Params())}
    for sname, kw in SCENARIOS.items():
        net = load(DATA, **kw)
        G = build_graph(net)
        lp = {v["name"]: k for k, v in net.landing_points.items()}
        rows = []
        for label, landing in ISLANDS:
            if landing not in lp:
                rows.append({"island": label, "cables": [], "k": 0, "note": "no cable lands here"})
                print(sname, "|", label, "no cable", flush=True)
                continue
            u = net.landing_points[lp[landing]]["unit"]
            row = {"island": label, "cables": sorted(external_cables(net, u)),
                   "landings": sorted(net.landing_points[l]["name"] for l in net.units[u]["landings"])}
            row["k"] = len(row["cables"])
            row["lambda"] = edge_cut_lambda(net, u, G)
            for r in RADII_KM:
                row[f"N_{int(r)}km"] = min_incidents(net, u, r, ZONE_KM, G)[0]
            p = Params()
            M = IslandModel(net, u, p.zone_km, G)
            cuts = M.minimal_cuts()
            row["independent"] = independent_risk(M, p, cuts)
            row["incident"] = incident_risk(M, p, cuts)
            rows.append(row)
            print(sname, "|", label, row["k"], row["lambda"], row["N_1km"], row["N_10km"], row["N_50km"],
                  "%.3g" % row["independent"]["hours_per_year"], "%.3g" % row["incident"]["hours_per_year"],
                  flush=True)
        out["scenarios"][sname] = rows
    save_json("exp3_portugal.json", out)
