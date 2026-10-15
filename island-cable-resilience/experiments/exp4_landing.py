"""Experiment 4: a counterfactual landing on São Miguel.

The planned New CAM Ring reaches São Miguel from Terceira and lands at Ponta Delgada, on the
south coast, like every other cable on the island. Here the same leg is redrawn to pass north
of the island and land near Ribeira Grande, about 15 km away on the north coast. Nothing else
changes. The site is illustrative, not a proposal: it was chosen only because it is on the
opposite coast and the leg arrives from the north-west.
"""
import dataclasses
import json
import os

from common import *  # noqa: F401,F403
from exp3_portugal import OLD_RING, PLANNED
from cableresilience.cuts import build_graph, edge_cut_lambda, external_cables, min_incidents
from cableresilience.data import load
from cableresilience.risk import IslandModel, Params, incident_risk, independent_risk

HYPO = {"id": "ribeira-grande-hypothetical", "name": "Ribeira Grande (hypothetical), Portugal",
        "lat": 37.8235, "lng": -25.5185}
# the Terceira -> São Miguel leg of the New CAM Ring, rerouted along the north coast
NORTH_TAIL = [[37.96, -25.85], [37.87, -25.62], [HYPO["lat"], HYPO["lng"]]]


def make_data(dst: Path) -> Path:
    src = DATA / "ixpavi_subsea-siting" / "public" / "data"
    out = dst / "ixpavi_subsea-siting" / "public" / "data"
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("cables.json", "landing-points.json") and not (out / f.name).exists():
            os.symlink(f.resolve(), out / f.name)
    if not (dst / "ne").exists():
        os.symlink((DATA / "ne").resolve(), dst / "ne")
    cables = json.loads((src / "cables.json").read_text())
    done = 0
    for c in cables:
        if c["id"] != "new-cam-ring":
            continue
        for i, path in enumerate(c["paths"]):
            a, b = path[0], path[-1]
            if abs(a[0] - 38.666) < 0.01 and abs(b[0] - 37.7396) < 0.01:      # Terceira -> São Miguel
                c["paths"][i] = path[:3] + NORTH_TAIL
                done += 1
    assert done == 1, done
    (out / "cables.json").write_text(json.dumps(cables))
    lps = json.loads((src / "landing-points.json").read_text())
    lps.append(HYPO)
    (out / "landing-points.json").write_text(json.dumps(lps))
    return dst


SCENARIOS = {
    "plus New CAM Ring and Nuvem, Terceira leg landed on the north coast": {"extra": PLANNED, "exclude": ()},
    "after the old ring retires, Terceira leg landed on the north coast": {"extra": PLANNED,
                                                                            "exclude": OLD_RING},
}

if __name__ == "__main__":
    data = make_data(RESULTS / "cf_data")
    out = {"scenarios": {}, "params": dataclasses.asdict(Params())}
    for sname, kw in SCENARIOS.items():
        net = load(data, **kw)
        G = build_graph(net)
        lp = {v["name"]: k for k, v in net.landing_points.items()}
        u = net.landing_points[lp["Ponta Delgada, Portugal"]]["unit"]
        assert net.landing_points[lp[HYPO["name"]]]["unit"] == u, "hypothetical landing not on São Miguel"
        row = {"island": "São Miguel", "cables": sorted(external_cables(net, u)),
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
        out["scenarios"][sname] = [row]
        print(sname, "|", row["k"], row["lambda"], row["N_1km"], row["N_10km"], row["N_50km"],
              "%.3g" % row["independent"]["hours_per_year"], "%.3g" % row["incident"]["hours_per_year"],
              row["landings"], flush=True)
    save_json("exp4_landing.json", out)
