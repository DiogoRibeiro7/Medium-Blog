"""Experiment 2: isolation risk of case-study islands under the independent and the incident
fault model, with sensitivity to the model's assumptions; then all islands at baseline."""
import dataclasses
import time

from common import *  # noqa: F401,F403
from cableresilience.cuts import build_graph, external_cables
from cableresilience.risk import IslandModel, Params, incident_risk, independent_risk

CASES = [
    ("São Miguel (Azores)", "Ponta Delgada, Portugal"),
    ("Terceira (Azores)", "Angra do Heroismo, Portugal"),
    ("Faial (Azores)", "Faial, Portugal"),
    ("Flores (Azores)", "Flores, Portugal"),
    ("Madeira", "Funchal, Portugal"),
    ("Porto Santo", "Porto Santo, Portugal"),
    ("Gran Canaria", "Gran Canaria, Canary Islands, Spain"),
    ("Tenerife", "Granadilla de Abona, Spain"),
    ("Malta", "Mellieha, Malta"),
    ("Cyprus", "Yeroskipos, Cyprus"),
    ("Iceland", "Seydisfjordur, Iceland"),
    ("Faroe Islands", "Torshavn, Faroe Islands"),
    ("Ireland", "Galway, Ireland"),
    ("Sicily", "Mazara del Vallo, Italy"),
    ("Sri Lanka", "Colombo, Sri Lanka"),
    ("Mauritius", "Baie du Tombeau, Mauritius"),
    ("Maldives (Malé)", "Male, Maldives"),
    ("Fiji (Viti Levu)", "Suva, Fiji"),
    ("Tonga (Tongatapu)", "Nuku'alofa, Tonga"),
    ("Bermuda", "Paget, Bermuda"),
    ("Cape Verde (Santiago)", "Praia, Cape Verde"),
    ("Guam", "Piti, Guam"),
    ("Taiwan", "Tanshui, Taiwan"),
    ("Puerto Rico", "San Juan, PR, United States"),
]

SENSITIVITY = {
    "baseline": {},
    "no regional incidents": {"regional_share": 0.0},
    "regional share 2%": {"regional_share": 0.02},
    "regional share 25%": {"regional_share": 0.25},
    "regional radius 5 km": {"regional_median_km": 5.0},
    "regional radius 40 km": {"regional_median_km": 40.0},
    "fault rate x3": {"faults_per_km_year": 3 * 200 / 1.5e6},
    "repairs start after 10 days": {"wait_median_days": 10.0},
    "repairs start after 50 days": {"wait_median_days": 50.0},
}


def run_case(net, G, u, p):
    M = IslandModel(net, u, p.zone_km, G)
    cuts = M.minimal_cuts()
    a = independent_risk(M, p, cuts)
    b = incident_risk(M, p, cuts)
    return M, a, b


if __name__ == "__main__":
    net = network()
    G = build_graph(net)
    lp = {v["name"]: k for k, v in net.landing_points.items()}
    out = {"params": dataclasses.asdict(Params()), "cases": [], "all_islands": []}
    t0 = time.time()
    for label, landing in CASES:
        if landing not in lp:
            print("missing", label)
            continue
        u = net.landing_points[lp[landing]]["unit"]
        row = {"label": label, "unit": u, "cables": sorted(external_cables(net, u)),
               "landings": [net.landing_points[l]["name"] for l in net.units[u]["landings"]]}
        for name, change in SENSITIVITY.items():
            p = dataclasses.replace(Params(), **change)
            M, a, b = run_case(net, G, u, p)
            row[name] = {"independent": a, "incident": b}
        row["segments_in_zone"] = len(M.edges)
        row["km_in_zone"] = float(M.length_in_zone.sum())
        out["cases"].append(row)
        base = row["baseline"]
        print(f"{label:24s} indep {base['independent']['hours_per_year']:.3g} h/yr | incident "
              f"{base['incident']['hours_per_year']:.3g} h/yr | {time.time() - t0:.0f}s", flush=True)
    save_json("exp2_risk.json", out)

    # every island with two or more cables, baseline only
    rows = load_json("exp1_structure.json")["rows"]
    p = Params()
    for r in rows:
        if r["k"] < 2:
            continue
        M, a, b = run_case(net, G, r["unit"], p)
        out["all_islands"].append({"unit": r["unit"], "label": r["label"], "k": r["k"],
                                   "lambda": r.get("lambda"), "N_1km": r["N_1km"], "N_10km": r["N_10km"],
                                   "N_50km": r["N_50km"],
                                   "independent_bound_h": a["upper_bound_hours_if_no_small_cut"],
                                   "min_cut_size": a["min_cut_size"],
                                   "independent_h": a["hours_per_year"], "incident_h": b["hours_per_year"],
                                   "single_h": b["single_incident_hours_per_year"],
                                   "independent_episodes": a["episodes_per_year"],
                                   "incident_episodes": b["episodes_per_year"]})
    save_json("exp2_risk.json", out)
    print("done", len(out["all_islands"]), round(time.time() - t0), "s")
