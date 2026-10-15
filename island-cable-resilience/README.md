# Correlated faults and island isolation on the submarine cable network

Code for the Medium article *Correlated Faults and Island Isolation on the Submarine Cable Network*.

The submarine cable map is turned into a graph (land masses, landing stations and branching units as nodes, drawn cable segments as edges), and every island with a path to a continent gets three numbers:

- **k**, the number of cable systems that land on it;
- **λ**, its edge connectivity, the fewest segment cuts that isolate it (a max-flow computation);
- **N(r)**, the fewest *incidents* of radius *r* km, centred within 100 km of its landing stations, that isolate it. An incident severs every segment within *r* of its centre. N(r) is solved exactly as a small integer programme with OR-Tools CP-SAT.

Two fault models then turn geometry into expected hours of isolation a year. Both break the same number of segments per year (one fault per 7,500 km, from industry fault counts) and share the same repair process; one lets segments fail independently, the other sends the faults as incidents of random position and size, with one shared wait for a repair ship per incident.

## Main results

| | |
|---|---|
| Islands analysed | 463 (176 with two or more cable systems) |
| Islands with ≥ 2 systems isolated by one 1 km incident | 63 (36%) |
| … by one 10 km incident | 103 (59%) |
| … by one 50 km incident | 137 (78%) |
| Median expected isolation, independent faults | < 0.1 s a year |
| Median expected isolation, incidents | 2.3 h a year |

For the Azores and Madeira, the planned Atlantic CAM ring and Google's Nuvem cut the independent model's isolation risk to almost nothing and the incident model's by about a quarter, because on São Miguel every new cable lands at Ponta Delgada beside the old ones. A counterfactual that lands one leg on the north coast does more for São Miguel than the whole replacement does.

## Running it

```bash
pip install -e ".[dev]"
./fetch_data.sh      # pinned cable and coastline data into ./data (about 20 MB)
python -m pytest -q  # synthetic checks, plus real-data checks once the data are fetched
./run_all.sh         # all experiments and figures, about 20 minutes
```

Set `CABLE_DATA` to keep the data elsewhere. Outputs go to `results/` (JSON and logs) and `figures/`; the versions committed here are the ones behind the article.

| File | What it does |
|---|---|
| `src/cableresilience/data.py` | Loads routes, landing points and coastlines; joins them geometrically; merges land masses linked by bridges and tunnels |
| `src/cableresilience/cuts.py` | Graph, edge connectivity λ, and the CP-SAT model for N(r) |
| `src/cableresilience/risk.py` | Independent-fault and incident models, minimal cuts, Monte Carlo |
| `experiments/exp1_structure.py` | k, λ and N(1), N(10), N(50) for every island |
| `experiments/exp2_risk.py` | Both fault models for 24 case islands under nine parameter settings, and for all 176 islands with two or more systems |
| `experiments/exp3_portugal.py` | The Azores and Madeira today, with the planned systems, and after the old ring retires |
| `experiments/exp4_landing.py` | Counterfactual: one leg of the new ring landed on São Miguel's north coast |
| `experiments/figures.py` | Figures 1 to 4 |
| `tests/` | The cut measures and both fault models on small synthetic networks, and spot checks on the real data |

## Data and licences

- Cable routes, landing points and system details: [TeleGeography Submarine Cable Map](https://www.submarinecablemap.com), licensed [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). The copy used was fetched on 24 September 2026 and committed to [ixpavi/subsea-siting](https://github.com/ixpavi/subsea-siting) (commit `98d377f`). It is fetched by `fetch_data.sh` and not redistributed here. The committed `figures/` and `results/` are derived from it and are therefore licensed CC BY-NC-SA 3.0, with attribution to TeleGeography.
- Coastlines: [Natural Earth](https://www.naturalearthdata.com) 1:10m land and minor islands, public domain (commit `ca96624` of nvkelso/natural-earth-vector).
- Code: Apache License 2.0, like the rest of this repository.

## Caveats

The map is schematic: routes are drawn for legibility and landing points are placed at town level, so N(1) measures whether cables land in the same town. Planned routes may change. Isolation is topological (any surviving path counts, whatever its capacity), satellite and microwave backup are not counted, the fault rate is a global average, and incidents are uniform within 100 km of the landings. The article discusses these in its *Limits* section.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt
