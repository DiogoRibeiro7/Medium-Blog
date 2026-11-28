# Same Aircraft, Different Crew: aircraft routing, crew pairing and disruption on a synthetic hub airline

Companion code for an article by Diogo Ribeiro on the mathematics behind airline operations. It plans a synthetic
Munich-hub airline (30 European outstations, 142 daily flights, one fleet, one crew base) in four ways, flies every
plan through 300 seeded two-day replications of seven disruption scenarios, and tests whether the plan that uses
aircraft and crews most efficiently is also the one that copes best when the day goes wrong.

- **Greedy**: first-in-first-out aircraft, then Chvátal's greedy set cover over legal pairings.
- **Sequential**: exact daily aircraft routing with a four-night maintenance rule (state-expanded network), then crew
  pairing by column generation with the routing fixed.
- **Integrated**: routing and pairing in one model, linked through short connections.
- **Robust**: the integrated model with a hub buffer and a penalty on expected propagated delay.

Crew legality follows the EASA flight time limitations (Regulation (EU) No 83/2014): Table 2 of ORO.FTL.205, the
minimum rest of ORO.FTL.235 and the commander's discretion of ORO.FTL.205(f). The opening case (Figure 1) is
the synthetic airline's own Porto rotation, with the crews the integrated plan assigns; every flight is fictional.

`PLAN.md` is the pre-registration, written after the code and before the main run, with the changes made after it.

## Results in one paragraph

Integration saved 0.62% of planned cost (€4,910 a day; H1 supported) and did not lower operational cost (H2
supported); on a stressed day its disruption cost was €9,809 a day higher than the sequential plan's. Under baseline
delays the robust plan's total cost was not lower than the sequential and integrated plans' (+€361 and +€4,261 a day;
H3 not supported), but it was far lower on bad days (−€23,569 a day against the integrated plan on a stressed day,
−€93,366 under a hub capacity cut). The best hub buffer was none in the calm and average delay climates and the
largest tested, 45 minutes, in the stressed one (H4 supported). A penalty from a point forecast of delays did no
better than no penalty, and neither did one from the predictive distribution under baseline delays (H5 half
supported), although the distributional penalty saved €24,129 a day on a stressed day against €6,975 for the point
forecast. Added after the results: a plan with no buffer and a penalty of weight 0.5 had the lowest expected total
cost under baseline delays (−€4,325 a day against the integrated plan, for 0.15% more planned cost).

## Files

- `PLAN.md`: the pre-registration.
- `configs/experiment.toml`: every parameter of the main run; `configs/smoke.toml` for continuous integration.
- `src/airline_ops/regulation/ftl.py`: EASA FTL Table 2, minimum rest and commander's discretion.
- `src/airline_ops/data/airports.py`: downloads the OpenFlights airport file at a pinned commit and checks its size
  and SHA-256 against `../data_sources/openflights-airports.json`; the bytes are cached in `.cache/`, never committed.
- `src/airline_ops/network/`: the network model and the synthetic network generator.
- `src/airline_ops/planning/`: duties and pairings, connection arcs, exact routing with maintenance states, the
  column generation master and pricing, the four planners and the plan validator.
- `src/airline_ops/simulation/`: the delay model, the scenarios and the discrete-event simulator with its recovery
  policy.
- `src/airline_ops/experiments/runner.py`: the registered experiment (`results/*.json`).
- `src/airline_ops/experiments/details.py`: paired re-evaluations added after the results (`results/details.json`).
- `src/airline_ops/experiments/summary.py`: the text summary every number in the article comes from
  (`results/summary.txt`).
- `src/airline_ops/experiments/digest.py`: the digest of the inputs that determine the results (`results/INPUTS`).
- `src/airline_ops/viz/figures.py`, `figures/`: the article's ten figures, as SVG and 200 dpi PNG.
- `tests/`: unit tests (FTL tables, network, planners including column-generation pricing against full enumeration,
  simulator, statistics).

## Run

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
ruff check . && ruff format --check . && mypy src && python -m pytest -q
python -m airline_ops run --config configs/experiment.toml --out results       # about 25 minutes on one core
python -m airline_ops details --config configs/experiment.toml --results results
python -m airline_ops summary --results results
python -m airline_ops figures --results results --out figures
```

The workflow `.github/workflows/airline-operations-optimization.yml` runs lint, static types, tests and a smoke run
on every change; on pushes to the branch and on manual runs it reruns the experiment when `results/INPUTS` no longer
matches the inputs, then the summary and the figures, and commits them.

## Data

- OpenFlights airport database (`airports.dat`), pinned to commit `a438edcb` of `jpatokal/openflights`, Open
  Database License; registered as an external source in `DiogoRibeiro7/data` (`external/openflights-airports`).
- EUROCONTROL CODA Digest, annual 2024 and November 2025: calibration targets (punctuality, mean and reactionary
  delay), quoted in the configuration.
- Typical scheduled block times on the Munich–Porto sector (180 minutes westbound, 170 eastbound), used only to
  calibrate the block-time model.

Everything else (timetable, flight numbers, passengers, pairings, costs) is synthetic and set in
`configs/experiment.toml`.
