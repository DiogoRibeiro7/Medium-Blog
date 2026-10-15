# Operational research in Python, 2026: a two-stage depot example

Companion code for the Medium article *Operational Research in Python, 2026: The Solver Stack and a Two-Stage Depot Example*.

## Question

A distributor leases depots for a season before it knows weekly demand, then routes pallets every week once orders are in, outsourcing what its own depots cannot cover. Does it matter whether the depots are chosen against a demand forecast or against the distribution of demand, and how would you know?

The project compares three ways of choosing the depots and scores every design on the same 50,000 fresh weeks:

1. **Mean forecast** — the deterministic plan against average weekly demand (also against the true mean, as an oracle).
2. **Zone-wise quantiles** — the same plan against each zone's 50th, 75th, 90th and 95th percentile.
3. **Sample average approximation (SAA)** — the two-stage stochastic program with 104 past weeks as equally likely scenarios.

It then re-solves SAA on ten independent histories of 10, 26, 52, 104 and 208 weeks to see how stable the chosen design is, how noisy the in-sample cost is, and to build a statistical optimality gap (Mak, Morton & Wood, 1999).

## Model

```
minimise    Σ_j f_j y_j  +  (1/S) Σ_s [ Σ_ij c_ij x_sij  +  p Σ_i z_si ]

subject to  Σ_j x_sij + z_si  =  d_si          every week s, every zone i
            Σ_i x_sij        ≤  u_j y_j        every week s, every depot j
            y_j ∈ {0, 1},   x_sij ≥ 0,   z_si ≥ 0
```

`y_j` leases depot j, `x_sij` ships from depot j to zone i in week s, `z_si` is outsourced at `p` = €140 per pallet. With a single row of demand (a point forecast) the same model is the deterministic plan.

## Data

Synthetic and seeded (seed `20261015`), so every design can be scored against the true demand process. Ten candidate depots (300–450 pallets a week, €3,570–5,860 a week in lease), thirty customer zones in three regions, transport at €6 + €0.90 per km. Weekly demand is each zone's mean multiplied by three independent mean-one lognormal shocks: common (σ = 0.15), regional (σ = 0.25) and zone-level (σ = 0.20).

## Run

From this folder:

```bash
python -m pip install -e ".[dev]"
pytest                          # model and consistency checks, ~10 s
cd experiments
python run_experiment.py        # ~6 min on two CPU cores
python make_figure.py           # draws Figure 1 from results/
```

Both scripts accept `--quick` for a reduced smoke run; quick runs write `*_quick.*` files and never overwrite the full results.

## Outputs

| Path | Contents |
|---|---|
| `src/depotsaa/instance.py` | the seeded instance and the weekly demand process |
| `src/depotsaa/model.py` | the two-stage model in linopy (solved by HiGHS) and the weekly recourse LP in highspy, re-solved from the previous basis |
| `experiments/run_experiment.py` | plans, scoring on 50,000 weeks, stability study, optimality gap |
| `experiments/make_figure.py` | Figure 1, drawn only from `results/` |
| `results/run_log.txt` | everything the full run printed; every number in the article is here |
| `results/summary.json` | the same numbers in machine-readable form |
| `results/weekly_costs.npz` | weekly cost of the mean-forecast and SAA designs on the 50,000 test weeks |
| `figures/fig1_planned_vs_realised.png` | Figure 1 |
| `tests/` | the checks below |

## What the tests check

* The instance is reproducible from its seed, and demand has the stated mean.
* The deterministic plan's objective equals lease cost plus the recourse LP at mean demand.
* The SAA objective equals the independent highspy re-scoring of the chosen design on the same weeks (two implementations of one quantity).
* Recourse cost at mean demand is no larger than its average over the weeks (Jensen's inequality for a convex LP value function).
* Adding the textbook linking rows `x ≤ d·y` does not change the optimum.

## Automation

`.github/workflows/operational-research-2026.yml` runs the tests and the quick versions of the experiment and figure when files in this folder change.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt.
