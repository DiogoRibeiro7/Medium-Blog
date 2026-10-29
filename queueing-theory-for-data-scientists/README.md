# Queueing Theory for Data Scientists: Why Accurate Demand Forecasts Still Produce Long Waits

Companion research for the Medium article [`article.md`](article.md).

## Question

If two systems have the same average arrival rate and average time per job,
should they have the same waiting time? No. The experiment isolates service
variability, arrival variability, utilization, and capacity. It compares an
event-by-event FCFS simulator with stationary queueing formulas, measures
long waits as well as means, and reports uncertainty between independent
simulation repetitions.

The analysis is intentionally narrower than a general call-centre model:
there is no abandonment, redial, changing arrival rate, shift schedule,
priority, or forecast estimation error. Those topics have separate
implementations in this repository:

- [`../mechanistic-vs-data-driven-generators/`](../mechanistic-vs-data-driven-generators/) —
  a richer simulated contact centre and a comparison of mechanistic with
  data-driven generators;
- [`../anonymous-bank-redials/`](../anonymous-bank-redials/) —
  real call-centre records and the inference problem created by redials.

## Model and scope

- A GI/G/c queue with infinite waiting room and FCFS, no priorities or losses.
- Baseline: 0.8 arrivals/min, mean service of 1 minute, one server
  (utilization 80%).
- Six scenarios: M/M/1; M/D/1; M/G/1 with lognormal service CV² = 4;
  GI/M/1 with hyperexponential renewal interarrivals CV² = 4; M/M/1 with
  15% shorter service; and M/M/2 with the original service distribution.
- Sixteen independent simulation repetitions, 12,000 warm-up arrivals plus
  60,000 measured arrivals per repetition; the same random arrival trace is
  used for each Poisson-arrival scenario within a repetition.
- Additional utilization sweep from 0.50 to 0.95 for three service-time
  distributions, with longer runs near saturation to reduce start-empty bias.
- Four complementary figure diagnostics: independently sampled input
  distributions, a single seeded queue trace, policy outcomes across three
  service metrics, and a separate run-length convergence study.
- Queue wait excludes service time. Figures show queue wait, not end-to-end
  system time, unless otherwise stated.

The values are **synthetic**, not observations from a real business.
Mathematical benchmark values are exact only under their stated assumptions.
The GI/M/1 benchmark is Kingman's heavy-traffic **approximation**.
Simulation confidence intervals use between-repetition standard errors and
do not correct model misspecification or residual finite-run bias.

## Reproduce

From this directory with Python 3.12:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis.py                  # full study and publication figures
python analysis.py --quick          # smoke run, separate *_quick files
```

The notebook [`queueing_theory.ipynb`](queueing_theory.ipynb) includes a
readable derivation, a hand-worked queue, and a quick executable replication.
It is executed in a fresh Python kernel in the project-specific GitHub Actions
workflow. It leaves full-study results untouched.

## Artifact contract

| File | Purpose |
| --- | --- |
| `article.md` | Medium-ready manuscript, numerical claims and references |
| `analysis.py` | Typed, seeded simulator, exact baselines, CSV/figure generation |
| `queueing_theory.ipynb` | Executable narrative notebook |
| `results/summary.csv` | Main comparison: mean wait, CI, p95, delay probability |
| `results/tails.csv` | Waiting-time exceedance curves |
| `results/sensitivity.csv` | Analytic and simulated utilization sweep |
| `results/input_variability.csv` | Sampled input survival curves (200,000 draws per distribution) |
| `results/queue_trace.csv` | Two 500-arrival sample paths, shared service draws |
| `results/convergence.csv` | Run-length sensitivity with sixteen independent repetitions |
| `figures/fig1_same_mean_different_wait.png` | Queue means and analytic comparison |
| `figures/fig2_utilization_cliff.png` | Nonlinear response to utilization |
| `figures/fig3_waiting_tails.png` | Waiting-time tail by scenario |
| `figures/fig4_input_variability.png` | Service/interarrival input distributions |
| `figures/fig5_queue_memory.png` | Burst and backlog dynamics on one trace |
| `figures/fig6_capacity_outcomes.png` | Three operational metrics under staffing policies |
| `figures/fig7_monte_carlo_convergence.png` | Uncertainty across simulation run lengths |
| `tests/test_analysis.py` | Analytical identities, trace invariants, notebook smoke |

Generated `*_quick` files are smoke-test byproducts and should not be used
as article evidence. Commit the full results/figures after running the full
command. The main seed is `20261029`, with independent diagnostic seed
namespaces; exact floating-point outputs may vary by
platform and dependency version, while the main quantitative relationships
must remain stable.

## Methodological limits

The hyperexponential interarrival times are iid: they introduce burstiness
but **not** serial correlation or a time-varying rate. The lognormal service
scenario has finite variance, not an infinite-variance heavy-tail model.
Warm-up is based on arrival counts, which need not remove all transient bias
near saturation. In operational deployments, balking, abandonment,
retries, priorities, service interruptions, shift changes, and uncertain
forecasts often dominate this pedagogical model.

The manuscript reports a designed experiment about mechanisms. It makes no
causal claim about a real organization.
