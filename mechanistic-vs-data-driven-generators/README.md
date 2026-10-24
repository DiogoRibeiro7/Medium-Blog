# Mechanistic and data-driven synthetic data generators, compared on a simulated contact centre

Companion code for the Medium article *Mechanistic and Data-Driven Synthetic Data Generators: The Trade-Offs, Measured on a Simulated Contact Centre*.

A data-driven generator estimates the distribution of the data you have; a mechanistic generator simulates the process that produces it. To measure what each kind gets right and wrong, this folder builds a contact centre whose truth is known, gives six generators the same interval table that a real centre would export, and asks them the same questions.

## The setup

**Reality** (`src/gentradeoffs/world.py`) is an event-by-event simulation of a contact centre open 08:00–20:00 on weekdays, reported in half-hour intervals. It is deliberately richer than the models fitted to it: fresh calls are Poisson within an interval but with a day-level and an interval-level shock on top of the intraday profile, weekday effect and announced campaign days; handle times are log-normal; patience is a mixture (a fifth of callers hang up quickly, the rest are log-normal); six in ten callers who hang up call again about ten minutes later; and the planners staff by a square-root rule on their forecast, revise it during the day when volume runs above or below forecast, and lose agents to absence. The queue itself (`src/gentradeoffs/queue.py`) is an exact first-come-first-served simulation, compiled with numba.

**The data** the generators see is the table a call-centre platform exports: one row per day and half hour with weekday, interval, campaign flag, offered calls, agents, abandonment share, service level (share answered within 20 s), mean time waited per call, and mean handle time.

**The generators** are fitted on that table only:

- mechanistic (`src/gentradeoffs/mechanistic.py`): Erlang C (exponential service, nobody hangs up) and Erlang A (exponential patience, rate estimated as abandonments over total waiting time), each with a Poisson demand model fitted to the table and the company's own staffing rule;
- data-driven (`src/gentradeoffs/data_driven.py`): a Gaussian copula with empirical marginals, sequential CART in the style of synthpop, SMOTE-style interpolation, and a full-covariance Gaussian mixture with the number of components chosen by BIC.

## Experiments

| Script | Question |
|--------|----------|
| `experiments/e1_fidelity.py` | How easily a classifier tells synthetic rows from held-out real ones, by length of training data. |
| `experiments/e2_surge.py` | What each generator says about fresh demand 10–50% above normal; Erlang A with redials of unknown probability; which of the analyst's assumptions carries the error. |
| `experiments/e3_intervention.py` | The effect of adding or removing agents, against the effect obtained by replaying the same callers. |
| `experiments/e4_error_budget.py` | Error in the long-run service level as the synthetic sample grows and as the real data grows. |
| `experiments/e5_calibration.py` | Three consistent estimators of Erlang A's patience rate and the different values they converge to. |
| `experiments/e6_privacy_cost.py` | Distance from synthetic rows to the closest training row; fitting and sampling time. |
| `experiments/e7_decision.py` | Staffing for a surge of uncertain size, with a point model, domain randomisation, a scenario-robust rule and a copula. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
cd experiments
python run_all.py          # about forty minutes on two cores
python make_figures.py
```

`run_all.py --quick` runs everything at tiny sizes in a couple of minutes; CI does that on every change. Every number quoted in the article is in `results/summary.txt`, and each experiment's raw output is in `results/*.json`. All randomness flows from one master seed through `numpy.random.SeedSequence`, so a rerun with the pinned versions reproduces the results on the same machine; across machines, floating-point differences in the compiled queue and in scikit-learn can move the last digits, which is why CI checks that the pipeline runs rather than diffing the numbers.

## Caveats

- Reality here is itself a simulator. It is richer than the generators fitted to it and the analyst never sees its parameters, but the conclusions are about this world, not about every contact centre.
- The data-driven generators are written out in full rather than taken from a library, so that their behaviour at the edge of the data is visible; neural synthesizers such as CTGAN or diffusion models are not included.
- Timings in `summary.txt` depend on the machine.

## License

Apache-2.0, as the rest of the repository.
