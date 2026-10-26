# Erlang A plus a learned correction: outputs, inputs or structure

Companion code for a Medium article on hybrid generators: a mechanistic queueing model with a learned correction added in one of three places, compared on a simulated contact centre whose truth is known.

It builds on the contact-centre simulator in [`../mechanistic-vs-data-driven-generators`](../mechanistic-vs-data-driven-generators) (package `gentradeoffs`), which the scripts here put on the import path. The world, the interval table the analyst sees and Erlang A's calibration are the same as there: reality has a day-level demand shock, log-normal handle times, a patience mixture and callers who redial; Erlang A has none of them.

## The hybrids (`src/hybridgen`)

- **Output discrepancy** (`discrepancy.py`, `hybrid.OutputHybrid`): Kennedy and O'Hagan's formulation, observations = Erlang A's expected outcome + a learned function of the inputs. The function is learned on the residuals of the service level, abandonment share and mean wait, with boosting, a zero-mean Gaussian process or a ridge regression, on the additive scale (and, for boosting, also on the logit/log scale).
- **Input correction** (`hybrid.EmpiricalDayDemand`): the Poisson demand model's day levels are resampled from the day levels in the data.
- **Structural slot** (`hybrid.GreyBox`): redials are added to Erlang A with a probability estimated from the interval table by indirect inference. The auxiliary statistic is the within-day slope of surplus offered calls on the previous half hour's abandonments; the mean delay before a redial is an assumption.

## Experiments

| Script | Question |
|--------|----------|
| `experiments/h1_fit.py` | Prediction of held-out days from their inputs, by training length; what each correction does beyond the data; discriminator AUC of generated days. |
| `experiments/h2_surge.py` | Service level when fresh demand is 10–50% above normal, for every hybrid, and for Erlang A given reality's own surge inputs. |
| `experiments/h3_redials.py` | The redial probability estimated from 30 to 810 days, in a world with no redials, and under a wrong delay assumption. |
| `experiments/h4_confounding.py` | Erlang A with its patience fixed at values from 100 to 450 s, with and without a learned correction: fit and surge forecast. |
| `experiments/h5_intervention.py` | The effect of adding or removing agents, each method used as a generator. |
| `experiments/h6_decision.py` | Staffing for a launch whose surge is uniform on 1.1–1.3. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
cd experiments
python run_all.py          # about an hour on two cores
python make_figures.py
```

`run_all.py --quick` runs everything at tiny sizes in a few minutes; CI does that on every change. Every number quoted in the article is in `results/summary.txt`; raw results are in `results/*.json`. All randomness flows from one master seed through `numpy.random.SeedSequence`.

## Caveats

- Reality is a simulator, and the structural slot tested here is the mechanism that the earlier experiment identified as the one that matters. A slot for the wrong mechanism is not tested beyond a wrong delay.
- The Gaussian process is fitted on at most 400 rows per target.

## License

Apache-2.0, as the rest of the repository.
