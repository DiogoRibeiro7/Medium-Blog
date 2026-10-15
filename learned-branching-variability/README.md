# Learned branching against performance variability

Companion code for the Medium article *Machine Learning Inside Optimisation Solvers: The State of the Art and a Seed-Controlled Test of Learned Branching*.

## Question

A learned branching rule for a mixed-integer solver is usually reported as a speed-up ratio against the solver's default rule, averaged over a few dozen instances. Mixed-integer solvers also change their run time when nothing but the random seed changes (performance variability). How large is a learned rule's measured gain compared with the "gain" the default solver shows against itself under a different seed, and does the gain survive moving to larger instances than the ones it was trained on?

## Design

* **Instances.** Random set covering in the Balas–Ho style used by Gasse et al. (2019): 500 rows × 1000 columns for training, validation and the first test set; 750 × 1000 for the second test set. Every split has its own seed range, so no test instance is seen in training.
* **Solver.** SCIP 10 through PySCIPOpt, cuts at the root only, no restarts, one thread, as in Gasse et al. The solver seed changes SCIP's random seed shift and permutes variables and constraints.
* **Expert data.** On 80 training and 20 validation instances, the default solver drives the search; at a random half of its nodes (at most 40 per instance) every candidate is scored by full strong branching without changing the solver state. A child whose bound reaches the cutoff counts as pruned, and gains are capped at the distance to the cutoff.
* **Learned rule.** A gradient-boosted regressor (scikit-learn) on fourteen cheap candidate features predicts each candidate's strong-branching score relative to the node's best; the rule branches on the highest prediction.
* **Evaluation.** Every test instance is solved with SCIP's default (reliability pseudocost branching), plain pseudocost branching and the learned rule under five solver seeds, and with full strong branching under two seeds on the training-size set.
* **Statistics.** Shifted geometric means (CPU seconds, shift 1; nodes, shift 100); paired instance-level ratios against the default with bootstrap 95% intervals over instances; the same ratio one seed at a time; and a placebo, the default against itself under every other seed.

## Results in brief

From `results/e2_log.txt` and `results/e4_log.txt` (CPU time ratios are paired shifted geometric means over 20 instances × 5 seeds, with 95% bootstrap intervals over instances):

| | 500 × 1000 (training size) | 750 × 1000 (larger) |
|---|---|---|
| Learned vs SCIP default, CPU | ×0.75 [0.71, 0.78] | ×0.76 [0.70, 0.84] |
| Pseudocost vs SCIP default, CPU | ×0.73 [0.70, 0.76] | ×0.79 [0.72, 0.88] |
| Learned vs pseudocost, CPU | ×1.02 [0.99, 1.04] | ×0.96 [0.93, 0.98] |
| Learned vs pseudocost, nodes | ×0.78 [0.73, 0.83] | ×0.72 [0.69, 0.74] |
| Learned vs pseudocost, CPU without the Python callback | ×0.94 [0.91, 0.97] | ×0.84 [0.81, 0.88] |
| Placebo: default vs default, other seed (range of 20 seed pairs) | 0.95–1.06 | 0.97–1.03 |
| Default, slowest/fastest seed on one instance (median, worst) | 1.22×, 1.88× | 1.13×, 1.28× |

On held-out nodes the learned scorer picks a candidate with the best strong-branching score at 34.8% of nodes (pseudocost score 27.1%, random 5.5%).

## Run

From this folder:

```bash
python -m pip install -e ".[dev]"
pytest                          # generators, features, strong branching, statistics
cd experiments
python e1_collect.py            # strong-branching data, ~22 min on two cores
python e2_train.py              # fit and validate the scorer, seconds
python e3_evaluate.py           # 640 solves, ~75 min on two cores (resumable)
python e4_analyse.py            # statistics and figures
```

Every script accepts `--quick` for a smoke run; quick runs write `*_quick.*` files and never overwrite the full results.

## Outputs

| Path | Contents |
|---|---|
| `src/learnbranch/instances.py` | set covering and capacitated facility location generators, PySCIPOpt models |
| `src/learnbranch/branching.py` | candidate features, capped strong-branching scores, the data collector and the learned rule |
| `src/learnbranch/solve.py` | solver settings, seeds, one solve under one policy |
| `experiments/` | the four steps above and `config.py` |
| `results/e1_log.txt`, `e2_log.txt`, `e4_log.txt` | everything the full run printed |
| `results/e2_metrics.json`, `e4_summary.json` | the same numbers in machine-readable form |
| `results/e3_runs.csv` | one row per solve: instance, policy, seed, CPU and wall time, nodes, status |
| `figures/` | the article's figures, drawn only from `results/` |

The strong-branching samples (`results/samples_*.npz`) and the fitted model are not committed; `e1_collect.py` and `e2_train.py` rebuild them deterministically.

## What the tests check

* Generated set-covering instances cover every row; facility capacities match the requested ratio.
* Features are finite and of the documented shape, and strong branching leaves the node's LP bound unchanged.
* A branching rule driven by a stand-in predictor reaches the same optimum as the default solver, and recording samples does not change the answer.
* The shifted geometric mean, the paired ratio (recovers a planted effect) and the placebo (symmetric) behave as specified.

## Automation

`.github/workflows/learned-branching-variability.yml` runs the tests and the quick version of the whole pipeline when files in this folder change.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt.
