# Computational reproducibility audit — Causal Inference Parts 1–5

Date: 1 November 2026.

## Scope and environment

The committed experiment scripts for Parts 1–5 were run independently with Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. Each simulation was run in its own temporary directory, and every regenerated CSV cell matched the corresponding committed `results.csv`, including the published decimal rounding. No experiment scripts or original results were changed.

| Part | Experiment | Replications and sizes | Regenerated rows | Outcome |
| --- | --- | --- | ---: | --- |
| 1 | Unadjusted/adjusted OLS versus randomization | 1,000 at n=100, 1,000, 10,000 | 9 | Exact committed CSV match |
| 2 | Selection and randomization with HC2 errors | 1,000 at n=200, 1,000, 5,000 | 9 | Exact committed CSV match |
| 3 | Propensity-score matching and hidden confounding | 300 per hidden-confounding scenario, n=1,500 | 10 | Exact committed CSV match |
| 4 | IPW, Hájek normalization and propensity clipping | 300 per overlap-strength scenario, n=1,500 | 30 | Exact committed CSV match |
| 5 | Adjustment for confounders, colliders, mediators, irrelevant controls | 1,000 at n=200, 1,000, 5,000 | 24 | Exact committed CSV match |

## Independent verification

From the repository root, run:

```bash
python verify_foundations_results.py
python verify_foundations_results.py --recompute
```

The first command only needs the Python standard library. It checks row counts, combinations of simulation conditions, sample-size metadata, reported bias identities and coverage values where applicable. The second runs all five original experiments with the numerical dependencies installed and checks every CSV field against the committed output. Runs are isolated: the source and published result files are not overwritten.

The comparator explicitly understands the intentional `nan` entries in Part 4, where an RMSE of an effect estimator is not meaningful for diagnostic quantities such as effective sample size or maximum weight. It also preserves rounding tolerance for Parts 1–4, which print four-decimal summaries.

## Scientific interpretation and limits

- Part 1 isolates confounding bias relative to both adjustment and randomized treatment.
- Part 2 demonstrates why treatment selection can reverse the sign of an observational association, while randomization and covariate adjustment recover the simulated effect.
- Part 3 explicitly includes hidden confounding; a close propensity-score match on observed covariates need not remove bias from an unobserved variable.
- Part 4 distinguishes weight instability from trimming-induced bias under deteriorating overlap; the diagnostics are not themselves treatment-effect estimates.
- Part 5 contrasts adjustment sets across distinct causal structures. The total-effect target changes in the mediator scenario, so the conditional direct coefficient should not be treated as the same estimand.

Reproducing the same seeded experiment establishes computational reproducibility in this environment. It does not substitute for evaluating sensitivity to different seeds, data-generating processes, misspecified propensity models, alternative coverage definitions, or finite-sample uncertainty.
