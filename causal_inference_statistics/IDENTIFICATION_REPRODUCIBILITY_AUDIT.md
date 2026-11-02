# Computational reproducibility audit — Causal Inference Parts 6–10

Audit performed: 2 November 2026.

The original seeded experiments for Parts 6–10 were independently executed in temporary directories using Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and scikit-learn 1.8.0. All 48 regenerated CSV rows agreed with the committed outputs, including individual numeric cells. The original experiment sources and their published results were not modified.

| Part | Method | Experimental conditions | Rows | Check |
| --- | --- | --- | ---: | --- |
| 6 | Difference-in-differences | parallel trend, post-period differential trend, anticipation | 3 | Passed |
| 7 | Instrumental variables | strong, moderate, weak, and exclusion-violating instruments; OLS/IV | 8 | Passed |
| 8 | Regression discontinuity | three data-generating scenarios × four bandwidths | 12 | Passed |
| 9 | Synthetic controls | stable, latent post-shift, anticipation × three estimators | 9 | Passed |
| 10 | Cross-fitted augmented IPW | correct/incorrect propensity and outcome nuisance models × four estimators | 16 | Passed |

## Reproduction commands

From repository root:

```bash
python verify_identification_results.py
python verify_identification_results.py --recompute
```

The default mode validates committed table structure, unique simulation conditions, replication counts, causal targets, bias identities and relevant probability/quantile ranges, using only the standard library. The full mode requires NumPy, SciPy and scikit-learn, executes the complete Monte Carlo experiments, and compares every regenerated CSV field against its committed counterpart to a tolerance of `1e-8`. Each experiment runs from its own temporary copy to avoid overwriting published results.

## Scientific interpretation

- Part 6: A pre-treatment placebo comparison is not a test of the unobserved post-treatment parallel-trends counterfactual. In particular, a differential trend appearing only after treatment can evade a pre-period placebo.
- Part 7: Instrument strength is distinct from the exclusion restriction. Even a strong first stage does not establish instrument validity; weak-instrument finite-sample bias must be interpreted separately.
- Part 8: Numerical stability across bandwidths is evidence about this specific simulation, not proof that continuity or no-sorting assumptions hold for observational data.
- Part 9: A good pre-period synthetic-control fit is necessary but not sufficient to justify the post-treatment counterfactual. Latent shifts and anticipation remain threats.
- Part 10: Double robustness depends on at least one correctly specified nuisance component, positivity, and appropriate estimation conditions. Cross-fitting reduces overfitting-related bias but does not repair violations of identification assumptions.

Exact computational reproduction establishes a dependable baseline for the article's numerical examples. It does not establish external validity or sensitivity robustness across different random seeds, environments, data-generating mechanisms or model specifications.
