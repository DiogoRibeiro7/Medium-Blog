# Reproducibility audit — Parts 16–18 and 20

Audit date: 2026-10-23.

These four Monte Carlo experiments have now been independently reproduced using the source equations and seed from the repository, with NumPy 2.3.5, SciPy and scikit-learn in a local Python environment (`OPENBLAS_NUM_THREADS=1`). The archived results were compared at the reported precision.

| Part | Scenarios | Reproductions | Status |
| --- | ---: | ---: | --- |
| 16 — Unmeasured confounding sensitivity | 4 strengths × 2 estimators | 400 datasets per strength | PASS |
| 17 — Negative controls / placebo | 3 causal structures × 3 tests | 500 datasets per structure | PASS |
| 18 — Missing outcomes | 3 observation mechanisms × 4 estimators | 300 datasets per mechanism | PASS |
| 20 — Treatment allocation | 4 policy rules | 500 independent training/test experiments | PASS |

Selected independently verified values:

| Part | Metric | Reproduced mean | Archived mean |
| --- | --- | ---: | ---: |
| 16 | Observed-only, hidden strength 1.0 | 1.7492655921 | 1.7492655921 |
| 16 | Oracle-adjusted, strength 1.0 | 1.0008521670 | 1.0008521670 |
| 17 | Shared confounder, negative control | 0.5010683333 | 0.5010683333 |
| 17 | Outcome-specific confounder, negative control | -0.0001512614 | -0.0001512614 |
| 18 | MAR complete case | 0.9956588372 | 0.9956588372 |
| 18 | MNAR response IPW | 0.9685150427 | 0.9685150427 |
| 20 | Baseline-risk policy value per treated person | -1.6002624802 | -1.6002624802 |
| 20 | Estimated-benefit policy value | 2.5989457476 | 2.5989457476 |

All reported mean and RMSE values for Part 16, all mean/RMSE/rejection values for Part 17, all mean/RMSE combinations for Part 18 and all mean/SD policy rankings for Part 20 matched the stored CSV to displayed precision.

## Scope

The simulations were reconstructed from the current GitHub-connected source definitions and independently run locally. This is not a byte-for-byte execution of an entire repository checkout or a CI certification. The validation checks reproducibility of the specific numerical comparisons, not identification assumptions or bibliography. Part 19 was corrected separately in PR #248 and is outside this four-part numerical rerun.

## Next audit phase

A numerical result matching its archive does not prove that every numeric assertion in the article matches the archive. Audit all 20 manuscripts against their committed CSVs, verify the scholarly references and prepare figures from reproducible source data. In particular, retain the distinction between a mathematically stipulated effect and a measured empirical effect.
